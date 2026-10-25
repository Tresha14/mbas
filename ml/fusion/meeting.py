"""Meeting head: per-participant engagement + contribution balance; trained on synthetic meetings,
generalisation test on AMI (real meetings, audio only).

Contribution balance has a closed-form reference (1 - |share - 1/n| * n / 2), so on AMI we can measure the
drop for that output exactly; AMI has no engagement labels, so only its distribution is reported there.

    python -m ml.fusion.meeting
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ml.common.config import CHECKPOINTS, PUBLIC, SPLITS, SYNTHETIC, seed_everything
from ml.common.train_utils import model_card, write_metrics
from ml.fusion.model import MEETING_FEATURES, MEETING_FEATURES_ABSOLUTE, MeetingHead
from ml.vision.engagement.window_models import Standardizer, fit


def add_relative(t: pd.DataFrame, turns: str = "n_turns", avg: str = "avg_turn_s") -> pd.DataFrame:
    t = t.copy()
    tot = t.groupby("meeting_id")[turns].transform("sum").clip(lower=1)
    t["turn_share_vs_fair"] = t[turns] / tot * t.n_participants
    t["turn_length_vs_meeting"] = t[avg] / t.groupby("meeting_id")[avg].transform("mean").clip(lower=1e-3)
    return t


def balance_reference(share: pd.Series, n: pd.Series) -> pd.Series:
    return (1 - (share - 1 / n).abs() * n / 2).clip(0, 1)


def synthetic_meeting_table() -> pd.DataFrame:
    s = pd.read_csv(SYNTHETIC / "meeting" / "meeting_participant_summary.csv")
    m = pd.read_csv(SYNTHETIC / "meeting" / "meetings.csv")[["meeting_id", "duration_min", "n_participants"]]
    mins = pd.read_csv(SYNTHETIC / "meeting" / "meeting_participant_minutes.csv")
    agg = mins.groupby(["meeting_id", "participant_id"]).agg(
        on_screen_ratio=("on_screen_ratio", "mean"), camera_on=("camera_on", "mean"),
        face_motion_energy=("face_motion_energy", "mean"), head_yaw_std_deg=("head_yaw_std_deg", "mean"),
        multitasking_share=("multitasking_flag", "mean")).reset_index()
    t = s.merge(m, on="meeting_id").merge(agg, on=["meeting_id", "participant_id"], how="left")
    t["share_vs_fair"] = t.talk_share * t.n_participants
    t["turns_per_min"] = t.n_turns / t.duration_min
    t["avg_turn_s"] = t.talk_time_s / t.n_turns.clip(lower=1)
    t = add_relative(t)
    sp = pd.read_csv(SPLITS / "synthetic" / "meeting_participants.csv")
    return t.merge(sp[["meeting_id", "participant_id", "split"]], on=["meeting_id", "participant_id"], how="left")


def ami_table() -> pd.DataFrame:
    """AMI per speaker, with turns rebuilt exactly like our conversation analyser (segments < 1 s apart merged)."""
    from ml.behaviour.conversation import turns_from_segments

    man = pd.read_parquet(PUBLIC / "ami" / "manifest.parquet")
    segs = pd.read_parquet(PUBLIC / "ami" / "turns.parquet")
    n_turns, talk = {}, {}
    for mid, g in segs.groupby("meeting_id"):
        by = {spk: list(zip(x.start_s, x.end_s)) for spk, x in g.groupby("speaker")}
        for tr in turns_from_segments(by):
            n_turns[(mid, tr.speaker)] = n_turns.get((mid, tr.speaker), 0) + 1
            talk[(mid, tr.speaker)] = talk.get((mid, tr.speaker), 0.0) + tr.duration_s
    key = list(zip(man.meeting_id, man.speaker))
    nt = pd.Series([n_turns.get(k, 0) for k in key])
    tt = pd.Series([talk.get(k, 0.0) for k in key])
    n = man.groupby("meeting_id").speaker.transform("count")
    dur_min = man.meeting_duration_s / 60
    t = pd.DataFrame({"meeting_id": man.meeting_id, "participant_id": man.subject_id, "talk_share": man.talk_share,
                      "n_participants": n, "share_vs_fair": man.talk_share * n,
                      "turns_per_min": nt / dur_min.clip(lower=1), "avg_turn_s": tt / nt.clip(lower=1),
                      "on_screen_ratio": 0.0, "camera_on": 0.0, "face_motion_energy": 0.0, "head_yaw_std_deg": 0.0,
                      "multitasking_share": 0.0, "n_turns": nt})
    return add_relative(t[t.talk_share > 0].reset_index(drop=True))


def train_head(tr, va, feats):
    st = Standardizer(tr[feats].to_numpy(float))
    y = lambda d: d[["mean_engagement", "contribution_balance"]].to_numpy(np.float32)  # noqa: E731
    # camera-off augmentation: copies of training rows with vision features zeroed (like AMI / camera off)
    tr_off = tr.copy()
    for c in ("on_screen_ratio", "face_motion_energy", "head_yaw_std_deg", "multitasking_share"):
        tr_off[c] = 0.0
    tr_off["camera_on"] = 0.0
    tr_all = pd.concat([tr, tr_off], ignore_index=True)
    model = fit(MeetingHead(len(feats)), st(tr_all[feats].to_numpy(float)), y(tr_all), F.mse_loss, epochs=300, lr=3e-3,
                bs=64, xv=st(va[feats].to_numpy(float)), yv=y(va))
    return model, st


def score(model, st, feats, te, ami):
    with torch.no_grad():
        p = model(torch.from_numpy(st(te[feats].to_numpy(float)))).numpy()
    r = {"synthetic_test": {"participants": len(te),
                            "engagement_mae": round(float(np.abs(p[:, 0] - te.mean_engagement).mean()), 4),
                            "balance_mae": round(float(np.abs(p[:, 1] - te.contribution_balance).mean()), 4)}}
    if ami is not None:
        with torch.no_grad():
            pa = model(torch.from_numpy(st(ami[feats].to_numpy(float)))).numpy()
        ref = balance_reference(ami.talk_share, ami.n_participants).to_numpy()
        mae = float(np.abs(pa[:, 1] - ref).mean())
        r["ami"] = {"participants": len(ami), "meetings": int(ami.meeting_id.nunique()),
                    "balance_mae_vs_formula": round(mae, 4),
                    "balance_mae_drop": round(mae - r["synthetic_test"]["balance_mae"], 4),
                    "engagement_pred_mean": round(float(pa[:, 0].mean()), 4),
                    "engagement_pred_std": round(float(pa[:, 0].std()), 4)}
    return r


def main(argv: list[str] | None = None) -> int:
    seed_everything(42)
    t = synthetic_meeting_table()
    tr, va, te = (t[t.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    ami = ami_table() if (PUBLIC / "ami" / "manifest.parquet").exists() else None
    model, st = train_head(tr, va, MEETING_FEATURES)
    abs_model, abs_st = train_head(tr, va, MEETING_FEATURES_ABSOLUTE)
    res = {"relative_turn_features (used)": score(model, st, MEETING_FEATURES, te, ami),
           "absolute_turn_features (first try)": score(abs_model, abs_st, MEETING_FEATURES_ABSOLUTE, te, ami),
           "note": "AMI is audio-only here (camera_on = 0) and has no engagement labels; balance is checked against "
                   "its closed-form definition 1 - |share - 1/n| * n / 2."}
    if ami is not None:
        res["feature_ranges_5_95pct"] = {
            name: {c: [round(float(d[c].quantile(q)), 2) for q in (0.05, 0.95)]
                   for c in ("turns_per_min", "avg_turn_s", "turn_share_vs_fair", "turn_length_vs_meeting")}
            for name, d in (("synthetic_train", tr), ("ami", ami))}
    ck = CHECKPOINTS / "meeting_head"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "std": st.state(), "features": MEETING_FEATURES}, ck / "best.pt")
    write_metrics("meeting_head", res)
    used, first = res["relative_turn_features (used)"], res["absolute_turn_features (first try)"]
    model_card("meeting_head", "Meeting head (per-participant engagement + contribution balance)", {
        "Task": "Meeting mode: engagement 0-1 and contribution balance 0-1 per participant, from talk share, relative "
                "turn features and (camera on) on-screen ratio, face motion, head movement, multitasking share.",
        "Model": "MLP 2 x 32 on standardised features; trained with camera-off copies so it works audio-only.",
        "Data": "synthetic meetings (participants split person-wise); generalisation test on AMI (171 real meetings).",
        "Results": {"synthetic test engagement MAE": used["synthetic_test"]["engagement_mae"],
                    "synthetic test balance MAE": used["synthetic_test"]["balance_mae"],
                    "AMI balance MAE (relative features)": used.get("ami", {}).get("balance_mae_vs_formula"),
                    "AMI balance MAE (absolute features, first try)": first.get("ami", {}).get("balance_mae_vs_formula")},
        "Finding": "Absolute turn rate and turn length did not transfer from the simulator to real meetings (AMI turns "
                   "are 3-6x shorter); within-meeting relative turn features fixed most of the drop.",
    })
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
