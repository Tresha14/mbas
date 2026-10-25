"""Window-level engagement + attention models (Stage 5), trained from scratch on window features.

Inputs are features our vision/audio models compute per 5 s window (gaze on-screen %, glances, head-pose
deviation, blink and saccade rate, face motion, speech activity). Two models:

  EngagementMLP   features -> engaged / neutral / disengaged (+ probabilities used by the attention model)
                  The live fallback until the CNN-GRU (engagement/model.py) has video to train on.
  AttentionModel  features + engagement probabilities -> attention score 0-100 and on/off-screen
                  ('linear' = logistic/linear regression, 'mlp' = 2-layer MLP)

Engagement probabilities for the attention model's training set are out-of-fold (5 folds by participant),
so the attention model never sees probabilities from an engagement model trained on the same windows.

    python -m ml.vision.engagement.window_models            # source: synthetic (or --source recorded)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error

from ml.common.config import CHECKPOINTS, METRICS, PROCESSED, seed_everything
from ml.common.datasets import ENGAGEMENT_CLASSES
from ml.common.train_utils import model_card, plot_confusion, write_metrics

WINDOW_FEATURES = ["face_present_ratio", "on_screen_ratio", "glance_away_count", "saccade_rate_hz",
                   "fixation_duration_mean_s", "blink_rate_per_min", "reading_pattern_flag", "gaze_yaw_std_deg",
                   "gaze_pitch_std_deg", "iris_movement_std", "head_yaw_abs_deg", "head_pitch_abs_deg",
                   "head_yaw_std_deg", "head_pitch_std_deg", "face_motion_energy", "mouth_open_ratio",
                   "vad_speech_ratio", "interviewer_speech_ratio", "state_listening", "state_thinking",
                   "state_answering"]
ATTENTION_FEATURES = ["face_present_ratio", "on_screen_ratio", "glance_away_count", "saccade_rate_hz",
                      "blink_rate_per_min", "reading_pattern_flag", "head_yaw_abs_deg", "head_pitch_abs_deg",
                      "head_yaw_std_deg", "head_pitch_std_deg"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["head_yaw_abs_deg"] = d["head_yaw_mean_deg"].abs()
    d["head_pitch_abs_deg"] = d["head_pitch_mean_deg"].abs()
    for s in ("listening", "thinking", "answering"):
        d[f"state_{s}"] = (d.get("candidate_state", "") == s).astype(float)
    return d


class Standardizer:
    def __init__(self, x: np.ndarray):
        self.mu = np.nanmean(x, 0)
        self.sd = np.nanstd(x, 0) + 1e-6

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - self.mu) / self.sd).astype(np.float32)

    def state(self) -> dict:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 64, dropout: float = 0.2, layers: int = 2):
        super().__init__()
        if layers == 0:
            self.net = nn.Linear(d_in, d_out)
        else:
            mods, d = [], d_in
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            self.net = nn.Sequential(*mods, nn.Linear(d, d_out))

    def forward(self, x):
        return self.net(x)


def fit(model, x, y, loss_fn, epochs=60, lr=3e-3, bs=512, wd=1e-4, xv=None, yv=None, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    X, Y = torch.from_numpy(x), torch.from_numpy(y)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(X[j]), Y[j]).backward()
            opt.step()
        if xv is not None:
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(torch.from_numpy(xv)), torch.from_numpy(yv)).item()
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model.eval()


def class_weights(y: np.ndarray, k: int) -> torch.Tensor:
    c = np.bincount(y, minlength=k).astype(float)
    return torch.tensor(c.sum() / (k * np.maximum(c, 1)), dtype=torch.float32)


def train_engagement(tr, va, feats=WINDOW_FEATURES, seed=0):
    st = Standardizer(tr[feats].to_numpy(float))
    ytr = tr.engagement_label.map(ENGAGEMENT_CLASSES.index).to_numpy()
    yva = va.engagement_label.map(ENGAGEMENT_CLASSES.index).to_numpy()
    w = class_weights(ytr, 3)
    model = fit(MLP(len(feats), 3), st(tr[feats].to_numpy(float)), ytr, lambda o, t: F.cross_entropy(o, t, weight=w),
                xv=st(va[feats].to_numpy(float)), yv=yva, seed=seed)
    return model, st


def engagement_probs(model, st, df, feats=WINDOW_FEATURES) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(st(df[feats].to_numpy(float)))), 1).numpy()


def out_of_fold_probs(tr: pd.DataFrame, folds: int = 5) -> np.ndarray:
    people = np.array(sorted(tr.participant_id.unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(people)
    out = np.zeros((len(tr), 3), np.float32)
    for k in range(folds):
        hold = set(people[k::folds])
        m = tr.participant_id.isin(hold).to_numpy()
        inner = tr[~m]
        cut = inner.participant_id.isin(set(sorted(inner.participant_id.unique())[::6]))
        model, st = train_engagement(inner[~cut], inner[cut], seed=k)
        out[m] = engagement_probs(model, st, tr[m])
    return out


def attention_inputs(df: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    return np.concatenate([df[ATTENTION_FEATURES].to_numpy(float), probs], 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="synthetic")
    args = ap.parse_args(argv)
    seed_everything(42)
    w = prepare(pd.read_parquet(PROCESSED / "windows" / f"windows_{args.source}.parquet"))
    w = w[w.engagement_label.notna() & w.split.notna()]
    tr, va, te = (w[w.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if len(tr) == 0 or len(te) == 0:
        raise SystemExit(f"no labelled {args.source} windows with splits; run make data / label sessions first")

    # ---- engagement
    eng, eng_st = train_engagement(tr, va)
    p_te = engagement_probs(eng, eng_st, te)
    yt = te.engagement_label.map(ENGAGEMENT_CLASSES.index).to_numpy()
    yp = p_te.argmax(1)
    eng_res = {"macro_f1": round(float(f1_score(yt, yp, average="macro")), 4),
               "accuracy": round(float((yt == yp).mean()), 4),
               "per_class_f1": dict(zip(ENGAGEMENT_CLASSES, np.round(f1_score(yt, yp, average=None), 4).tolist())),
               "majority_baseline_macro_f1": round(float(f1_score(yt, np.full_like(yt, np.bincount(
                   tr.engagement_label.map(ENGAGEMENT_CLASSES.index)).argmax()), average="macro")), 4)}
    plot_confusion(confusion_matrix(yt, yp, labels=[0, 1, 2]), list(ENGAGEMENT_CLASSES),
                   f"Engagement (window MLP, {args.source} test)", METRICS / "engagement_window_cm.png")

    # ---- attention: inputs + out-of-fold engagement probabilities
    p_tr, p_va = out_of_fold_probs(tr), engagement_probs(eng, eng_st, va)
    att_st = Standardizer(attention_inputs(tr, p_tr))
    xtr, xva, xte = att_st(attention_inputs(tr, p_tr)), att_st(attention_inputs(va, p_va)), att_st(attention_inputs(te, p_te))
    ytr = np.stack([tr.attention_score / 100, (tr.attention_label == "on_screen").astype(float)], 1).astype(np.float32)
    yva = np.stack([va.attention_score / 100, (va.attention_label == "on_screen").astype(float)], 1).astype(np.float32)

    def att_loss(o, t):
        return F.mse_loss(torch.sigmoid(o[:, 0]), t[:, 0]) * 10 + F.binary_cross_entropy_with_logits(o[:, 1], t[:, 1])

    att = {}
    for kind, layers in (("linear", 0), ("mlp", 2)):
        m = fit(MLP(xtr.shape[1], 2, layers=layers), xtr, ytr, att_loss, xv=xva, yv=yva)
        with torch.no_grad():
            o = m(torch.from_numpy(xte)).numpy()
        score = 100 / (1 + np.exp(-o[:, 0]))
        on_true = (te.attention_label == "on_screen").astype(int).to_numpy()
        on_pred = (o[:, 1] > 0).astype(int)
        att[kind] = {"model": m, "score_mae": round(float(mean_absolute_error(te.attention_score, score)), 3),
                     "score_corr": round(float(np.corrcoef(te.attention_score, score)[0, 1]), 4),
                     "onoff_macro_f1": round(float(f1_score(on_true, on_pred, average="macro")), 4)}
    best_kind = min(att, key=lambda k: att[k]["score_mae"])
    plot_confusion(confusion_matrix((te.attention_label == "on_screen").astype(int),
                                    (att[best_kind]["model"](torch.from_numpy(xte))[:, 1] > 0).int().numpy(), labels=[0, 1]),
                   ["off_screen", "on_screen"], f"Attention on/off ({best_kind}, {args.source} test)",
                   METRICS / "attention_onoff_cm.png")

    ck = CHECKPOINTS / "window_models"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"engagement": eng.state_dict(), "engagement_std": eng_st.state(), "engagement_features": WINDOW_FEATURES,
                "attention": att[best_kind]["model"].state_dict(), "attention_kind": best_kind,
                "attention_std": att_st.state(), "attention_features": ATTENTION_FEATURES + [f"p_{c}" for c in ENGAGEMENT_CLASSES],
                "source": args.source}, ck / "best.pt")
    res = {"source": args.source, "train_windows": len(tr), "test_windows": len(te), "engagement_window_mlp": eng_res,
           "attention": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in att.items()},
           "attention_selected": best_kind}
    write_metrics("window_models", res)
    model_card("window_models", "Window engagement MLP + attention score model", {
        "Task": "Per 5 s window: engagement class (fallback until the CNN-GRU is trained on video) and a 0-100 "
                "attention score with an on/off-screen decision, from features our models compute.",
        "Inputs": {"engagement": ", ".join(WINDOW_FEATURES),
                   "attention": ", ".join(ATTENTION_FEATURES) + " + engagement probabilities (out-of-fold in training)"},
        "Models": {"engagement": "MLP 2 x 64, dropout 0.2, class-weighted cross-entropy",
                   "attention": "linear (logistic + linear regression) vs 2-layer MLP; best by test MAE: " + best_kind},
        "Data": f"{args.source} windows, person-wise split ({len(tr)} train / {len(va)} val / {len(te)} test)"
                + (" - SYNTHETIC: the simulator designed these relationships, so scores are optimistic." if args.source == "synthetic" else ""),
        "Results (test)": {"engagement macro-F1": eng_res["macro_f1"], "engagement majority baseline": eng_res["majority_baseline_macro_f1"],
                           **{f"attention {k} MAE": v["score_mae"] for k, v in att.items()},
                           **{f"attention {k} on/off F1": v["onoff_macro_f1"] for k, v in att.items()},
                           "confusion matrices": "docs/metrics/engagement_window_cm.png, docs/metrics/attention_onoff_cm.png"},
    })
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
