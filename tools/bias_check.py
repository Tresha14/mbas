"""Bias check: model scores and errors across gender, accent and glasses groups on the TEST participants.

  answers  fusion effectiveness (predicted vs label) per group           (synthetic now; ours when labelled)
  windows  engagement-MLP accuracy and attention MAE per group
  gaze     angular error per MPIIGaze test person (no demographics in MPIIGaze; reported per person)
Writes docs/bias_check.md + docs/metrics/bias_check.json. A gap > 5 points (or > 5 % accuracy) is flagged.

    python -m tools.bias_check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.common.config import CHECKPOINTS, METRICS, PROCESSED, SYNTHETIC, save_json  # noqa: E402
from ml.common.datasets import FUSION_FEATURES, FusionDataset, fusion_table  # noqa: E402

GROUPS = ("gender", "accent", "glasses")
GAP = 5.0


def people() -> pd.DataFrame:
    return pd.read_csv(SYNTHETIC / "interview" / "participants.csv")[["participant_id", *GROUPS]]


def fusion_by_group() -> dict:
    from ml.fusion.model import FusionNet

    p = CHECKPOINTS / "fusion" / "best.pt"
    if not p.exists():
        return {"note": "fusion not trained"}
    st = torch.load(p, map_location="cpu", weights_only=False)
    m = FusionNet(d=st["cfg"]["model"]["d"], heads=st["cfg"]["model"]["heads"])
    m.load_state_dict(st["model"])
    m.eval()
    table = fusion_table("synthetic")
    stats = {k: (np.array(a), np.array(b)) for k, (a, b) in st["stats"].items()}
    te = FusionDataset("test", table=table, stats=stats)
    with torch.no_grad():
        o = m({k: torch.from_numpy(te.x[k]) for k in FUSION_FEATURES})
    d = te.t[["participant_id"]].assign(pred=o["scores"][:, 0].numpy() * 100, true=te.y[:, 0] * 100).merge(people(), on="participant_id")
    return _groups(d, lambda g: {"n": len(g), "mean_pred": round(float(g.pred.mean()), 2), "mean_label": round(float(g.true.mean()), 2),
                                  "mae": round(float((g.pred - g.true).abs().mean()), 2)}, "mae")


def windows_by_group() -> dict:
    from ml.vision.engagement.window_models import MLP, prepare

    p = CHECKPOINTS / "window_models" / "best.pt"
    if not p.exists():
        return {"note": "window models not trained"}
    st = torch.load(p, map_location="cpu", weights_only=False)
    w = prepare(pd.read_parquet(PROCESSED / "windows" / "windows_synthetic.parquet"))
    w = w[w.split == "test"].merge(people(), on="participant_id")
    eng = MLP(len(st["engagement_features"]), 3)
    eng.load_state_dict(st["engagement"])
    eng.eval()
    x = np.nan_to_num((w[st["engagement_features"]].to_numpy(float) - np.array(st["engagement_std"]["mu"])) /
                      np.array(st["engagement_std"]["sd"])).astype(np.float32)
    with torch.no_grad():
        pred = eng(torch.from_numpy(x)).argmax(1).numpy()
    w = w.assign(correct=(pred == w.engagement_label.map(["disengaged", "neutral", "engaged"].index).to_numpy()).astype(float) * 100)
    return _groups(w, lambda g: {"n": len(g), "engagement_accuracy_pct": round(float(g.correct.mean()), 2)}, "engagement_accuracy_pct")


def gaze_by_person() -> dict:
    p = METRICS / "gaze_iris.json"
    return json.loads(p.read_text()).get("angular_error_per_person_deg", {}) if p.exists() else {"note": "gaze not evaluated"}


def _groups(d: pd.DataFrame, fn, key: str) -> dict:
    out = {}
    for g in GROUPS:
        rows = {str(k): fn(x) for k, x in d.groupby(g) if len(x) >= 20}
        vals = [r[key] for r in rows.values()]
        out[g] = {"groups": rows, "max_gap": round(max(vals) - min(vals), 2) if vals else None,
                  "flag": bool(vals and max(vals) - min(vals) > GAP)}
    return out


def main(argv: list[str] | None = None) -> int:
    res = {"fusion_effectiveness": fusion_by_group(), "engagement_windows": windows_by_group(), "gaze_per_person": gaze_by_person(),
           "data": "synthetic test participants (person-wise split); rerun after our sessions are labelled"}
    save_json(res, METRICS / "bias_check.json")
    lines = ["# Bias check", "", f"Data: {res['data']}. Groups with fewer than 20 items are omitted. A gap above {GAP} "
             "points between groups is flagged.", ""]
    for sec, title in (("fusion_effectiveness", "Communication-effectiveness error (MAE, points)"),
                       ("engagement_windows", "Engagement accuracy (%)")):
        lines += [f"## {title}", ""]
        r = res[sec]
        if "note" in r:
            lines += [r["note"], ""]
            continue
        for g in GROUPS:
            lines += [f"**{g}** - max gap {r[g]['max_gap']}{' FLAGGED' if r[g]['flag'] else ''}", "",
                      "| group | " + " | ".join(next(iter(r[g]["groups"].values()), {}).keys()) + " |",
                      "| --- |" + " ---: |" * len(next(iter(r[g]["groups"].values()), {}))]
            lines += [f"| {k} | " + " | ".join(str(v) for v in row.values()) + " |" for k, row in r[g]["groups"].items()]
            lines.append("")
    lines += ["## Gaze error per held-out MPIIGaze person (deg)", "", str(res["gaze_per_person"]), "",
              "## Notes", "",
              "- The synthetic simulator generates traits independently of demographics, so gaps there mostly measure "
              "noise; the real test is our recordings (accent and glasses are recorded in meta.json).",
              "- Engagement and attention from webcams can disadvantage people with disabilities, different eye-contact "
              "norms or poor cameras; follow-up flags stay explainable and decisions stay with the interviewer."]
    (ROOT / "docs" / "bias_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if k == "data" else {g: v[g]["max_gap"] for g in GROUPS} if isinstance(v, dict) and "gender" in v else v)
                      for k, v in res.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
