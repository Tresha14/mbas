"""Learned per-answer behaviour models (random init), trained on labelled answers.

  SpeakingPatternMLP   prosody + pause + hesitation features -> confident / balanced / hesitant / rushed / monotone
  HesitationIndex      non-negative weighted sum of standardised hesitation features (+ bias) -> 0-100,
                       weights learned against the labelled hesitation index (interpretable: one weight per feature);
                       compared with a 2-layer MLP

Only features our live pipeline computes are used (no simulator-only columns).
Also validates the rule-based text hesitation counters against the dataset's counts.

    python -m ml.behaviour.answer_models
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error

from ml.behaviour.text_hesitation import text_hesitation
from ml.common.config import CHECKPOINTS, METRICS, seed_everything
from ml.common.datasets import SPEAKING_PATTERNS, fusion_table
from ml.common.train_utils import model_card, plot_confusion, write_metrics
from ml.vision.engagement.window_models import MLP, Standardizer, class_weights, fit

PATTERN_FEATURES = ["speaking_rate_wpm", "articulation_rate_wpm", "pitch_std_hz", "pitch_range_hz", "monotony_index",
                    "energy_std_db", "volume_consistency", "speech_rate_variability", "pause_ratio", "long_pause_count",
                    "mean_pause_duration_s", "response_latency_s", "filler_rate_per_100w", "disfluency_rate_per_100w",
                    "hedge_rate_per_100w", "restart_count", "repetition_count"]
HESITATION_FEATURES = ["response_latency_s", "long_pause_count", "long_pause_total_s", "pause_ratio",
                       "filler_rate_per_100w", "repetition_count", "restart_count", "hedge_rate_per_100w",
                       "prolongation_count", "block_count", "disfluency_rate_per_100w"]


def add_rates(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["hedge_rate_per_100w"] = 100 * t.hedge_count / t.word_count.clip(lower=1)
    return t


class HesitationIndex(nn.Module):
    """index = 100 * sigmoid(b + sum softplus(w_i) * x_i): more of any hesitation feature never lowers it."""

    def __init__(self, n: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return (x * F.softplus(self.w)).sum(1, keepdim=True) + self.b

    def weights(self) -> np.ndarray:
        return F.softplus(self.w).detach().numpy()


def validate_text_rules(t: pd.DataFrame, n: int = 1500) -> dict:
    s = t.sample(min(n, len(t)), random_state=0)
    ours = pd.DataFrame([text_hesitation(x) for x in s.answer_text])
    out = {}
    for c in ("filler_count", "repetition_count", "restart_count", "hedge_count"):
        a, b = ours[c].to_numpy(float), s[c].to_numpy(float)
        out[c] = {"pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 4), "mae": round(float(np.abs(a - b).mean()), 3)}
    return out


def main(argv: list[str] | None = None) -> int:
    seed_everything(42)
    t = add_rates(fusion_table("synthetic"))
    tr, va, te = (t[t.split == s].reset_index(drop=True) for s in ("train", "val", "test"))

    # ---- speaking pattern
    st_p = Standardizer(tr[PATTERN_FEATURES].to_numpy(float))
    yp = {s: d.speaking_pattern.map(SPEAKING_PATTERNS.index).to_numpy() for s, d in (("tr", tr), ("va", va), ("te", te))}
    w = class_weights(yp["tr"], len(SPEAKING_PATTERNS))
    sp = fit(MLP(len(PATTERN_FEATURES), len(SPEAKING_PATTERNS)), st_p(tr[PATTERN_FEATURES].to_numpy(float)), yp["tr"],
             lambda o, y: F.cross_entropy(o, y, weight=w), epochs=120, xv=st_p(va[PATTERN_FEATURES].to_numpy(float)),
             yv=yp["va"])
    with torch.no_grad():
        pred = sp(torch.from_numpy(st_p(te[PATTERN_FEATURES].to_numpy(float)))).argmax(1).numpy()
    sp_res = {"macro_f1": round(float(f1_score(yp["te"], pred, average="macro")), 4),
              "per_class_f1": dict(zip(SPEAKING_PATTERNS, np.round(f1_score(yp["te"], pred, average=None,
                                                                            labels=range(5)), 4).tolist())),
              "majority_baseline_macro_f1": round(float(f1_score(yp["te"], np.full_like(yp["te"], np.bincount(yp["tr"]).argmax()),
                                                                 average="macro")), 4)}
    plot_confusion(confusion_matrix(yp["te"], pred, labels=range(5)), list(SPEAKING_PATTERNS),
                   "Speaking pattern (synthetic test)", METRICS / "speaking_pattern_cm.png")

    # ---- hesitation index
    st_h = Standardizer(tr[HESITATION_FEATURES].to_numpy(float))
    xh = {s: st_h(d[HESITATION_FEATURES].to_numpy(float)) for s, d in (("tr", tr), ("va", va), ("te", te))}
    yh = {s: (d.hesitation_index.to_numpy(np.float32) / 100)[:, None] for s, d in (("tr", tr), ("va", va), ("te", te))}
    loss = lambda o, y: F.mse_loss(torch.sigmoid(o), y)  # noqa: E731
    hi = fit(HesitationIndex(len(HESITATION_FEATURES)), xh["tr"], yh["tr"], loss, epochs=150, lr=1e-2, xv=xh["va"], yv=yh["va"])
    hm = fit(MLP(len(HESITATION_FEATURES), 1), xh["tr"], yh["tr"], loss, epochs=150, xv=xh["va"], yv=yh["va"])
    with torch.no_grad():
        p_lin = 100 * torch.sigmoid(hi(torch.from_numpy(xh["te"]))).numpy()[:, 0]
        p_mlp = 100 * torch.sigmoid(hm(torch.from_numpy(xh["te"]))).numpy()[:, 0]
    y_true = te.hesitation_index.to_numpy()
    hi_res = {"weighted_sum_mae": round(float(mean_absolute_error(y_true, p_lin)), 3),
              "weighted_sum_r": round(float(np.corrcoef(y_true, p_lin)[0, 1]), 4),
              "mlp_mae": round(float(mean_absolute_error(y_true, p_mlp)), 3),
              "mean_baseline_mae": round(float(np.abs(y_true - tr.hesitation_index.mean()).mean()), 3),
              "weights": dict(zip(HESITATION_FEATURES, np.round(hi.weights(), 4).tolist()))}
    rules = validate_text_rules(t)

    ck = CHECKPOINTS / "behaviour"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"speaking_pattern": sp.state_dict(), "pattern_features": PATTERN_FEATURES, "pattern_std": st_p.state(),
                "hesitation_index": hi.state_dict(), "hesitation_features": HESITATION_FEATURES,
                "hesitation_std": st_h.state()}, ck / "best.pt")
    res = {"source": "synthetic", "train_answers": len(tr), "test_answers": len(te), "speaking_pattern": sp_res,
           "hesitation_index": hi_res, "text_rule_validation": rules}
    write_metrics("behaviour", res)
    top = sorted(hi_res["weights"].items(), key=lambda kv: -kv[1])[:5]
    model_card("behaviour", "Speaking-pattern classifier + hesitation index", {
        "Task": "Per answer: speaking pattern (confident / balanced / hesitant / rushed / monotone) and a 0-100 "
                "hesitation index; inputs come from the pause model, prosody extractor and text/audio hesitation.",
        "Models": {"speaking pattern": "MLP 2 x 64, class-weighted CE",
                   "hesitation index": "100 * sigmoid(b + sum softplus(w_i) x_i) (monotone, interpretable) vs MLP"},
        "Data": f"synthetic answers, person-wise split ({len(tr)} / {len(va)} / {len(te)}) - SYNTHETIC; re-train on our "
                "labels with the same script once sessions are labelled.",
        "Results (test)": {"speaking pattern macro-F1": sp_res["macro_f1"], "majority baseline": sp_res["majority_baseline_macro_f1"],
                           "hesitation index MAE (weighted sum)": hi_res["weighted_sum_mae"], "hesitation index MAE (MLP)": hi_res["mlp_mae"],
                           "mean baseline MAE": hi_res["mean_baseline_mae"]},
        "Largest hesitation weights": ", ".join(f"{k} {v:.2f}" for k, v in top),
        "Text hesitation rules vs dataset counts (Pearson r)": {k: v["pearson_r"] for k, v in rules.items()},
    })
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
