"""Evaluate the disfluency CRNN: per-class F1 (threshold tuned on val), ROC-AUC, prevalence baseline.

    python -m ml.behaviour.disfluency.eval
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from ml.behaviour.disfluency.model import LABELS, DisfluencyCRNN
from ml.common.config import CHECKPOINTS
from ml.common.datasets import DisfluencyDataset
from ml.common.train_utils import benchmark, count_params, model_card, plot_history, write_metrics


def load(dev="cpu") -> DisfluencyCRNN:
    st = torch.load(CHECKPOINTS / "disfluency" / "best.pt", map_location="cpu", weights_only=False)
    mc = st["cfg"]["model"]
    m = DisfluencyCRNN(mc["width"], mc["hidden"], mc["dropout"])
    # the log-mel front end has no learned weights (older checkpoints stored torchaudio's window/filter buffers)
    m.load_state_dict({k: v for k, v in st["model"].items() if not k.startswith("front.")}, strict=False)
    return m.to(dev).eval()


@torch.no_grad()
def predict(model, ds, dev):
    P, Y = [], []
    for b in DataLoader(ds, batch_size=128, num_workers=4):
        P.append(torch.sigmoid(model(b["wav"].to(dev))).cpu())
        Y.append(b["labels"])
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def main(argv: list[str] | None = None) -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load(dev)
    pv, yv = predict(model, DisfluencyDataset("val"), dev)
    te = DisfluencyDataset("test")
    pt, yt = predict(model, te, dev)
    res = {"test_clips": len(te), "per_class": {}}
    for k, name in enumerate(LABELS):
        grid = np.linspace(0.05, 0.95, 37)
        f1s = [f1_score(yv[:, k], pv[:, k] >= g, zero_division=0) for g in grid]
        thr = float(grid[int(np.argmax(f1s))])
        prev = float(yt[:, k].mean())
        res["per_class"][name] = {
            "f1": round(float(f1_score(yt[:, k], pt[:, k] >= thr, zero_division=0)), 4), "threshold": round(thr, 3),
            "auc": round(float(roc_auc_score(yt[:, k], pt[:, k])), 4) if 0 < yt[:, k].sum() < len(yt) else None,
            "positives": int(yt[:, k].sum()), "prevalence": round(prev, 4),
            "always_positive_f1": round(2 * prev / (1 + prev), 4)}
    res["macro_f1"] = round(float(np.mean([v["f1"] for v in res["per_class"].values()])), 4)
    res["thresholds"] = {k: v["threshold"] for k, v in res["per_class"].items()}
    cpu = load()
    torch.set_num_threads(4)
    with torch.no_grad():
        res["cpu_latency_torch_4threads"] = benchmark(lambda: cpu(torch.zeros(1, 48000)))
    res["params"] = count_params(cpu)
    write_metrics("disfluency", res)
    plot_history("disfluency", ["train_loss", "val_loss"])
    model_card("disfluency", "Disfluency CRNN", {
        "Task": "Audio-side hesitation: filled pause, repetition, prolongation, block in each 3 s window of the "
                "candidate's speech (counts feed the behaviour JSON and the hesitation index).",
        "Architecture": {"front end": "80-band log-mel, per-clip normalisation, SpecAugment (train)",
                         "network": "4 conv blocks -> BiGRU(96) -> attention pooling -> 4 sigmoid outputs",
                         "parameters": f"{res['params']:,}"},
        "Training data": "SEP-28k (podcast clips, label present when >= 2 of 3 annotators marked it), split by "
                         "episode; class-balanced BCE (pos_weight). Our hesitation spans are added once labelled.",
        "Results (test episodes)": {f"{k} F1 / AUC": f"{v['f1']} / {v['auc']} (always-yes F1 {v['always_positive_f1']})"
                                    for k, v in res["per_class"].items()},
        "Limitations": ["SEP-28k is mostly people who stutter, recorded on podcasts; Indian-English fillers such as "
                        "'matlab' are covered by the text-side rules until our labelled answers are added.",
                        "Blocks and prolongations are rare in interviews; their F1 is reported for completeness."],
    })
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
