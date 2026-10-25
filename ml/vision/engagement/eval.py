"""Evaluate the engagement CNN-GRU on our test windows: macro-F1, per-class confusion matrix, and the
comparison "fine-tuned from DAiSEE" vs "trained from scratch on our data" (plus DAiSEE test for the pre-trained model).

    python -m ml.vision.engagement.eval
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from ml.common.config import CHECKPOINTS, METRICS
from ml.common.datasets import ENGAGEMENT_CLASSES, EngagementClipDataset
from ml.common.train_utils import model_card, plot_confusion, write_metrics
from ml.vision.engagement.model import EngagementCNNGRU


def load(name: str, dev):
    p = CHECKPOINTS / name / "best.pt"
    if not p.exists():
        return None, None
    st = torch.load(p, map_location="cpu", weights_only=False)
    mc = st["cfg"]["model"]
    m = EngagementCNNGRU(mc["width"], mc["dim"], mc["hidden"], mc["dropout"])
    m.load_state_dict(st["model"])
    return m.to(dev).eval(), st["cfg"]


@torch.no_grad()
def predict(model, ds, dev):
    p, t = [], []
    for b in DataLoader(ds, batch_size=16, num_workers=4):
        p.append(model(b["frames"].to(dev))[0].argmax(1).cpu())
        t.append(b["engagement"])
    return torch.cat(t).numpy(), torch.cat(p).numpy()


def main(argv: list[str] | None = None) -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res, table = {}, []
    for name, source, label in (("engagement_daisee", "daisee", "DAiSEE pre-trained, DAiSEE test"),
                                ("engagement_daisee", "recorded", "DAiSEE only (no fine-tune), our test"),
                                ("engagement_scratch", "recorded", "ours from scratch, our test"),
                                ("engagement", "recorded", "DAiSEE pre-train + our fine-tune, our test")):
        m, cfg = load(name, dev)
        if m is None:
            table.append({"setting": label, "macro_f1": None, "note": f"checkpoints/{name} missing"})
            continue
        try:
            ds = EngagementClipDataset("test", source, cfg["data"]["n_frames"], cfg["data"]["size"])
        except FileNotFoundError as e:
            table.append({"setting": label, "macro_f1": None, "note": str(e).splitlines()[0]})
            continue
        if not len(ds):
            table.append({"setting": label, "macro_f1": None, "note": "no test clips"})
            continue
        y, p = predict(m, ds, dev)
        f1 = round(float(f1_score(y, p, average="macro")), 4)
        table.append({"setting": label, "macro_f1": f1, "clips": len(ds),
                      "per_class_f1": dict(zip(ENGAGEMENT_CLASSES, np.round(f1_score(y, p, average=None, labels=[0, 1, 2]), 4).tolist()))})
        plot_confusion(confusion_matrix(y, p, labels=[0, 1, 2]), list(ENGAGEMENT_CLASSES), label,
                       METRICS / f"engagement_cm_{name}_{source}.png")
    res["comparison"] = table
    write_metrics("engagement", res)
    model_card("engagement", "Engagement CNN-GRU", {
        "Task": "Engaged / neutral / disengaged per 5 s window from 16 webcam frames; auxiliary boredom and "
                "confusion heads (DAiSEE levels).",
        "Architecture": "4-block CNN frame encoder (112x112) -> BiGRU(128) -> mean + last state -> 3 heads; "
                        "weighted cross-entropy (class imbalance).",
        "Training": "Pre-train on DAiSEE (person-wise split), fine-tune on our labelled windows; ablation from scratch.",
        "Results": {r["setting"]: (r["macro_f1"] if r["macro_f1"] is not None else r.get("note")) for r in table},
        "Status": "Waiting for data: DAiSEE needs a signed request and our windows need labels. Until then the "
                  "live system uses the window-feature engagement MLP (docs/model_cards/window_models.md).",
    })
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
