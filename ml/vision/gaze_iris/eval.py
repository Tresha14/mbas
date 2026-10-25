"""Evaluate the gaze + iris CNN on held-out MPIIGaze persons (and our labelled windows when available).

Reports mean / median angular error (deg), per-person error, iris-centre L1, and the on/off-screen F1 +
confusion matrix on every sample that has an on/off label (our recordings; MPIIGaze has none).

    python -m ml.vision.gaze_iris.eval
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.common.config import CHECKPOINTS, METRICS
from ml.common.datasets import GazeDataset
from ml.common.train_utils import benchmark, count_params, model_card, plot_confusion, plot_history, write_metrics
from ml.vision.gaze_iris.model import GazeIrisNet, angular_error_deg
from ml.vision.gaze_iris.train import iris_targets


def load_model(dev="cpu"):
    st = torch.load(CHECKPOINTS / "gaze_iris" / "best.pt", map_location="cpu", weights_only=False)
    mc = st["cfg"]["model"]
    m = GazeIrisNet(mc["width"], mc["dim"], mc["dropout"])
    m.load_state_dict(st["model"])
    return m.to(dev).eval(), st


@torch.no_grad()
def predict(model, ds, dev, batch=512):
    G, I, L, T, P, ON, OM = [], [], [], [], [], [], []
    for b in DataLoader(ds, batch_size=batch):
        g, iris, logit = model(b["left"].to(dev), b["right"].to(dev), b["head"].to(dev))
        G.append(g.cpu()), I.append(iris.cpu()), L.append(logit.cpu()), T.append(b["gaze"])
        ON.append(b["onscreen"]), OM.append(b["onscreen_mask"])
    return (torch.cat(G), torch.cat(I), torch.cat(L), torch.cat(T), torch.cat(ON), torch.cat(OM))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, st = load_model(dev)
    ds = GazeDataset(args.split)
    g, iris, logit, true, on, om = predict(model, ds, dev)
    err = angular_error_deg(g, true).numpy()
    per = {p: round(float(err[ds.person == p].mean()), 3) for p in sorted(set(ds.person))}
    iris_l1 = float((iris - iris_targets(true, st["iris_model"])).abs().mean())
    res = {"split": args.split, "samples": len(ds), "persons": sorted(set(ds.person)),
           "angular_error_mean_deg": round(float(err.mean()), 3), "angular_error_median_deg": round(float(np.median(err)), 3),
           "angular_error_per_person_deg": per, "iris_l1_vs_eyeball_model": round(iris_l1, 4),
           "iris_model": st["iris_model"], "params": count_params(model)}
    # baseline: always predict the mean training gaze
    tr = GazeDataset("train")
    mean_g = torch.from_numpy(tr.gaze.mean(0, keepdims=True)).repeat(len(true), 1)
    res["baseline_mean_gaze_error_deg"] = round(float(angular_error_deg(mean_g, true).mean()), 3)
    m = om.numpy() > 0
    if m.any():
        from sklearn.metrics import confusion_matrix, f1_score

        yt, yp = on.numpy()[m].astype(int), (torch.sigmoid(logit).numpy()[m] > 0.5).astype(int)
        res["onscreen_f1"] = round(float(f1_score(yt, yp, average="macro")), 4)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        plot_confusion(cm, ["off_screen", "on_screen"], "Gaze on/off-screen", METRICS / "gaze_iris_onscreen_cm.png")
    else:
        res["onscreen_f1"] = None
        res["onscreen_note"] = ("no on/off-screen labels in MPIIGaze; F1 is reported once our labelled "
                                "windows / Columbia Gaze exist. Live on/off uses the per-session calibration box.")
    cpu, _ = load_model("cpu")
    torch.set_num_threads(4)
    x = torch.zeros(1, 1, 36, 60)
    h = torch.zeros(1, 2)
    with torch.no_grad():
        res["cpu_latency_torch_4threads"] = benchmark(lambda: cpu(x, x, h))
    write_metrics("gaze_iris", res)
    plot_history("gaze_iris", ["train_loss", "val_loss", "train_ang_deg", "val_ang_deg"])
    cfg = st["cfg"]
    model_card("gaze_iris", "Gaze + iris CNN", {
        "Task": "Estimate gaze direction (pitch, yaw), the iris centre in each eye crop and on/off-screen from two "
                "36x60 eye crops plus head pose. Feeds attention signals: off-screen %, glance-away count, saccade "
                "rate, blink rate, reading-pattern flag (ml/vision/gaze_iris/signals.py).",
        "Architecture": {"eye encoder": "4 conv blocks (32-64-128-128), shared by both eyes",
                         "trunk": "concat(2 x 128 + head pose 2) -> 256 -> 128", "heads": "gaze 2, iris 4, on-screen 1",
                         "parameters": f"{res['params']:,}"},
        "Training data": "MPIIGaze normalised eye images, person-wise split (train/val/test persons in "
                         "data/splits/public/mpiigaze). Flip augmentation swaps eyes and mirrors yaw.",
        "Iris targets": f"eyeball model fitted to a classical dark-pupil detector: {st['iris_model']}",
        "Training": {"epochs": cfg["train"]["epochs"], "batch": cfg["train"]["batch_size"],
                     "loss": "L1 gaze + 0.5 L1 iris + BCE on-screen (masked)", "curves": "docs/metrics/gaze_iris_curves.png"},
        "Results (held-out persons)": {"mean angular error": f"{res['angular_error_mean_deg']} deg",
                                       "median angular error": f"{res['angular_error_median_deg']} deg",
                                       "mean-gaze baseline": f"{res['baseline_mean_gaze_error_deg']} deg",
                                       "on/off-screen F1": res["onscreen_f1"] if res["onscreen_f1"] is not None else
                                       "pending our labelled windows (see note)",
                                       "CPU latency": f"{res['cpu_latency_torch_4threads']['median_ms']} ms"},
        "Limitations": ["MPIIGaze crops are perspective-normalised; our webcam crops use roll correction only, so "
                        "a domain gap remains until fine-tuning on our 60 s calibration captures.",
                        "On/off-screen in the live system: per-session calibration box on predicted gaze "
                        "(signals.ScreenCalibration), not the untrained on-screen head.",
                        "Glasses with strong reflections and very dim light reduce accuracy (bias check, Stage 10)."],
    })
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
