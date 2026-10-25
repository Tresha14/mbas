"""Train the head-pose CNN from scratch; evaluation on the held-out images runs at the end.

    python -m ml.vision.headpose.train
    python -m ml.vision.headpose.train --set train.epochs=1 data.limit=2000
Writes runs/headpose, checkpoints/headpose/{last,best}.pt, docs/metrics/headpose.json, docs/model_cards/headpose.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.common.config import device, load_config, parse_overrides, seed_everything
from ml.common.train_utils import (Run, benchmark, cosine_lr, count_params, model_card, plot_history, set_lr,
                                   write_metrics)
from ml.vision.headpose.model import HeadPoseDataset, HeadPoseNet, load

HERE = Path(__file__).resolve().parent


def run_epoch(model, dl, dev, opt=None, scaler=None, sched=None, step=0, amp=True, run=None):
    train = opt is not None
    model.train(train)
    err, n, tot = np.zeros(3), 0, 0.0
    for x, y in dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev)
        with torch.autocast(dev.type, enabled=amp and dev.type == "cuda"), torch.set_grad_enabled(train):
            p = model(x).float()
        loss = F.smooth_l1_loss(p, y, beta=2.0)
        if train:
            set_lr(opt, sched(step))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if run and step % 100 == 0:
                run.scalars(step, **{"train/loss": loss.item(), "lr": opt.param_groups[0]["lr"]})
            step += 1
        err += (p.detach() - y).abs().sum(0).cpu().numpy()
        tot += loss.item() * len(y)
        n += len(y)
    return tot / max(n, 1), err / max(n, 1), step


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg["seed"])
    dev = device()
    tc, dc = cfg["train"], cfg["data"]
    tr, va = HeadPoseDataset("train", True, dc["limit"]), HeadPoseDataset("val", False, dc["limit"] and dc["limit"] // 10)
    kw = dict(batch_size=tc["batch_size"], num_workers=dc["workers"], persistent_workers=dc["workers"] > 0,
              pin_memory=dev.type == "cuda")
    tr_dl, va_dl = DataLoader(tr, shuffle=True, drop_last=True, **kw), DataLoader(va, **kw)
    model = HeadPoseNet(cfg["model"]["width"], cfg["model"]["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=tc["amp"] and dev.type == "cuda")
    run = Run(cfg["name"], cfg, resume=args.resume)
    start, step, best = 1, 0, float("inf")
    if args.resume and (st := run.load_last()):
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"])
        start, step, best = st["epoch"] + 1, st["step"], st["best"]
    total = tc["epochs"] * len(tr_dl)
    sched = lambda s: cosine_lr(s, total, tc["lr"], tc["warmup_steps"])  # noqa: E731
    run.log(f"headpose: {count_params(model):,} params, {len(tr)} train / {len(va)} val faces")
    for epoch in range(start, tc["epochs"] + 1):
        t = time.time()
        tl, te, step = run_epoch(model, tr_dl, dev, opt, scaler, sched, step, tc["amp"], run)
        with torch.no_grad():
            vl, ve, _ = run_epoch(model, va_dl, dev, amp=tc["amp"])
        mae = float(ve.mean())
        is_best = mae < best
        best = min(best, mae)
        run.epoch_end(epoch, train_loss=tl, val_loss=vl, val_mae_deg=mae, val_yaw_mae=ve[0], val_pitch_mae=ve[1],
                      val_roll_mae=ve[2])
        run.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step, "best": best,
                  "cfg": cfg}, best=is_best)
        run.log(f"epoch {epoch}/{tc['epochs']}  val MAE yaw {ve[0]:.2f} pitch {ve[1]:.2f} roll {ve[2]:.2f} deg"
                f"{' *' if is_best else ''} ({time.time() - t:.0f} s)")
    run.close()

    # ---- evaluation on held-out images
    best_m = load(dev=dev)
    te_ds = HeadPoseDataset("test")
    with torch.no_grad():
        _, test_err, _ = run_epoch(best_m, DataLoader(te_ds, **kw), dev, amp=False)
    base = np.abs(te_ds.df[["yaw", "pitch", "roll"]].to_numpy() -
                  tr.df[["yaw", "pitch", "roll"]].to_numpy().mean(0)).mean(0)
    cpu = load()
    torch.set_num_threads(4)
    with torch.no_grad():
        speed = benchmark(lambda: cpu(torch.zeros(1, 3, 64, 64)))
    res = {"test_faces": len(te_ds), "test_mae_deg": {"yaw": round(float(test_err[0]), 3), "pitch": round(float(test_err[1]), 3),
                                                       "roll": round(float(test_err[2]), 3)},
           "baseline_mean_pose_mae_deg": {k: round(float(v), 3) for k, v in zip(("yaw", "pitch", "roll"), base)},
           "label_source": "solvePnP on RetinaFace landmarks (geometric pseudo-labels)", "params": count_params(cpu),
           "cpu_latency_torch_4threads": speed}
    write_metrics("headpose", res)
    plot_history("headpose", ["train_loss", "val_loss", "val_mae_deg"])
    model_card("headpose", "Head-pose CNN", {
        "Task": "Yaw, pitch and roll of the candidate's head from a 64x64 face crop; input to the gaze CNN, the "
                "attention score and the head-movement features.",
        "Architecture": {"layers": "4 conv blocks (32-64-128-128) + global average pool + MLP 64 -> 3",
                         "parameters": f"{res['params']:,}", "output": "degrees (tanh-free, scaled x30)"},
        "Training data": f"{len(tr)} WIDER FACE train faces >= 40 px with RetinaFace landmarks; labels from solvePnP "
                         "on a generic 5-point 3-D face (ml/vision/headpose/geometry.py). Box jitter, photometric "
                         "augmentation, horizontal flip (yaw and roll negated). Split by image.",
        "Results (held-out images)": {"MAE yaw": f"{res['test_mae_deg']['yaw']} deg", "MAE pitch": f"{res['test_mae_deg']['pitch']} deg",
                                      "MAE roll": f"{res['test_mae_deg']['roll']} deg",
                                      "mean-pose baseline (yaw/pitch/roll)": str(res["baseline_mean_pose_mae_deg"]),
                                      "CPU latency": f"{speed['median_ms']} ms"},
        "Limitations": ["Labels are geometric estimates from a generic face model, so errors are measured against "
                        "pseudo-labels; absolute accuracy against ground truth needs our scripted 'look at camera' "
                        "segments (0, 0, 0) or a calibrated rig.",
                        "Pitch from 5 landmarks is the least reliable angle (depends on the face's proportions)."],
    })
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
