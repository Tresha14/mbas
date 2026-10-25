"""Train the gaze + iris CNN from scratch on MPIIGaze (person-wise split).

Iris-centre targets: a classical dark-pupil detector (eye_crops.iris_centroid) is run on training crops and a
linear eyeball model iris = a + b * sin(angle) is fitted to it; the fitted (smooth, geometric) value is the
target for every sample. On/off-screen is trained only on samples that carry that label (onscreen_mask).

    python -m ml.vision.gaze_iris.train
    python -m ml.vision.gaze_iris.train --set train.epochs=2 data.limit=20000     # live retrain demo
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
from ml.common.datasets import GazeDataset
from ml.common.train_utils import Run, cosine_lr, count_params, plot_history, set_lr
from ml.vision.gaze_iris.eye_crops import iris_centroid
from ml.vision.gaze_iris.model import GazeIrisNet, angular_error_deg

HERE = Path(__file__).resolve().parent


def fit_iris_model(ds: GazeDataset, n: int = 6000, seed: int = 0) -> dict:
    """Least-squares fit iris_x = ax + bx sin(yaw), iris_y = ay + by sin(pitch) on confident detections."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    xs, ys, sy, sp = [], [], [], []
    for i in idx:
        for img in (ds.left[i], ds.right[i]):
            x, y, c = iris_centroid(img)
            if c > 0.35:
                xs.append(x), ys.append(y)
                sp.append(np.sin(ds.gaze[i, 0])), sy.append(np.sin(ds.gaze[i, 1]))
    A = np.stack([np.ones(len(sy)), sy], 1)
    bx = np.linalg.lstsq(A, np.array(xs), rcond=None)[0]
    B = np.stack([np.ones(len(sp)), sp], 1)
    by = np.linalg.lstsq(B, np.array(ys), rcond=None)[0]
    r_x = float(np.corrcoef(sy, xs)[0, 1]) if len(xs) > 2 else 0.0
    r_y = float(np.corrcoef(sp, ys)[0, 1]) if len(ys) > 2 else 0.0
    return {"ax": float(bx[0]), "bx": float(bx[1]), "ay": float(by[0]), "by": float(by[1]),
            "n": len(xs), "corr_x_yaw": round(r_x, 3), "corr_y_pitch": round(r_y, 3)}


def iris_targets(gaze: torch.Tensor, m: dict) -> torch.Tensor:
    x = m["ax"] + m["bx"] * torch.sin(gaze[:, 1])
    y = m["ay"] + m["by"] * torch.sin(gaze[:, 0])
    return torch.stack([x, y, x, y], 1).clamp(0.05, 0.95)


def run_epoch(model, loader, dev, iris_m, cfg, opt=None, scaler=None, sched=None, run=None, step=0):
    train = opt is not None
    model.train(train)
    tc = cfg["train"]
    sums = {"loss": 0.0, "ang": 0.0, "iris": 0.0}
    n = 0
    for b in loader:
        left, right, head, gaze = (b[k].to(dev, non_blocking=True) for k in ("left", "right", "head", "gaze"))
        on, on_m = b["onscreen"].to(dev), b["onscreen_mask"].to(dev)
        with torch.autocast(dev.type, enabled=tc["amp"] and dev.type == "cuda"), torch.set_grad_enabled(train):
            g, iris, logit = model(left, right, head)
        g, iris, logit = g.float(), iris.float(), logit.float()
        l_gaze = F.l1_loss(g, gaze)
        l_iris = F.l1_loss(iris, iris_targets(gaze, iris_m))
        l_on = (F.binary_cross_entropy_with_logits(logit, on, reduction="none") * on_m).sum() / on_m.sum().clamp(min=1)
        loss = l_gaze + tc["iris_weight"] * l_iris + tc["onscreen_weight"] * l_on
        if train:
            set_lr(opt, sched(step))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if run and step % 100 == 0:
                run.scalars(step, **{"train/loss": loss.item(), "train/gaze_l1": l_gaze.item(), "lr": opt.param_groups[0]["lr"]})
            step += 1
        k = len(gaze)
        sums["loss"] += loss.item() * k
        sums["ang"] += angular_error_deg(g.detach(), gaze).sum().item()
        sums["iris"] += l_iris.item() * k
        n += k
    return {k: v / max(n, 1) for k, v in sums.items()}, step


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
    tr_ds = GazeDataset("train", augment=True, limit=dc["limit"])
    va_ds = GazeDataset("val", augment=False, limit=dc["limit"] and dc["limit"] // 5)
    iris_m = fit_iris_model(tr_ds)
    kw = dict(batch_size=tc["batch_size"], num_workers=dc["workers"], persistent_workers=dc["workers"] > 0,
              pin_memory=dev.type == "cuda")
    tr_dl = DataLoader(tr_ds, shuffle=True, drop_last=True, **kw)
    va_dl = DataLoader(va_ds, shuffle=False, **kw)
    model = GazeIrisNet(cfg["model"]["width"], cfg["model"]["dim"], cfg["model"]["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=tc["amp"] and dev.type == "cuda")
    run = Run(cfg["name"], cfg, resume=args.resume)
    start, step, best = 1, 0, float("inf")
    if args.resume and (st := run.load_last()):
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start, step, best, iris_m = st["epoch"] + 1, st["step"], st["best"], st["iris_model"]
    total = tc["epochs"] * len(tr_dl)
    sched = lambda s: cosine_lr(s, total, tc["lr"], tc["warmup_steps"])  # noqa: E731
    run.log(f"gaze_iris: {count_params(model):,} params; {len(tr_ds)} train / {len(va_ds)} val samples; iris model {iris_m}")
    for epoch in range(start, tc["epochs"] + 1):
        t = time.time()
        tr, step = run_epoch(model, tr_dl, dev, iris_m, cfg, opt, scaler, sched, run, step)
        with torch.no_grad():
            va, _ = run_epoch(model, va_dl, dev, iris_m, cfg)
        is_best = va["ang"] < best
        best = min(best, va["ang"])
        run.epoch_end(epoch, train_loss=tr["loss"], val_loss=va["loss"], train_ang_deg=tr["ang"], val_ang_deg=va["ang"],
                      val_iris_l1=va["iris"])
        run.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step, "best": best,
                  "cfg": cfg, "iris_model": iris_m}, best=is_best)
        run.log(f"epoch {epoch}/{tc['epochs']}  train {tr['ang']:.2f} deg  val {va['ang']:.2f} deg"
                f"{' *' if is_best else ''}  ({time.time() - t:.0f} s)")
    plot_history(cfg["name"], ["train_loss", "val_loss", "train_ang_deg", "val_ang_deg"])
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
