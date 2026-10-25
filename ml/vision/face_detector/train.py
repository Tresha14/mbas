"""Train the face + landmark detector from scratch on WIDER FACE.

    python -m ml.vision.face_detector.train                       # full run (config.yaml)
    python -m ml.vision.face_detector.train --resume
    python -m ml.vision.face_detector.train --set train.epochs=2 data.limit=300   # smoke / live retrain
Logs: runs/face_detector/ (TensorBoard), checkpoints/face_detector/{last,best}.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.common.config import device, load_config, parse_overrides, seed_everything
from ml.common.datasets import FaceDataset, face_collate
from ml.common.train_utils import Run, cosine_lr, count_params, plot_history, set_lr
from ml.vision.face_detector.model import FaceDetector, detection_loss

HERE = Path(__file__).resolve().parent


def to_dev(batch, dev):
    imgs, boxes, kps, masks = batch
    return imgs.to(dev, non_blocking=True), [b.to(dev) for b in boxes], [k.to(dev) for k in kps], [m.to(dev) for m in masks]


def run_epoch(model, loader, dev, opt=None, scaler=None, sched=None, kp_w=0.5, amp=True, run=None, step0=0):
    train = opt is not None
    model.train(train)
    tot = {"cls": 0.0, "box": 0.0, "kp": 0.0}
    n = 0
    step = step0
    for batch in loader:
        imgs, boxes, kps, masks = to_dev(batch, dev)
        with torch.autocast(dev.type, enabled=amp and dev.type == "cuda"), torch.set_grad_enabled(train):
            logits, deltas, kpd = model(imgs)
        lc, lb, lk = detection_loss(logits.float(), deltas.float(), kpd.float(), boxes, kps, masks, model.anchors)
        loss = lc + lb + kp_w * lk
        if train:
            if sched:
                set_lr(opt, sched(step))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if run and step % 50 == 0:
                run.scalars(step, **{"train/loss": loss.item(), "train/cls": lc.item(), "train/box": lb.item(),
                                     "train/kp": float(lk), "lr": opt.param_groups[0]["lr"]})
            step += 1
        tot["cls"] += lc.item()
        tot["box"] += lb.item()
        tot["kp"] += float(lk)
        n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}, step


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg["seed"])
    dev = device()
    tc, dc = cfg["train"], cfg["data"]

    train_ds = FaceDataset("train", cfg["model"]["size"], augment=True, min_face=dc["min_face"], limit=dc["limit"])
    val_ds = FaceDataset("val", cfg["model"]["size"], augment=False, min_face=dc["min_face"],
                         limit=dc["limit"] and max(dc["limit"] // 5, 20))
    kw = dict(batch_size=tc["batch_size"], num_workers=dc["workers"], collate_fn=face_collate,
              persistent_workers=dc["workers"] > 0, pin_memory=dev.type == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)

    model = FaceDetector(cfg["model"]["size"], cfg["model"]["width"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=tc["amp"] and dev.type == "cuda")
    run = Run(cfg["name"], cfg, resume=args.resume)
    start, step, best = 1, 0, float("inf")
    if args.resume and (st := run.load_last()):
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start, step, best = st["epoch"] + 1, st["step"], st["best"]
        run.log(f"resumed from epoch {st['epoch']}")
    total = tc["epochs"] * len(train_dl)
    sched = lambda s: cosine_lr(s, total, tc["lr"], tc["warmup_steps"])  # noqa: E731
    run.log(f"face_detector: {count_params(model):,} params, {len(train_ds)} train / {len(val_ds)} val images, {dev}")

    for epoch in range(start, tc["epochs"] + 1):
        t = time.time()
        tr, step = run_epoch(model, train_dl, dev, opt, scaler, sched, tc["kp_weight"], tc["amp"], run, step)
        with torch.no_grad():
            va, _ = run_epoch(model, val_dl, dev, kp_w=tc["kp_weight"], amp=tc["amp"])
        val_loss = va["cls"] + va["box"] + tc["kp_weight"] * va["kp"]
        is_best = val_loss < best
        best = min(best, val_loss)
        run.epoch_end(epoch, train_loss=tr["cls"] + tr["box"] + tc["kp_weight"] * tr["kp"], val_loss=val_loss,
                      val_cls=va["cls"], val_box=va["box"], val_kp=va["kp"])
        run.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step, "best": best,
                  "cfg": cfg}, best=is_best)
        run.log(f"epoch {epoch}/{tc['epochs']}  train cls {tr['cls']:.3f} box {tr['box']:.3f} kp {tr['kp']:.3f}  "
                f"val {val_loss:.3f}{' *' if is_best else ''}  ({time.time() - t:.0f} s)")
    plot_history(cfg["name"], ["train_loss", "val_loss", "val_cls", "val_box", "val_kp"])
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
