"""Train the disfluency CRNN from scratch on SEP-28k (split by episode).

    python -m ml.behaviour.disfluency.train
    python -m ml.behaviour.disfluency.train --set train.epochs=1 data.limit=2000
Evaluate: python -m ml.behaviour.disfluency.eval
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

from ml.behaviour.disfluency.model import DisfluencyCRNN
from ml.common.config import device, load_config, parse_overrides, seed_everything
from ml.common.datasets import DisfluencyDataset
from ml.common.train_utils import Run, cosine_lr, count_params, plot_history, set_lr

HERE = Path(__file__).resolve().parent


def run_epoch(model, dl, dev, pos_w, opt=None, scaler=None, sched=None, step=0, amp=True, run=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for b in dl:
        x, y = b["wav"].to(dev, non_blocking=True), b["labels"].to(dev)
        with torch.autocast(dev.type, enabled=amp and dev.type == "cuda"), torch.set_grad_enabled(train):
            logits = model(x).float()
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)
        if train:
            set_lr(opt, sched(step))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if run and step % 50 == 0:
                run.scalars(step, **{"train/loss": loss.item(), "lr": opt.param_groups[0]["lr"]})
            step += 1
        tot += loss.item() * len(y)
        n += len(y)
    return tot / max(n, 1), step


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg["seed"])
    dev = device()
    tc, dc, mc = cfg["train"], cfg["data"], cfg["model"]
    tr = DisfluencyDataset("train", dc["limit"], augment=True)
    va = DisfluencyDataset("val", dc["limit"] and dc["limit"] // 5)
    y = tr.targets()
    pos_w = torch.tensor(np.clip((len(y) - y.sum(0)) / np.maximum(y.sum(0), 1), 1, 20), dtype=torch.float32, device=dev)
    kw = dict(batch_size=tc["batch_size"], num_workers=dc["workers"], persistent_workers=dc["workers"] > 0,
              pin_memory=dev.type == "cuda")
    tr_dl, va_dl = DataLoader(tr, shuffle=True, drop_last=True, **kw), DataLoader(va, **kw)
    model = DisfluencyCRNN(mc["width"], mc["hidden"], mc["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=tc["amp"] and dev.type == "cuda")
    run = Run(cfg["name"], cfg, resume=args.resume)
    start, step, best = 1, 0, float("inf")
    if args.resume and (st := run.load_last()):
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"])
        start, step, best = st["epoch"] + 1, st["step"], st["best"]
    total = tc["epochs"] * len(tr_dl)
    sched = lambda s: cosine_lr(s, total, tc["lr"], tc["warmup_steps"])  # noqa: E731
    run.log(f"disfluency: {count_params(model):,} params, {len(tr)} train / {len(va)} val clips, "
            f"positives {y.sum(0).astype(int).tolist()}, pos_weight {pos_w.tolist()}")
    for epoch in range(start, tc["epochs"] + 1):
        t = time.time()
        tl, step = run_epoch(model, tr_dl, dev, pos_w, opt, scaler, sched, step, tc["amp"], run)
        with torch.no_grad():
            vl, _ = run_epoch(model, va_dl, dev, pos_w, amp=tc["amp"])
        is_best = vl < best
        best = min(best, vl)
        run.epoch_end(epoch, train_loss=tl, val_loss=vl)
        run.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step, "best": best,
                  "cfg": cfg}, best=is_best)
        run.log(f"epoch {epoch}/{tc['epochs']} train {tl:.4f} val {vl:.4f}{' *' if is_best else ''} ({time.time() - t:.0f} s)")
    plot_history(cfg["name"], ["train_loss", "val_loss"])
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
