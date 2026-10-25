"""Train the engagement CNN-GRU: pre-train on DAiSEE, then fine-tune on our labelled windows.

    python -m ml.vision.engagement.train --stage pretrain          # DAiSEE  -> runs/engagement_daisee
    python -m ml.vision.engagement.train --stage finetune          # ours, from the DAiSEE checkpoint
    python -m ml.vision.engagement.train --stage scratch           # ours, random init (ablation)
Evaluation + "with vs without DAiSEE pre-training" table: python -m ml.vision.engagement.eval
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

from ml.common.config import CHECKPOINTS, device, load_config, parse_overrides, seed_everything
from ml.common.datasets import EngagementClipDataset
from ml.common.train_utils import Run, cosine_lr, count_params, plot_history, set_lr
from ml.vision.engagement.model import EngagementCNNGRU

HERE = Path(__file__).resolve().parent
RUN_NAMES = {"pretrain": "engagement_daisee", "finetune": "engagement", "scratch": "engagement_scratch"}


def weights(ds) -> torch.Tensor:
    c = np.bincount(ds.labels(), minlength=3).astype(float)
    return torch.tensor(c.sum() / (3 * np.maximum(c, 1)), dtype=torch.float32)


def run_epoch(model, dl, dev, w, aux_w, opt=None, scaler=None, sched=None, step=0, amp=True):
    train = opt is not None
    model.train(train)
    tot, correct, n = 0.0, 0, 0
    preds, trues = [], []
    for b in dl:
        x, y = b["frames"].to(dev, non_blocking=True), b["engagement"].to(dev)
        bo, co = b["boredom"].to(dev), b["confusion"].to(dev)
        with torch.autocast(dev.type, enabled=amp and dev.type == "cuda"), torch.set_grad_enabled(train):
            e, bl, cl = model(x)
        e, bl, cl = e.float(), bl.float(), cl.float()
        loss = F.cross_entropy(e, y, weight=w)
        if (bo >= 0).any():  # auxiliary heads only where the levels exist (DAiSEE)
            m = bo >= 0
            loss = loss + aux_w * (F.cross_entropy(bl[m], bo[m]) + F.cross_entropy(cl[m], co[m]))
        if train:
            set_lr(opt, sched(step))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
        tot += loss.item() * len(y)
        preds.append(e.argmax(1).cpu()), trues.append(y.cpu())
        n += len(y)
    from sklearn.metrics import f1_score

    p, t = torch.cat(preds).numpy(), torch.cat(trues).numpy()
    return tot / max(n, 1), float(f1_score(t, p, average="macro")), step


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--stage", choices=list(RUN_NAMES), default="pretrain")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg["seed"])
    dev = device()
    tc, dc, mc = cfg["train"], cfg["data"], cfg["model"]
    source = "daisee" if args.stage == "pretrain" else "recorded"
    mk = lambda s, aug: EngagementClipDataset(s, source, dc["n_frames"], dc["size"], aug, dc["limit"])  # noqa: E731
    tr, va = mk("train", True), mk("val", False)
    if not len(tr):
        raise SystemExit(f"no {source} training clips (DAiSEE needs tools.datasets.daisee; ours need labelled sessions)")
    kw = dict(batch_size=tc["batch_size"], num_workers=dc["workers"], persistent_workers=dc["workers"] > 0,
              pin_memory=dev.type == "cuda")
    tr_dl, va_dl = DataLoader(tr, shuffle=True, drop_last=True, **kw), DataLoader(va, **kw)
    model = EngagementCNNGRU(mc["width"], mc["dim"], mc["hidden"], mc["dropout"]).to(dev)
    epochs, lr = (tc["epochs"], tc["lr"]) if args.stage == "pretrain" else (tc["finetune_epochs"], tc["finetune_lr"])
    if args.stage == "finetune":
        st = torch.load(CHECKPOINTS / "engagement_daisee" / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
    name = RUN_NAMES[args.stage]
    run = Run(name, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=tc["amp"] and dev.type == "cuda")
    total = epochs * len(tr_dl)
    sched = lambda s: cosine_lr(s, total, lr, min(200, total // 10))  # noqa: E731
    w = weights(tr).to(dev)
    run.log(f"{name}: {count_params(model):,} params, {len(tr)} train / {len(va)} val clips, class weights {w.tolist()}")
    best, step = -1.0, 0
    for epoch in range(1, epochs + 1):
        t = time.time()
        tl, tf, step = run_epoch(model, tr_dl, dev, w, tc["aux_weight"], opt, scaler, sched, step, tc["amp"])
        with torch.no_grad():
            vl, vf, _ = run_epoch(model, va_dl, dev, w, tc["aux_weight"], amp=tc["amp"])
        is_best = vf > best
        best = max(best, vf)
        run.epoch_end(epoch, train_loss=tl, val_loss=vl, train_macro_f1=tf, val_macro_f1=vf)
        run.save({"model": model.state_dict(), "epoch": epoch, "best": best, "cfg": cfg, "stage": args.stage}, best=is_best)
        run.log(f"epoch {epoch}/{epochs} val macro-F1 {vf:.3f}{' *' if is_best else ''} ({time.time() - t:.0f} s)")
    plot_history(name, ["train_loss", "val_loss", "train_macro_f1", "val_macro_f1"])
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
