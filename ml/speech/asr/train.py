"""Train the Conformer-CTC ASR from scratch (LibriSpeech train-clean-100 + Common Voice Indian English + our transcripts).

Training plan for one 8-16 GB GPU (also Colab/Kaggle T4):
  * duration-bucketed batches of `batch_seconds` audio, fp16 autocast, gradient clipping
  * Noam schedule (warm-up then 1/sqrt(step)), SpecAugment + speed perturbation 0.9/1.0/1.1
  * dynamic chunk training (half the batches use chunked attention) so the same model streams
  * a checkpoint every epoch (epoch_XXX.pt) and last.pt; --resume continues after any interruption
  * final model = average of the last `average_last` epoch checkpoints (checkpoints/asr/averaged.pt)
On the RTX 4060 laptop one epoch of 100 h takes ~15-25 min; 40 epochs ~ 12-16 h.

    python -m ml.speech.asr.train
    python -m ml.speech.asr.train --resume
    python -m ml.speech.asr.train --set train.epochs=1 data.limit=500        # smoke test
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from ml.common import text
from ml.common.config import CHECKPOINTS, device, load_config, parse_overrides, seed_everything
from ml.common.datasets import ASRDataset, asr_collate
from ml.common.train_utils import Run, count_params, noam_lr, plot_history, set_lr
from ml.speech.asr.decode import greedy, wer
from ml.speech.asr.model import ConformerCTC

HERE = Path(__file__).resolve().parent


class DurationBatchSampler(Sampler):
    """Batches of similar-length utterances with at most `max_seconds` of audio each, shuffled per epoch."""

    def __init__(self, durations: np.ndarray, max_seconds: float, shuffle: bool = True, seed: int = 0):
        order = np.argsort(durations)
        self.batches, cur, tot = [], [], 0.0
        for i in order:
            d = float(durations[i])
            if cur and (tot + d > max_seconds or len(cur) * max(d, 1) > max_seconds):
                self.batches.append(cur)
                cur, tot = [], 0.0
            cur.append(int(i))
            tot += d
        if cur:
            self.batches.append(cur)
        self.shuffle, self.epoch, self.seed = shuffle, 0, seed

    def __iter__(self):
        b = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(b)
        self.epoch += 1
        return iter(b)

    def __len__(self):
        return len(self.batches)


def ctc_loss(model, batch, dev, chunk=None, left=4):
    wav, wl = batch["wav"].to(dev, non_blocking=True), batch["wav_len"].to(dev)
    tok, tl = batch["tokens"].to(dev), batch["tok_len"].to(dev)
    logp, ol = model(wav, wl, chunk, left)
    loss = F.ctc_loss(logp.transpose(0, 1), tok, ol, tl, blank=0, reduction="sum", zero_infinity=True)
    return loss / max(int(tl.sum()), 1), logp, ol


@torch.no_grad()
def greedy_wer(model, dl, dev, tok) -> float:
    model.eval()
    errs, n = 0, 0
    for b in dl:
        logp, ol = model(b["wav"].to(dev), b["wav_len"].to(dev))
        for i, ref in enumerate(b["text"]):
            hyp = tok.decode(greedy(logp[i, : ol[i]].cpu().numpy()))
            e, r = wer(ref, hyp)
            errs, n = errs + e, n + r
    return errs / max(n, 1)


def average_checkpoints(paths: list[Path], out: Path) -> Path:
    avg = None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if avg is None:
            avg = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k in avg:
                avg[k] += sd[k].float()
    for k in avg:
        avg[k] /= len(paths)
        if sd[k].dtype in (torch.int64, torch.long):
            avg[k] = sd[k]
    st = torch.load(paths[-1], map_location="cpu", weights_only=False)
    st["model"] = avg
    st["averaged_from"] = [p.name for p in paths]
    torch.save(st, out)
    return out


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
    tok = text.load("asr")
    tr = ASRDataset("train", tok, tuple(dc["sources"]), dc["max_s"], augment=True, limit=dc["limit"])
    va = ASRDataset("val", tok, tuple(dc["sources"]), dc["max_s"], limit=tc["val_items"])
    sampler = DurationBatchSampler(tr.durations() * 1.0, dc["batch_seconds"])
    tr_dl = DataLoader(tr, batch_sampler=sampler, collate_fn=asr_collate, num_workers=dc["workers"],
                       persistent_workers=dc["workers"] > 0, pin_memory=dev.type == "cuda")
    va_dl = DataLoader(va, batch_sampler=DurationBatchSampler(va.durations(), 400, shuffle=False),
                       collate_fn=asr_collate, num_workers=2)
    model = ConformerCTC(len(tok), mc["d"], mc["heads"], mc["layers"], mc["kernel"], dropout=mc["dropout"],
                         time_masks=mc.get("time_masks", 5)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=tc["amp"] and dev.type == "cuda")
    run = Run(cfg["name"], cfg, resume=args.resume)
    start, step, best = 1, 0, float("inf")
    if args.resume and (st := run.load_last()):
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"])
        start, step, best = st["epoch"] + 1, st["step"], st["best"]
        sampler.epoch = st["epoch"]
        run.log(f"resumed after epoch {st['epoch']} (step {step})")
    factor = tc["peak_lr"] / noam_lr(tc["warmup_steps"], mc["d"], tc["warmup_steps"])
    hours = tr.durations().sum() / 3600
    run.log(f"asr: {count_params(model):,} params, vocab {len(tok)}, {len(tr)} train utts ({hours:.1f} h), "
            f"{len(sampler)} batches/epoch, {len(va)} val utts")
    rng = random.Random(cfg["seed"])
    for epoch in range(start, tc["epochs"] + 1):
        t = time.time()
        model.train()
        tl, nb = 0.0, 0
        for b in tr_dl:
            chunk = (rng.choice(tc["chunk_choices"])
                     if epoch >= tc.get("chunk_start_epoch", 0) and rng.random() < tc["dynamic_chunk_p"] else None)
            set_lr(opt, noam_lr(step + 1, mc["d"], tc["warmup_steps"], factor))
            with torch.autocast(dev.type, enabled=tc["amp"] and dev.type == "cuda"):
                loss, _, _ = ctc_loss(model, b, dev, chunk, tc["left_chunks"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if step % 50 == 0:
                run.scalars(step, **{"train/ctc": loss.item(), "lr": opt.param_groups[0]["lr"], "train/grad_norm": float(gnorm)})
            tl += loss.item()
            nb += 1
            step += 1
        val_wer = greedy_wer(model, va_dl, dev, tok)
        is_best = val_wer < best
        best = min(best, val_wer)
        run.epoch_end(epoch, train_ctc=tl / max(nb, 1), val_wer_greedy=val_wer)
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step, "best": best,
                 "cfg": cfg, "vocab": len(tok)}
        run.save(state, best=is_best)
        torch.save({k: v for k, v in state.items() if k != "opt"}, run.ckpt_dir / f"epoch_{epoch:03d}.pt")
        run.log(f"epoch {epoch}/{tc['epochs']} ctc {tl / max(nb, 1):.3f} val WER {val_wer:.3f}{' *' if is_best else ''} "
                f"({(time.time() - t) / 60:.1f} min)")
    eps = sorted(run.ckpt_dir.glob("epoch_*.pt"))[-tc["average_last"]:]
    if eps:
        out = average_checkpoints(eps, run.ckpt_dir / "averaged.pt")
        m2 = ConformerCTC(len(tok), mc["d"], mc["heads"], mc["layers"], mc["kernel"], dropout=mc["dropout"]).to(dev)
        m2.load_state_dict(torch.load(out, map_location="cpu", weights_only=False)["model"])
        run.log(f"averaged {len(eps)} checkpoints -> val WER {greedy_wer(m2, va_dl, dev, tok):.3f}")
    plot_history(cfg["name"], ["train_ctc", "val_wer_greedy"])
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
