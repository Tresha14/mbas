"""Train the answer-quality transformer from scratch.

Training answers = labelled answers (synthetic now, ours later) + synthetic off-topic pairs: a question
paired with an answer written for a different question (relevance 0.05, quality 1, follow-up
'topic_not_covered'), which teaches the model to check the answer against the question.

    python -m ml.response.answer_quality.train
Evaluate (+ TF-IDF baseline): python -m ml.response.answer_quality.eval
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.common import text
from ml.common.config import device, load_config, parse_overrides, seed_everything
from ml.common.datasets import AnswerDataset, answer_collate, answers_table
from ml.common.train_utils import Run, cosine_lr, count_params, plot_history, set_lr
from ml.response.answer_quality.model import AnswerQualityModel

HERE = Path(__file__).resolve().parent


def off_topic_pairs(t: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = t.sample(frac=min(frac, 1.0), random_state=seed).reset_index(drop=True)
    donors = t.reset_index(drop=True)
    rows = []
    for r in base.itertuples(index=False):
        for _ in range(10):
            d = donors.iloc[int(rng.integers(len(donors)))]
            if d.question_id != r.question_id:
                break
        rows.append({**r._asdict(), "answer_id": f"{r.answer_id}_off", "transcript": d.transcript, "relevance": 0.05,
                     "quality_1to5": 1, "follow_up_needed": 1, "follow_up_reason": "topic_not_covered",
                     "completeness": 0.0, "star_situation": 0.0 if pd.notna(r.star_situation) else np.nan,
                     "star_task": 0.0 if pd.notna(r.star_task) else np.nan,
                     "star_action": 0.0 if pd.notna(r.star_action) else np.nan,
                     "star_result": 0.0 if pd.notna(r.star_result) else np.nan})
    return pd.DataFrame(rows)


def losses(out, b):
    l_rel = F.mse_loss(torch.sigmoid(out["relevance"]), b["relevance"]) * 5
    l_q = F.cross_entropy(out["quality"], b["quality"])
    l_star = (F.binary_cross_entropy_with_logits(out["star"], b["star"], reduction="none") * b["star_mask"]).sum() \
        / b["star_mask"].sum().clamp(min=1)
    l_c = (F.mse_loss(torch.sigmoid(out["completeness"]), b["completeness"], reduction="none") * b["completeness_mask"]).sum() \
        / b["completeness_mask"].sum().clamp(min=1) * 5
    l_f = F.binary_cross_entropy_with_logits(out["follow_up"], b["follow_up"])
    l_r = F.cross_entropy(out["reason"], b["reason"])
    return l_rel + l_q + 0.5 * l_star + 0.5 * l_c + l_f + l_r


def run_epoch(model, dl, dev, sep, opt=None, scaler=None, sched=None, step=0, amp=True):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for b in dl:
        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast(dev.type, enabled=amp and dev.type == "cuda"), torch.set_grad_enabled(train):
            out = model(b["ids"], sep)
        out = {k: v.float() for k, v in out.items()}
        loss = losses(out, b)
        if train:
            set_lr(opt, sched(step))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
        tot += loss.item() * len(b["ids"])
        n += len(b["ids"])
    return tot / max(n, 1), step


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg["seed"])
    dev = device()
    tc, mc = cfg["train"], cfg["model"]
    tok = text.load("answer")
    table = answers_table(cfg["source"])
    tr_t = table[table.split == "train"]
    aug = off_topic_pairs(tr_t, cfg["data"]["off_topic_pairs"]).assign(split="train")
    full = pd.concat([table, aug], ignore_index=True)
    tr = AnswerDataset("train", tok, table=full, max_len=mc["max_len"])
    va = AnswerDataset("val", tok, table=full, max_len=mc["max_len"])
    kw = dict(batch_size=tc["batch_size"], collate_fn=answer_collate)
    tr_dl, va_dl = DataLoader(tr, shuffle=True, drop_last=True, **kw), DataLoader(va, **kw)
    model = AnswerQualityModel(tr.vocab_size, mc["d_model"], mc["heads"], mc["layers"], mc["ff"], mc["max_len"],
                               mc["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=tc["amp"] and dev.type == "cuda")
    run = Run(cfg["name"], cfg)
    total = tc["epochs"] * len(tr_dl)
    sched = lambda s: cosine_lr(s, total, tc["lr"], tc["warmup_steps"])  # noqa: E731
    run.log(f"answer_quality: {count_params(model):,} params, vocab {tr.vocab_size}, {len(tr)} train "
            f"({len(aug)} off-topic pairs) / {len(va)} val")
    best, step = float("inf"), 0
    for epoch in range(1, tc["epochs"] + 1):
        t = time.time()
        tl, step = run_epoch(model, tr_dl, dev, tr.sep, opt, scaler, sched, step, tc["amp"])
        with torch.no_grad():
            vl, _ = run_epoch(model, va_dl, dev, tr.sep, amp=tc["amp"])
        is_best = vl < best
        best = min(best, vl)
        run.epoch_end(epoch, train_loss=tl, val_loss=vl)
        run.save({"model": model.state_dict(), "epoch": epoch, "best": best, "cfg": cfg, "vocab": tr.vocab_size,
                  "sep": tr.sep, "cls": tr.cls}, best=is_best)
        run.log(f"epoch {epoch}/{tc['epochs']} train {tl:.3f} val {vl:.3f}{' *' if is_best else ''} ({time.time() - t:.0f} s)")
    plot_history(cfg["name"], ["train_loss", "val_loss"])
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
