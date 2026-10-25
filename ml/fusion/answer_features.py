"""Answer-model outputs as fusion features, without leakage.

train split: 5-fold cross-fitting by participant (transformer retrained on 4/5 of the training people,
             predicts the held-out fifth) -> out-of-fold predictions
val / test:  the full answer-quality checkpoint
Output: data/processed/features/answer_model_<source>.parquet with ans_* columns per answer_id.

    python -m ml.fusion.answer_features --source synthetic
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from ml.common import text
from ml.common.config import PROCESSED, device, load_config, seed_everything
from ml.common.datasets import FOLLOW_UP_REASONS, AnswerDataset, answer_collate, answers_table
from ml.response.answer_quality import train as aq_train
from ml.response.answer_quality.eval import load as load_aq
from ml.response.answer_quality.model import AnswerQualityModel

OUT = PROCESSED / "features"
ANS_COLUMNS = (["ans_relevance", "ans_quality", "ans_completeness", "ans_star_count", "ans_follow_up_p"]
               + [f"ans_reason_{r}" for r in FOLLOW_UP_REASONS])


@torch.no_grad()
def predict_table(model, tok, table: pd.DataFrame, sep: int, dev, max_len: int = 256) -> pd.DataFrame:
    ds = AnswerDataset(table.split.iloc[0], tok, table=table, max_len=max_len)
    rows = []
    for b in torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=answer_collate):
        o = model(b["ids"].to(dev), sep)
        rel = torch.sigmoid(o["relevance"]).cpu().numpy()
        q = AnswerQualityModel.expected_quality(o["quality"]).cpu().numpy()
        comp = torch.sigmoid(o["completeness"]).cpu().numpy()
        star = (torch.sigmoid(o["star"]) > 0.5).sum(1).cpu().numpy()
        fu = torch.sigmoid(o["follow_up"]).cpu().numpy()
        rs = torch.softmax(o["reason"], 1).cpu().numpy()
        for i in range(len(rel)):
            rows.append([rel[i], q[i], comp[i], star[i], fu[i], *rs[i]])
    return pd.DataFrame(rows, columns=ANS_COLUMNS).assign(answer_id=ds.t.answer_id.to_numpy())


def train_fold(cfg, tok, table, dev, epochs: int):
    tr_t = table[table.split == "train"]
    aug = aq_train.off_topic_pairs(tr_t, cfg["data"]["off_topic_pairs"]).assign(split="train")
    full = pd.concat([table, aug], ignore_index=True)
    mc, tc = cfg["model"], cfg["train"]
    tr = AnswerDataset("train", tok, table=full, max_len=mc["max_len"])
    dl = torch.utils.data.DataLoader(tr, batch_size=tc["batch_size"], shuffle=True, drop_last=True, collate_fn=answer_collate)
    model = AnswerQualityModel(tr.vocab_size, mc["d_model"], mc["heads"], mc["layers"], mc["ff"], mc["max_len"],
                               mc["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    total = epochs * len(dl)
    sched = lambda s: aq_train.cosine_lr(s, total, tc["lr"], tc["warmup_steps"])  # noqa: E731
    step = 0
    for _ in range(epochs):
        _, step = aq_train.run_epoch(model, dl, dev, tr.sep, opt, scaler, sched, step)
    return model.eval(), tr.sep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args(argv)
    seed_everything(42)
    dev = device()
    cfg = load_config(aq_train.HERE / "config.yaml")
    tok = text.load("answer")
    table = answers_table(args.source)
    tr = table[table.split == "train"]
    people = np.array(sorted(tr.participant_id.unique()))
    np.random.default_rng(0).shuffle(people)
    parts = []
    for k in range(args.folds):
        hold = set(people[k::args.folds])
        fold_t = table.copy()
        is_hold = fold_t.participant_id.isin(hold) & (fold_t.split == "train")
        fold_t.loc[is_hold, "split"] = "holdout"
        fold_t.loc[fold_t.split.isin(["val", "test"]), "split"] = "unused"
        model, sep = train_fold(cfg, tok, fold_t, dev, args.epochs)
        parts.append(predict_table(model, tok, fold_t[fold_t.split == "holdout"].assign(split="holdout"), sep, dev))
        print(f"fold {k + 1}/{args.folds}: {len(parts[-1])} out-of-fold answers", flush=True)
    full, st = load_aq(dev)
    for s in ("val", "test"):
        parts.append(predict_table(full, tok, table[table.split == s], st["sep"], dev))
    out = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"answer_model_{args.source}.parquet"
    out.to_parquet(p, index=False)
    print(f"wrote {p} ({len(out)} answers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
