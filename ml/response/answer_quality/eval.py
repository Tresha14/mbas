"""Evaluate the answer-quality transformer vs the TF-IDF baseline on the test participants.

Metrics: relevance MAE / Pearson r, quality exact accuracy / MAE / quadratic-weighted kappa, STAR F1,
completeness MAE, follow-up F1 and reason macro-F1.

    python -m ml.response.answer_quality.eval
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score
from torch.utils.data import DataLoader

from ml.common import text
from ml.common.config import CHECKPOINTS
from ml.common.datasets import FOLLOW_UP_REASONS, AnswerDataset, answer_collate, answers_table
from ml.common.train_utils import count_params, model_card, plot_history, write_metrics
from ml.response.answer_quality.baseline_tfidf import TfidfBaseline
from ml.response.answer_quality.model import AnswerQualityModel


def load(dev="cpu"):
    st = torch.load(CHECKPOINTS / "answer_quality" / "best.pt", map_location="cpu", weights_only=False)
    mc = st["cfg"]["model"]
    m = AnswerQualityModel(st["vocab"], mc["d_model"], mc["heads"], mc["layers"], mc["ff"], mc["max_len"], mc["dropout"])
    m.load_state_dict(st["model"])
    return m.to(dev).eval(), st


@torch.no_grad()
def predict(model, ds, sep, dev) -> dict:
    out = {k: [] for k in ("relevance", "quality", "quality_expected", "star", "completeness", "follow_up", "reason")}
    for b in DataLoader(ds, batch_size=64, collate_fn=answer_collate):
        o = model(b["ids"].to(dev), sep)
        out["relevance"].append(torch.sigmoid(o["relevance"]).cpu())
        out["quality"].append(o["quality"].argmax(-1).cpu() + 1)
        out["quality_expected"].append(AnswerQualityModel.expected_quality(o["quality"]).cpu())
        out["star"].append(torch.sigmoid(o["star"]).cpu())
        out["completeness"].append(torch.sigmoid(o["completeness"]).cpu())
        out["follow_up"].append(torch.sigmoid(o["follow_up"]).cpu())
        out["reason"].append(o["reason"].argmax(-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def score(t, p: dict) -> dict:
    y_rel, y_q = t.relevance.to_numpy(), t.quality_1to5.astype(int).to_numpy()
    reason = p["reason"] if p["reason"].dtype.kind in "OU" else np.array(FOLLOW_UP_REASONS)[p["reason"]]
    r = {"relevance_mae": round(float(np.abs(p["relevance"] - y_rel).mean()), 4),
         "relevance_r": round(float(np.corrcoef(p["relevance"], y_rel)[0, 1]), 4),
         "quality_accuracy": round(float((p["quality"] == y_q).mean()), 4),
         "quality_mae": round(float(np.abs(p["quality_expected"] - y_q).mean()), 4),
         "quality_qwk": round(float(cohen_kappa_score(y_q, p["quality"], weights="quadratic")), 4),
         "follow_up_f1": round(float(f1_score(t.follow_up_needed.astype(int), p["follow_up"] > 0.5)), 4),
         "reason_macro_f1": round(float(f1_score(t.follow_up_reason, reason, average="macro")), 4)}
    if "star" in p:
        cols = ["star_situation", "star_task", "star_action", "star_result"]
        m = t[cols].notna().to_numpy()
        if m.any():
            yt = t[cols].fillna(0).to_numpy()[m] > 0.5
            r["star_f1"] = round(float(f1_score(yt, p["star"][m] > 0.5)), 4)
        mc = t.completeness.notna().to_numpy()
        if mc.any():
            r["completeness_mae"] = round(float(np.abs(p["completeness"][mc] - t.completeness.to_numpy()[mc]).mean()), 4)
    return r


def main(argv: list[str] | None = None) -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, st = load(dev)
    tok = text.load("answer")
    table = answers_table(st["cfg"]["source"])
    te_t = table[table.split == "test"].reset_index(drop=True)
    ds = AnswerDataset("test", tok, table=table, max_len=st["cfg"]["model"]["max_len"])
    ours = score(te_t, predict(model, ds, st["sep"], dev))
    base = TfidfBaseline().fit(table[table.split == "train"])
    baseline = score(te_t, base.predict(te_t))
    # off-topic check: every test question paired with an answer to a different question
    from ml.response.answer_quality.train import off_topic_pairs

    off = off_topic_pairs(te_t, 1.0, seed=7).assign(split="test")
    off_ds = AnswerDataset("test", tok, table=off, max_len=st["cfg"]["model"]["max_len"])
    p_off = predict(model, off_ds, st["sep"], dev)
    better = {k: (ours[k] < baseline[k]) if k.endswith("mae") else (ours[k] > baseline[k]) for k in baseline}
    res = {"source": st["cfg"]["source"], "test_answers": len(te_t), "transformer": ours, "tfidf_baseline": baseline,
           "transformer_better": better, "params": count_params(model),
           "off_topic_test": {"mean_relevance": round(float(p_off["relevance"].mean()), 4),
                              "flagged_follow_up": round(float((p_off["follow_up"] > 0.5).mean()), 4)}}
    write_metrics("answer_quality", res)
    plot_history("answer_quality", ["train_loss", "val_loss"])
    mc = st["cfg"]["model"]
    model_card("answer_quality", "Answer-quality transformer", {
        "Task": "From the question and the answer transcript: relevance 0-1, quality 1-5, STAR components "
                "(behavioural questions), completeness, follow-up yes/no and reason.",
        "Architecture": {"encoder": f"{mc['layers']} pre-norm transformer layers, d_model {mc['d_model']}, {mc['heads']} heads, FF {mc['ff']}",
                         "input": "[CLS] question [SEP] answer, SentencePiece answer vocabulary (ours), segment embeddings",
                         "parameters": f"{res['params']:,}"},
        "Training data": f"{st['cfg']['source']} labelled answers (person-wise split) + synthetic off-topic pairs "
                         "(question with another question's answer). SYNTHETIC for now.",
        "Results (test) - transformer vs TF-IDF baseline": {k: f"{ours[k]} vs {baseline[k]}" for k in baseline},
        "Off-topic check": f"mean predicted relevance {res['off_topic_test']['mean_relevance']}, "
                           f"{res['off_topic_test']['flagged_follow_up']:.0%} flagged for follow-up",
        "Notes": ["The synthetic answers are built from key-point sentences, so bag-of-words baselines are strong; "
                  "the gap to the transformer should be re-measured on our transcripts.",
                  "MIT Interview transcripts (on request) are added through answers_table once converted."],
    })
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
