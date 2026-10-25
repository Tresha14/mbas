"""Synthetic dataset labels -> the same parquet schema export_labels.py writes.

Lets Stage 3+ run end to end before our own sessions are labelled. Annotator 1
is the label column in interview/*.csv; annotator 2 comes from annotations/
(20% of sessions). HIDDEN columns are never copied.

    python -m tools.synthetic_to_labels
Output: data/processed/labels/synthetic/*.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_labels import HESITATION_TYPES, primary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "raw" / "mbas_synthetic_dataset"
OUT = ROOT / "data" / "processed" / "labels" / "synthetic"
ANN1, ANN2 = "ANN1", "ANN2"


def convert(syn: Path = SYN, out: Path = OUT) -> dict[str, int]:
    iv = syn / "interview"
    win = pd.read_csv(iv / "windows.csv")
    ans = pd.read_csv(iv / "answers.csv")
    hes = pd.read_csv(iv / "hesitation_events.csv")
    bank = pd.read_csv(syn / "reference" / "question_bank.csv")[["question_id", "text"]]
    w2 = pd.read_csv(syn / "annotations" / "windows_annotator2.csv")
    a2 = pd.read_csv(syn / "annotations" / "answers_annotator2.csv")

    wcols = ["window_id", "session_id", "participant_id", "start_s", "end_s", "answer_id",
             "attention_label", "engagement_label"]
    w1 = win[wcols].assign(window_index=win["window_id"].str.split("_W").str[1].astype(int),
                           flags="", annotator=ANN1, annotation_rank=0)
    w2 = (win[[c for c in wcols if c not in ("attention_label", "engagement_label")]]
          .merge(w2[["window_id", "attention_label", "engagement_label"]], on="window_id")
          .assign(window_index=lambda d: d["window_id"].str.split("_W").str[1].astype(int),
                  flags="", annotator=ANN2, annotation_rank=1))
    windows_all = pd.concat([w1, w2], ignore_index=True)

    acols = ["answer_id", "session_id", "participant_id", "question_id", "question_category",
             "question_start_s", "answer_end_s"]
    base = ans[acols].merge(bank, on="question_id", how="left").rename(columns={
        "question_category": "category", "question_start_s": "ask_s", "answer_end_s": "end_s", "text": "question_text"})
    lab1 = ans[["answer_id", "answer_text", "quality_1to5", "relevance", "follow_up_needed",
                "follow_up_reason", "speaking_pattern"]].rename(columns={"answer_text": "transcript"})
    a1 = base.merge(lab1, on="answer_id").assign(annotator=ANN1, annotation_rank=0)
    a2 = (base.merge(ans[["answer_id", "answer_text"]].rename(columns={"answer_text": "transcript"}), on="answer_id")
          .merge(a2[["answer_id", "quality_1to5", "relevance", "follow_up_needed", "follow_up_reason",
                     "speaking_pattern"]], on="answer_id")
          .assign(annotator=ANN2, annotation_rank=1))
    answers_all = pd.concat([a1, a2], ignore_index=True)

    h = hes[hes["event_type"].isin(HESITATION_TYPES)].merge(
        ans[["answer_id", "participant_id"]], on="answer_id", how="left")
    hes_all = h[["session_id", "answer_id", "participant_id", "event_type", "start_s", "end_s", "token"]].assign(
        annotator=ANN1, annotation_rank=0)

    out.mkdir(parents=True, exist_ok=True)
    files = {
        "windows_annotations.parquet": windows_all, "answers_annotations.parquet": answers_all,
        "hesitation_annotations.parquet": hes_all,
        "windows.parquet": primary(windows_all, "window_id"), "answers.parquet": primary(answers_all, "answer_id"),
        "hesitation_events.parquet": hes_all.drop(columns=["annotation_rank"]),
    }
    for name, df in files.items():
        df.to_parquet(out / name, index=False)
    return {k: len(v) for k, v in files.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SYN))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    for k, v in convert(Path(args.src), Path(args.out)).items():
        print(f"{args.out}/{k}: {v} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
