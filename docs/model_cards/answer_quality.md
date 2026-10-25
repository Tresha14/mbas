# Model card: Answer-quality transformer

_Generated 2026-10-25 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

From the question and the answer transcript: relevance 0-1, quality 1-5, STAR components (behavioural questions), completeness, follow-up yes/no and reason.

## Architecture

| | |
| --- | --- |
| encoder | 4 pre-norm transformer layers, d_model 256, 4 heads, FF 1024 |
| input | [CLS] question [SEP] answer, SentencePiece answer vocabulary (ours), segment embeddings |
| parameters | 3,520,018 |

## Training data

synthetic labelled answers (person-wise split) + synthetic off-topic pairs (question with another question's answer). SYNTHETIC for now.

## Results (test) - transformer vs TF-IDF baseline

| | |
| --- | --- |
| relevance_mae | 0.0672 vs 0.0719 |
| relevance_r | 0.9477 vs 0.9248 |
| quality_accuracy | 0.6913 vs 0.5814 |
| quality_mae | 0.3907 vs 0.4546 |
| quality_qwk | 0.8574 vs 0.7726 |
| follow_up_f1 | 0.8829 vs 0.8523 |
| reason_macro_f1 | 0.5485 vs 0.4981 |

## Off-topic check

mean predicted relevance 0.0982, 97% flagged for follow-up

## Notes

- The synthetic answers are built from key-point sentences, so bag-of-words baselines are strong; the gap to the transformer should be re-measured on our transcripts.
- MIT Interview transcripts (on request) are added through answers_table once converted.
