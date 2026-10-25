# Model card: Multimodal fusion network

_Generated 2026-10-25 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

Communication effectiveness 0-100 with clarity, confidence, engagement, relevance and fluency sub-scores, follow-up yes/no + reason, with the top-3 reasons for every score.

## Architecture

| | |
| --- | --- |
| encoders | one MLP per modality (vision, prosody, hesitation, content, conversation) -> 64-d token |
| fusion | learned query, 4-head attention over modality tokens (weights = modality importance) + mean pool |
| heads | 6 sigmoid scores, follow-up logit, 6-way reason |
| parameters | 85,837 |
| regularisation | dropout 0.3, modality dropout 0.15 |

## Data

synthetic answers, person-wise split; content modality = out-of-fold answer-model outputs. SYNTHETIC for now: retrain on MIT Interview ratings + our labels with the same script.

## Ablation (test, effectiveness MAE / follow-up F1)

| | |
| --- | --- |
| video only | 10.712 / 0.7474 |
| audio only | 7.818 / 0.7896 |
| text only | 8.169 / 0.8765 |
| all modalities | 5.876 / 0.9586 |

## Baselines

| | |
| --- | --- |
| ridge_effectiveness_mae | 5.709 |
| mean_effectiveness_mae | 12.486 |

## All modalities (test)

| | |
| --- | --- |
| effectiveness MAE | 5.876 |
| Pearson r | 0.8917 |
| follow-up F1 | 0.9586 |
| reason macro-F1 | 0.6367 |
| mean modality attention | {'vision': 0.20469999313354492, 'prosody': 0.05979999899864197, 'hesitation': 0.10189999639987946, 'content': 0.5889999866485596, 'conversation': 0.044599998742341995} |

## Explanations

Shapley values by permutation sampling (48 orderings) -> templates (ml/fusion/explain.py); 100% of test answers get 3 reasons (examples: docs/metrics/fusion_explanations.json).

## Use

Decision support for the interviewer, never an automatic reject.
