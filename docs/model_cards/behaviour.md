# Model card: Speaking-pattern classifier + hesitation index

_Generated 2026-10-25 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

Per answer: speaking pattern (confident / balanced / hesitant / rushed / monotone) and a 0-100 hesitation index; inputs come from the pause model, prosody extractor and text/audio hesitation.

## Models

| | |
| --- | --- |
| speaking pattern | MLP 2 x 64, class-weighted CE |
| hesitation index | 100 * sigmoid(b + sum softplus(w_i) x_i) (monotone, interpretable) vs MLP |

## Data

synthetic answers, person-wise split (2255 / 451 / 473) - SYNTHETIC; re-train on our labels with the same script once sessions are labelled.

## Results (test)

| | |
| --- | --- |
| speaking pattern macro-F1 | 0.8098 |
| majority baseline | 0.1129 |
| hesitation index MAE (weighted sum) | 4.317 |
| hesitation index MAE (MLP) | 2.796 |
| mean baseline MAE | 13.84 |

## Largest hesitation weights

filler_rate_per_100w 0.33, disfluency_rate_per_100w 0.22, prolongation_count 0.12, response_latency_s 0.11, repetition_count 0.10

## Text hesitation rules vs dataset counts (Pearson r)

| | |
| --- | --- |
| filler_count | 0.8183 |
| repetition_count | 0.9604 |
| restart_count | 0.9434 |
| hedge_count | 0.8684 |
