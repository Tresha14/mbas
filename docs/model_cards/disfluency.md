# Model card: Disfluency CRNN

_Generated 2026-10-25 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

Audio-side hesitation: filled pause, repetition, prolongation, block in each 3 s window of the candidate's speech (counts feed the behaviour JSON and the hesitation index).

## Architecture

| | |
| --- | --- |
| front end | 80-band log-mel, per-clip normalisation, SpecAugment (train) |
| network | 4 conv blocks -> BiGRU(96) -> attention pooling -> 4 sigmoid outputs |
| parameters | 1,008,357 |

## Training data

SEP-28k (podcast clips, label present when >= 2 of 3 annotators marked it), split by episode; class-balanced BCE (pos_weight). Our hesitation spans are added once labelled.

## Results (test episodes)

| | |
| --- | --- |
| filler F1 / AUC | 0.6804 / 0.8667 (always-yes F1 0.3966) |
| repetition F1 / AUC | 0.4033 / 0.7019 (always-yes F1 0.337) |
| prolongation F1 / AUC | 0.4618 / 0.8521 (always-yes F1 0.1728) |
| block F1 / AUC | 0.293 / 0.6723 (always-yes F1 0.2135) |

## Limitations

- SEP-28k is mostly people who stutter, recorded on podcasts; Indian-English fillers such as 'matlab' are covered by the text-side rules until our labelled answers are added.
- Blocks and prolongations are rare in interviews; their F1 is reported for completeness.
