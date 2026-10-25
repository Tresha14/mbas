# Model card: Meeting head (per-participant engagement + contribution balance)

_Generated 2026-10-25 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

Meeting mode: engagement 0-1 and contribution balance 0-1 per participant, from talk share, relative turn features and (camera on) on-screen ratio, face motion, head movement, multitasking share.

## Model

MLP 2 x 32 on standardised features; trained with camera-off copies so it works audio-only.

## Data

synthetic meetings (participants split person-wise); generalisation test on AMI (171 real meetings).

## Results

| | |
| --- | --- |
| synthetic test engagement MAE | 0.0381 |
| synthetic test balance MAE | 0.0283 |
| AMI balance MAE (relative features) | 0.0317 |
| AMI balance MAE (absolute features, first try) | 0.1201 |

## Finding

Absolute turn rate and turn length did not transfer from the simulator to real meetings (AMI turns are 3-6x shorter); within-meeting relative turn features fixed most of the drop.
