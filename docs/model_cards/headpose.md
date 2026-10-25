# Model card: Head-pose CNN

_Generated 2026-10-25 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

Yaw, pitch and roll of the candidate's head from a 64x64 face crop; input to the gaze CNN, the attention score and the head-movement features.

## Architecture

| | |
| --- | --- |
| layers | 4 conv blocks (32-64-128-128) + global average pool + MLP 64 -> 3 |
| parameters | 591,331 |
| output | degrees (tanh-free, scaled x30) |

## Training data

20700 WIDER FACE train faces >= 40 px with RetinaFace landmarks; labels from solvePnP on a generic 5-point 3-D face (ml/vision/headpose/geometry.py). Box jitter, photometric augmentation, horizontal flip (yaw and roll negated). Split by image.

## Results (held-out images)

| | |
| --- | --- |
| MAE yaw | 10.753 deg |
| MAE pitch | 10.697 deg |
| MAE roll | 5.484 deg |
| mean-pose baseline (yaw/pitch/roll) | {'yaw': 26.008, 'pitch': 16.434, 'roll': 10.485} |
| CPU latency | 7.669 ms |

## Limitations

- Labels are geometric estimates from a generic face model, so errors are measured against pseudo-labels; absolute accuracy against ground truth needs our scripted 'look at camera' segments (0, 0, 0) or a calibrated rig.
- Pitch from 5 landmarks is the least reliable angle (depends on the face's proportions).
