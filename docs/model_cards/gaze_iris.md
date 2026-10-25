# Model card: Gaze + iris CNN

_Generated 2026-10-25 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

Estimate gaze direction (pitch, yaw), the iris centre in each eye crop and on/off-screen from two 36x60 eye crops plus head pose. Feeds attention signals: off-screen %, glance-away count, saccade rate, blink rate, reading-pattern flag (ml/vision/gaze_iris/signals.py).

## Architecture

| | |
| --- | --- |
| eye encoder | 4 conv blocks (32-64-128-128), shared by both eyes |
| trunk | concat(2 x 128 + head pose 2) -> 256 -> 128 |
| heads | gaze 2, iris 4, on-screen 1 |
| parameters | 1,141,287 |

## Training data

MPIIGaze normalised eye images, person-wise split (train/val/test persons in data/splits/public/mpiigaze). Flip augmentation swaps eyes and mirrors yaw.

## Iris targets

eyeball model fitted to a classical dark-pupil detector: {'ax': 0.48728672809456614, 'bx': -0.22662725765336847, 'ay': 0.40219017641269905, 'by': -0.1708512532725962, 'n': 11842, 'corr_x_yaw': -0.307, 'corr_y_pitch': -0.211}

## Training

| | |
| --- | --- |
| epochs | 25 |
| batch | 256 |
| loss | L1 gaze + 0.5 L1 iris + BCE on-screen (masked) |
| curves | docs/metrics/gaze_iris_curves.png |

## Results (held-out persons)

| | |
| --- | --- |
| mean angular error | 5.284 deg |
| median angular error | 5.188 deg |
| mean-gaze baseline | 9.635 deg |
| on/off-screen F1 | pending our labelled windows (see note) |
| CPU latency | 36.602 ms |

## Limitations

- MPIIGaze crops are perspective-normalised; our webcam crops use roll correction only, so a domain gap remains until fine-tuning on our 60 s calibration captures.
- On/off-screen in the live system: per-session calibration box on predicted gaze (signals.ScreenCalibration), not the untrained on-screen head.
- Glasses with strong reflections and very dim light reduce accuracy (bias check, Stage 10).
