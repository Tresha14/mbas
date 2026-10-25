# Model card: Face + landmark detector (BlazeFace-style)

_Generated 2026-10-25 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

Detect faces and 5 landmarks (eyes, nose, mouth corners) in a webcam frame; eye landmarks feed the eye-crop extractor for the gaze/iris CNN.

## Architecture

| | |
| --- | --- |
| type | single-shot, BlazeBlocks (depthwise 5x5 + pointwise, residual) |
| input | 128x128 RGB, letterboxed |
| anchors | 896 (16x16 x 2 + 8x8 x 6), square |
| parameters | 382,740 |
| heads | face logit, box (cx, cy, w, h), 5 key points |

## Training data

WIDER FACE train (90%) with RetinaFace 5-point landmarks; random face-centred square crops (1.4-6x face size), photometric jitter, horizontal flip with landmark swap.

## Training

| | |
| --- | --- |
| epochs | 60 |
| batch | 64 |
| optimiser | AdamW lr 0.002, cosine, warm-up 500 |
| loss | focal (alpha .25, gamma 2) + smooth-L1 box + 0.5 x smooth-L1 landmarks |
| curves | docs/metrics/face_detector_curves.png |
| logs | runs/face_detector (TensorBoard) |

## Results (WIDER val = our test split)

| | |
| --- | --- |
| webcam-style crops AP@0.5 | 0.8574 |
| webcam-style crops recall@0.5 | 0.7347 |
| full images AP@0.5 (faces >= 8% of input) | 0.5345 |
| landmark NME (val, inter-ocular) | 117.9912 |
| CPU latency (torch, 4 threads) | 8.136 ms (122.9 fps) |
| PR curve | docs/metrics/face_detector_pr.png |

## Limitations

- Faces smaller than ~10 px at 128 px input are not detected (not needed for webcams).
- Not yet fine-tuned on our own frames: the 1,500-frame key-point set from our recordings is the next step once sessions exist (FaceDataset accepts the same format).
- WIDER is crowd/event photography; webcam lighting (backlit, dim) is covered only by photometric augmentation until our recordings are added.
