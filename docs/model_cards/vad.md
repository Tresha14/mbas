# Model card: Voice activity detector

_Generated 2026-10-25 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

Speech / non-speech every 10 ms on each participant's audio track; feeds the ASR, the pause model and the conversation analyser.

## Architecture

| | |
| --- | --- |
| front end | 40-band log-mel (25 ms / 10 ms) |
| network | 4 dilated 1-D conv layers (64 ch, receptive field ~330 ms) |
| post-processing | hysteresis 0.5 / 0.35, 200 ms hangover, 120 ms minimum speech |
| parameters | 45,377 |

## Training data

LibriSpeech (+ Common Voice) utterances placed with random silences in 8 s mixtures, mixed with recorded noise (data/raw/noise) or generated room noise (white/pink/brown/hum/babble) at 0-25 dB SNR; targets from the clean utterances' energy mask.

## Results (test mixtures, unseen speakers)

| | |
| --- | --- |
| frame F1 raw | 0.9726 |
| frame F1 smoothed | 0.9547 |
| energy-threshold baseline | 0.7818 |
| CPU latency per 640 ms chunk | 3.473 ms |

## Limitations

Targets are energy masks of clean read speech; our sessions (spontaneous speech, laptop mics) are the real test once labelled; record lab/hostel noise into data/raw/noise to replace generated noise.
