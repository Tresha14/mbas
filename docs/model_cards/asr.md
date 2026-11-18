# Model card: Conformer-CTC speech recogniser

_Generated 2026-11-19 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

Streaming transcription of each participant's audio track (< 2 s delay); word timestamps for the transcript panel, the pause model and the fusion features.

## Architecture

| | |
| --- | --- |
| front end | 80-band log-mel, per-utterance CMVN, SpecAugment (2 freq x 27, 10 time x 5%) |
| encoder | Conv2d x2 subsampling (40 ms) + 12 Conformer blocks, d_model 176, 4 heads, conv kernel 31 |
| output | CTC over our SentencePiece vocabulary (1000 pieces) |
| parameters | 9,724,648 |

## Training

LibriSpeech train-clean-100 (+ Common Voice Indian English + our transcripts when available); speed perturbation 0.9/1.0/1.1, Noam schedule, fp16, dynamic chunk training (streaming), checkpoint averaging.

## Language model

our interpolated Kneser-Ney 4-gram (ml/speech/lm/ngram.py), alpha 0.3, beta 1.0

## WER (greedy / beam / beam+LM)

| | |
| --- | --- |
| librispeech_test_clean | 0.159 / 0.1573 / 0.1366 |
| common_voice_indian_test | no data yet |
| our_sessions_test | no data yet |

## Whisper-small (benchmark only, not in the product)

not run

## Why a gap to Whisper is expected

Whisper-small was trained on 680,000 h of audio; ours on ~100 h. The gap comes from training data size, not from the architecture.
