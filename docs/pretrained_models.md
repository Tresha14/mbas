# Pre-trained models used (everything else is trained by us)

Every model in this project is trained from scratch by us **except the two optional pre-trained models below** (Whisper
for speech, MediaPipe for the face). Both are off by default and clearly labelled wherever they are used.

| Model | What for | Source | Licence | Size | Switch |
| --- | --- | --- | --- | --- | --- |
| Whisper `base.en` (CTranslate2 build `Systran/faster-whisper-base.en`, run with the `faster-whisper` package) | Live transcript of what the candidate and interviewer say | OpenAI Whisper; conversion by Systran on Hugging Face | MIT (Whisper, faster-whisper and the converted weights) | about 150 MB | `MBAS_ASR=whisper` in `.env` (default `own`) |
| MediaPipe FaceLandmarker (`face_landmarker.task`, Google) | Face found in the frame, head pose, eye direction, blink scores | Google MediaPipe model zoo | Apache-2.0 | 3.7 MB | `MBAS_FACE=mediapipe` in `.env` (default `own`) |

## Why Whisper is there

Our own recogniser (a Conformer-CTC trained on about 100 h of LibriSpeech, with our 4-gram language model) reaches a
word error rate of 0.137 on LibriSpeech test-clean, but it was trained only on clean, read audiobook speech. On real
laptop and phone microphones, with room noise and accents, it produces unreadable text. Measured on a real call
(interview 49, 60 s of speech): our model gave "woman'in being bevib can you leave f and whly's pie…" with and
without the language model; Whisper gave "SQL database in a structured database, where there is no SQLR document like
structure…". There is no word-for-word reference for that call, so this is a qualitative comparison, not a WER.
The audio itself was checked and is clean (normal level, no clipping or dropped chunks, normal pitch range).

## How it behaves

- It transcribes each finished utterance once (the voice detector closes an utterance after 0.7 s of silence, or at
  12 s), so a line appears about a second after the speaker pauses; there are no live partial words.
- Whisper can invent text (such as "Thank you.") from silence or noise. Segments it marks as probably not speech,
  repetitive loops and stock phrases on very short clips are dropped; silence produced no text in our test.
- The cockpit shows "Speech recognition here is a pre-trained model (Whisper)" when it is active.
- If the package or model is missing, the worker falls back to our own recogniser and says so in its log.
- **Fillers.** Whisper leaves out spoken "um" and "uh" unless it is prompted. The worker gives it a short prompt full of
  fillers, so "um", "uh", "hmm" and "like" reach the transcript and the filler rate is not always zero. If it ever echoes
  the prompt back on noise, that text is discarded. `MBAS_WHISPER_PROMPT=off` in `.env` turns the prompt off.
- **Runs in the background.** Whisper transcribes on its own thread. The audio loop never waits for it, so a slow recogniser
  delays the words a little but does not lose audio. At the end of an interview the worker processes any audio still
  queued and waits for Whisper to finish.

## Setup

```powershell
.\.venv\Scripts\python.exe -m pip install faster-whisper
```
The model downloads once (about 150 MB) on first use into the Hugging Face cache. Then set `MBAS_ASR=whisper` in `.env`
and restart the workers (`.\tasks.ps1 workers`); the log shows `[speech] recogniser: Whisper base.en (pre-trained)`.
`MBAS_WHISPER_MODEL=small.en` is more accurate and about three times slower on CPU.

## Everything the speech worker still computes with our own models

Voice-activity detection, the disfluency model, pitch and pause measurements, the behaviour model, the answer-quality
model and fusion do not depend on the recogniser; only the words (and so word-based measures such as filler and hedge
counts and the answer text given to the answer-quality model) come from Whisper when it is on.


## MediaPipe face tracking

**Why.** Our own face detector is small (AP 0.857, recall 0.735 on WIDER) and our eye-openness estimate loses blinks
behind glasses. On one real recording of a candidate wearing glasses (15 s, the same frames through both pipelines):

| | our detector | MediaPipe |
| --- | --- | --- |
| windows with a face | 47 % | 100 % |
| blink rate | 0.0 / min (blinks lost) | 21.8 / min |
| time per frame (CPU) | 31 ms | 15 ms |

This is one clip, not a benchmark. There is no ground truth for the blinks, but 22 per minute is a normal human rate.

**What it replaces.** Only the per-frame face / head-pose / gaze / blink step. The 5 s window features (glances, reading
pattern, saccades, iris movement), the report and the engagement / attention models work unchanged on its output. Those
two models were trained on features from our own pipeline, so their outputs can shift a little with MediaPipe.

**Setup.**
```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps mediapipe matplotlib
```
`--no-deps` on purpose: mediapipe's metadata asks for `opencv-contrib-python`, which would sit on top of the OpenCV that is
already installed and can break video decoding; the face-landmark API does not use OpenCV. (`pip check` then notes the
unmet requirement; that is expected.) Put `face_landmarker.task` (3.7 MB, Google's published file) in
`checkpoints/mediapipe/`, set `MBAS_FACE=mediapipe` in `.env` and restart the workers. The cockpit then says
"Face detected (MediaPipe)". If the package or the file is missing, the worker logs it and uses our own models.

**Sign conventions** (checked against the nose position in real frames): head yaw is positive towards the subject's left,
head pitch positive when looking down; gaze uses the convention of `ScreenCalibration` (pitch negative looking down).


## Why the transcript fell behind on a real Meet call (fixed)

A real Meet interview (196 s) had a transcript and speech stats that stopped at 106 s. Cause: answer analysis takes about
10 s (the disfluency model alone 6 s) and ran inside the loop that reads the audio, so every question stalled the loop; at
the end the worker handled the "stop" message before the audio still waiting in the queue and threw that audio away.
Now answer analysis and Whisper run on their own threads, and queued audio is processed before an interview is closed.
The same recording replayed in real time through the fixed workers: transcript to 201 s (was 106 s), candidate words 135
(was 61), answers analysed 4 (was 1), filler / hedge rate present in 34 of 40 speech-stat rows (was 7 of 21).
Filler and hedge rates are now computed over the last minute of words, from 8 words on (before: 30 s and 15 words, which
usually gave "no value").
