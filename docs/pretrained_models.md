# Pre-trained models used (everything else is trained by us)

Every model in this project is trained from scratch by us **except the optional Whisper speech recogniser below**.
It is off by default and clearly labelled wherever it is used.

| Model | What for | Source | Licence | Size | Switch |
| --- | --- | --- | --- | --- | --- |
| Whisper `base.en` (CTranslate2 build `Systran/faster-whisper-base.en`, run with the `faster-whisper` package) | Live transcript of what the candidate and interviewer say | OpenAI Whisper; conversion by Systran on Hugging Face | MIT (Whisper, faster-whisper and the converted weights) | about 150 MB | `MBAS_ASR=whisper` in `.env` (default `own`) |

## Why it is there

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
