# InterLens (MBAS): project status and handoff

_Last updated 19 November 2026. Code: https://github.com/Tresha14/mbas (branch `main`)._

InterLens (formal title: MBAS, Multimodal Behavioural Analysis System) is an interview and meeting copilot. While a candidate answers, it
watches the video, listens to the audio and reads the transcript, then tells the interviewer how attentive, engaged,
fluent and confident the candidate is, which answers need a follow-up, and finally gives a rated report.

**The hard rule of the project:** no pre-trained models and no AI APIs. Every model below is our own PyTorch code,
trained from random weights on public raw datasets, the synthetic dataset and our own recordings.

---

## 1. Short version

| Part | Status |
| --- | --- |
| 11 of 12 models | **trained and evaluated** |
| Engagement CNN-GRU (12th model) | waiting for the DAiSEE dataset (needs a request form); the app uses the engagement MLP instead meanwhile |
| Web app (interviewer cockpit, candidate room, meeting mode, report, feedback) | **done** |
| Google Meet / Zoom / Teams interviews | **done** (needs one real-call test) |
| Detailed rated report (grade A-E, blink rate, voice breaks, fillers, pauses, ...) | **done** |
| Automated tests | **63 pass** |
| Real recorded sessions + labelling | **left: needs people** |
| Two-laptop live test + Meet / Zoom / Teams call test | **left: needs people** |
| User study | **left: needs people** |
| Final project report (Word) + viva deck (PowerPoint) | **left**: outlines exist in `docs/` |

---

## 2. The models

All numbers are on held-out test data. Lower is better for error, MAE and WER; higher is better for F1, AP, QWK.
Full details: `docs/model_report.md` and `docs/model_cards/`.

| # | Model | What it does | Trained on | Epochs | Test result |
| --- | --- | --- | --- | ---: | --- |
| 1 | Face + landmark detector | finds the face and 5 points (eyes, nose, mouth) in each frame | WIDER Face | 60 | AP@0.5 0.857, recall 0.735 |
| 2 | Head-pose CNN | head turn left/right, up/down, tilt | WIDER Face (geometric pseudo-labels) | 20 | error yaw 10.8°, pitch 10.7°, roll 5.5° (baseline 26°/16°/10°) |
| 3 | Gaze + iris CNN | where the eyes look + iris position | MPIIGaze | 25 | angular error 5.3° (baseline 9.6°) |
| 4 | Engagement MLP + attention score | engaged / neutral / disengaged and a 0-100 attention score every 5 s | synthetic dataset | - | engagement macro-F1 0.746, attention MAE 3.2 |
| 5 | Engagement CNN-GRU | engagement straight from video clips | DAiSEE (not received yet) | - | **not trained yet** |
| 6 | Voice activity detector (VAD) | speech / silence every 10 ms | LibriSpeech mixed with room noise | 8 | frame F1 0.955 (energy baseline 0.782) |
| 7 | Speech recogniser (Conformer-CTC ASR) + our 4-gram language model | live transcript | LibriSpeech (100 h) | 40 | word error rate 0.137 with our LM (0.159 without) |
| 8 | Disfluency CRNN | hears fillers, repetitions, stretched sounds, blocks | SEP-28k | 25 | macro-F1 0.467 (filler 0.694) |
| 9 | Speaking pattern + hesitation index | fluent / hesitant / rushed / monotone, 0-100 hesitation | synthetic dataset | - | pattern macro-F1 0.810, hesitation MAE 4.3 |
| 10 | Answer-quality transformer | relevance, quality, STAR structure of the answer text | synthetic dataset | 40 | QWK 0.857 (TF-IDF baseline 0.773) |
| 11 | Multimodal fusion network | combines video + voice + text into an answer score, 5 sub-scores, follow-up suggestion and top-3 reasons | synthetic dataset | 150 | score MAE 5.9 (baseline 12.5), follow-up F1 0.959 |
| 12 | Meeting head | engagement and contribution balance per meeting participant | synthetic meetings, tested on AMI | - | AMI balance MAE 0.032 |

Rule-based parts (no learned weights): pause model, prosody (pace, pitch, expressiveness, voice breaks, jitter),
text hesitation (fillers, hedges, repeats), conversation analyser (talk time, turns, interruptions), report rules.

**Honest caveats for the report and viva**
- Models 4, 9, 10, 11 and 12 are trained on the **synthetic** dataset; label them SYNTHETIC wherever their numbers
  appear.
- The ASR is trained on read English audiobooks. Indian-accented interview speech will be harder; we have no test
  recordings for that yet.
- Vision models are not yet fine-tuned on our own webcam frames.

---

## 3. The product (what the app does)

Start everything with `.\tasks.ps1 start-all` (needs Docker Desktop running), then open http://localhost:3020.

| Page | Who uses it | What it shows |
| --- | --- | --- |
| Dashboard | interviewer | create interviews or meetings, copy the candidate link |
| Candidate room | candidate | consent, camera/mic check, video call, screen share; **the candidate never sees any analytics** |
| Cockpit | interviewer | live video, live transcript with fillers highlighted, attention / engagement / hesitation gauges, gaze plot, alerts (looking away, reading from notes), question list, per-answer scores with reasons, follow-up suggestions, confidence-vs-content nudge |
| External call | interviewer | run the interview on **Google Meet, Zoom (browser or desktop app) or Teams**; MBAS captures the call and analyses it the same way |
| Meeting mode | meeting host | per-person talk time, engagement and contribution balance |
| Report | interviewer | see below |
| Feedback | candidate | strengths and 3 improvement tips that the interviewer reviewed and released |

**The report** (after the interview):
- Final rating 0-100 and grade A-E from six scores: content, fluency, composure, delivery, eye contact, responsiveness.
- Behaviour details, each with a typical range and a good / watch / concern label:
  - eyes: blink rate, eye contact %, glances away, gaze steadiness, rapid eye movements, iris movement, reading pattern;
  - voice: pace, expressiveness, voice breaks (cracks) with timestamps, voice shakiness (jitter), volume;
  - words: filler words counted per word ("um" ×9, "hmm" ×2), overused words, hedging ("I think", "maybe"), repeats;
  - pauses: response delay, long pauses (> 1 s), very long pauses (> 3 s), longest silence.
- Extras: confidence-vs-content chart ("sounds sure but off target" / "knows it but sounds unsure"), "under pressure"
  comparison of easier vs harder questions, moments to rewatch that jump the replay, PDF export of the whole report.

---

## 4. Work done in the latest session (18-19 Nov 2026)

1. Finished training: disfluency 14 → 25 epochs (F1 0.460 → 0.467); ASR 13 → 40 epochs (WER 0.345 → 0.137 on test).
2. Zoom and Teams support (Meet already existed): one "External call" menu in the cockpit.
3. Detailed rated report: scorecard + grade, all the behaviour metrics above, confidence-vs-content, under pressure,
   moments to rewatch, print-all PDF.
4. New signals: voice-break and jitter detection (tuned on real speech so normal speech scores near zero), longest
   and very long pauses, per-word filler / crutch-word breakdown.
5. Bug fix: the live voice detector treated quiet room noise as speech, which broke response delay and pause numbers.
6. Tests added; 63 pass. Everything committed and pushed.

**19 Nov, after merging Aryan's commits (live flags, models page, preflight, ...):**
7. Fixed the answer-quality crash: the answer tokenizer had been overwritten by an 8,000-piece one; the original
   1,130-piece tokenizer was rebuilt and verified (test scores identical to the original: QWK 0.8574). The tokenizers
   are now **in git**, and `python -m ml.common.text` no longer overwrites existing ones (`--force` to rebuild).
8. Fixed lost audio at the start of the first interview: the workers loaded their models (~20 s) before marking
   where to start reading, so the candidate's first ~20 s were skipped. They now read from the start message and
   load the models when they start. Live test (fake candidate through the real LiveKit room): full transcript from
   the first word, answer scored with all models, report grade built, 0 worker errors; vision 24 ms/frame.
9. 82 tests pass; `python -m tools.preflight --services` is READY.

---

## 5. What is left

### Needs people (cannot be done on the computer alone)
1. **Record real sessions** with the recorder (`.\tasks.ps1 recorder-api` + `recorder-web`), a few participants.
   Follow `docs/data_protocol.md`; everyone signs the consent form.
2. **Label them** (`docs/annotation_guide.md`): about 1,500 face key-point frames and the gaze on/off-screen windows.
   Then fine-tune the face detector and report gaze F1 on our own data.
3. **Live test** (`docs/e2e_test.md`): two laptops over the LAN with real cameras, one meeting with 3+ people, and
   one interview each over Google Meet, Zoom (browser and desktop app) and Teams.
4. **User study** (`docs/user_study/README.md`): a few people use the app and fill in
   `docs/user_study/responses_template.csv` (save it as `responses.csv`); then run
   `python -m tools.user_study docs/user_study/responses.csv` for the results page.
5. **Demo video** of the full flow (`docs/demo_script.md`).

### Needs a dataset request
6. **DAiSEE**: request it from the authors (academic form), put it in `data/raw/public/daisee`, then train the
   engagement CNN-GRU: `python -m ml.vision.engagement.train --stage pretrain`, then `--stage finetune` once our
   windows are labelled, and `python -m ml.vision.engagement.eval`. Optional: the app already works without it.

### Writing
7. **Project report** (Word, 60-80 pages) from `docs/project_report_outline.md`.
8. **Viva deck** (PowerPoint, 15 slides) from `docs/viva_deck_outline.md`.
   Both need the team names, guide name and college name. Real-session, live-test and user-study results go in
   once items 1-4 are done; until then mark them as future work.

---

## 6. Practical notes for whoever continues

- **Trained model files are not on GitHub** (too big). Get `mbas_models_2026-11-19.zip` (121 MB, v2 with the fixed
  tokenizer) from Tresha and extract it into the `mbas/` folder; it fills `checkpoints/`, `ml/export/onnx/` and
  `data/processed/`. Then `python -m tools.preflight` must say READY.
- Start everything: Docker Desktop, then `.\tasks.ps1 start-all` (or `api`, `workers`, `web` in three terminals),
  open http://localhost:3020. Before a demo: `python -m tools.preflight --services`.
- Setup: `README.md` → *Setup (Windows, PowerShell)*. Python venv in `.venv`, frontend in `frontend/` (`npm install`).
- Run the tests: `.\tasks.ps1 test`.
- Retrain a model: commands per model in `README.md` → *Stages 4-8*; every training script supports `--resume`.
- Long training on a laptop: keep it **plugged in** and set Windows sleep to **Never** while plugged in. Sleep or
  battery mode pauses or slows training (this happened once).
- Push to `Tresha14/mbas` branch `main`.
