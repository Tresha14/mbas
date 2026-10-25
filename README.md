# MBAS — Multimodal Behavioural Analysis System

An AI-assisted interview or meeting analysis system using video, audio, and conversational signals. The system
analyzes attention, engagement, hesitation, speaking patterns, and responses to assess communication effectiveness
and identify areas requiring follow-up.

**Hard rule:** no pre-trained models or AI APIs. Every model is defined in PyTorch and trained from random
initialisation on our own recorded dataset plus public raw datasets.

## Build status (10-stage plan)

| Stage | Content | Status |
| --- | --- | --- |
| 1 | Repo skeleton, docker compose, dataset recorder, `check_session.py` | done (see below) |
| 2 | Dataset protocol, annotation, public datasets, splits | tools done; recording + labelling in progress (see `docs/dataset_card.md`) |
| 3 | Data pipeline + features (`ml/common`) | done: `.\tasks.ps1 data` |
| 4–5 | Vision: face + landmarks, head pose, gaze/iris, engagement, attention, ONNX export | trained on WIDER / MPIIGaze / synthetic; CNN-GRU waits for DAiSEE or our labels |
| 6 | VAD + Conformer-CTC ASR + our 4-gram LM + streaming + mic demo | code done; trains on LibriSpeech (`.\tasks.ps1 train-all vad lm asr`) |
| 7 | Disfluency CRNN, pauses, prosody, speaking patterns, conversation, behaviour JSON | done (SEP-28k + synthetic) |
| 8 | Answer-quality transformer vs TF-IDF, fusion + ablation + explanations, meeting head on AMI | done (synthetic + AMI) |
| 9 | FastAPI + workers + LiveKit bot + Next.js cockpit / candidate room / meeting mode | done |
| 10 | Reports + release flow, model report, bias check, user-study kit, one-command start, viva material | done; user study and demo video need people |

Results for every model: `docs/model_report.md` (generated), per-model cards in `docs/model_cards/`, numbers in
`docs/metrics/*.json`. Anything trained on the synthetic dataset is labelled SYNTHETIC there.

## Repo layout

```
data/raw/                     recorded sessions (git-ignored) + mbas_synthetic_dataset/ (committed)
data/processed  data/splits
recorder/                     dataset recording app
  backend/                    FastAPI: sessions, LiveKit tokens, chunk uploads, finalize
  web/                        Next.js 15: /interviewer and /candidate
  questions.json              40 interview questions (HR, behavioural, technical)
tools/check_session.py        verifies a recorded session folder
tools/make_questions.py       builds recorder/questions.json from the synthetic question bank
tools/import_to_labelstudio.py, export_labels.py, agreement.py, make_splits.py, dataset_stats.py   Stage 2
tools/datasets/               public dataset download + conversion scripts
infra/labelstudio/            Label Studio labelling configs
ml/{common,vision,speech,behaviour,response,fusion,export}/
backend/  frontend/           product app (Stage 9)
infra/                        livekit.yaml, Caddyfile
docs/                         model cards, metrics, schemas
tests/
```

## Synthetic dataset

`data/raw/mbas_synthetic_dataset/` holds the simulated tabular dataset (120 participants, 289 interview sessions,
40 meetings, person-wise splits). See its `README.md` and `DATA_DICTIONARY.md`. It contains **features and labels,
not raw video/audio**; the vision and speech models still need real frames and audio (Stage 2). Columns marked
HIDDEN (`hidden_ground_truth/`) must never be used as model inputs. Always label it as synthetic in reports.

## Setup (Windows, PowerShell)

Prerequisites: Python 3.11, Node 20+, Docker Desktop, Chrome or Edge.

```powershell
cd C:\minor_prjct_final\mbas
python -m venv .venv
.\tasks.ps1 install                # torch (CUDA 12.4) + requirements.txt
copy .env.example .env             # then edit MBAS_LAN_IP for two-laptop recording
.\tasks.ps1 recorder-web-install
```

On Linux/macOS use the same targets through `make` (`make venv install torch-cuda recorder-web-install`).

Host ports default to 5442 (Postgres), 6389 (Redis), 8010 (recorder API), 3010 (recorder web), 7880-7882 (LiveKit),
8443 (HTTPS). They are set in `.env`, chosen so they don't clash with other local projects on 3000/5432/6379/8000.

## Run the recorder

Three terminals:

```powershell
.\tasks.ps1 up              # postgres, redis, livekit, caddy
.\tasks.ps1 recorder-api    # FastAPI on :8010
.\tasks.ps1 recorder-web    # Next.js on :3010
```

**Same laptop (quick test):** open `http://localhost:3010/interviewer` in one window and the candidate link it
shows in another window.

**Two laptops (real session):**

1. On the host laptop run `ipconfig`, put its IPv4 address in `.env` as `MBAS_LAN_IP`, then `.\tasks.ps1 down; .\tasks.ps1 up`.
2. Allow inbound TCP 8443, 7880, 7881 and UDP 7882 in Windows Firewall on the host laptop.
3. Interviewer opens `https://<LAN_IP>:8443/interviewer`, candidate opens `https://<LAN_IP>:8443/candidate?room=<CODE>`.
   Both accept the self-signed certificate warning once (browsers only allow camera/mic on https or localhost).

### Session flow

1. Interviewer and candidate each pass the consent screen. The candidate also enters participant ID, age band,
   gender, glasses and accent, which are saved to `meta.json`. Recording and tokens are refused until both consent.
2. Candidate clicks **Share screen** (keep it on for the whole session).
3. Interviewer clicks **Start recording**. Both browsers sync to the server clock and start their recorders.
4. **Run calibration** shows a 60 s moving dot on the candidate screen (9-point fixation then smooth pursuit),
   saved to `calibration.json`, one row per animation frame.
5. Use **Ask** / **Answer ended** on each question, which saves the events to `question_events.json`.
6. **Stop recording**, wait for uploads to finish, then **Finalize & check** runs `check_session` and shows the result.

### What one session folder contains (`data/raw/<session_id>/`)

| File | Content |
| --- | --- |
| `candidate_webcam.webm` | candidate camera, VP8, 1280x720 at 15 fps |
| `candidate_screen.webm` | candidate screen share, VP8, 15 fps |
| `candidate_mic.wav` / `interviewer_mic.wav` | 16 kHz mono PCM16, one file per participant |
| `question_events.json` | `ask` / `answer_end` with question id, text, category |
| `calibration.json` | dot `x, y, x_norm, y_norm, phase` per frame + screen size |
| `meta.json` | participants, consent, client clock offsets, file list |
| `timeline.json` | recording window and start/end of every track |

All timestamps are milliseconds on the recorder server clock (`server_epoch_ms`). Each browser estimates its offset
NTP-style (minimum-RTT of several `/api/time` pings) before recording.

### Verify a session

```powershell
.\tasks.ps1 check data\raw\<session_id>
# or: .\.venv\Scripts\python -m tools.check_session data\raw\<session_id> --json report.json
```

The check covers: required files, consent and meta fields, 16 kHz mono audio, webcam resolution and fps,
frame gaps, each track covering at least 95% of the recording window, required track durations within 1 s,
**sync drift < 100 ms** per track (media duration vs shared-clock duration), question-event pairing, and
calibration length, rate and bounds. Use `--lenient` to only warn when the screen share or calibration is missing.

## Stage 2: dataset construction and annotation

Documents: `docs/data_protocol.md` (how to run a session, consent text, storage rules),
`docs/annotation_guide.md` (label definitions and examples) and `docs/dataset_card.md` (generated statistics).

```powershell
.\tasks.ps1 labelstudio          # Label Studio on http://localhost:8090 (login: LABEL_STUDIO_USERNAME/PASSWORD in .env)
.\tasks.ps1 labelstudio-import   # cut clips for every checked session, create the 4 projects, import new tasks
#   ... annotators label in the browser (docs/annotation_guide.md) ...
.\tasks.ps1 labels               # export -> parquet, synthetic labels, kappa, person-wise splits, dataset card
.\tasks.ps1 datasets             # download + convert WIDER FACE, MPIIGaze, LibriSpeech, SEP-28k, AMI (~45 GB)
```

| Step | Tool | Output |
| --- | --- | --- |
| Clips + tasks | `tools/import_to_labelstudio.py` | `data/processed/labelstudio/{media,tasks}`; one timeline task per session (mark exceptions only), per-answer audio/video; `--per-window` adds 5 s window clips |
| Weak labels | `tools/session_log.py` | windows inside the scripted segments of `data/raw/session_log.csv` + calibration are labelled automatically |
| Double labelling | same | 20% of sessions (at least one) go to the `(double-labelled)` projects; assignment kept in `assignment.json` |
| Export | `tools/export_labels.py --pull` | `data/processed/labels/recorded/{windows,answers,hesitation_events}.parquet` + `*_annotations.parquet` (all annotators) |
| Synthetic labels | `tools/synthetic_to_labels.py` | same schema in `data/processed/labels/synthetic/` |
| Agreement | `tools/agreement.py` | Cohen's kappa per label, `docs/metrics/agreement_<source>.json` |
| Splits | `tools/make_splits.py` | `data/splits/<source>/` 70/15/15 by participant, stratified by gender, leakage check |
| Dataset card | `tools/dataset_stats.py` | `docs/dataset_card.md`, `docs/metrics/dataset_stats.json` |
| Public data | `tools/datasets/*.py` | `data/processed/public/<name>/manifest.parquet` + `info.json` (see `tools/datasets/README.md`) |

Second annotator account: create it in Label Studio (Organization → invite link), or with
`docker exec mbas-labelstudio-1 label-studio reset_password --username <email> --password <pw>` for an existing user.
Label Studio 1.23 disables legacy API tokens, so the tools log in with the username/password from `.env`
(or use a personal access token in `LABEL_STUDIO_TOKEN`).

The synthetic dataset's meeting-level `split` column is not person-wise (some meeting participants are interview
test participants); `make_splits` therefore assigns meeting rows by participant and marks such meetings `mixed`.

## Stage 3: data pipeline and features

`.\tasks.ps1 data` (= `python -m ml.common.make_data`) turns everything into model-ready tables; `.\tasks.ps1 eda`
rebuilds `notebooks/eda.ipynb`.

| Module (`ml/common/`) | Content |
| --- | --- |
| `sync.py` | `SessionClock`: webcam frame times, audio slices and events on the shared session clock |
| `windowing.py` | 5 s / 2.5 s windows with labels, hesitation counts and person-wise split |
| `video_features.py` | ffmpeg/OpenCV decoding at 15 fps, letterbox, normalisation, augmentation (flip mirrors gaze yaw) |
| `audio_features.py` | 80-band log-mel (25/10 ms), MFCC-40, YIN pitch, RMS, SpecAugment, speed perturbation, noise |
| `text.py` | ASR text normalisation; SentencePiece unigram tokenizers (asr 1000, answer 8000) trained from scratch |
| `datasets.py` | Face, Gaze, EngagementClip, VAD, ASR, Disfluency, Answer, Fusion datasets |
| `config.py` | paths, seeds, YAML configs with `key=value` overrides |

## Stages 4-8: models

| Model | Code | Data now | Train / evaluate |
| --- | --- | --- | --- |
| Face + 5 landmarks (BlazeFace-style, 0.38 M) | `ml/vision/face_detector` | WIDER FACE + RetinaFace landmarks | `python -m ml.vision.face_detector.train` / `.eval` |
| Head pose CNN | `ml/vision/headpose` | WIDER faces, solvePnP labels | `python -m ml.vision.headpose.train` |
| Gaze + iris CNN | `ml/vision/gaze_iris` | MPIIGaze (person-wise) | `.train` / `.eval` |
| Engagement CNN-GRU | `ml/vision/engagement` | DAiSEE (request) / our windows | `.train --stage pretrain|finetune|scratch`, `.eval` |
| Window engagement MLP + attention score | `ml/vision/engagement/window_models.py` | synthetic windows | `python -m ml.vision.engagement.window_models` |
| VAD | `ml/speech/vad` | LibriSpeech + noise | `python -m ml.speech.vad.train` |
| Conformer-CTC ASR + streaming | `ml/speech/asr` | LibriSpeech (+ Common Voice, ours) | `.train [--resume]` / `.eval` |
| 4-gram Kneser-Ney LM (ours, ARPA) | `ml/speech/lm/ngram.py` | LibriSpeech text + interview text | `python -m ml.speech.lm.ngram` |
| Disfluency CRNN | `ml/behaviour/disfluency` | SEP-28k | `.train` / `.eval` |
| Speaking pattern + hesitation index | `ml/behaviour/answer_models.py` | synthetic answers | `python -m ml.behaviour.answer_models` |
| Answer-quality transformer (+ TF-IDF baseline) | `ml/response/answer_quality` | synthetic answers + off-topic pairs | `.train` / `.eval` |
| Fusion network + Shapley reasons | `ml/fusion` | synthetic answers (out-of-fold answer features) | `python -m ml.fusion.answer_features`, `python -m ml.fusion.train` |
| Meeting head | `ml/fusion/meeting.py` | synthetic meetings, AMI test | `python -m ml.fusion.meeting` |

`.\tasks.ps1 train-all` runs everything one model at a time (low RAM). `.\tasks.ps1 export` exports to ONNX and
benchmarks latency (`docs/metrics/onnx_latency.md`). Demos: `.\tasks.ps1 demo-webcam`, `.\tasks.ps1 demo-mic`
(needs `pip install sounddevice`). Viva: `.\tasks.ps1 retrain-live` retrains the VAD and gaze CNN from random init in minutes.

## Stages 9-10: the product

```
candidate browser --WebRTC--> LiveKit SFU <--WebRTC-- interviewer browser
                                  |                          ^
                    hidden analysis bot (backend/workers/bot.py)   | WebSocket /ws/analytics (interviewer tokens only)
                     frames 10 fps | PCM 640 ms                    |
                         v         v                               |
                     Redis streams --> vision worker ----------> analytics:{id} --> FastAPI (backend/main.py)
                                   --> speech/answer worker ---^        PostgreSQL: users, interviews, participants,
                                                                        questions, transcript_segments, window_scores,
                                                                        answer_scores, follow_ups, reports
```

```powershell
.\tasks.ps1 web-install             # once
.\tasks.ps1 start-all               # postgres, redis, livekit, caddy + API (:8020) + workers + web (:3020)
# or fully in containers (CPU):  docker compose --profile app up -d --build
```

Open http://localhost:3020, create an interviewer account, create an interview and send the candidate link (for a
second laptop: `https://<LAN_IP>:8444/join/...` after setting `MBAS_LAN_IP` and opening TCP 8444 in the firewall).

| Page | Who | What |
| --- | --- | --- |
| `/dashboard` | interviewer / meeting host | create + schedule interviews or meetings, copy the candidate link |
| `/join/<invite>` -> `/room/<id>` | candidate | consent, camera/mic check, video call, screen share, timer; **no analytics** |
| `/cockpit/<id>` | interviewer | video + shared screen, live transcript (fillers highlighted), attention / engagement / hesitation / confidence gauges, iris-gaze plot, alerts, speaking chips, timeline with follow-up markers, question checklist, notes, per-answer scores + reasons |
| `/meeting/<id>` | meeting host | participant grid, talk-time bars, per-person engagement |
| `/report/<id>` | interviewer | radar, replay synced with the transcript, per-question cards, follow-ups, PDF export, candidate-feedback editor + release |
| `/feedback/<token>` | candidate | released strengths + 3 improvements with timestamps and tips |

Candidate tokens are rejected by the analytics WebSocket on the server (tested in `tests/test_backend.py`); `/metrics`
shows per-model and end-to-end latency. Stage 10 documents: `docs/model_report.md`, `docs/bias_check.md`,
`docs/user_study/`, `docs/project_report_outline.md`, `docs/viva_deck_outline.md`, `docs/demo_script.md`
(`.\tasks.ps1 reports` regenerates them).

## Tests

```powershell
.\tasks.ps1 test
```

## Ethics

Recording requires explicit consent from both people; data falls under India's DPDP Act 2023. Recorded sessions are
git-ignored and must stay on project machines. Scores produced later are decision support for the interviewer,
never an automatic reject.
