# Merge report: older project (`minor_project`) vs this project

Read-only comparison made on 2026-11-19. Nothing in `minor_project` was changed. Nothing has been ported yet; the
list below is what is worth porting and what is not, for approval.

## The two projects in one paragraph

**minor_project** (Sep-Oct 2026): a Streamlit app with an `analyzer/` package. Pre-trained models: faster-whisper
(speech), Silero VAD, MediaPipe face landmarks. Rule-based scoring (fusion, flags, feedback), an OBS-free Windows
window capture, automatic question detection, a post-session HTML report, an `evaluation/` package, and 20 test files.
**This project**: FastAPI + Redis + LiveKit + Next.js, with our own trained models (vision, VAD, ASR with an n-gram
LM, disfluency, answer quality, fusion, meeting head), a real video call, cockpit, meeting mode, report with a rated
scorecard, candidate feedback release, and 87 tests. This project is the base; the older one is only a source of ideas.

## What the older project has that this one lacks

| # | Feature (older project file) | In this project? | Value | Effort / risk | Recommendation |
|---|---|---|---|---|---|
| 1 | **Combined flags**: a moment is flagged only when several signals get worse together (pause + hedging + looking away), judged against the speaker's own median baseline (`analyzer/flags.py`) | No. We flag single events (look away, long pause), which is noisy | High for the report: fewer, better "moments to review" | Low-medium: pure functions, no new dependencies, testable | **Port** into the report |
| 2 | **Automatic question detection** from the interviewer's speech (`question_detector.py`, `questions.extract_question_text`) | No. The interviewer clicks Ask and Answer ended every time | High for Meet / Zoom / Teams calls, where questions are free-form | Medium: needs the interviewer track in the speech worker and a way to create the question; risk of false questions | **Port behind a switch, off by default**; do after the demo |
| 3 | **Validation against human ratings**: rater agreement and system-vs-human correlation (`evaluation/statistics.py`, `validation.py`, `ratings_template.csv`) | Partly: `tools/agreement.py` covers label agreement; there is no system-vs-human correlation | High for the viva ("does the score match people?") | Low: standalone script | **Port** as a tool |
| 4 | **Self-contained HTML report** to download (`analyzer/report.py`) | No. We have a PDF print only | Medium: easy to send or archive | Low-medium | Port if time allows |
| 5 | **Whisper speech recognition** as an alternative to our own ASR | No (own ASR, WER 0.137 on LibriSpeech, visible spelling errors on unusual voices) | High for transcript quality | High: new dependency, CUDA or CPU speed, different streaming design, and it is a pre-trained model | **Not now.** If done, only as an optional, clearly labelled backend |
| 6 | **MediaPipe face landmarks and blendshapes** (blink, gaze) | No. Our blink detector loses blinks for people wearing glasses | Medium: would fix blink readings | High: new dependency and a second vision path | **Not now**; same rule as 5 |
| 7 | OBS-free **window capture** (`windows-capture`) | Replaced by browser screen capture ("External call") | Low | n/a | **Skip** |
| 8 | Streamlit UI, SQLite event log, rule-only fusion, coaching tips | Replaced by the Next.js app, Postgres and our trained fusion / tips | None | n/a | **Skip** |
| 9 | Documents: `PROJECT_REPORT.md`, `TESTING_GUIDE.md`, synopsis, slides | We have `docs/project_report_outline.md`, `docs/viva_deck_outline.md`, `docs/demo_script.md` | Medium: text and testing steps can be reused | Low, but they describe the older architecture | Reuse selectively, rewrite for this architecture |

## Things to keep honest if pre-trained models are ever brought in

Our documents and the demo script say every model was trained by us. Items 5 and 6 would add pre-trained models.
If they are added they must be optional, off by default, labelled in the UI and in `docs/`, with name, source and
licence recorded. Items 1 to 4 add no pre-trained models.

## Things that must not regress while merging

Empty or video-only reports are not graded; rows with no data show "-"; windows without a face are not averaged
in; the ASR clock is anchored to the first chunk; the speech worker survives an answer-model failure; everyone sees
their own video and has mute / camera buttons; `https://<LAN_IP>:8444` works from a phone.
