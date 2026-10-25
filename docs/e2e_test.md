# End-to-end system test (Stage 9 "done when")

Date: 2026-10-25. Setup: one laptop (RTX 4060, 32 GB RAM), `docker compose up postgres redis livekit caddy`, API on
:8020, `backend.workers.run_all`, Next.js production build on :3020, two Chrome tabs (interviewer + candidate).
The candidate camera was a looping video of a frontal WIDER FACE photo and the microphone a LibriSpeech recording,
injected as fake devices (no physical camera available to the test harness). The ASR was not trained yet.

| Check | Result |
| --- | --- |
| Candidate joins via invite link with consent, camera/mic check, call + screen share | pass |
| Interviewer sees candidate camera + shared screen side by side | pass |
| LiveKit bot joins hidden, forwards 10 fps frames + 640 ms audio to Redis | pass |
| Live gauges (attention, hesitation, confidence), gaze plot inside the screen box, speaking chips, timeline | pass |
| Ask / Answer ended -> answer card with effectiveness score + top-3 reasons in ~3.4 s | pass |
| Candidate token on `/ws/analytics/{id}` | rejected by the server (HTTP 403) |
| Report: radar, replay timeline, per-question card, candidate-feedback editor, release | pass |
| Candidate opens released feedback link | pass (strengths + improvement with tip) |
| `/metrics` latency | vision 21 ms/frame, camera-to-cockpit 71 ms, answer analysis 3.4 s |
| Worker errors during the run | 0 |

Bugs found and fixed during the test:
- A candidate and an interviewer signed in on the same browser: the stored interviewer login took priority over the
  candidate's per-tab token (`frontend/lib/api.ts`), so the candidate page joined with the interviewer identity.
- PostgreSQL rejects `NaN` in JSON columns (SQLite in the unit tests accepted it): values are cleaned before storing,
  and a failing message no longer stops a worker loop.

Known limits seen in the run: engagement read 0% because the window engagement MLP is trained on synthetic features
(domain gap until our sessions are labelled); WPM / transcript need the ASR (training queued).
Still to do with people: a two-laptop run over the LAN with real cameras and a 3+ person meeting.
