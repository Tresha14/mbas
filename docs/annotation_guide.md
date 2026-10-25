# MBAS annotation guide

Version 1.0. Labels are made in Label Studio (`.\tasks.ps1 labelstudio`, then <http://localhost:8090>). Read the
whole guide once before your first session, and keep it open while labelling. When the guide and your instinct
disagree, follow the guide. When the guide does not cover a case, choose the closest rule, tick **unsure** (windows)
and write the case in the team chat so the guide can be updated.

## 1. Workflow

| Project | One task is | Who labels |
| --- | --- | --- |
| `MBAS windows` | one 5 s window of the candidate webcam (with candidate audio) | one annotator |
| `MBAS windows (double-labelled)` | same, for the 20% overlap sessions | **both** annotators, independently |
| `MBAS answers` | one question + answer (candidate mic audio + small video) | one annotator |
| `MBAS answers (double-labelled)` | same, overlap sessions | **both** annotators, independently |

Rules:

- Never label your own session (as candidate or interviewer).
- In the double-labelled projects, do not look at the other annotator's labels and do not discuss the session until
  both have finished. Agreement (Cohen's kappa) is computed from these projects.
- Windows overlap by 2.5 s, so neighbouring windows share half their video. Label each window on its own; it is normal
  that neighbours differ.
- Watch each window clip completely at normal speed at least once. Answer tasks: listen to the whole answer once
  before marking spans.
- Take a break every 45 minutes; label quality drops fast after that.

After labelling: `.\tasks.ps1 labels` (export → parquet → kappa → splits → dataset card).

## 2. Window labels (every 5 s window)

### 2.1 Attention: `on_screen` / `off_screen`

Where the candidate's **eyes** point for **most of the 5 s (more than 2.5 s)**.

| Label | Definition | Examples |
| --- | --- | --- |
| `on_screen` | Gaze is on the laptop screen or the webcam, including reading the question panel, looking at the interviewer's video, short blinks and glances under 1 s. | Looking at the interviewer while listening; reading the question on screen; looking at the camera; a 0.5 s glance at the keyboard. |
| `off_screen` | Gaze is away from the screen for more than half the window. | Looking at the phone (scripted segment), looking up at the ceiling to think for 3 s, turning to someone in the room, eyes closed for more than 2.5 s, face out of frame. |

Edge cases:

- **Thinking glance:** looking up or to the side while thinking counts as `off_screen` only if it lasts more than half
  the window.
- **Reading from another screen** (eyes sweep left-right on a second monitor or a paper below the camera): `off_screen`.
- **Face not visible** (out of frame, covered by a hand): `off_screen` and tick `face_not_visible`.
- **Video frozen or black:** pick the closest label, tick `bad_video`.

### 2.2 Engagement: `engaged` / `neutral` / `disengaged`

How involved the candidate looks **and sounds** in the conversation during these 5 s. Judge the behaviour, not how
good the answer is.

| Label | Definition | Typical cues |
| --- | --- | --- |
| `engaged` | Actively involved: speaking with energy, or listening attentively and responding to the interviewer. | Expressive face, nods, gestures, steady eye contact, lively voice, leans forward. |
| `neutral` | Present and following, but with low expressiveness. The default for calm, normal behaviour. | Still face, quiet listening, flat but fluent speech, normal posture. |
| `disengaged` | Attention or effort has clearly dropped. | Looking away repeatedly, slumping, yawning, fidgeting with the phone, very long silent pauses without thinking cues, bored or absent expression, one-word replies in a flat voice. |

Edge cases:

- Nervous but trying hard (fast speech, fidgeting hands, but eyes on the interviewer): `engaged`.
- Silent thinking with a focused face before answering: `neutral`. It becomes `disengaged` only if there are clear
  signs that they have given up (sighing, looking away, shrugging).
- Calm, introverted people are **not** disengaged. If you hesitate between `neutral` and `disengaged`, choose `neutral`.
- Scripted segments: "look at phone" is `disengaged`; calibration and reading aloud are `neutral` unless clearly
  otherwise.

### 2.3 Flags (optional)

`face_not_visible`, `bad_video`, `unsure`. Flags do not replace a label: always choose attention and engagement.

## 3. Hesitation spans (answer tasks, drawn on the waveform)

Select the label, then drag over the time span on the waveform. Spans cover **only the candidate's speech**; ignore
the interviewer's voice if it leaks into the candidate's microphone. Zoom in so span edges are within about 100 ms.

| Label | Definition | Mark from–to | Examples |
| --- | --- | --- | --- |
| `filler` | A filled pause: a sound or word that fills time and carries no meaning in that position. | start to end of the filler sound/word | "um", "uh", "hmm", "er", "matlab", "basically", "actually", "like", "you know", "so" (only when used as a filler, not as a real connector) |
| `long_pause` | Silence of **more than 1 s inside the answer** (after the candidate has started answering). | from the end of the last word to the start of the next | 1.4 s silence mid-sentence. Not the silence before the answer starts: that is response latency, measured automatically. |
| `repetition` | A word, part-word or phrase repeated without adding meaning. | first occurrence to the end of the last repetition | "I I I worked", "we used the the database", "it was, it was a project" |
| `restart` | The candidate abandons a sentence and starts again differently (false start, self-correction). | start of the abandoned part to the start of the new version | "We were going to — so what I did was…", "It has three — no, four layers." |

Edge cases:

- A filler inside a long pause ("……um……"): mark both `long_pause` (the silence parts) and `filler` (the "um").
- "Actually" or "basically" with real meaning ("Actually, the result was negative" = contrast) is not a filler.
- Hindi/regional fillers count the same way as English ones; write them as heard in the transcript.
- Stretched sounds ("sooooo") without a pause: mark as `filler` if it fills time, otherwise nothing.

## 4. Answer labels

### 4.1 Manual transcript

Type **only the candidate's words**, verbatim, lowercase is fine:

- Keep fillers, repetitions and restarts exactly as spoken: `um so i i worked on a — we built a portal`.
- Mark an abandoned word with a dash `—`. Unintelligible speech: `[inaudible]`. Laughter or cough: `[laugh]`, `[cough]`.
- Hindi or other-language words in Latin script as heard (`matlab`, `achha`).
- Numbers as words (`twenty twenty four`), abbreviations as spoken (`s q l` or `sequel`, as said).

### 4.2 Quality (1–5 stars)

How good the answer is **as an interview answer to this question**, ignoring accent and grammar mistakes.

| Score | Meaning |
| --- | --- |
| 5 | Excellent: directly answers, well structured (behavioural: clear Situation, Task, Action, Result), specific details, concise. |
| 4 | Good: answers the question with some structure and specifics; minor gaps. |
| 3 | Adequate: answers in general terms; missing structure or detail, or some rambling. |
| 2 | Weak: mostly vague or partly off-topic; key parts of the question not addressed. |
| 1 | Very poor: no real answer, wrong or off-topic, or "I don't know" without any attempt. |

Example (Q01 "Tell me about yourself"):
- 5: "I'm a final-year computer engineering student focusing on ML. I built a recommendation system for our college
  library that 300 students use, and I interned at a startup on FastAPI back-ends. I'm looking for a backend role."
- 3: "I'm a student, I like coding, I did some projects and an internship, I'm hard-working."
- 1: "Um… I don't know what to say… I'm from Pune."

### 4.3 Relevance (0–1, steps of 0.05)

The share of the answer that is about what was asked. `1.0` = everything relates to the question; `0.5` = about half
is generic or about something else; `0.0` = nothing relates. Quality and relevance differ: a short but fully on-topic
answer can have relevance 1.0 and quality 2.

### 4.4 Follow-up needed + reason

**Would a careful interviewer ask a follow-up question here?** Choose `yes` if any of the reasons below clearly
applies, otherwise `no`. With `yes`, pick the **main** reason:

| Reason | Use when | Example |
| --- | --- | --- |
| `vague` | The answer stays generic, no concrete example or detail. | "I am a team player and I always communicate well." |
| `low_confidence` | Content may be fine, but delivery shows strong uncertainty: many hedges, very long latency, much hesitation. | "I think… maybe it's… I'm not sure, perhaps a process is… like a program?" |
| `attention_drop` | The candidate seemed distracted, looked away or read from somewhere during the answer. | Eyes on the phone during most of the answer. |
| `topic_not_covered` | An important part of the question was not answered. | Asked "process vs thread, and when would you use each?", only defines a process. |
| `contradiction` | The answer contradicts itself or an earlier answer. | "I led the team" then "I wasn't really involved in decisions." |

## 5. Quality control

- Each annotator labels one practice session (not counted) and compares with the guide author before starting.
- After every 10 sessions, `.\tasks.ps1 labels` recomputes kappa. If any window or answer kappa drops below 0.6, the
  team meets, discusses the disagreements (Label Studio → filter by task) and updates this guide (bump the version).
- Target kappa: attention ≥ 0.8, engagement ≥ 0.6, quality (quadratic) ≥ 0.6, follow-up ≥ 0.6.
