# MBAS data collection protocol

Version 1.0 (Stage 2). Every recording session follows this document exactly, so sessions are comparable and the
labels in `docs/annotation_guide.md` mean the same thing across the dataset.

**Target:** 60–100 mock interviews from at least 30 different participants (each person records 1–3 sessions on
different days), 10–15 minutes each.

---

## 1. Roles

| Role | Who | Does |
| --- | --- | --- |
| Interviewer | team member | runs the recorder, reads the questions, presses **Ask** / **Answer ended** |
| Candidate | volunteer participant | answers as in a real interview |
| Session lead | interviewer (or a third team member) | consent, setup checklist, session log |

Never interview someone you are close friends with in their first session: familiarity changes eye contact and
hesitation.

## 2. Before the session

### Equipment and software

- Host laptop runs `.\tasks.ps1 up`, `.\tasks.ps1 recorder-api`, `.\tasks.ps1 recorder-web` (README, "Run the recorder").
- Candidate uses their own laptop with its built-in webcam where possible (that is what the product will see).
  Record the device type in the session log.
- Both laptops on the same Wi-Fi; candidate opens `https://<LAN_IP>:8443/candidate?room=<CODE>`.
- Wired or headset microphone for the interviewer is fine; the **candidate uses the laptop microphone** unless the
  session log says otherwise.

### Lighting and environment variety

The models must work in normal student rooms, not a studio. Rotate the conditions so that across the whole dataset
each one is at least 20% of sessions:

| Condition | How to set it up |
| --- | --- |
| `good` | room lights on, face lit from the front (window or lamp behind the laptop) |
| `dim` | only one weak light, no direct light on the face |
| `backlit` | window or lamp **behind** the candidate |
| `noisy` | fan on, door open to a corridor, or quiet background music (not speech) |
| `glasses` | candidates who wear glasses keep them on (they are recorded in `meta.json`) |

Write the condition(s) in the session log. Do not ask anyone to remove glasses, change clothes or change their
normal sitting position.

### Setup checklist (session lead)

1. Candidate seated about 50–70 cm from the screen, face fully in frame, top of the head visible.
2. Close notification pop-ups, set the candidate's phone on silent but **keep it on the desk** (needed for 4.2).
3. Candidate's screen: the browser tab with the recorder, full screen. They will share **the entire screen**.
4. Quick audio check: both people say "testing" and the levels move in the recorder.

## 3. Consent (before anything is recorded)

The recorder will not start until both people accept the consent screen. Read the text below aloud to the
candidate, let them read it on screen and answer any questions. If they decline, stop; nothing is stored.

> **Consent to record (MBAS project, version v1)**
>
> This is a final-year student project that builds software to analyse mock interviews and meetings. We would like to
> record this mock interview: your webcam video, your microphone, your shared screen and the interviewer's microphone.
>
> - The recording is used only to build and test our own models and to write our project report. It is not a real job
>   interview and does not affect any grade, job or evaluation of you.
> - We also store your participant ID (not your name), age band, gender, whether you wear glasses and your accent.
> - Recordings are kept on project computers only, are never uploaded to public websites and are deleted within
>   12 months of the end of the project.
> - Team members may watch the recording to label it (for example "looking at the screen" or "hesitation"). These
>   labels describe this mock interview only and are not a judgement of you as a person.
> - Short still images or anonymised numbers may appear in our report or viva. Your face will not be shown without
>   asking you again in writing.
> - You can stop at any time, and you can ask us to delete your recording later by giving your participant ID to
>   any team member. We will delete it within 7 days.
> - This data is processed under India's Digital Personal Data Protection Act 2023. Contact: the project team.
>
> I have read this and agree to be recorded.

Participant IDs are `P001`, `P002`, ... in the order people first take part. Keep the ID-to-name list **on paper or in
a separate encrypted file**, never in the repo or in `data/`. Interviewers use IDs from `P900` upwards.

## 4. Session script (10–15 minutes)

Interviewer clicks **Start recording** once both people have joined and the candidate is sharing the screen. Then:

| # | Segment | Length | What happens | Why |
| --- | --- | --- | --- | --- |
| 1 | Gaze calibration | 60 s | Interviewer clicks **Run calibration**. Candidate follows the dot with the eyes, head still. | Gaze/iris ground truth (`calibration.json`) |
| 2 | Look at camera | 30 s | "Please look straight into your webcam lens for 30 seconds." | on-screen / eye-contact reference |
| 3 | Look at phone | 20 s | "Now pick up your phone and look at it as if reading a message." | off-screen reference |
| 4 | Read aloud | ~60 s | Candidate reads the paragraph in 4.1 from the screen. | reading-pattern flag, fluent speech for ASR |
| 5 | Interview questions | 8–12 min | 8–12 questions, see 4.2. | main data |
| 6 | Unprepared answer | 1–2 min | One question they could not prepare for (see 4.2). | hesitation, low-confidence answers |

For segments 2–4 the interviewer says the instruction and **writes the start and end time** (recorder clock shown on
screen) in the session log, so the scripted labels can be attached later.

### 4.1 Read-aloud paragraph

> Good morning. My name is not important for this recording, but the way I speak is. I am reading this paragraph so
> that the system can hear my normal voice at a steady pace. Interviews often begin with simple questions about
> education and projects, and then move to problem solving, teamwork and technical topics such as databases,
> operating systems, networks and machine learning. A clear answer usually explains the situation, the task, the
> action that was taken and the result. Thank you for helping us build and test this project.

### 4.2 Question order

Use the recorder's question panel (`recorder/questions.json`, 40 questions):

1. Always start with **Q01** ("Tell me about yourself").
2. Then 2 HR, 3 behavioural, 3 technical questions, chosen to match the candidate's degree; vary the choice between
   sessions so every question is asked in at least 5 sessions overall.
3. End segment 5 with one more behavioural question.
4. Segment 6 (unprepared): one technical question outside the candidate's field (for example Q36 for a non-CS
   student), or Q37–Q40.
5. A participant's second and third sessions use questions they have not answered before.

Press **Ask** when you *start* reading a question and **Answer ended** when the candidate has finished (wait about one
second of silence). Do not cut the candidate off; if you must interrupt, still press **Answer ended** afterwards.
Interviewer behaviour: neutral, friendly, the same short acknowledgement after every answer ("Okay, thank you."). No
follow-up questions, hints or feedback during the session.

### 4.3 Ending

1. **Stop recording**, wait until the upload counters stop, click **Finalize & check**.
2. If the check fails, note the error in the session log; do not delete the session.
3. Thank the candidate. Answer questions about the project, but do not tell them how they "did".

## 5. Session log

One row per session in `data/raw/session_log.csv` (git-ignored, contains no names):

```
session_id,participant_id,interviewer_id,date,device,lighting,noise,glasses,calib_ok,look_camera_start_s,look_camera_end_s,look_phone_start_s,look_phone_end_s,read_start_s,read_end_s,unprepared_question_id,notes
```

Times are seconds from the recording start (shown on the interviewer page).

## 6. Storage and data protection rules

- Sessions are written to `data/raw/<session_id>/` on the host laptop. `data/raw/` and `data/processed/` are
  git-ignored: **never commit, upload to public cloud drives, or send recordings over chat apps**.
- Back up to one encrypted external drive or the college's private storage only. Copy with the whole folder so the
  shared clock and `meta.json` stay together.
- Labelling clips (`data/processed/labelstudio/`) are derived personal data and follow the same rules.
- Deletion request: delete `data/raw/<session_id>/`, the session's clips under `data/processed/labelstudio/media/`,
  its tasks in Label Studio and its rows in `data/processed/labels/`, then re-run `tools/make_splits.py` and
  `tools/dataset_stats.py`. Note the deletion date in the session log.
- Only aggregated numbers and anonymised plots go into reports. Any face image in a report needs separate written
  permission.
- Delete all raw recordings within 12 months of the project's end.

## 7. After the session (same day)

```powershell
.\tasks.ps1 check data\raw\<session_id>          # must pass (or a note in the log)
.\tasks.ps1 labelstudio-import                    # clips + tasks for annotation
```

Then follow `docs/annotation_guide.md`.
