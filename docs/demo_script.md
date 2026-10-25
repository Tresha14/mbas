# 5-minute demo script

**Before (10 min earlier):** `.\tasks.ps1 start-all` on the host laptop. Log in as the interviewer at
http://localhost:3020, create the interview "Demo - SDE intern" and copy the candidate link. Open it on the second
laptop (`https://<LAN_IP>:8444/join/...`) and leave it on the consent page. Then open `/metrics` in a spare tab.

| Time | Who | Action | Say |
| --- | --- | --- | --- |
| 0:00 | Presenter | Dashboard on the projector | "MBAS analyses interviews from video, audio and conversation, and every model here was trained by us from scratch." |
| 0:20 | Candidate | Ticks consent, checks camera/mic, joins, shares screen | "The candidate consents first, and their page shows only a normal video call." |
| 0:45 | Interviewer | Opens the cockpit, clicks **Start analysis** | "Only the interviewer's token can open the analytics channel; candidate tokens are refused by the server." |
| 1:00 | Interviewer | Clicks **Ask** on Q01 "Tell me about yourself" | Point at the transcript appearing within about 2 s, with fillers highlighted. |
| 1:40 | Candidate | Answers; looks at a phone for 4 s mid-answer | Point at the gaze plot leaving the screen box, the off-screen alert and the attention gauge dropping. |
| 2:20 | Interviewer | Clicks **Answer ended** | The answer card: effectiveness score, speaking pattern, delay, fillers, **top-3 reasons**, and a follow-up flag (attention drop). |
| 3:00 | Interviewer | Asks Q11 (behavioural, STAR); candidate gives a vague answer | Show the "vague" follow-up and the STAR components. |
| 3:40 | Interviewer | Ends the interview, opens **Report** | Radar of sub-scores, timeline replay (click a marker to jump in the transcript), per-question cards. |
| 4:15 | Interviewer | Edits one candidate tip, clicks **Release to candidate** | "The candidate gets reviewed feedback with timestamps, not a vague rejection." |
| 4:35 | Candidate | Opens the feedback link | Strengths and three concrete improvements. |
| 4:50 | Presenter | `/metrics` tab | "Per-model latency; end-to-end delay stays under 2 seconds." |

**Backup:** if the network fails, play `docs/demo.mp4`. Record it the day before with OBS: both laptops side by side.
