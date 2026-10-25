# MBAS user study kit

Participants: **10 interviewers** (faculty, seniors, placement-cell members) and **10 candidates** (students). Each
interviewer runs one 10-minute mock interview with MBAS; each candidate takes part in one and reads their released
feedback. Everyone fills in the questionnaire below right after their last step. It takes about 5 minutes.

Collect answers in a copy of `responses_template.csv`, one row per person, then run
`python -m tools.user_study docs/user_study/responses.csv`. That writes `summary.json` and `results.md`.

## Consent (read aloud)

> This questionnaire is anonymous and voluntary. It asks how usable and fair MBAS felt to you. You can skip any
> question or stop at any time. Answers are used only in our project report.

## Part A: System Usability Scale (SUS)

1 = strongly disagree … 5 = strongly agree. (Standard SUS items; do not change the wording.)

| # | Item |
| --- | --- |
| sus1 | I think that I would like to use this system frequently. |
| sus2 | I found the system unnecessarily complex. |
| sus3 | I thought the system was easy to use. |
| sus4 | I think that I would need the support of a technical person to be able to use this system. |
| sus5 | I found the various functions in this system were well integrated. |
| sus6 | I thought there was too much inconsistency in this system. |
| sus7 | I would imagine that most people would learn to use this system very quickly. |
| sus8 | I found the system very cumbersome to use. |
| sus9 | I felt very confident using the system. |
| sus10 | I needed to learn a lot of things before I could get going with this system. |

## Part B: Usefulness and fairness

1 = strongly disagree … 5 = strongly agree.

| column | Interviewers | Candidates |
| --- | --- | --- |
| useful_overall | The live dashboard helped me run a better interview. | The feedback report helped me understand how the interview went. |
| useful_followups | The follow-up suggestions pointed at answers worth probing. | The three improvement areas are things I can act on. |
| trust_scores | The scores and their reasons matched my own impression. | The feedback matched how I felt the interview went. |
| fair_to_candidate | The scores looked fair to the candidate (not biased by accent, glasses, camera). | I felt the analysis was fair to me. |
| feedback_actionable | The candidate report is something I would send after editing. | The tips were concrete enough to practise. |
| privacy_ok | It is clear what is recorded and who sees it. | I was comfortable with what was recorded and who could see it. |

**comment:** What was most useful, and what would you change? (free text)

## Reporting

Report the SUS mean and SD per role, with the adjective grade (> 68 is above average, > 80.3 is excellent), the mean
of each Part B item, and three representative comments. Keep the raw CSV out of the repository if it contains names.
