# Viva deck: 15 slides

| # | Slide | Content | Source |
| --- | --- | --- | --- |
| 1 | Title | MBAS: interview & meeting copilot, team, guide | - |
| 2 | Problem statement | Faculty description quoted word for word; the earlier rejection ("used pre-trained models") | build plan |
| 3 | Requirement traceability | Requirement → our model → stage table | build plan |
| 4 | What the product does | Screenshot: interviewer cockpit (video + screen, transcript, gauges, gaze plot, timeline) | live demo |
| 5 | Architecture | LiveKit → bot → Redis streams → workers → interviewer-only WebSocket; candidate sees nothing | README |
| 6 | Our dataset | Sessions, participants, hours, kappa, demographics, splits by person; public datasets for pre-training | dataset_card.md |
| 7 | Vision models | Face + landmarks (AP, FPS), head pose (MAE), gaze (angular error vs baseline), sample frames | model_report.md |
| 8 | Attention & engagement | Window models, attention score, confusion matrices, CNN-GRU status / DAiSEE comparison | model_report.md |
| 9 | Our speech recogniser | Conformer-CTC, own 4-gram LM, WER table greedy / beam / beam+LM, Whisper row (comparison only) | asr.json |
| 10 | Hesitation & speaking patterns | Disfluency CRNN per-class F1, hesitation index weights, speaking-pattern confusion matrix | behaviour.json |
| 11 | Responses & fusion | Transformer vs TF-IDF table, ablation table (video / audio / text / all) | fusion.json |
| 12 | Explainability | One answer: score + top-3 reasons + modality weights; follow-up flags | fusion_explanations.json |
| 13 | Meeting mode & generalisation | Talk-time bars; AMI test: absolute features failed, relative features fixed it | meeting_head.json |
| 14 | Fairness, ethics, user study | Bias check gaps, consent & DPDP, decision support only, SUS scores | bias_check.md, user_study |
| 15 | Live retrain + Q&A | `.\tasks.ps1 retrain-live` running on screen: VAD + gaze from random init in minutes | live_retrain.py |

Keep one number per claim on every slide, and cite its file in small print. Show the synthetic-data label wherever
synthetic results appear.
