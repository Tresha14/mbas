# Final project report: outline

**Title:** Multimodal Behavioural Analysis System: an AI-assisted interview and meeting copilot trained from scratch

Aim for about 60-80 pages. Every number below comes from a generated file in `docs/` (cite the file in a footnote).

1. **Abstract** (1 page): problem, approach (own models, own data), headline results, ethics.
2. **Introduction**
   - The faculty problem statement, quoted word for word, and the requirement traceability table (from the build plan).
   - Why "no pre-trained models": the earlier rejection, and what training from scratch proves.
   - Contributions: dataset + annotation protocol, 12 models, real-time system, feedback reports.
3. **Related work**: face/gaze estimation (BlazeFace, MPIIGaze), engagement (DAiSEE), ASR (Conformer, CTC, n-gram
   LMs), disfluency (SEP-28k), interview analysis (MIT Interview), meeting analysis (AMI), fairness in affective computing.
4. **Dataset** (`docs/dataset_card.md`, `docs/data_protocol.md`, `docs/annotation_guide.md`)
   - Recording protocol, consent, DPDP Act 2023 storage rules.
   - Labelling: timeline workflow, weak labels from scripted segments, double labelling, kappa table.
   - Public datasets and their licences; synthetic dataset and why it is labelled as such.
   - Person-wise splits and the leakage check (including the synthetic meeting-split finding).
5. **System architecture**: LiveKit SFU, analysis bot, Redis streams, workers, interviewer-only WebSocket, database.
   One diagram, plus a sequence diagram of one answer.
6. **Models** (one section each, from `docs/model_report.md` and `docs/model_cards/`): task, architecture diagram,
   parameters, data, training (curves), test results, failure cases.
   6.1 Face + landmark detector · 6.2 Head pose · 6.3 Gaze + iris · 6.4 Engagement (CNN-GRU + window MLP) and attention
   score · 6.5 VAD · 6.6 Conformer-CTC + our 4-gram LM + beam search · 6.7 Disfluency CRNN, pauses, prosody,
   speaking pattern, hesitation index · 6.8 Answer-quality transformer vs TF-IDF · 6.9 Fusion network, Shapley
   explanations · 6.10 Meeting head and the AMI generalisation test.
7. **Real-time system evaluation**: latency table (`docs/metrics/onnx_latency.md`, `/metrics`), end-to-end delay,
   and a test showing the candidate browser receives no analytics traffic.
8. **Feedback reports**: interviewer report, candidate feedback templates, release workflow.
9. **Evaluation**
   - Ablation (video / audio / text / all), baselines, generalisation (unseen participants, AMI).
   - Bias check (`docs/bias_check.md`).
   - User study (SUS + usefulness/fairness, `docs/user_study/results.md`).
   - Whisper benchmark row (comparison only) and why the gap exists.
10. **Ethics and limitations**: consent, decision support only, webcam bias, synthetic-data caveats, data deletion.
11. **Conclusion and future work**: more recorded sessions, fine-tuning on our frames, larger LM, INT8 on CPU.
12. **Appendices**: questionnaires, annotation guide, commands to reproduce every number (`.\tasks.ps1 ...`).
