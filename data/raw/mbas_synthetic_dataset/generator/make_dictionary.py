"""Builds DATA_DICTIONARY.md for every CSV in the dataset. Fails loudly if a column has no description."""
import glob
import os
import sys

import pandas as pd

D = {
    # ids / time
    "participant_id": "Participant ID (P001...). Splits are by this ID.",
    "session_id": "Interview session ID (S0001...).",
    "answer_id": "Answer ID = <session>_A<question order>.",
    "question_id": "Question ID from reference/question_bank.csv.",
    "window_id": "5-second window ID = <session>_W<index>.",
    "segment_id": "Transcript segment ID; suffix _Q question, _R candidate response, _K interviewer acknowledgement.",
    "meeting_id": "Meeting ID (M001...).",
    "start_s": "Start time in seconds from session/meeting start.",
    "end_s": "End time in seconds from session/meeting start.",
    "duration_s": "Duration in seconds.",
    "split": "train / val / test (person-wise).",
    # participants
    "age_band": "Age group.", "gender": "male / female / non_binary.", "glasses": "Wears glasses (affects eye/iris models).",
    "accent": "First-language influenced English accent.", "english_proficiency": "0-1 self/assessor rated English proficiency.",
    "education": "Degree programme.", "work_experience_years": "Years of work experience.",
    "skin_tone_fitzpatrick": "Fitzpatrick skin type 1-6 (for vision bias checks).", "consent_given": "Recording consent recorded.",
    # sessions
    "mode": "interview or meeting.", "session_index": "1st/2nd/3rd session of this participant (practice effect).",
    "device": "Capture device.", "camera_quality": "0-1 camera quality score.", "lighting": "good / dim / backlit.",
    "lighting_lux": "Estimated illuminance (lux).", "background_noise_db": "Background noise level (dB SPL).",
    "network_jitter_ms": "WebRTC network jitter (ms).", "video_fps": "Video frame rate analysed.",
    "audio_sample_rate_hz": "Audio sample rate.", "n_questions": "Questions asked in the session.",
    "n_windows": "Number of 5 s windows in the session.", "talk_time_ratio_candidate": "Fraction of session time candidate spoke.",
    "mean_response_latency_s": "Mean response latency across answers (s).",
    "overall_effectiveness": "Session-level communication effectiveness 0-100 (mean over answers).",
    "mean_clarity": "Mean clarity sub-score.", "mean_confidence": "Mean confidence sub-score.",
    "mean_engagement": "Mean engagement sub-score.", "mean_relevance": "Mean relevance sub-score.", "mean_fluency": "Mean fluency sub-score.",
    "follow_up_count": "Answers flagged for follow-up.", "interviewer_rating_1to10": "Interviewer overall rating 1-10.",
    "recommendation": "strong_hire / hire / borderline / no_hire.",
    # windows
    "candidate_state": "listening (interviewer asking), thinking (before answering), answering, idle.",
    "face_present_ratio": "Fraction of frames with a face detected.", "face_detection_conf": "Mean face-detector confidence 0-1.",
    "head_yaw_mean_deg": "Mean head yaw (deg, + = right). Empty if face absent.", "head_pitch_mean_deg": "Mean head pitch (deg, + = up).",
    "head_roll_mean_deg": "Mean head roll (deg).", "head_yaw_std_deg": "Head yaw std (movement/expressiveness).",
    "head_pitch_std_deg": "Head pitch std (nodding).", "gaze_yaw_mean_deg": "Mean horizontal gaze angle (deg; 0 = camera/screen centre).",
    "gaze_pitch_mean_deg": "Mean vertical gaze angle (deg; negative = looking down).", "gaze_yaw_std_deg": "Horizontal gaze std (deg).",
    "gaze_pitch_std_deg": "Vertical gaze std (deg).", "iris_left_x": "Left iris centre x in eye crop (0-1, 0.5 = centre).",
    "iris_left_y": "Left iris centre y in eye crop (0-1).", "iris_right_x": "Right iris centre x in eye crop (0-1).",
    "iris_right_y": "Right iris centre y in eye crop (0-1).", "iris_movement_std": "Std of iris position (iris movement amount).",
    "on_screen_ratio": "Fraction of frames gaze is on screen.", "glance_away_count": "Number of glances away from screen.",
    "saccade_rate_hz": "Rapid eye movements per second.", "fixation_duration_mean_s": "Mean fixation duration (s).",
    "blink_rate_per_min": "Blinks per minute (stress/fatigue indicator).", "eye_aspect_ratio_mean": "Mean eye aspect ratio (eye openness).",
    "reading_pattern_flag": "1 = repeated horizontal sweeps (likely reading from another screen).",
    "face_motion_energy": "0-1 facial motion energy.", "mouth_open_ratio": "0-1 mouth opening (speaking proxy).",
    "smile_intensity": "0-1 smile intensity.", "brow_raise_intensity": "0-1 brow raise intensity.",
    "vad_speech_ratio": "Fraction of candidate audio frames containing speech.",
    "interviewer_speech_ratio": "Fraction of interviewer audio frames containing speech.",
    "rms_energy_db": "Candidate audio RMS energy (dBFS).", "pitch_mean_hz": "Mean F0 pitch (Hz); empty when not speaking.",
    "pitch_std_hz": "F0 pitch std (Hz).", "snr_db": "Signal-to-noise ratio (dB).", "words_in_window": "Candidate words spoken in window.",
    "fillers_in_window": "Filler words in window.", "attention_label": "LABEL on_screen / off_screen (annotator 1).",
    "attention_score": "LABEL attention 0-100.", "engagement_label": "LABEL engaged / neutral / disengaged.",
    "boredom_level": "LABEL 0-3 (DAiSEE style).", "confusion_level": "LABEL 0-3.", "frustration_level": "LABEL 0-3.",
    "distraction_event": "1 = distraction (phone, looking away, leaving frame).",
    # answers
    "question_category": "hr / behavioural / technical.", "question_difficulty": "0-1 question difficulty.",
    "question_order": "Position of question in session.", "question_start_s": "Interviewer starts asking (s).",
    "question_end_s": "Interviewer finishes asking (s).", "answer_start_s": "Candidate starts answering (s).",
    "answer_end_s": "Candidate stops answering (s).", "answer_text": "Verbatim transcript incl. fillers, repetitions, restarts, hedges.",
    "response_latency_s": "Delay between question end and answer start (s).", "answer_duration_s": "Answer duration (s).",
    "word_count": "Words in transcript.", "speaking_rate_wpm": "Words per minute incl. pauses.",
    "articulation_rate_wpm": "Words per minute excluding pauses.", "long_pause_count": "Pauses > 1 s.",
    "long_pause_total_s": "Total long-pause time (s).", "pause_ratio": "Fraction of answer time silent.",
    "mean_pause_duration_s": "Mean long-pause duration (s).", "filler_count": "Filled pauses (um, uh, like, basically, matlab...).",
    "filler_rate_per_100w": "Fillers per 100 words.", "repetition_count": "Word repetitions.", "restart_count": "Sentence restarts.",
    "hedge_count": "Hedges (I think, maybe, not sure...).", "prolongation_count": "Sound prolongations (audio).",
    "block_count": "Speech blocks (audio).", "disfluency_rate_per_100w": "All disfluencies per 100 words.",
    "pitch_range_hz": "F0 range (Hz).", "monotony_index": "0-1, higher = more monotone voice.",
    "energy_mean_db": "Mean energy (dBFS).", "energy_std_db": "Energy std (dB).", "volume_consistency": "0-1 volume steadiness.",
    "speech_rate_variability": "0-1 variation of speech rate.", "jitter_pct": "Pitch jitter % (vocal stress).",
    "shimmer_pct": "Amplitude shimmer % (vocal stress).", "asr_confidence_mean": "Mean ASR word confidence 0-1.",
    "interrupted_by_interviewer": "1 = interviewer cut in.", "overlap_s": "Overlapping speech (s).",
    "mean_on_screen_ratio": "Mean on-screen gaze ratio during thinking+answering.",
    "mean_attention_score": "Mean attention score during thinking+answering.", "engaged_window_fraction": "Fraction of windows engaged.",
    "disengaged_window_fraction": "Fraction of windows disengaged.", "mean_blink_rate_per_min": "Mean blink rate.",
    "total_glance_away": "Total glances away.", "mean_gaze_yaw_std_deg": "Mean gaze yaw std.", "mean_head_yaw_std_deg": "Mean head yaw std.",
    "reading_pattern_detected": "1 = reading pattern in any window of the answer.", "mean_confusion_level": "Mean confusion level.",
    "key_points_covered": "Expected key points covered (0-4).", "vague_sentence_count": "Generic/vague sentences.",
    "off_topic_sentence_count": "Off-topic sentences.", "contains_contradiction": "1 = self-contradicting statement.",
    "keyword_overlap": "Fraction of expected keywords present.", "relevance": "LABEL relevance 0-1.", "completeness": "LABEL completeness 0-1.",
    "quality_1to5": "LABEL answer quality 1-5.", "star_situation": "LABEL STAR Situation present (behavioural only).",
    "star_task": "LABEL STAR Task present.", "star_action": "LABEL STAR Action present.", "star_result": "LABEL STAR Result present.",
    "hesitation_index": "LABEL hesitation 0-100.", "speaking_pattern": "LABEL confident / balanced / hesitant / rushed / monotone.",
    "clarity_score": "LABEL clarity 0-100.", "confidence_score": "LABEL confidence 0-100.", "engagement_score": "LABEL engagement 0-100.",
    "relevance_score": "LABEL relevance 0-100.", "fluency_score": "LABEL fluency 0-100.",
    "communication_effectiveness": "LABEL communication effectiveness 0-100 (main target).",
    "follow_up_needed": "LABEL 1 = answer needs follow-up.",
    "follow_up_reason": "LABEL none / vague / low_confidence / attention_drop / topic_not_covered / contradiction (meetings: participation reasons).",
    # events / segments / turns
    "event_type": "filler, repetition, restart, hedge, long_pause, prolongation, block, delayed_response.",
    "token": "Word(s) involved, if any.", "word_index": "Token position in answer_text.", "source": "Signal the event is detectable from.",
    "speaker": "candidate / interviewer.", "text": "Spoken text.", "asr_confidence": "ASR confidence for the segment.",
    "turn_index": "Order of turn.", "gap_from_previous_s": "Silence before this turn (s).",
    "overlap_with_previous_s": "Overlap with previous turn (s).", "is_interruption": "1 = overlap > 0.5 s.",
    # meetings
    "meeting_type": "standup / design_review / brainstorm / client_call / project_sync.", "duration_min": "Meeting length (min).",
    "n_participants": "Participants.", "host_id": "Host participant ID.", "n_turns": "Speaking turns.",
    "talk_balance_gini": "Gini of talk time (0 = perfectly balanced).", "meeting_effectiveness": "LABEL meeting effectiveness 0-100.",
    "minute_index": "Minute of meeting.", "camera_on": "1 = camera on (vision columns empty when off).",
    "speaking_time_s": "Seconds spoken in this minute.", "turns_taken": "Turns in this minute.",
    "multitasking_flag": "1 = mostly looking away (likely multitasking).", "speaker_id": "Participant speaking.",
    "role": "host / participant.", "talk_time_s": "Total talk time (s).", "talk_share": "Share of meeting talk time.",
    "mean_engagement": "LABEL mean engagement 0-1.", "contribution_balance": "0-1, 1 = fair share of talk.",
    "owner_id": "Action item owner.", "action_item": "Action item text.", "raised_at_s": "When raised (s).", "status": "Action item status.",
    # annotations / reference / hidden
    "annotator_id": "Second annotator ID (compare with annotator 1 labels for Cohen's kappa).",
    "category": "Question category.", "difficulty": "Question difficulty 0-1.",
    "key_point_1": "Expected key point 1 (STAR Situation for behavioural).", "key_point_2": "Expected key point 2 (Task).",
    "key_point_3": "Expected key point 3 (Action).", "key_point_4": "Expected key point 4 (Result).",
    "expected_keywords": "Pipe-separated expected keywords.",
    "confidence": "HIDDEN latent trait 0-1 - never use as a feature.", "nervousness": "HIDDEN latent trait 0-1.",
    "preparedness": "HIDDEN latent trait 0-1.", "attentiveness": "HIDDEN latent trait 0-1.", "fluency": "HIDDEN latent trait 0-1.",
    "expressiveness": "HIDDEN latent trait 0-1.", "speech_rate": "HIDDEN latent trait 0-1.", "accent_strength": "HIDDEN latent trait 0-1.",
    "reads_from_screen_tendency": "HIDDEN: participant tends to read answers from another screen.",
}
for i in range(1, 14):
    D[f"mfcc_{i}_mean"] = f"Mean MFCC coefficient {i} over the answer (speaker/voice timbre)."

root = sys.argv[1]
lines, missing = ["# Data dictionary\n", "Columns marked LABEL are training targets. HIDDEN columns are simulation ground truth for analysis only.\n"], []
for f in sorted(glob.glob(f"{root}/**/*.csv", recursive=True)):
    df = pd.read_csv(f, nrows=200)
    rel = os.path.relpath(f, root)
    lines.append(f"\n## {rel}\n\n| Column | Type | Description |\n| --- | --- | --- |")
    for c in df.columns:
        if c not in D:
            missing.append(f"{rel}:{c}")
        lines.append(f"| `{c}` | {df[c].dtype} | {D.get(c, '')} |")
open(f"{root}/DATA_DICTIONARY.md", "w").write("\n".join(lines) + "\n")
print("missing:", missing)
