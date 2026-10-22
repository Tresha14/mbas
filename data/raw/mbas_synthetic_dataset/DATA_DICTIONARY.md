# Data dictionary

Columns marked LABEL are training targets. HIDDEN columns are simulation ground truth for analysis only.


## annotations/answers_annotator2.csv

| Column | Type | Description |
| --- | --- | --- |
| `answer_id` | str | Answer ID = <session>_A<question order>. |
| `session_id` | str | Interview session ID (S0001...). |
| `quality_1to5` | int64 | LABEL answer quality 1-5. |
| `relevance` | float64 | LABEL relevance 0-1. |
| `follow_up_needed` | int64 | LABEL 1 = answer needs follow-up. |
| `follow_up_reason` | str | LABEL none / vague / low_confidence / attention_drop / topic_not_covered / contradiction (meetings: participation reasons). |
| `speaking_pattern` | str | LABEL confident / balanced / hesitant / rushed / monotone. |
| `annotator_id` | str | Second annotator ID (compare with annotator 1 labels for Cohen's kappa). |

## annotations/windows_annotator2.csv

| Column | Type | Description |
| --- | --- | --- |
| `window_id` | str | 5-second window ID = <session>_W<index>. |
| `session_id` | str | Interview session ID (S0001...). |
| `attention_label` | str | LABEL on_screen / off_screen (annotator 1). |
| `engagement_label` | str | LABEL engaged / neutral / disengaged. |
| `annotator_id` | str | Second annotator ID (compare with annotator 1 labels for Cohen's kappa). |

## hidden_ground_truth/participant_latent_traits.csv

| Column | Type | Description |
| --- | --- | --- |
| `participant_id` | str | Participant ID (P001...). Splits are by this ID. |
| `confidence` | float64 | HIDDEN latent trait 0-1 - never use as a feature. |
| `nervousness` | float64 | HIDDEN latent trait 0-1. |
| `preparedness` | float64 | HIDDEN latent trait 0-1. |
| `attentiveness` | float64 | HIDDEN latent trait 0-1. |
| `fluency` | float64 | HIDDEN latent trait 0-1. |
| `expressiveness` | float64 | HIDDEN latent trait 0-1. |
| `speech_rate` | float64 | HIDDEN latent trait 0-1. |
| `accent_strength` | float64 | HIDDEN latent trait 0-1. |
| `reads_from_screen_tendency` | bool | HIDDEN: participant tends to read answers from another screen. |

## interview/answers.csv

| Column | Type | Description |
| --- | --- | --- |
| `answer_id` | str | Answer ID = <session>_A<question order>. |
| `session_id` | str | Interview session ID (S0001...). |
| `participant_id` | str | Participant ID (P001...). Splits are by this ID. |
| `question_id` | str | Question ID from reference/question_bank.csv. |
| `question_category` | str | hr / behavioural / technical. |
| `question_difficulty` | float64 | 0-1 question difficulty. |
| `question_order` | int64 | Position of question in session. |
| `question_start_s` | float64 | Interviewer starts asking (s). |
| `question_end_s` | float64 | Interviewer finishes asking (s). |
| `answer_start_s` | float64 | Candidate starts answering (s). |
| `answer_end_s` | float64 | Candidate stops answering (s). |
| `answer_text` | str | Verbatim transcript incl. fillers, repetitions, restarts, hedges. |
| `response_latency_s` | float64 | Delay between question end and answer start (s). |
| `answer_duration_s` | float64 | Answer duration (s). |
| `word_count` | int64 | Words in transcript. |
| `speaking_rate_wpm` | float64 | Words per minute incl. pauses. |
| `articulation_rate_wpm` | float64 | Words per minute excluding pauses. |
| `long_pause_count` | int64 | Pauses > 1 s. |
| `long_pause_total_s` | float64 | Total long-pause time (s). |
| `pause_ratio` | float64 | Fraction of answer time silent. |
| `mean_pause_duration_s` | float64 | Mean long-pause duration (s). |
| `filler_count` | int64 | Filled pauses (um, uh, like, basically, matlab...). |
| `filler_rate_per_100w` | float64 | Fillers per 100 words. |
| `repetition_count` | int64 | Word repetitions. |
| `restart_count` | int64 | Sentence restarts. |
| `hedge_count` | int64 | Hedges (I think, maybe, not sure...). |
| `prolongation_count` | int64 | Sound prolongations (audio). |
| `block_count` | int64 | Speech blocks (audio). |
| `disfluency_rate_per_100w` | float64 | All disfluencies per 100 words. |
| `pitch_mean_hz` | float64 | Mean F0 pitch (Hz); empty when not speaking. |
| `pitch_std_hz` | float64 | F0 pitch std (Hz). |
| `pitch_range_hz` | float64 | F0 range (Hz). |
| `monotony_index` | float64 | 0-1, higher = more monotone voice. |
| `energy_mean_db` | float64 | Mean energy (dBFS). |
| `energy_std_db` | float64 | Energy std (dB). |
| `volume_consistency` | float64 | 0-1 volume steadiness. |
| `speech_rate_variability` | float64 | 0-1 variation of speech rate. |
| `jitter_pct` | float64 | Pitch jitter % (vocal stress). |
| `shimmer_pct` | float64 | Amplitude shimmer % (vocal stress). |
| `snr_db` | float64 | Signal-to-noise ratio (dB). |
| `asr_confidence_mean` | float64 | Mean ASR word confidence 0-1. |
| `mfcc_1_mean` | float64 | Mean MFCC coefficient 1 over the answer (speaker/voice timbre). |
| `mfcc_2_mean` | float64 | Mean MFCC coefficient 2 over the answer (speaker/voice timbre). |
| `mfcc_3_mean` | float64 | Mean MFCC coefficient 3 over the answer (speaker/voice timbre). |
| `mfcc_4_mean` | float64 | Mean MFCC coefficient 4 over the answer (speaker/voice timbre). |
| `mfcc_5_mean` | float64 | Mean MFCC coefficient 5 over the answer (speaker/voice timbre). |
| `mfcc_6_mean` | float64 | Mean MFCC coefficient 6 over the answer (speaker/voice timbre). |
| `mfcc_7_mean` | float64 | Mean MFCC coefficient 7 over the answer (speaker/voice timbre). |
| `mfcc_8_mean` | float64 | Mean MFCC coefficient 8 over the answer (speaker/voice timbre). |
| `mfcc_9_mean` | float64 | Mean MFCC coefficient 9 over the answer (speaker/voice timbre). |
| `mfcc_10_mean` | float64 | Mean MFCC coefficient 10 over the answer (speaker/voice timbre). |
| `mfcc_11_mean` | float64 | Mean MFCC coefficient 11 over the answer (speaker/voice timbre). |
| `mfcc_12_mean` | float64 | Mean MFCC coefficient 12 over the answer (speaker/voice timbre). |
| `mfcc_13_mean` | float64 | Mean MFCC coefficient 13 over the answer (speaker/voice timbre). |
| `interrupted_by_interviewer` | int64 | 1 = interviewer cut in. |
| `overlap_s` | float64 | Overlapping speech (s). |
| `mean_on_screen_ratio` | float64 | Mean on-screen gaze ratio during thinking+answering. |
| `mean_attention_score` | float64 | Mean attention score during thinking+answering. |
| `engaged_window_fraction` | float64 | Fraction of windows engaged. |
| `disengaged_window_fraction` | float64 | Fraction of windows disengaged. |
| `mean_blink_rate_per_min` | float64 | Mean blink rate. |
| `total_glance_away` | int64 | Total glances away. |
| `mean_gaze_yaw_std_deg` | float64 | Mean gaze yaw std. |
| `mean_head_yaw_std_deg` | float64 | Mean head yaw std. |
| `reading_pattern_detected` | int64 | 1 = reading pattern in any window of the answer. |
| `mean_confusion_level` | float64 | Mean confusion level. |
| `key_points_covered` | int64 | Expected key points covered (0-4). |
| `vague_sentence_count` | int64 | Generic/vague sentences. |
| `off_topic_sentence_count` | int64 | Off-topic sentences. |
| `contains_contradiction` | int64 | 1 = self-contradicting statement. |
| `keyword_overlap` | float64 | Fraction of expected keywords present. |
| `relevance` | float64 | LABEL relevance 0-1. |
| `completeness` | float64 | LABEL completeness 0-1. |
| `quality_1to5` | int64 | LABEL answer quality 1-5. |
| `star_situation` | float64 | LABEL STAR Situation present (behavioural only). |
| `star_task` | float64 | LABEL STAR Task present. |
| `star_action` | float64 | LABEL STAR Action present. |
| `star_result` | float64 | LABEL STAR Result present. |
| `hesitation_index` | float64 | LABEL hesitation 0-100. |
| `speaking_pattern` | str | LABEL confident / balanced / hesitant / rushed / monotone. |
| `clarity_score` | float64 | LABEL clarity 0-100. |
| `confidence_score` | float64 | LABEL confidence 0-100. |
| `engagement_score` | float64 | LABEL engagement 0-100. |
| `relevance_score` | float64 | LABEL relevance 0-100. |
| `fluency_score` | float64 | LABEL fluency 0-100. |
| `communication_effectiveness` | float64 | LABEL communication effectiveness 0-100 (main target). |
| `follow_up_needed` | int64 | LABEL 1 = answer needs follow-up. |
| `follow_up_reason` | str | LABEL none / vague / low_confidence / attention_drop / topic_not_covered / contradiction (meetings: participation reasons). |

## interview/conversation_turns.csv

| Column | Type | Description |
| --- | --- | --- |
| `session_id` | str | Interview session ID (S0001...). |
| `turn_index` | int64 | Order of turn. |
| `speaker` | str | candidate / interviewer. |
| `answer_id` | str | Answer ID = <session>_A<question order>. |
| `start_s` | float64 | Start time in seconds from session/meeting start. |
| `end_s` | float64 | End time in seconds from session/meeting start. |
| `duration_s` | float64 | Duration in seconds. |
| `gap_from_previous_s` | float64 | Silence before this turn (s). |
| `overlap_with_previous_s` | float64 | Overlap with previous turn (s). |
| `is_interruption` | int64 | 1 = overlap > 0.5 s. |

## interview/hesitation_events.csv

| Column | Type | Description |
| --- | --- | --- |
| `session_id` | str | Interview session ID (S0001...). |
| `answer_id` | str | Answer ID = <session>_A<question order>. |
| `event_type` | str | filler, repetition, restart, hedge, long_pause, prolongation, block, delayed_response. |
| `token` | str | Word(s) involved, if any. |
| `word_index` | float64 | Token position in answer_text. |
| `start_s` | float64 | Start time in seconds from session/meeting start. |
| `end_s` | float64 | End time in seconds from session/meeting start. |
| `source` | str | Signal the event is detectable from. |

## interview/participants.csv

| Column | Type | Description |
| --- | --- | --- |
| `participant_id` | str | Participant ID (P001...). Splits are by this ID. |
| `age_band` | str | Age group. |
| `gender` | str | male / female / non_binary. |
| `glasses` | bool | Wears glasses (affects eye/iris models). |
| `accent` | str | First-language influenced English accent. |
| `english_proficiency` | float64 | 0-1 self/assessor rated English proficiency. |
| `education` | str | Degree programme. |
| `work_experience_years` | int64 | Years of work experience. |
| `skin_tone_fitzpatrick` | int64 | Fitzpatrick skin type 1-6 (for vision bias checks). |
| `consent_given` | bool | Recording consent recorded. |
| `split` | str | train / val / test (person-wise). |

## interview/sessions.csv

| Column | Type | Description |
| --- | --- | --- |
| `session_id` | str | Interview session ID (S0001...). |
| `participant_id` | str | Participant ID (P001...). Splits are by this ID. |
| `split` | str | train / val / test (person-wise). |
| `mode` | str | interview or meeting. |
| `session_index` | int64 | 1st/2nd/3rd session of this participant (practice effect). |
| `device` | str | Capture device. |
| `camera_quality` | float64 | 0-1 camera quality score. |
| `lighting` | str | good / dim / backlit. |
| `lighting_lux` | float64 | Estimated illuminance (lux). |
| `background_noise_db` | float64 | Background noise level (dB SPL). |
| `network_jitter_ms` | float64 | WebRTC network jitter (ms). |
| `video_fps` | int64 | Video frame rate analysed. |
| `audio_sample_rate_hz` | int64 | Audio sample rate. |
| `duration_s` | float64 | Duration in seconds. |
| `n_questions` | int64 | Questions asked in the session. |
| `n_windows` | int64 | Number of 5 s windows in the session. |
| `talk_time_ratio_candidate` | float64 | Fraction of session time candidate spoke. |
| `mean_response_latency_s` | float64 | Mean response latency across answers (s). |
| `overall_effectiveness` | float64 | Session-level communication effectiveness 0-100 (mean over answers). |
| `mean_clarity` | float64 | Mean clarity sub-score. |
| `mean_confidence` | float64 | Mean confidence sub-score. |
| `mean_engagement` | float64 | LABEL mean engagement 0-1. |
| `mean_relevance` | float64 | Mean relevance sub-score. |
| `mean_fluency` | float64 | Mean fluency sub-score. |
| `follow_up_count` | int64 | Answers flagged for follow-up. |
| `interviewer_rating_1to10` | int64 | Interviewer overall rating 1-10. |
| `recommendation` | str | strong_hire / hire / borderline / no_hire. |

## interview/transcript_segments.csv

| Column | Type | Description |
| --- | --- | --- |
| `session_id` | str | Interview session ID (S0001...). |
| `segment_id` | str | Transcript segment ID; suffix _Q question, _R candidate response, _K interviewer acknowledgement. |
| `answer_id` | str | Answer ID = <session>_A<question order>. |
| `speaker` | str | candidate / interviewer. |
| `start_s` | float64 | Start time in seconds from session/meeting start. |
| `end_s` | float64 | End time in seconds from session/meeting start. |
| `text` | str | Spoken text. |
| `asr_confidence` | float64 | ASR confidence for the segment. |

## interview/windows.csv

| Column | Type | Description |
| --- | --- | --- |
| `window_id` | str | 5-second window ID = <session>_W<index>. |
| `session_id` | str | Interview session ID (S0001...). |
| `participant_id` | str | Participant ID (P001...). Splits are by this ID. |
| `answer_id` | str | Answer ID = <session>_A<question order>. |
| `question_id` | str | Question ID from reference/question_bank.csv. |
| `start_s` | float64 | Start time in seconds from session/meeting start. |
| `end_s` | float64 | End time in seconds from session/meeting start. |
| `candidate_state` | str | listening (interviewer asking), thinking (before answering), answering, idle. |
| `face_present_ratio` | float64 | Fraction of frames with a face detected. |
| `face_detection_conf` | float64 | Mean face-detector confidence 0-1. |
| `head_yaw_mean_deg` | float64 | Mean head yaw (deg, + = right). Empty if face absent. |
| `head_pitch_mean_deg` | float64 | Mean head pitch (deg, + = up). |
| `head_roll_mean_deg` | float64 | Mean head roll (deg). |
| `head_yaw_std_deg` | float64 | Head yaw std (movement/expressiveness). |
| `head_pitch_std_deg` | float64 | Head pitch std (nodding). |
| `gaze_yaw_mean_deg` | float64 | Mean horizontal gaze angle (deg; 0 = camera/screen centre). |
| `gaze_pitch_mean_deg` | float64 | Mean vertical gaze angle (deg; negative = looking down). |
| `gaze_yaw_std_deg` | float64 | Horizontal gaze std (deg). |
| `gaze_pitch_std_deg` | float64 | Vertical gaze std (deg). |
| `iris_left_x` | float64 | Left iris centre x in eye crop (0-1, 0.5 = centre). |
| `iris_left_y` | float64 | Left iris centre y in eye crop (0-1). |
| `iris_right_x` | float64 | Right iris centre x in eye crop (0-1). |
| `iris_right_y` | float64 | Right iris centre y in eye crop (0-1). |
| `iris_movement_std` | float64 | Std of iris position (iris movement amount). |
| `on_screen_ratio` | float64 | Fraction of frames gaze is on screen. |
| `glance_away_count` | int64 | Number of glances away from screen. |
| `saccade_rate_hz` | float64 | Rapid eye movements per second. |
| `fixation_duration_mean_s` | float64 | Mean fixation duration (s). |
| `blink_rate_per_min` | float64 | Blinks per minute (stress/fatigue indicator). |
| `eye_aspect_ratio_mean` | float64 | Mean eye aspect ratio (eye openness). |
| `reading_pattern_flag` | int64 | 1 = repeated horizontal sweeps (likely reading from another screen). |
| `face_motion_energy` | float64 | 0-1 facial motion energy. |
| `mouth_open_ratio` | float64 | 0-1 mouth opening (speaking proxy). |
| `smile_intensity` | float64 | 0-1 smile intensity. |
| `brow_raise_intensity` | float64 | 0-1 brow raise intensity. |
| `vad_speech_ratio` | float64 | Fraction of candidate audio frames containing speech. |
| `interviewer_speech_ratio` | float64 | Fraction of interviewer audio frames containing speech. |
| `rms_energy_db` | float64 | Candidate audio RMS energy (dBFS). |
| `pitch_mean_hz` | float64 | Mean F0 pitch (Hz); empty when not speaking. |
| `pitch_std_hz` | float64 | F0 pitch std (Hz). |
| `snr_db` | float64 | Signal-to-noise ratio (dB). |
| `words_in_window` | int64 | Candidate words spoken in window. |
| `fillers_in_window` | int64 | Filler words in window. |
| `attention_label` | str | LABEL on_screen / off_screen (annotator 1). |
| `attention_score` | float64 | LABEL attention 0-100. |
| `engagement_label` | str | LABEL engaged / neutral / disengaged. |
| `boredom_level` | int64 | LABEL 0-3 (DAiSEE style). |
| `confusion_level` | int64 | LABEL 0-3. |
| `frustration_level` | int64 | LABEL 0-3. |
| `distraction_event` | int64 | 1 = distraction (phone, looking away, leaving frame). |

## meeting/meeting_action_items.csv

| Column | Type | Description |
| --- | --- | --- |
| `meeting_id` | str | Meeting ID (M001...). |
| `owner_id` | str | Action item owner. |
| `action_item` | str | Action item text. |
| `raised_at_s` | float64 | When raised (s). |
| `status` | str | Action item status. |

## meeting/meeting_participant_minutes.csv

| Column | Type | Description |
| --- | --- | --- |
| `meeting_id` | str | Meeting ID (M001...). |
| `participant_id` | str | Participant ID (P001...). Splits are by this ID. |
| `minute_index` | int64 | Minute of meeting. |
| `camera_on` | int64 | 1 = camera on (vision columns empty when off). |
| `speaking_time_s` | float64 | Seconds spoken in this minute. |
| `turns_taken` | int64 | Turns in this minute. |
| `on_screen_ratio` | float64 | Fraction of frames gaze is on screen. |
| `head_yaw_std_deg` | float64 | Head yaw std (movement/expressiveness). |
| `blink_rate_per_min` | float64 | Blinks per minute (stress/fatigue indicator). |
| `face_motion_energy` | float64 | 0-1 facial motion energy. |
| `smile_intensity` | float64 | 0-1 smile intensity. |
| `multitasking_flag` | int64 | 1 = mostly looking away (likely multitasking). |
| `attention_score` | float64 | LABEL attention 0-100. |
| `engagement_label` | str | LABEL engaged / neutral / disengaged. |

## meeting/meeting_participant_summary.csv

| Column | Type | Description |
| --- | --- | --- |
| `meeting_id` | str | Meeting ID (M001...). |
| `participant_id` | str | Participant ID (P001...). Splits are by this ID. |
| `role` | str | host / participant. |
| `talk_time_s` | float64 | Total talk time (s). |
| `talk_share` | float64 | Share of meeting talk time. |
| `n_turns` | int64 | Speaking turns. |
| `mean_engagement` | float64 | LABEL mean engagement 0-1. |
| `contribution_balance` | float64 | 0-1, 1 = fair share of talk. |
| `follow_up_needed` | int64 | LABEL 1 = answer needs follow-up. |
| `follow_up_reason` | str | LABEL none / vague / low_confidence / attention_drop / topic_not_covered / contradiction (meetings: participation reasons). |

## meeting/meeting_turns.csv

| Column | Type | Description |
| --- | --- | --- |
| `meeting_id` | str | Meeting ID (M001...). |
| `turn_index` | int64 | Order of turn. |
| `speaker_id` | str | Participant speaking. |
| `start_s` | float64 | Start time in seconds from session/meeting start. |
| `end_s` | float64 | End time in seconds from session/meeting start. |
| `duration_s` | float64 | Duration in seconds. |
| `overlap_with_previous_s` | float64 | Overlap with previous turn (s). |
| `is_interruption` | int64 | 1 = overlap > 0.5 s. |

## meeting/meetings.csv

| Column | Type | Description |
| --- | --- | --- |
| `meeting_id` | str | Meeting ID (M001...). |
| `meeting_type` | str | standup / design_review / brainstorm / client_call / project_sync. |
| `duration_min` | int64 | Meeting length (min). |
| `n_participants` | int64 | Participants. |
| `host_id` | str | Host participant ID. |
| `n_turns` | int64 | Speaking turns. |
| `talk_balance_gini` | float64 | Gini of talk time (0 = perfectly balanced). |
| `meeting_effectiveness` | float64 | LABEL meeting effectiveness 0-100. |
| `split` | str | train / val / test (person-wise). |

## reference/question_bank.csv

| Column | Type | Description |
| --- | --- | --- |
| `question_id` | str | Question ID from reference/question_bank.csv. |
| `category` | str | Question category. |
| `difficulty` | float64 | Question difficulty 0-1. |
| `text` | str | Spoken text. |
| `key_point_1` | str | Expected key point 1 (STAR Situation for behavioural). |
| `key_point_2` | str | Expected key point 2 (Task). |
| `key_point_3` | str | Expected key point 3 (Action). |
| `key_point_4` | str | Expected key point 4 (Result). |
| `expected_keywords` | str | Pipe-separated expected keywords. |

## splits/test_participants.csv

| Column | Type | Description |
| --- | --- | --- |
| `participant_id` | str | Participant ID (P001...). Splits are by this ID. |

## splits/train_participants.csv

| Column | Type | Description |
| --- | --- | --- |
| `participant_id` | str | Participant ID (P001...). Splits are by this ID. |

## splits/val_participants.csv

| Column | Type | Description |
| --- | --- | --- |
| `participant_id` | str | Participant ID (P001...). Splits are by this ID. |
