# Model card: Window engagement MLP + attention score model

_Generated 2026-10-25 by the training/evaluation scripts. Trained from random initialisation; no pre-trained weights._

## Task

Per 5 s window: engagement class (fallback until the CNN-GRU is trained on video) and a 0-100 attention score with an on/off-screen decision, from features our models compute.

## Inputs

| | |
| --- | --- |
| engagement | face_present_ratio, on_screen_ratio, glance_away_count, saccade_rate_hz, fixation_duration_mean_s, blink_rate_per_min, reading_pattern_flag, gaze_yaw_std_deg, gaze_pitch_std_deg, iris_movement_std, head_yaw_abs_deg, head_pitch_abs_deg, head_yaw_std_deg, head_pitch_std_deg, face_motion_energy, mouth_open_ratio, vad_speech_ratio, interviewer_speech_ratio, state_listening, state_thinking, state_answering |
| attention | face_present_ratio, on_screen_ratio, glance_away_count, saccade_rate_hz, blink_rate_per_min, reading_pattern_flag, head_yaw_abs_deg, head_pitch_abs_deg, head_yaw_std_deg, head_pitch_std_deg + engagement probabilities (out-of-fold in training) |

## Models

| | |
| --- | --- |
| engagement | MLP 2 x 64, dropout 0.2, class-weighted cross-entropy |
| attention | linear (logistic + linear regression) vs 2-layer MLP; best by test MAE: mlp |

## Data

synthetic windows, person-wise split (32881 train / 6505 val / 6848 test) - SYNTHETIC: the simulator designed these relationships, so scores are optimistic.

## Results (test)

| | |
| --- | --- |
| engagement macro-F1 | 0.7464 |
| engagement majority baseline | 0.1782 |
| attention linear MAE | 3.79 |
| attention mlp MAE | 3.156 |
| attention linear on/off F1 | 0.94 |
| attention mlp on/off F1 | 0.9501 |
| confusion matrices | docs/metrics/engagement_window_cm.png, docs/metrics/attention_onoff_cm.png |
