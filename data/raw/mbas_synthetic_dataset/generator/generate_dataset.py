"""
MBAS synthetic dataset generator.

Simulates interview and meeting sessions with hidden (latent) candidate traits,
then generates every observable signal the system's models produce or consume:
  - 5-second windows of vision + audio features with attention/engagement labels
  - per-answer transcripts (with real disfluencies inserted), prosody, hesitation,
    response features and all response / effectiveness / follow-up labels
  - hesitation events, transcript segments, conversation turns
  - meeting-mode data (per participant per minute, turns, summaries, action items)
  - a second annotator's labels for inter-annotator agreement
  - person-wise train / val / test splits

Usage:
    python generate_dataset.py --participants 120 --meetings 40 --seed 42 --out ../mbas_synthetic_dataset
"""
import argparse
import json
import math
import os
import re

import numpy as np
import pandas as pd

from question_bank import (QUESTIONS, VAGUE_SENTENCES, CONTRADICTION_SENTENCES, OPENERS,
                           FILLERS, HEDGES, RESTART_PREFIXES, INTERVIEWER_ACKS,
                           MEETING_ACTION_TEMPLATES)

WIN = 5.0     # window length (s)
HOP = 2.5     # window hop (s)
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been it its this that
these those i we you he she they my our your their me us them do does did have has had not no so very can will
would should could about into over under than then there here which who what when where why how also each
such using used use""".split())


def clip(x, lo=0.0, hi=1.0):
    return float(min(max(x, lo), hi))


def keywords(points, n=6):
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", " ".join(points).lower())
    seen, out = set(), []
    for w in words:
        if len(w) >= 5 and w not in STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:n]


# --------------------------------------------------------------------------- participants
def make_participants(rng, n):
    rows, latent = [], []
    accents = ["gujarati_english", "hindi_english", "marathi_english", "tamil_english",
               "bengali_english", "telugu_english", "neutral_indian_english"]
    for i in range(n):
        pid = f"P{i + 1:03d}"
        gender = rng.choice(["male", "female", "non_binary"], p=[.52, .45, .03])
        prof = clip(rng.normal(0.65, 0.15), 0.2, 0.98)
        confidence = clip(rng.beta(4, 3))
        nervous = clip(0.85 - 0.6 * confidence + rng.normal(0, 0.12))
        prepared = clip(0.7 * rng.beta(3, 2.5) + 0.3 * confidence)
        attentive = clip(rng.beta(5, 2.2))
        fluency = clip(0.5 * prof + 0.3 * confidence + 0.2 * rng.random())
        expressive = clip(rng.beta(3, 3))
        speech_rate = clip(rng.beta(3, 3) + 0.15 * (nervous - 0.5))
        accent_strength = clip(1 - prof + rng.normal(0, 0.1))
        cheater = rng.random() < 0.08
        rows.append(dict(
            participant_id=pid,
            age_band=rng.choice(["18-21", "22-25", "26-30", "31-40"], p=[.35, .45, .15, .05]),
            gender=gender,
            glasses=bool(rng.random() < 0.42),
            accent=rng.choice(accents, p=[.25, .22, .10, .10, .08, .10, .15]),
            english_proficiency=round(prof, 3),
            education=rng.choice(["btech_cse", "btech_it", "btech_ece", "mca", "bsc_cs", "mtech"],
                                 p=[.4, .15, .12, .13, .1, .1]),
            work_experience_years=int(rng.choice([0, 0, 0, 1, 2, 3, 5])),
            skin_tone_fitzpatrick=int(rng.choice([2, 3, 4, 5, 6], p=[.05, .25, .4, .22, .08])),
            consent_given=True,
        ))
        latent.append(dict(participant_id=pid, confidence=confidence, nervousness=nervous,
                           preparedness=prepared, attentiveness=attentive, fluency=fluency,
                           expressiveness=expressive, speech_rate=speech_rate,
                           accent_strength=accent_strength, reads_from_screen_tendency=cheater))
    return pd.DataFrame(rows), pd.DataFrame(latent)


def make_splits(rng, participants):
    ids = participants.participant_id.tolist()
    rng.shuffle(ids)
    n = len(ids)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    split = {p: "train" for p in ids[:n_tr]}
    split.update({p: "val" for p in ids[n_tr:n_tr + n_va]})
    split.update({p: "test" for p in ids[n_tr + n_va:]})
    return split


# --------------------------------------------------------------------------- answers (text)
def make_answer_text(rng, q, k, h, contradiction, other_points):
    """Return token list, events and content metadata for one answer."""
    n_cov = int(np.clip(round(k * 4 + rng.normal(0, 0.6)), 0, 4))
    if q["category"] == "behavioural":
        # STAR: usually told in order; weaker answers drop Task/Result first
        order = [0, 2, 1, 3] if n_cov < 4 else [0, 1, 2, 3]
        covered = sorted(order[:n_cov])
    else:
        covered = sorted(rng.choice(4, n_cov, replace=False).tolist())
    sentences = [rng.choice(OPENERS)]
    content = [q["points"][i] for i in covered]
    n_vague = int(np.clip(rng.poisson(0.4 + 3.0 * (1 - k)), 0, 4)) if n_cov < 4 else int(rng.random() < 0.15)
    n_off = int(rng.random() < max(0, 0.45 - k)) + int(rng.random() < max(0, 0.2 - k))
    vague = rng.choice(VAGUE_SENTENCES, n_vague, replace=False).tolist() if n_vague else []
    off = rng.choice(other_points, n_off, replace=False).tolist() if n_off else []
    body = content + vague + off
    # vague sentences tend to come first when the candidate is unsure
    if k < 0.4:
        body = vague + content + off
    sentences += body
    if contradiction:
        sentences.insert(rng.integers(2, len(sentences) + 1), rng.choice(CONTRADICTION_SENTENCES))
    if len(sentences) == 1:
        sentences.append(rng.choice(VAGUE_SENTENCES))

    tokens, events = [], []
    p_fill = 0.01 + 0.10 * h
    p_rep = 0.003 + 0.03 * h
    p_restart = 0.03 + 0.25 * h
    p_hedge = 0.05 + 0.30 * (1 - k)
    for si, s in enumerate(sentences):
        words = s.split()
        if si > 0 and rng.random() < p_hedge:
            hedge = rng.choice(HEDGES)
            events.append(("hedge", len(tokens), hedge))
            tokens += hedge.split()
        if si > 0 and rng.random() < p_restart:
            r = rng.choice(RESTART_PREFIXES)
            events.append(("restart", len(tokens), r + " —"))
            tokens += (r + " —").split()
        for w in words:
            if rng.random() < p_fill:
                f = rng.choice(FILLERS)
                events.append(("filler", len(tokens), f))
                tokens += f.split()
            if rng.random() < p_rep:
                events.append(("repetition", len(tokens), w))
                tokens.append(w)
            tokens.append(w)
    meta = dict(covered=covered, n_vague=n_vague, n_off=n_off, contradiction=contradiction)
    return tokens, events, meta


# --------------------------------------------------------------------------- main simulation
def simulate(args):
    rng = np.random.default_rng(args.seed)
    qmap = {q["id"]: q for q in QUESTIONS}
    hr = [q["id"] for q in QUESTIONS if q["category"] == "hr" and q["id"] not in ("Q01", "Q12")]
    beh = [q["id"] for q in QUESTIONS if q["category"] == "behavioural"]
    tech = [q["id"] for q in QUESTIONS if q["category"] == "technical"]

    participants, latent = make_participants(rng, args.participants)
    split = make_splits(rng, participants)
    participants["split"] = participants.participant_id.map(split)
    lat = latent.set_index("participant_id").to_dict("index")
    pinfo = participants.set_index("participant_id").to_dict("index")

    sessions, answers, windows, events_rows, segments, turns = [], [], [], [], [], []
    sid_n = 0
    for pid in participants.participant_id:
        L, P = lat[pid], pinfo[pid]
        n_sess = int(rng.choice([2, 3], p=[.6, .4]))
        speaker_mfcc = rng.normal(0, 1, 13) * np.linspace(8, 1, 13) + np.r_[-250, 120, rng.normal(0, 10, 11)]
        base_pitch = {"male": 120, "female": 210, "non_binary": 165}[P["gender"]] * rng.normal(1, 0.08)
        for s_idx in range(n_sess):
            sid_n += 1
            sid = f"S{sid_n:04d}"
            practice = 0.05 * s_idx
            prep = clip(L["preparedness"] + practice)
            nerv = clip(L["nervousness"] - 0.06 * s_idx)
            device = rng.choice(["laptop_webcam", "external_webcam", "phone"], p=[.65, .25, .10])
            cam_q = clip(rng.normal({"laptop_webcam": .7, "external_webcam": .88, "phone": .62}[device], .08))
            lighting = rng.choice(["good", "dim", "backlit"], p=[.65, .22, .13])
            lux = float({"good": rng.normal(350, 60), "dim": rng.normal(90, 25),
                         "backlit": rng.normal(180, 50)}[lighting])
            noise_db = float(np.clip(rng.normal(38, 6), 28, 60))
            reads_session = L["reads_from_screen_tendency"] and rng.random() < 0.7
            qs = ["Q01"] + rng.choice(hr, 2, replace=False).tolist() + rng.choice(beh, 3, replace=False).tolist() \
                + rng.choice(tech, 4, replace=False).tolist() + ["Q12"]

            t = 3.0  # greeting offset
            sess_answers, sess_segments = [], []
            for qi, qid in enumerate(qs):
                q = qmap[qid]
                aid = f"{sid}_A{qi + 1:02d}"
                # ---- interviewer asks
                q_dur = len(q["text"].split()) / 2.5 + abs(rng.normal(1.0, 0.4))
                q_start, q_end = t, t + q_dur
                sess_segments.append(dict(session_id=sid, segment_id=f"{aid}_Q", answer_id=aid, speaker="interviewer",
                                          start_s=round(q_start, 2), end_s=round(q_end, 2), text=q["text"],
                                          asr_confidence=round(clip(rng.normal(.93, .02)), 3)))
                # ---- latent answer state
                k = clip(prep - q["difficulty"] + 0.45 + rng.normal(0, 0.12))   # knowledge of this question
                h = clip(0.55 * nerv + 0.35 * (1 - k) + 0.25 * (1 - L["fluency"]) - 0.15 + rng.normal(0, .05))
                reading = bool(reads_session and q["category"] == "technical" and rng.random() < 0.6)
                if reading:
                    k = clip(k + 0.3)
                contradiction = bool(rng.random() < 0.03 + 0.04 * (1 - k))
                other_pts = [p for oq in QUESTIONS if oq["id"] != qid for p in oq["points"]]
                tokens, evs, meta = make_answer_text(rng, q, k, h, contradiction, other_pts)
                words = len(tokens)
                latency = float(np.clip(rng.lognormal(math.log(0.8 + 2.5 * (1 - k) + 1.5 * nerv), 0.35), 0.3, 15))
                art_wpm = float(np.clip(rng.normal(115 + 45 * L["speech_rate"] + 15 * nerv - 10 * h + (8 if reading else 0), 10), 85, 205))
                speech_time = words / art_wpm * 60
                n_long = int(rng.poisson(speech_time / 60 * (0.8 + 5 * h)))
                long_pauses = rng.lognormal(math.log(1.6), 0.35, n_long) if n_long else np.array([])
                n_fill = sum(1 for e in evs if e[0] == "filler")
                n_prolong = int(rng.poisson(speech_time / 60 * (0.2 + 2.5 * h * (1 - L["fluency"]))))
                n_block = int(rng.poisson(speech_time / 60 * (0.05 + 1.2 * h * (1 - L["fluency"]))))
                short_pause_total = float(rng.normal(0.08, 0.02) * speech_time)
                duration = speech_time + long_pauses.sum() + short_pause_total + 0.35 * n_fill + 0.5 * n_prolong + 0.8 * n_block
                a_start = q_end + latency
                a_end = a_start + duration
                text = " ".join(tokens)
                sess_segments.append(dict(session_id=sid, segment_id=f"{aid}_R", answer_id=aid, speaker="candidate",
                                          start_s=round(a_start, 2), end_s=round(a_end, 2), text=text,
                                          asr_confidence=None))
                # hesitation events with timestamps
                for etype, idx, tok in evs:
                    et = a_start + (idx / max(words, 1)) * duration
                    ed = {"filler": .35, "repetition": .3, "restart": .6, "hedge": .4}[etype]
                    events_rows.append(dict(session_id=sid, answer_id=aid, event_type=etype, token=tok,
                                            word_index=idx, start_s=round(et, 2), end_s=round(et + ed, 2), source="text+audio"))
                for lp in long_pauses:
                    et = a_start + rng.random() * duration
                    events_rows.append(dict(session_id=sid, answer_id=aid, event_type="long_pause", token="",
                                            word_index=None, start_s=round(et, 2), end_s=round(et + lp, 2), source="audio"))
                for etype, n_e, ed in (("prolongation", n_prolong, .5), ("block", n_block, .8)):
                    for _ in range(n_e):
                        et = a_start + rng.random() * duration
                        events_rows.append(dict(session_id=sid, answer_id=aid, event_type=etype, token="",
                                                word_index=None, start_s=round(et, 2), end_s=round(et + ed, 2), source="audio"))
                if latency > 3.0:
                    events_rows.append(dict(session_id=sid, answer_id=aid, event_type="delayed_response", token="",
                                            word_index=None, start_s=round(q_end, 2), end_s=round(a_start, 2), source="audio"))
                # interruption by interviewer on long rambling answers
                interrupted = bool(duration > 150 and rng.random() < 0.35)
                ack = rng.choice(INTERVIEWER_ACKS)
                gap = abs(rng.normal(0.8, 0.4))
                ack_start = a_end - (1.2 if interrupted else -gap)
                ack_end = ack_start + len(ack.split()) / 2.5 + 0.4
                sess_segments.append(dict(session_id=sid, segment_id=f"{aid}_K", answer_id=aid, speaker="interviewer",
                                          start_s=round(ack_start, 2), end_s=round(ack_end, 2), text=ack,
                                          asr_confidence=round(clip(rng.normal(.93, .02)), 3)))
                t = ack_end + abs(rng.normal(1.5, 0.6))
                sess_answers.append(dict(
                    aid=aid, q=q, k=k, h=h, reading=reading, meta=meta, tokens=tokens, evs=evs, text=text,
                    q_start=q_start, q_end=q_end, a_start=a_start, a_end=a_end, latency=latency, art_wpm=art_wpm,
                    speech_time=speech_time, long_pauses=long_pauses, n_fill=n_fill, n_prolong=n_prolong,
                    n_block=n_block, duration=duration, interrupted=interrupted, words=words,
                    short_pause_total=short_pause_total))
            sess_dur = t + 3.0

            # ---------------- windows
            distract_until = -1
            sess_windows = []
            for w_i, ws in enumerate(np.arange(0, sess_dur - WIN + 1e-6, HOP)):
                we, wc = ws + WIN, ws + WIN / 2
                progress = wc / sess_dur
                state, cur = "idle", None
                for a in sess_answers:
                    if a["q_start"] <= wc < a["q_end"]:
                        state, cur = "listening", a
                    elif a["q_end"] <= wc < a["a_start"]:
                        state, cur = "thinking", a
                    elif a["a_start"] <= wc < a["a_end"]:
                        state, cur = "answering", a
                    if cur:
                        break
                if cur is None:
                    # between turns: attach to the previous answer
                    prev = [a for a in sess_answers if a["a_end"] <= wc]
                    cur = prev[-1] if prev else sess_answers[0]
                fatigue = progress * 0.15 * (1 - L["attentiveness"])
                k, h = cur["k"], cur["h"]
                reading = cur["reading"] and state == "answering"
                if wc > distract_until and rng.random() < 0.025 * (1.3 - L["attentiveness"]):
                    distract_until = wc + rng.uniform(2, 8)
                distracted = wc <= distract_until
                p_on = {"listening": 0.62 + 0.33 * L["attentiveness"],
                        "thinking": 0.35 + 0.35 * L["attentiveness"] - 0.2 * (1 - k),
                        "answering": 0.45 + 0.45 * L["attentiveness"] - 0.2 * h,
                        "idle": 0.55 + 0.35 * L["attentiveness"]}[state] - fatigue
                if distracted:
                    p_on = 0.1
                if reading:
                    p_on = 0.2
                p_on = clip(p_on, 0.02, 0.98)
                on_ratio = clip(rng.beta(p_on * 12 + 0.5, (1 - p_on) * 12 + 0.5))
                eng_lat = clip(0.45 * L["attentiveness"] + 0.3 * L["expressiveness"] + 0.15 * L["confidence"]
                               - fatigue - (0.25 if distracted else 0) + rng.normal(0, 0.09) + 0.05)
                eng = "engaged" if eng_lat > 0.6 else ("disengaged" if eng_lat < 0.38 else "neutral")
                boredom = int(np.clip(round(3 * clip(0.75 - eng_lat + rng.normal(0, .1))), 0, 3))
                conf_lvl = (1 - k) * (0.9 if state in ("thinking", "answering") else 0.3) * (cur["q"]["difficulty"] + 0.3)
                confusion = int(np.clip(round(3 * clip(conf_lvl - 0.1 + rng.normal(0, .1))), 0, 3))
                frustration = int(np.clip(round(3 * clip(0.6 * conf_lvl * nerv + rng.normal(0, .08))), 0, 3))
                # gaze / head
                side = rng.choice([-1, 1])
                if reading:
                    gyaw, gpitch = rng.normal(side * 22, 3), rng.normal(-4, 3)
                elif distracted:
                    gyaw, gpitch = rng.normal(side * 15, 10), rng.normal(-22, 6)
                else:
                    off = 1 - on_ratio
                    gyaw = rng.normal(0, 4) + off * side * rng.normal(18, 5)
                    gpitch = rng.normal(-2, 3) + off * rng.normal(8, 6) * rng.choice([-1, 1])
                gyaw_std = abs(2 + 7 * (1 - on_ratio) + 3 * nerv + (6 if reading else 0) + rng.normal(0, 1))
                gpitch_std = abs(1.5 + 4 * (1 - on_ratio) + 2 * nerv + rng.normal(0, .8))
                hyaw = 0.5 * gyaw + rng.normal(0, 3)
                hpitch = 0.4 * gpitch + rng.normal(0, 3)
                hroll = rng.normal(0, 3)
                hyaw_std = abs(1.5 + 4 * eng_lat * L["expressiveness"] + 2 * nerv + rng.normal(0, .7))
                irx_l = 0.5 + gyaw / 90 * 0.35 + rng.normal(0, .015)
                iry_l = 0.5 - gpitch / 90 * 0.30 + rng.normal(0, .015)
                irx_r = irx_l + rng.normal(0.0, .01)
                iry_r = iry_l + rng.normal(0.0, .01)
                blink = float(np.clip(rng.normal(15 + 12 * nerv - (6 if reading else 0) + 8 * fatigue, 4), 3, 45))
                speaking = state == "answering"
                vad = clip(rng.normal(0.78 - 0.4 * h, .08)) if speaking else (
                    clip(rng.normal(0.7, .1)) if state == "listening" else clip(rng.normal(0.05, .04)))
                face_present = 1.0 if not (distracted and rng.random() < 0.15) else clip(rng.uniform(0.1, 0.6))
                det_conf = clip(rng.normal(0.6 + 0.35 * cam_q - (0.12 if lighting == "dim" else 0)
                                           - (0.08 if lighting == "backlit" else 0) - (0.03 if P["glasses"] else 0), .04))
                n_words_w = 0
                n_fill_w = 0
                if speaking:
                    frac = (min(we, cur["a_end"]) - max(ws, cur["a_start"])) / max(cur["duration"], 1e-6)
                    n_words_w = int(round(cur["words"] * max(frac, 0)))
                    n_fill_w = sum(1 for e in cur["evs"] if e[0] == "filler" and
                                   ws <= cur["a_start"] + e[1] / max(cur["words"], 1) * cur["duration"] < we)
                att_label = "on_screen" if (on_ratio >= 0.6 and not reading and not distracted and face_present > 0.5) else "off_screen"
                if rng.random() < 0.04:
                    att_label = "off_screen" if att_label == "on_screen" else "on_screen"
                att_score = 100 * clip(0.55 * on_ratio + 0.2 * (1 - min(abs(hyaw), 45) / 45) + 0.25 * eng_lat
                                       - (0.3 if reading else 0) - (0.2 if distracted else 0) + rng.normal(0, .03))
                row = dict(
                    window_id=f"{sid}_W{w_i:04d}", session_id=sid, participant_id=pid, answer_id=cur["aid"],
                    question_id=cur["q"]["id"], start_s=round(ws, 2), end_s=round(we, 2), candidate_state=state,
                    # vision
                    face_present_ratio=round(face_present, 3), face_detection_conf=round(det_conf, 3),
                    head_yaw_mean_deg=round(hyaw, 2), head_pitch_mean_deg=round(hpitch, 2), head_roll_mean_deg=round(hroll, 2),
                    head_yaw_std_deg=round(hyaw_std, 2), head_pitch_std_deg=round(abs(1 + 3 * eng_lat + rng.normal(0, .5)), 2),
                    gaze_yaw_mean_deg=round(gyaw, 2), gaze_pitch_mean_deg=round(gpitch, 2),
                    gaze_yaw_std_deg=round(gyaw_std, 2), gaze_pitch_std_deg=round(gpitch_std, 2),
                    iris_left_x=round(irx_l, 4), iris_left_y=round(iry_l, 4), iris_right_x=round(irx_r, 4), iris_right_y=round(iry_r, 4),
                    iris_movement_std=round(abs(0.01 + gyaw_std / 90 * 0.35 + rng.normal(0, .003)), 4),
                    on_screen_ratio=round(on_ratio, 3),
                    glance_away_count=int(rng.poisson(5 * (1 - on_ratio) + 0.3)),
                    saccade_rate_hz=round(abs(2 + 2 * nerv + 3 * reading + rng.normal(0, .5)), 2),
                    fixation_duration_mean_s=round(clip(0.35 + 0.3 * L["attentiveness"] - 0.15 * nerv - 0.1 * reading + rng.normal(0, .04), .08, 1.2), 3),
                    blink_rate_per_min=round(blink, 1),
                    eye_aspect_ratio_mean=round(clip(0.29 - 0.0015 * (blink - 15) - 0.03 * fatigue + rng.normal(0, .01), .12, .4), 3),
                    reading_pattern_flag=int(reading or (rng.random() < 0.01)),
                    face_motion_energy=round(clip(0.15 + 0.45 * eng_lat + 0.15 * speaking + rng.normal(0, .06)), 3),
                    mouth_open_ratio=round(clip((0.38 if speaking else 0.05) + rng.normal(0, .05)), 3),
                    smile_intensity=round(clip(0.6 * L["expressiveness"] * eng_lat + rng.normal(0, .06)), 3),
                    brow_raise_intensity=round(clip(0.25 * eng_lat + 0.12 * confusion + rng.normal(0, .05)), 3),
                    # audio (candidate track)
                    vad_speech_ratio=round(vad if speaking else clip(rng.normal(0.03, .02)), 3),
                    interviewer_speech_ratio=round(vad if state == "listening" else clip(rng.normal(0.03, .03)), 3),
                    rms_energy_db=round(rng.normal(-22 + 4 * L["confidence"], 2) if speaking else rng.normal(-55, 3), 2),
                    pitch_mean_hz=round(base_pitch * rng.normal(1 + 0.05 * nerv, 0.03), 1) if speaking else None,
                    pitch_std_hz=round(base_pitch * (0.06 + 0.14 * L["expressiveness"]) * rng.normal(1, .1), 1) if speaking else None,
                    snr_db=round(float(np.clip(rng.normal(-22 + 4 * L["confidence"] + 70 - noise_db, 3), 0, 45)), 1),
                    words_in_window=n_words_w, fillers_in_window=n_fill_w,
                    # labels
                    attention_label=att_label, attention_score=round(att_score, 1),
                    engagement_label=eng, boredom_level=boredom, confusion_level=confusion, frustration_level=frustration,
                    distraction_event=int(distracted),
                )
                if face_present < 0.5:
                    for c in ["head_yaw_mean_deg", "head_pitch_mean_deg", "head_roll_mean_deg", "gaze_yaw_mean_deg",
                              "gaze_pitch_mean_deg", "iris_left_x", "iris_left_y", "iris_right_x", "iris_right_y"]:
                        row[c] = None
                sess_windows.append(row)
            windows += sess_windows
            wdf = pd.DataFrame(sess_windows)

            # ---------------- answer-level features + labels
            sess_rows = []
            for a in sess_answers:
                q, meta, k, h = a["q"], a["meta"], a["k"], a["h"]
                aw = wdf[(wdf.answer_id == a["aid"]) & (wdf.candidate_state.isin(["answering", "thinking"]))]
                if aw.empty:
                    aw = wdf[wdf.answer_id == a["aid"]]
                n_cov = len(meta["covered"])
                ev_types = [e[0] for e in a["evs"]]
                n_rep, n_rst, n_hdg = ev_types.count("repetition"), ev_types.count("restart"), ev_types.count("hedge")
                n_long = len(a["long_pauses"])
                pause_total = float(a["long_pauses"].sum() + a["short_pause_total"])
                eff_wpm = a["words"] / a["duration"] * 60
                pitch_mean = base_pitch * rng.normal(1 + 0.05 * nerv, 0.02)
                pitch_std = base_pitch * (0.06 + 0.14 * L["expressiveness"]) * rng.normal(1, .08)
                monotony = clip(1 - (pitch_std / base_pitch) / 0.22)
                energy_std = 3 + 4 * L["expressiveness"] + rng.normal(0, .5)
                jitter = clip(0.6 + 1.4 * nerv + rng.normal(0, .15), .2, 3.5)
                shimmer = clip(2.5 + 3 * nerv + rng.normal(0, .5), 1, 9)
                asr_conf = clip(0.93 - 0.12 * L["accent_strength"] - 0.004 * (noise_db - 35) - 0.05 * h + rng.normal(0, .02), .5, .99)
                disfl_per100 = 100 * (a["n_fill"] + n_rep + n_rst + a["n_prolong"] + a["n_block"]) / max(a["words"], 1)
                filler_rate = a["n_fill"] / max(a["words"], 1)
                mean_on = float(aw.on_screen_ratio.mean())
                mean_att = float(aw.attention_score.mean())
                frac_eng = float((aw.engagement_label == "engaged").mean())
                reading_any = int(aw.reading_pattern_flag.max())
                # labels
                relevance = clip(0.15 + 0.75 * n_cov / 4 - 0.18 * meta["n_off"] - 0.04 * meta["n_vague"] + rng.normal(0, .05))
                completeness = clip(n_cov / 4 + rng.normal(0, .04))
                hes_idx = 100 * clip(0.35 * min(filler_rate / 0.08, 1) + 0.2 * min(pause_total / a["duration"] / 0.35, 1)
                                     + 0.15 * min(a["latency"] / 8, 1) + 0.15 * min((n_rep + n_rst) / max(a["words"], 1) / 0.04, 1)
                                     + 0.15 * min((a["n_prolong"] + a["n_block"]) / max(a["speech_time"] / 60, .1) / 4, 1)
                                     + rng.normal(0, .03))
                clarity = 100 * clip(0.45 * relevance + 0.25 * (1 - min(abs(eff_wpm - 140) / 60, 1))
                                     + 0.2 * (1 - min(filler_rate * 8, 1)) + 0.1 * asr_conf + rng.normal(0, .03))
                confidence_s = 100 * clip(0.3 * (1 - hes_idx / 100) + 0.2 * (1 - min(a["latency"] / 10, 1)) + 0.2 * mean_on
                                          + 0.15 * clip(1 - energy_std / 12) + 0.15 * (1 - min(jitter / 3, 1)) + rng.normal(0, .03))
                engagement_s = 100 * clip(0.5 * frac_eng + 0.3 * clip((pitch_std / base_pitch) / 0.2) + 0.2 * mean_att / 100 + rng.normal(0, .03))
                fluency_s = 100 * clip(1 - disfl_per100 / 15 - 0.3 * pause_total / a["duration"] + 0.1 + rng.normal(0, .03))
                relevance_s = 100 * relevance
                effectiveness = float(np.clip(0.22 * clarity + 0.20 * confidence_s + 0.15 * engagement_s + 0.28 * relevance_s
                                              + 0.15 * fluency_s - (8 if reading_any else 0) - (6 if meta["contradiction"] else 0)
                                              + rng.normal(0, 4), 0, 100))
                quality = int(np.clip(round(1 + 4 * clip(0.5 * relevance + 0.3 * completeness + 0.2 * clarity / 100 + rng.normal(0, .07))), 1, 5))
                if hes_idx > 55:
                    pattern = "hesitant"
                elif eff_wpm > 138:
                    pattern = "rushed"
                elif monotony > 0.55:
                    pattern = "monotone"
                elif hes_idx < 30 and confidence_s > 62:
                    pattern = "confident"
                else:
                    pattern = "balanced"
                if rng.random() < 0.05:
                    pattern = rng.choice(["confident", "rushed", "monotone", "hesitant", "balanced"])
                if meta["contradiction"]:
                    reason = "contradiction"
                elif relevance < 0.35 and meta["n_off"] > 0:
                    reason = "topic_not_covered"
                elif completeness <= 0.3 or meta["n_vague"] >= 2:
                    reason = "vague"
                elif mean_on < 0.45 or reading_any:
                    reason = "attention_drop"
                elif confidence_s < 50:
                    reason = "low_confidence"
                else:
                    reason = "none"
                if rng.random() < 0.04:
                    reason = rng.choice(["none", "vague", "low_confidence", "attention_drop"])
                star = {}
                for j, tag in enumerate(["situation", "task", "action", "result"]):
                    star[f"star_{tag}"] = int(j in meta["covered"]) if q["category"] == "behavioural" else None
                row = dict(
                    answer_id=a["aid"], session_id=sid, participant_id=pid, question_id=q["id"],
                    question_category=q["category"], question_difficulty=q["difficulty"], question_order=int(a["aid"][-2:]),
                    question_start_s=round(a["q_start"], 2), question_end_s=round(a["q_end"], 2),
                    answer_start_s=round(a["a_start"], 2), answer_end_s=round(a["a_end"], 2),
                    answer_text=a["text"],
                    # timing / fluency
                    response_latency_s=round(a["latency"], 2), answer_duration_s=round(a["duration"], 2),
                    word_count=a["words"], speaking_rate_wpm=round(eff_wpm, 1), articulation_rate_wpm=round(a["art_wpm"], 1),
                    long_pause_count=n_long, long_pause_total_s=round(float(a["long_pauses"].sum()), 2),
                    pause_ratio=round(pause_total / a["duration"], 3),
                    mean_pause_duration_s=round(float(a["long_pauses"].mean()) if n_long else 0.0, 2),
                    filler_count=a["n_fill"], filler_rate_per_100w=round(100 * filler_rate, 2),
                    repetition_count=n_rep, restart_count=n_rst, hedge_count=n_hdg,
                    prolongation_count=a["n_prolong"], block_count=a["n_block"],
                    disfluency_rate_per_100w=round(disfl_per100, 2),
                    # prosody / voice
                    pitch_mean_hz=round(pitch_mean, 1), pitch_std_hz=round(pitch_std, 1),
                    pitch_range_hz=round(pitch_std * rng.normal(3.8, .4), 1), monotony_index=round(monotony, 3),
                    energy_mean_db=round(rng.normal(-22 + 4 * L["confidence"], 1.5), 2), energy_std_db=round(energy_std, 2),
                    volume_consistency=round(clip(1 - energy_std / 12), 3),
                    speech_rate_variability=round(clip(0.05 + 0.25 * h + rng.normal(0, .03)), 3),
                    jitter_pct=round(jitter, 3), shimmer_pct=round(shimmer, 3),
                    snr_db=round(float(np.clip(rng.normal(70 - noise_db - 22 + 4 * L["confidence"], 3), 0, 45)), 1),
                    asr_confidence_mean=round(asr_conf, 3),
                    **{f"mfcc_{i + 1}_mean": round(float(speaker_mfcc[i] + rng.normal(0, 2)), 2) for i in range(13)},
                    # conversation
                    interrupted_by_interviewer=int(a["interrupted"]),
                    overlap_s=round(1.2 if a["interrupted"] else max(0.0, rng.normal(0.1, 0.15)), 2),
                    # vision aggregates over answer
                    mean_on_screen_ratio=round(mean_on, 3), mean_attention_score=round(mean_att, 1),
                    engaged_window_fraction=round(frac_eng, 3),
                    disengaged_window_fraction=round(float((aw.engagement_label == "disengaged").mean()), 3),
                    mean_blink_rate_per_min=round(float(aw.blink_rate_per_min.mean()), 1),
                    total_glance_away=int(aw.glance_away_count.sum()),
                    mean_gaze_yaw_std_deg=round(float(aw.gaze_yaw_std_deg.mean()), 2),
                    mean_head_yaw_std_deg=round(float(aw.head_yaw_std_deg.mean()), 2),
                    reading_pattern_detected=reading_any,
                    mean_confusion_level=round(float(aw.confusion_level.mean()), 2),
                    # content
                    key_points_covered=n_cov, vague_sentence_count=meta["n_vague"], off_topic_sentence_count=meta["n_off"],
                    contains_contradiction=int(meta["contradiction"]),
                    keyword_overlap=round(len(set(keywords(q["points"])) & set(w.lower().strip(".,;") for w in a["tokens"]))
                                          / max(len(keywords(q["points"])), 1), 3),
                    # labels
                    relevance=round(relevance, 3), completeness=round(completeness, 3), quality_1to5=quality, **star,
                    hesitation_index=round(hes_idx, 1), speaking_pattern=pattern,
                    clarity_score=round(clarity, 1), confidence_score=round(confidence_s, 1),
                    engagement_score=round(engagement_s, 1), relevance_score=round(relevance_s, 1),
                    fluency_score=round(fluency_s, 1), communication_effectiveness=round(effectiveness, 1),
                    follow_up_needed=int(reason != "none"), follow_up_reason=reason,
                )
                sess_rows.append(row)
            answers += sess_rows
            # segments asr confidence for candidate rows
            amap = {r["answer_id"]: r for r in sess_rows}
            for sg in sess_segments:
                if sg["speaker"] == "candidate":
                    sg["asr_confidence"] = amap[sg["answer_id"]]["asr_confidence_mean"]
            segments += sess_segments
            # conversation turns
            sg_sorted = sorted(sess_segments, key=lambda x: x["start_s"])
            prev = None
            for ti, sg in enumerate(sg_sorted):
                overlap = max(0.0, prev["end_s"] - sg["start_s"]) if prev else 0.0
                turns.append(dict(session_id=sid, turn_index=ti, speaker=sg["speaker"], answer_id=sg["answer_id"],
                                  start_s=sg["start_s"], end_s=sg["end_s"], duration_s=round(sg["end_s"] - sg["start_s"], 2),
                                  gap_from_previous_s=round(max(0.0, sg["start_s"] - prev["end_s"]), 2) if prev else None,
                                  overlap_with_previous_s=round(overlap, 2), is_interruption=int(overlap > 0.5)))
                prev = sg
            sdf = pd.DataFrame(sess_rows)
            eff = float(sdf.communication_effectiveness.mean())
            rating = int(np.clip(round(eff / 10 + rng.normal(0, 0.8)), 1, 10))
            rec = "strong_hire" if eff >= 72 else "hire" if eff >= 60 else "borderline" if eff >= 48 else "no_hire"
            sessions.append(dict(
                session_id=sid, participant_id=pid, split=split[pid], mode="interview", session_index=s_idx + 1,
                device=device, camera_quality=round(cam_q, 3), lighting=lighting, lighting_lux=round(lux, 0),
                background_noise_db=round(noise_db, 1), network_jitter_ms=round(abs(rng.normal(25, 15)), 1),
                video_fps=15, audio_sample_rate_hz=16000, duration_s=round(sess_dur, 1), n_questions=len(qs),
                n_windows=len(sess_windows),
                talk_time_ratio_candidate=round(sum(a["duration"] for a in sess_answers) / sess_dur, 3),
                mean_response_latency_s=round(float(sdf.response_latency_s.mean()), 2),
                overall_effectiveness=round(eff, 1),
                mean_clarity=round(float(sdf.clarity_score.mean()), 1),
                mean_confidence=round(float(sdf.confidence_score.mean()), 1),
                mean_engagement=round(float(sdf.engagement_score.mean()), 1),
                mean_relevance=round(float(sdf.relevance_score.mean()), 1),
                mean_fluency=round(float(sdf.fluency_score.mean()), 1),
                follow_up_count=int(sdf.follow_up_needed.sum()),
                interviewer_rating_1to10=rating, recommendation=rec,
            ))
        if int(pid[1:]) % 20 == 0:
            print(f"  simulated {pid}")

    return dict(participants=participants, latent=latent, sessions=pd.DataFrame(sessions),
                answers=pd.DataFrame(answers), windows=pd.DataFrame(windows),
                hesitation_events=pd.DataFrame(events_rows), transcript_segments=pd.DataFrame(segments),
                conversation_turns=pd.DataFrame(turns), rng=rng, lat=lat, split=split)


# --------------------------------------------------------------------------- meetings
def simulate_meetings(data, n_meet, rng):
    lat, split = data["lat"], data["split"]
    pids = list(lat.keys())
    meetings, minutes, mturns, summary, actions = [], [], [], [], []
    for m in range(n_meet):
        mid = f"M{m + 1:03d}"
        mtype = rng.choice(["standup", "design_review", "brainstorm", "client_call", "project_sync"])
        dur_min = int({"standup": rng.integers(10, 18), "design_review": rng.integers(30, 50),
                       "brainstorm": rng.integers(25, 45), "client_call": rng.integers(20, 40),
                       "project_sync": rng.integers(20, 35)}[mtype])
        n = int(rng.integers(3, 7))
        members = rng.choice(pids, n, replace=False).tolist()
        host = members[0]
        dominance = np.array([0.6 * lat[p]["confidence"] + 0.4 * rng.random() + (0.3 if p == host else 0) for p in members])
        share = np.exp(dominance * 3)
        share /= share.sum()
        talk = {p: 0.0 for p in members}
        ti, t = 0, 0.0
        prev_sp = None
        while t < dur_min * 60:
            sp = rng.choice(members, p=share)
            if sp == prev_sp:
                continue
            d = float(rng.lognormal(math.log(12 + 25 * share[members.index(sp)]), 0.6))
            overlap = float(max(0, rng.normal(0.2, 0.6))) if rng.random() < 0.15 else 0.0
            mturns.append(dict(meeting_id=mid, turn_index=ti, speaker_id=sp, start_s=round(t - overlap, 2),
                               end_s=round(t - overlap + d, 2), duration_s=round(d, 2),
                               overlap_with_previous_s=round(overlap, 2), is_interruption=int(overlap > 0.5)))
            talk[sp] += d
            t += d - overlap + abs(rng.normal(0.6, 0.4))
            ti += 1
            prev_sp = sp
        total_talk = sum(talk.values())
        shares = np.array([talk[p] / total_talk for p in members])
        gini = float(np.abs(shares[:, None] - shares[None, :]).sum() / (2 * n * shares.sum()))
        per_eng = []
        for p in members:
            L = lat[p]
            camera_on = rng.random() < 0.85
            engs = []
            for mi in range(dur_min):
                fat = mi / dur_min * 0.2 * (1 - L["attentiveness"])
                e = clip(0.4 * L["attentiveness"] + 0.25 * L["expressiveness"] + 0.2 * (talk[p] / total_talk * n)
                         * 0.5 - fat + rng.normal(0, .1) + 0.1)
                lab = "engaged" if e > 0.6 else ("disengaged" if e < 0.38 else "neutral")
                engs.append(e)
                on = clip(rng.beta((0.55 + 0.4 * L["attentiveness"] - fat) * 10 + .5, (0.45 - 0.4 * L["attentiveness"] + fat) * 10 + .5))
                speak_s = float(np.clip(rng.normal(talk[p] / dur_min, 8), 0, 60))
                minutes.append(dict(
                    meeting_id=mid, participant_id=p, minute_index=mi, camera_on=int(camera_on),
                    speaking_time_s=round(speak_s, 1), turns_taken=int(rng.poisson(speak_s / 15)),
                    on_screen_ratio=round(on, 3) if camera_on else None,
                    head_yaw_std_deg=round(abs(1.5 + 4 * e + rng.normal(0, .7)), 2) if camera_on else None,
                    blink_rate_per_min=round(float(np.clip(rng.normal(15 + 10 * L["nervousness"] + 10 * fat, 4), 3, 45)), 1) if camera_on else None,
                    face_motion_energy=round(clip(0.15 + 0.45 * e + rng.normal(0, .06)), 3) if camera_on else None,
                    smile_intensity=round(clip(0.6 * L["expressiveness"] * e + rng.normal(0, .06)), 3) if camera_on else None,
                    multitasking_flag=int(camera_on and on < 0.35),
                    attention_score=round(100 * clip(0.6 * (on if camera_on else 0.5) + 0.4 * e), 1),
                    engagement_label=lab))
            per_eng.append(float(np.mean(engs)))
        meff = float(np.clip(100 * (0.45 * np.mean(per_eng) + 0.35 * (1 - gini) + 0.2 * rng.random()), 0, 100))
        meetings.append(dict(meeting_id=mid, meeting_type=mtype, duration_min=dur_min, n_participants=n, host_id=host,
                             n_turns=ti, talk_balance_gini=round(gini, 3), meeting_effectiveness=round(meff, 1),
                             split="train" if rng.random() < .7 else ("val" if rng.random() < .5 else "test")))
        for p, sh, e in zip(members, shares, per_eng):
            low_part = sh < 0.5 / n
            disen = e < 0.42
            reason = "low_participation_and_disengaged" if (low_part and disen) else "low_participation" if low_part \
                else "disengaged" if disen else "dominating_discussion" if sh > 2.2 / n else "none"
            summary.append(dict(meeting_id=mid, participant_id=p, role="host" if p == host else "participant",
                                talk_time_s=round(talk[p], 1), talk_share=round(float(sh), 3),
                                n_turns=sum(1 for x in mturns if x["meeting_id"] == mid and x["speaker_id"] == p),
                                mean_engagement=round(e, 3),
                                contribution_balance=round(float(1 - abs(sh - 1 / n) * n / 2), 3),
                                follow_up_needed=int(reason != "none"), follow_up_reason=reason))
        for _ in range(int(rng.integers(1, 5))):
            owner = rng.choice(members)
            actions.append(dict(meeting_id=mid, owner_id=owner,
                                action_item=rng.choice(MEETING_ACTION_TEMPLATES).format(
                                    owner=owner, day=rng.choice(["Monday", "Wednesday", "Friday", "end of week"])),
                                raised_at_s=round(float(rng.uniform(0.3, 1.0) * dur_min * 60), 1),
                                status="open"))
    return dict(meetings=pd.DataFrame(meetings), meeting_participant_minutes=pd.DataFrame(minutes),
                meeting_turns=pd.DataFrame(mturns), meeting_participant_summary=pd.DataFrame(summary),
                meeting_action_items=pd.DataFrame(actions))


# --------------------------------------------------------------------------- annotator 2
def second_annotator(data, rng):
    sessions = data["sessions"].session_id.tolist()
    chosen = set(rng.choice(sessions, int(0.2 * len(sessions)), replace=False))
    w = data["windows"][data["windows"].session_id.isin(chosen)][["window_id", "session_id", "attention_label", "engagement_label"]].copy()
    flip_a = rng.random(len(w)) < 0.07
    w.loc[flip_a, "attention_label"] = np.where(w.loc[flip_a, "attention_label"] == "on_screen", "off_screen", "on_screen")
    adj = {"engaged": "neutral", "neutral": rng.choice(["engaged", "disengaged"]), "disengaged": "neutral"}
    flip_e = rng.random(len(w)) < 0.15
    w.loc[flip_e, "engagement_label"] = w.loc[flip_e, "engagement_label"].map(adj)
    w["annotator_id"] = "ANN2"
    a = data["answers"][data["answers"].session_id.isin(chosen)][["answer_id", "session_id", "quality_1to5", "relevance",
                                                                    "follow_up_needed", "follow_up_reason", "speaking_pattern"]].copy()
    a["quality_1to5"] = np.clip(a["quality_1to5"] + rng.choice([-1, 0, 0, 0, 0, 1], len(a)), 1, 5)
    a["relevance"] = np.clip(a["relevance"] + rng.normal(0, .07, len(a)), 0, 1).round(3)
    flip_f = rng.random(len(a)) < 0.1
    a.loc[flip_f, "follow_up_needed"] = 1 - a.loc[flip_f, "follow_up_needed"]
    a.loc[flip_f & (a.follow_up_needed == 0), "follow_up_reason"] = "none"
    a.loc[flip_f & (a.follow_up_needed == 1) & (a.follow_up_reason == "none"), "follow_up_reason"] = "vague"
    a["annotator_id"] = "ANN2"
    return w, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", type=int, default=120)
    ap.add_argument("--meetings", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="../mbas_synthetic_dataset")
    args = ap.parse_args()
    print("Simulating interviews...")
    data = simulate(args)
    rng = data["rng"]
    print("Simulating meetings...")
    meet = simulate_meetings(data, args.meetings, rng)
    print("Second annotator...")
    w2, a2 = second_annotator(data, rng)

    out = args.out
    for sub in ["interview", "meeting", "annotations", "splits", "reference", "hidden_ground_truth"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    data["participants"].to_csv(f"{out}/interview/participants.csv", index=False)
    data["sessions"].to_csv(f"{out}/interview/sessions.csv", index=False)
    data["answers"].to_csv(f"{out}/interview/answers.csv", index=False)
    data["windows"].to_csv(f"{out}/interview/windows.csv", index=False)
    data["hesitation_events"].to_csv(f"{out}/interview/hesitation_events.csv", index=False)
    data["transcript_segments"].to_csv(f"{out}/interview/transcript_segments.csv", index=False)
    data["conversation_turns"].to_csv(f"{out}/interview/conversation_turns.csv", index=False)
    for k, v in meet.items():
        v.to_csv(f"{out}/meeting/{k}.csv", index=False)
    w2.to_csv(f"{out}/annotations/windows_annotator2.csv", index=False)
    a2.to_csv(f"{out}/annotations/answers_annotator2.csv", index=False)
    data["latent"].round(3).to_csv(f"{out}/hidden_ground_truth/participant_latent_traits.csv", index=False)
    qb = pd.DataFrame([dict(question_id=q["id"], category=q["category"], difficulty=q["difficulty"], text=q["text"],
                            key_point_1=q["points"][0], key_point_2=q["points"][1], key_point_3=q["points"][2],
                            key_point_4=q["points"][3], expected_keywords="|".join(keywords(q["points"])))
                       for q in QUESTIONS])
    qb.to_csv(f"{out}/reference/question_bank.csv", index=False)
    for name in ["train", "val", "test"]:
        ids = [p for p, s in data["split"].items() if s == name]
        pd.DataFrame({"participant_id": sorted(ids)}).to_csv(f"{out}/splits/{name}_participants.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
