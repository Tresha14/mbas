"""Per-answer feature assembler (Stage 8.2): our model outputs -> the fusion network's feature vector.

Every FUSION_FEATURES column (ml/common/datasets.py) and where it comes from:
  vision        mean_on_screen_ratio, mean_attention_score, engaged/disengaged_window_fraction,
                mean_blink_rate_per_min, total_glance_away, mean_gaze_yaw_std_deg, mean_head_yaw_std_deg,
                reading_pattern_detected                      <- window scores (vision worker) inside the answer
  prosody       speaking_rate_wpm ... speech_rate_variability <- ml/behaviour/prosody.py on the answer audio
                snr_db                                        <- speech vs non-speech energy (VAD)
                mfcc_1_mean ... mfcc_13_mean                  <- librosa MFCC means over speech frames
  hesitation    response_latency_s ... disfluency_rate_per_100w <- pause model + text rules + disfluency CRNN
  content       answer_duration_s, word_count                 <- pause model / transcript
                keyword_overlap, question_difficulty          <- question bank (expected keywords)
                asr_confidence_mean                           <- mean max-probability of the ASR frames
                ans_*                                         <- answer-quality transformer
  conversation  interrupted_by_interviewer, overlap_s         <- conversation analyser
"""
from __future__ import annotations

import csv
from functools import lru_cache

import numpy as np

from ml.common.config import SYNTHETIC


@lru_cache(maxsize=1)
def question_bank() -> dict[str, dict]:
    p = SYNTHETIC / "reference" / "question_bank.csv"
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return {r["question_id"]: r for r in csv.DictReader(f)}


def keyword_overlap(question_id: str, transcript: str) -> float:
    q = question_bank().get(question_id)
    if not q or not q.get("expected_keywords"):
        return float("nan")
    kws = [k.strip().lower() for k in q["expected_keywords"].split("|") if k.strip()]
    t = transcript.lower()
    return sum(k in t for k in kws) / max(len(kws), 1)


def snr_db(audio: np.ndarray, speech: np.ndarray) -> float:
    n = min(len(audio) // 160, len(speech))
    if n < 10:
        return float("nan")
    fr = audio[: n * 160].reshape(n, 160)
    e = (fr ** 2).mean(1) + 1e-10
    s, ns = e[speech[:n]], e[~speech[:n]]
    if len(s) < 5 or len(ns) < 5:
        return float("nan")
    return float(10 * np.log10(s.mean() / ns.mean()))


def mfcc_means(audio: np.ndarray, speech: np.ndarray | None) -> dict:
    from ml.common.audio_features import mfcc

    if len(audio) < 1600:
        return {}
    m = mfcc(audio)[:13]
    if speech is not None and speech.any():
        k = min(m.shape[1], len(speech))
        sel = speech[:k]
        if sel.sum() > 5:
            m = m[:, :k][:, sel]
    return {f"mfcc_{i + 1}_mean": float(v) for i, v in enumerate(m.mean(1))}


def vision_stats(windows: list[dict]) -> dict:
    if not windows:
        return {}
    f = lambda k: [w["features"].get(k) for w in windows if w["features"].get(k) is not None]  # noqa: E731
    eng = [w.get("engagement") for w in windows]
    return {"mean_on_screen_ratio": float(np.mean(f("on_screen_ratio"))) if f("on_screen_ratio") else np.nan,
            "mean_attention_score": float(np.mean([w["attention"] for w in windows if w.get("attention") is not None]))
            if any(w.get("attention") is not None for w in windows) else np.nan,
            "engaged_window_fraction": eng.count("engaged") / len(eng),
            "disengaged_window_fraction": eng.count("disengaged") / len(eng),
            "mean_blink_rate_per_min": float(np.mean(f("blink_rate_per_min"))) if f("blink_rate_per_min") else np.nan,
            "total_glance_away": float(np.sum(f("glance_away_count"))) if f("glance_away_count") else np.nan,
            "mean_gaze_yaw_std_deg": float(np.mean(f("gaze_yaw_std_deg"))) if f("gaze_yaw_std_deg") else np.nan,
            "mean_head_yaw_std_deg": float(np.mean(f("head_yaw_std_deg"))) if f("head_yaw_std_deg") else np.nan,
            "reading_pattern_detected": float(max(f("reading_pattern_flag") or [0]))}


def assemble(behaviour: dict, answer: dict | None, windows: list[dict], conversation: dict, question_id: str,
             transcript: str, audio: np.ndarray, speech: np.ndarray | None, asr_conf: float | None) -> dict:
    t, pr, th, ah = behaviour["timing"], behaviour["prosody"], behaviour["text_hesitation"], behaviour["audio_hesitation"]
    q = question_bank().get(question_id, {})
    feats = {**vision_stats(windows), **{k: pr.get(k, np.nan) for k in (
        "speaking_rate_wpm", "articulation_rate_wpm", "pitch_mean_hz", "pitch_std_hz", "pitch_range_hz", "monotony_index",
        "energy_mean_db", "energy_std_db", "volume_consistency", "speech_rate_variability")}}
    feats["snr_db"] = snr_db(audio, speech) if speech is not None else np.nan
    feats.update(mfcc_means(audio, speech))
    feats.update({k: t.get(k, np.nan) for k in ("response_latency_s", "long_pause_count", "long_pause_total_s",
                                                "pause_ratio", "mean_pause_duration_s")})
    feats.update({k: th.get(k, np.nan) for k in ("filler_count", "filler_rate_per_100w", "repetition_count",
                                                 "restart_count", "hedge_count", "disfluency_rate_per_100w")})
    feats["prolongation_count"] = ah.get("prolongation_count", 0)
    feats["block_count"] = ah.get("block_count", 0)
    feats.update(answer_duration_s=t.get("answer_duration_s", np.nan), word_count=th.get("word_count", 0),
                 keyword_overlap=keyword_overlap(question_id, transcript),
                 question_difficulty=float(q["difficulty"]) if q.get("difficulty") else np.nan,
                 asr_confidence_mean=asr_conf if asr_conf is not None else np.nan)
    if answer:
        feats.update(ans_relevance=answer["relevance"], ans_quality=answer["quality"], ans_completeness=answer["completeness"],
                     ans_star_count=answer["star_count"], ans_follow_up_p=answer["follow_up_p"],
                     **{f"ans_reason_{k}": v for k, v in answer["reason_probs"].items() if k != "none"})
    feats["interrupted_by_interviewer"] = float(conversation.get("interrupted", 0))
    feats["overlap_s"] = float(conversation.get("overlap_s", 0.0))
    return feats
