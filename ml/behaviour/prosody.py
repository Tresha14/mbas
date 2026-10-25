"""Prosody features per answer (signal processing): speaking rate, pitch, energy, monotony, consistency."""
from __future__ import annotations

import numpy as np

from ml.common.audio_features import HOP, SR, pitch_yin, rms_db


def syllable_rate(x: np.ndarray, sr: int = SR) -> float:
    """Energy-envelope peaks per second of speech (a syllable-rate proxy that does not need ASR)."""
    db = rms_db(x)
    env = np.convolve(db, np.ones(5) / 5, mode="same")
    thr = np.percentile(env, 40)
    peaks = (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > thr + 3)
    speech_s = max((env > thr).sum() * HOP / sr, 1e-3)
    return float(peaks.sum() / speech_s)


def prosody(x: np.ndarray, n_words: int | None = None, speech_s: float | None = None, sr: int = SR) -> dict:
    """x: the candidate's answer audio. n_words from the transcript / ASR; speech_s from the pause model."""
    dur = len(x) / sr
    f0 = pitch_yin(x, sr)
    voiced = f0[~np.isnan(f0)]
    db = rms_db(x)
    active = db[db > np.percentile(db, 30)] if len(db) else db
    seg = int(2 * sr)
    rates = [syllable_rate(x[i:i + seg], sr) for i in range(0, max(len(x) - seg, 1), seg)] if len(x) > seg else [syllable_rate(x, sr)]
    p_mean = float(voiced.mean()) if len(voiced) else float("nan")
    p_std = float(voiced.std()) if len(voiced) else float("nan")
    semitone_std = float(np.std(12 * np.log2(voiced / np.median(voiced)))) if len(voiced) > 5 else float("nan")
    out = {
        "duration_s": round(dur, 3),
        "pitch_mean_hz": round(p_mean, 2), "pitch_std_hz": round(p_std, 2),
        "pitch_range_hz": round(float(np.percentile(voiced, 95) - np.percentile(voiced, 5)), 2) if len(voiced) else float("nan"),
        # monotony: 1 when pitch variation is below ~1 semitone, 0 above ~4 semitones
        "monotony_index": round(float(np.clip((4 - semitone_std) / 3, 0, 1)), 4) if not np.isnan(semitone_std) else float("nan"),
        "energy_mean_db": round(float(active.mean()), 2) if len(active) else float("nan"),
        "energy_std_db": round(float(active.std()), 2) if len(active) else float("nan"),
        "volume_consistency": round(float(np.clip(1 - active.std() / 15, 0, 1)), 4) if len(active) else float("nan"),
        "syllable_rate": round(syllable_rate(x, sr), 3),
        "speech_rate_variability": round(float(np.std(rates) / (np.mean(rates) + 1e-6)), 4),
    }
    if n_words is not None:
        out["speaking_rate_wpm"] = round(60 * n_words / max(dur, 1e-3), 2)
        if speech_s:
            out["articulation_rate_wpm"] = round(60 * n_words / max(speech_s, 1e-3), 2)
    return out
