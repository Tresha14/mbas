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


def _voiced_runs(f0: np.ndarray, bridge: int = 3) -> list[tuple[int, np.ndarray]]:
    """Voiced stretches as (start frame, f0), with unvoiced gaps of <= `bridge` frames filled by interpolation
    (the energy gate drops single 10 ms frames inside vowels)."""
    v = ~np.isnan(f0)
    if v.sum() < 2:
        return []
    idx = np.flatnonzero(v)
    f = f0.copy()
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        a, b = idx[g], idx[g + 1]
        if b - a - 1 <= bridge:
            f[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [f0[a], f0[b]])
    v = ~np.isnan(f)
    edges = np.flatnonzero(np.diff(np.r_[0, v.astype(int), 0]))
    return [(a, f[a:b]) for a, b in zip(edges[::2], edges[1::2])]


def voice_breaks(f0: np.ndarray, sr: int = SR, jump_st: float = 7.0, side: int = 12, steady_st: float = 0.5,
                 max_transition: int = 6) -> tuple[int, list[float]]:
    """Voice breaks ("cracks"): inside one voiced stretch, pitch is steady for 120 ms (std <= `steady_st` semitones),
    then, after at most `max_transition` frames (60 ms) of transition, is steady again for 120 ms at a level
    >= `jump_st` semitones away. Jumps of about an octave (10.5-13.5 st) are left out because they are mostly
    pitch-tracker octave errors; jumps closer than 300 ms count once. Thresholds tuned so LibriSpeech read speech
    gives median 0 and 90th percentile 5.5 breaks per voiced minute.
    Returns (count, times in seconds from the clip start)."""
    times = []
    for a, run in _voiced_runs(f0):
        st = 12 * np.log2(run / np.median(run))
        steady = np.array([st[i:i + side].std() <= steady_st for i in range(max(len(st) - side + 1, 0))], bool)
        for i in range(side, len(st) - side + 1):
            if not steady[i - side]:
                continue
            lo = np.median(st[i - side:i])
            for w in range(max_transition + 1):
                if i + w + side > len(st) or not steady[i + w]:
                    continue
                jump = abs(np.median(st[i + w:i + w + side]) - lo)
                if jump >= jump_st and not 10.5 <= jump <= 13.5:
                    t = (a + i + w) * HOP / sr          # where the new pitch level starts
                    if not times or t - times[-1] > 0.3:
                        times.append(round(float(t), 2))
                    break
    return len(times), times


def pitch_jitter_pct(f0: np.ndarray) -> float:
    """Frame-level pitch perturbation: mean |second difference of the period| / mean period (%), inside voiced
    stretches of >= 100 ms, skipping pitch-tracker jumps (> 2 semitones between frames). The second difference
    ignores smooth intonation, so this measures shakiness. LibriSpeech: median 1.1 %, 90th percentile 1.7 %.
    (Clinical jitter is per glottal cycle; this is the 10 ms-frame approximation.)"""
    d, p = [], []
    for _, run in _voiced_runs(f0):
        if len(run) < 10:
            continue
        ok = np.abs(np.diff(12 * np.log2(run))) < 2
        T = 1.0 / run
        d.append(np.abs(T[2:] - 2 * T[1:-1] + T[:-2])[ok[1:] & ok[:-1]])
        p.append(T)
    if not d or not np.concatenate(d).size:
        return float("nan")
    return float(np.concatenate(d).mean() / np.concatenate(p).mean() * 100)


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
    n_breaks, break_t = voice_breaks(f0, sr)
    voiced_min = len(voiced) * HOP / sr / 60
    out.update(voice_break_count=n_breaks, voice_break_times_s=break_t,
               voice_break_rate_per_min=round(n_breaks / voiced_min, 2) if voiced_min > 0.02 else float("nan"),
               pitch_jitter_pct=round(pitch_jitter_pct(f0), 3))
    if n_words is not None:
        out["speaking_rate_wpm"] = round(60 * n_words / max(dur, 1e-3), 2)
        if speech_s:
            out["articulation_rate_wpm"] = round(60 * n_words / max(speech_s, 1e-3), 2)
    return out
