"""Pause model from VAD output: response latency, mid-answer long pauses, pause ratio.

Input: per-frame speech probability or booleans for the candidate track (10 ms frames, session time),
and the question timing (question end = interviewer stops speaking after 'Ask').
"""
from __future__ import annotations

import numpy as np

FRAME_S = 0.01


def segments(speech: np.ndarray, t0: float = 0.0, min_speech_s: float = 0.12, min_gap_s: float = 0.25) -> list[tuple[float, float]]:
    """Boolean frames -> merged speech segments [(start_s, end_s)] in session time."""
    s = np.asarray(speech, bool)
    out, start = [], None
    for i, v in enumerate(np.append(s, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([t0 + start * FRAME_S, t0 + i * FRAME_S])
            start = None
    merged = []
    for seg in out:
        if merged and seg[0] - merged[-1][1] < min_gap_s:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    return [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= min_speech_s]


def question_end(interviewer_segs: list[tuple[float, float]], ask_s: float, answer_end_s: float) -> float:
    """The interviewer's last speech that starts within 15 s of 'Ask' and before the answer ends."""
    ends = [b for a, b in interviewer_segs if ask_s - 0.5 <= a <= min(ask_s + 15, answer_end_s)]
    return max(ends) if ends else ask_s


def answer_pauses(cand_segs: list[tuple[float, float]], q_end_s: float, answer_end_s: float,
                  long_pause_s: float = 1.0) -> dict:
    segs = [(max(a, q_end_s), min(b, answer_end_s)) for a, b in cand_segs if b > q_end_s and a < answer_end_s]
    if not segs:
        return {"response_latency_s": round(answer_end_s - q_end_s, 3), "answer_start_s": None, "speech_s": 0.0,
                "long_pause_count": 0, "long_pause_total_s": 0.0, "mean_pause_duration_s": 0.0, "pause_ratio": 1.0,
                "answer_duration_s": 0.0}
    start, end = segs[0][0], segs[-1][1]
    gaps = [b[0] - a[1] for a, b in zip(segs, segs[1:])]
    longs = [g for g in gaps if g > long_pause_s]
    speech = sum(b - a for a, b in segs)
    dur = max(end - start, 1e-6)
    return {"response_latency_s": round(start - q_end_s, 3), "answer_start_s": round(start, 3),
            "speech_s": round(speech, 3), "long_pause_count": len(longs), "long_pause_total_s": round(sum(longs), 3),
            "mean_pause_duration_s": round(float(np.mean(longs)), 3) if longs else 0.0,
            "pause_ratio": round(1 - speech / dur, 4), "answer_duration_s": round(dur, 3)}
