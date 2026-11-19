"""Combined flags: moments where SEVERAL signals got worse together, judged against the speaker's own baseline.

Ported from the older project (minor_project/analyzer/flags.py) and adapted to this project's data. A single
deviation (one long pause, one glance away) is normal in any conversation, so flagging each one is noise. What is
worth a follow-up is co-occurrence: a long pause AND hedged phrasing AND looking away in the same stretch. And
"worse" is measured against THIS speaker's own median, not a universal ideal, because gaze and pace vary with
glasses, camera position and personal style (a fairness point).

How it works
  1. Each reading is turned into a 0-1 "how good is this" score with fixed, listed thresholds (NORM below).
  2. A signal's baseline is the MEDIAN of its readings over the whole interview (robust to a few bad stretches).
     A signal with fewer than MIN_BASELINE readings has no baseline and cannot fire; it is never guessed.
  3. A 30 s window slides over the interview in 15 s steps. A signal deviates in a window when the window's
     average is at least NORM_DROP worse than its baseline AND is itself below NORM_MAX (so a consistently great
     speaker is not flagged for noise). A long-pause flag inside the window counts as one more deviating signal.
  4. A window with at least MIN_SIGNALS deviating signals is flagged; overlapping windows are merged.

Limits, stated on the report too: the baseline comes from the whole interview after the fact, and a speaker who is
uniformly weak deviates from nothing, so they get no combined flags (the rated scorecard covers that case). Windows
in which no face was seen are ignored for the camera signals (absence has its own "face not in frame" flag), and
blink rate is not used (a weak signal that the eye tracker often loses).

Pure functions over plain dicts; no I/O.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

WINDOW_S, STEP_S = 30.0, 15.0
NORM_DROP, NORM_MAX = 0.25, 0.60
MIN_SIGNALS, MIN_BASELINE = 2, 3

# signal -> (short phrase for the text). Order is the display order.
PHRASES = {
    "long_pause": "a long pause",
    "pause_rate": "more pausing than usual",
    "hedge_rate": "hedged phrasing",
    "filler_rate": "more filler words",
    "camera_facing": "looking away from the screen",
    "wpm": "an unusual speaking pace",
    "pose_stability": "restless head movement",
}
_METRICS = tuple(k for k in PHRASES if k != "long_pause")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _lower_better(v: float, good: float, bad: float) -> float:
    return 1.0 if v <= good else 0.0 if v >= bad else _clamp((bad - v) / (bad - good))


def _band(v: float, floor: float, lo: float, hi: float, ceiling: float) -> float:
    if v <= floor or v >= ceiling:
        return 0.0
    if v < lo:
        return _clamp((v - floor) / (lo - floor))
    return 1.0 if v <= hi else _clamp((ceiling - v) / (ceiling - hi))


# raw reading -> 0-1 "how good is this" (same thresholds the older project's fusion used)
NORM = {
    "camera_facing": lambda v: _clamp(v),                      # fraction of the window on screen
    "pose_stability": lambda v: _lower_better(v, 2.0, 12.0),   # head-turn spread in degrees
    "wpm": lambda v: _band(v, 60.0, 110.0, 165.0, 220.0),
    "pause_rate": lambda v: _lower_better(v, 0.15, 0.45),      # fraction of time paused
    "filler_rate": lambda v: _lower_better(v, 1.0, 8.0),       # per 100 words
    "hedge_rate": lambda v: _lower_better(v, 1.0, 10.0),       # per 100 words
}


@dataclass
class CombinedFlag:
    t_start: float
    t_end: float
    signals: list[str] = field(default_factory=list)
    question_order: int | None = None
    question_ref: int | None = None


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _series(windows: list[dict], speech: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """{signal: [(t_s, normalised 0-1 score)]} from the stored vision windows and speaking stats."""
    out: dict[str, list[tuple[float, float]]] = {k: [] for k in _METRICS}
    for w in windows:
        ft = w.get("features") or {}
        face = ft.get("face_present_ratio")
        if face is not None and _ok(face) and face < 0.5:
            continue                                                   # no face seen: nothing to judge
        t = w.get("end_s")
        if not _ok(t):
            continue
        if _ok(w.get("on_screen_ratio")):
            out["camera_facing"].append((t, NORM["camera_facing"](float(w["on_screen_ratio"]))))
        if _ok(ft.get("head_yaw_std_deg")):
            out["pose_stability"].append((t, NORM["pose_stability"](float(ft["head_yaw_std_deg"]))))
    for s in speech:
        t = s.get("t_s")
        if not _ok(t):
            continue
        for key, field_name in (("wpm", "wpm"), ("pause_rate", "pause_ratio"), ("filler_rate", "fillers_per_100w"),
                                ("hedge_rate", "hedges_per_100w")):
            if _ok(s.get(field_name)):
                out[key].append((t, NORM[key](float(s[field_name]))))
    return out


def _spans(duration: float) -> list[tuple[float, float]]:
    out, lo = [], 0.0
    while lo < duration:
        hi = min(lo + WINDOW_S, duration)
        if out and hi - lo < WINDOW_S / 2:      # a short tail is covered by the previous, half-overlapping window
            break
        out.append((lo, hi))
        if hi >= duration:
            break
        lo += STEP_S
    return out


def find_combined_flags(windows: list[dict], speech: list[dict], flags: list[dict], cards: list[dict] | None = None) -> list[dict]:
    """The flagged moments, in time order, ready for the report:
    {t_start, t_end, time, signals, phrases, text, question_order, question_ref}."""
    series = _series(windows, speech)
    pauses = [f["t_s"] for f in flags if f.get("kind") == "long_pause" and _ok(f.get("t_s"))]
    times = [t for rows in series.values() for t, _ in rows] + pauses
    duration = max(times) if times else 0.0
    if duration <= 0:
        return []
    base = {k: (statistics.median(v for _, v in rows) if len(rows) >= MIN_BASELINE else None) for k, rows in series.items()}

    raw: list[CombinedFlag] = []
    for lo, hi in _spans(duration):
        fired = []
        for k in _METRICS:
            vals = [v for t, v in series[k] if lo <= t < hi]
            if base[k] is None or not vals:
                continue
            score = sum(vals) / len(vals)
            if score <= base[k] - NORM_DROP and score <= NORM_MAX:
                fired.append(k)
        if any(lo <= t < hi for t in pauses):
            fired.append("long_pause")
        if len(fired) >= MIN_SIGNALS:
            raw.append(CombinedFlag(lo, hi, fired))

    order = list(PHRASES)
    merged: list[CombinedFlag] = []
    for f in raw:
        if merged and f.t_start <= merged[-1].t_end:
            m = merged[-1]
            m.t_end = max(m.t_end, f.t_end)
            m.signals = sorted(set(m.signals) | set(f.signals), key=order.index)
        else:
            merged.append(CombinedFlag(f.t_start, f.t_end, sorted(f.signals, key=order.index)))

    out = []
    for f in merged:
        best, best_overlap = None, 0.0
        for c in cards or []:
            a, b = c.get("ask_s"), c.get("end_s")
            if not _ok(a):
                continue
            overlap = min(f.t_end, b if _ok(b) else float("inf")) - max(f.t_start, a)
            if overlap > best_overlap:
                best, best_overlap = c, overlap
        phrases = [PHRASES[s] for s in f.signals]
        joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
        out.append({"t_start": round(f.t_start, 1), "t_end": round(f.t_end, 1), "time": f"{_mmss(f.t_start)}–{_mmss(f.t_end)}",
                    "signals": f.signals, "phrases": phrases, "text": f"{joined} together",
                    "question_order": best.get("order") if best else None, "question_ref": best.get("question_ref") if best else None})
    return out
