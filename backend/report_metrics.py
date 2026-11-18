"""Detailed behaviour metrics, the rated scorecard and the analysis extras for the interviewer report.

behaviour_sections  eyes (blink rate, eye contact, glances, gaze stability, saccades, iris movement, reading),
                    voice (pace, expressiveness, voice breaks, pitch jitter, volume), fluency (fillers with a word
                    breakdown, crutch words, hedges, repetitions, audio disfluencies, hesitation index) and pauses
                    (response delay, long and very long pauses, longest silence). Every metric has a typical range
                    and a status (good / watch / concern) from transparent thresholds.
scorecard           six 0-100 dimensions from those metrics, a weighted overall rating and a letter grade.
confidence_matrix   each answer placed on content quality vs vocal confidence ("knows it but sounds unsure",
                    "sounds sure but off target").
pressure_response   how delivery changes between the easier and the harder half of the questions.
key_moments         the timestamps worth rewatching (best / weakest answer, longest silence, voice breaks, ...).

All rules, no learned weights; the thresholds are listed next to each metric so the report explains itself.
"""
from __future__ import annotations

import math

import numpy as np

from ml.behaviour.text_hesitation import word_breakdown

WIN_S, HOP_S = 5.0, 2.5          # vision windows overlap by half (backend/workers/vision_worker.py)


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _mean(xs) -> float | None:
    xs = [float(x) for x in xs if _ok(x)]
    return float(np.mean(xs)) if xs else None


def _r(v, d=1):
    return round(float(v), d) if _ok(v) else None


def _mmss(t) -> str | None:
    return f"{int(t // 60):02d}:{int(t % 60):02d}" if _ok(t) else None


def _lin(v, best: float, worst: float) -> float | None:
    """0-100: 100 at `best`, 0 at `worst`, linear in between (works for either direction)."""
    if not _ok(v):
        return None
    return float(np.clip((v - worst) / (best - worst), 0, 1) * 100)


def _band(v, lo: float, hi: float, zero_lo: float, zero_hi: float) -> float | None:
    """0-100: 100 inside [lo, hi], falling linearly to 0 at zero_lo / zero_hi."""
    if not _ok(v):
        return None
    if v < lo:
        return _lin(v, lo, zero_lo)
    if v > hi:
        return _lin(v, hi, zero_hi)
    return 100.0


def _status(v, good, watch, higher_is_better: bool = False) -> str:
    """good / watch / concern from two cut points (or two ranges for 'inside is good')."""
    if not _ok(v):
        return "neutral"
    if isinstance(good, tuple):
        if good[0] <= v <= good[1]:
            return "good"
        return "watch" if watch[0] <= v <= watch[1] else "concern"
    if higher_is_better:
        return "good" if v >= good else "watch" if v >= watch else "concern"
    return "good" if v <= good else "watch" if v <= watch else "concern"


def _row(key, label, value, unit="", typical="", status="neutral", note="", at=None, extra=None) -> dict:
    out = {"key": key, "label": label, "value": value, "unit": unit, "typical": typical, "status": status, "note": note}
    if at is not None:
        out["at_s"], out["at"] = at, _mmss(at)
    if extra:
        out["extra"] = extra
    return out


# --------------------------------------------------------------------------------------------- per-answer extras
def card_details(card: dict, answer: dict, windows: list[dict]) -> dict:
    """Extra per-answer fields: pauses, voice, word breakdown and the eye metrics inside the answer."""
    beh = answer.get("behaviour") or {}
    tm, pr, th, ah = (beh.get(k) or {} for k in ("timing", "prosody", "text_hesitation", "audio_hesitation"))
    ask, end = card.get("ask_s") or 0, card.get("end_s") or (card.get("ask_s") or 0)
    inside = [w for w in windows if w["end_s"] > ask and w["start_s"] < end]
    feat = lambda k: [w.get("features", {}).get(k) for w in inside]  # noqa: E731
    on = _mean([w.get("on_screen_ratio") for w in inside])
    return {
        "long_pauses": tm.get("long_pause_count"), "longest_pause_s": _r(tm.get("longest_pause_s"), 2),
        "longest_pause_at_s": tm.get("longest_pause_at_s"), "very_long_pauses": tm.get("very_long_pause_count"),
        "pause_ratio": _r(tm.get("pause_ratio"), 3), "answer_duration_s": _r(tm.get("answer_duration_s"), 1),
        "voice_breaks": pr.get("voice_break_count"), "voice_break_times_s": pr.get("voice_break_times_s") or [],
        "voice_break_rate": _r(pr.get("voice_break_rate_per_min")), "pitch_jitter_pct": _r(pr.get("pitch_jitter_pct"), 2),
        "monotony": _r(pr.get("monotony_index"), 3), "volume_consistency": _r(pr.get("volume_consistency"), 3),
        "pitch_mean_hz": _r(pr.get("pitch_mean_hz")), "word_count": th.get("word_count"),
        "filler_rate": _r(th.get("filler_rate_per_100w")), "hedges": th.get("hedge_count"),
        "repetitions": th.get("repetition_count"), "restarts": th.get("restart_count"),
        "prolongations": ah.get("prolongation_count"), "blocks": ah.get("block_count"),
        "words": word_breakdown(card.get("transcript") or ""),
        "eyes": {"blink_rate": _r(_mean(feat("blink_rate_per_min"))), "on_screen_pct": _r(on * 100) if on is not None else None,
                 "reading": any(feat("reading_pattern_flag"))},
    }


# --------------------------------------------------------------------------------------------- sections
def _eyes(windows: list[dict]) -> tuple[list[dict], dict]:
    f = lambda k: [w.get("features", {}).get(k) for w in windows]  # noqa: E731
    on = _mean([w.get("on_screen_ratio") for w in windows])
    on_pct = on * 100 if on is not None else None
    blink = _mean(f("blink_rate_per_min"))
    blink_series = [x for x in f("blink_rate_per_min") if _ok(x)]
    glances = _mean(f("glance_away_count"))
    glances_pm = glances / WIN_S * 60 if glances is not None else None
    gaze_std = _mean(f("gaze_yaw_std_deg"))
    sacc = _mean(f("saccade_rate_hz"))
    iris = _mean(f("iris_movement_std"))
    head = _mean(f("head_yaw_std_deg"))
    face = _mean(f("face_present_ratio"))
    reading_w = [w for w in windows if (w.get("features") or {}).get("reading_pattern_flag")]
    reading_s = len(reading_w) * HOP_S
    rows = [
        _row("blink_rate", "Blink rate", _r(blink), "/min", "8-30 in conversation",
             _status(blink, (8, 30), (5, 40)),
             "Much higher than usual often goes with stress or tiredness; very low with staring or reading.",
             extra={"min": _r(min(blink_series)) if blink_series else None,
                    "max": _r(max(blink_series)) if blink_series else None}),
        _row("eye_contact", "Eye contact", _r(on_pct), "%", "70 % or more of the time on screen",
             _status(on_pct, 70, 50, higher_is_better=True), "Share of video frames with the gaze inside the calibrated screen."),
        _row("glances", "Glances away", _r(glances_pm), "/min", "4 or fewer per minute", _status(glances_pm, 4, 8),
             "Looks away from the screen lasting 0.3 s or more."),
        _row("gaze_stability", "Gaze steadiness", _r(gaze_std), "° spread", "under 6°", _status(gaze_std, 6, 10),
             "Spread of the horizontal gaze angle within 5 s windows; a restless gaze spreads more."),
        _row("saccades", "Rapid eye movements", _r(sacc, 2), "/s", "about 1-3 per second", _status(sacc, (0.5, 3.5), (0.2, 5)),
             "Saccades: quick jumps of the eyes faster than 60°/s."),
        _row("iris_movement", "Iris movement", _r(iris, 3), "", "lower is steadier", "neutral",
             "Spread of the iris position inside the eye crop (from the gaze + iris model)."),
        _row("reading", "Reading pattern", _r(reading_s, 1), "s", "0 s", _status(reading_s, 0, 10),
             "Time with left-to-right sweeps and fast returns, as when reading from another screen or notes.",
             at=reading_w[0]["start_s"] if reading_w else None),
        _row("face_visible", "Face visible", _r(face * 100) if face is not None else None, "%", "90 % or more",
             _status(face * 100 if face is not None else None, 90, 75, higher_is_better=True), ""),
        _row("head_movement", "Head movement", _r(head), "° spread", "under 10°", _status(head, 10, 18),
             "Spread of the head turn (yaw) within 5 s windows."),
    ]
    raw = {"blink": blink, "on_pct": on_pct, "glances_pm": glances_pm, "reading_s": reading_s,
           "first_reading_s": reading_w[0]["start_s"] if reading_w else None}
    return rows, raw


def _voice(cards: list[dict]) -> tuple[list[dict], dict]:
    w = [(c.get("answer_duration_s") or 0) for c in cards]
    wmean = lambda k: (float(np.average([c[k] for c in cards if _ok(c.get(k))],  # noqa: E731
                                        weights=[max(x, 1) for c, x in zip(cards, w) if _ok(c.get(k))]))
                       if any(_ok(c.get(k)) for c in cards) else None)
    wpm = wmean("wpm")
    mono = wmean("monotony")
    expr = (1 - mono) * 100 if mono is not None else None
    breaks = sum(c.get("voice_breaks") or 0 for c in cards)
    br_rate = wmean("voice_break_rate")
    jitter = wmean("pitch_jitter_pct")
    vol = wmean("volume_consistency")
    pitch = wmean("pitch_mean_hz")
    times = sorted(t for c in cards for t in c.get("voice_break_times_s") or [])
    rows = [
        _row("pace", "Speaking pace", _r(wpm, 0), "wpm", "110-170 words per minute", _status(wpm, (110, 170), (90, 190)), ""),
        _row("expressiveness", "Voice expressiveness", _r(expr, 0), "/100", "50 or more",
             _status(expr, 50, 25, higher_is_better=True), "Pitch variation; low values sound flat or monotone."),
        _row("voice_breaks", "Voice breaks (cracks)", breaks, "", "6 or fewer per voiced minute",
             _status(br_rate, 6, 15) if breaks else ("good" if cards else "neutral"),
             f"{_r(br_rate)} per voiced minute. Sudden jumps of 7+ semitones between two steady stretches of voice; "
             "usually tension or a dry throat." if br_rate is not None else "",
             at=times[0] if times else None, extra={"times": [_mmss(t) for t in times[:12]]}),
        _row("jitter", "Voice steadiness (jitter)", _r(jitter, 2), "%", "1.7 % or less (read speech median 1.1 %)",
             _status(jitter, 1.7, 2.5), "Frame-to-frame pitch shakiness after removing normal intonation."),
        _row("volume", "Volume consistency", _r(vol * 100 if vol is not None else None, 0), "/100", "60 or more",
             _status(vol * 100 if vol is not None else None, 60, 40, higher_is_better=True), ""),
        _row("pitch", "Average pitch", _r(pitch, 0), "Hz", "", "neutral", "For reference only; not scored."),
    ]
    return rows, {"wpm": wpm, "expr": expr, "break_rate": br_rate, "breaks": breaks, "jitter": jitter, "vol": vol,
                  "break_times": times}


def _fluency(cards: list[dict], cand_text: str) -> tuple[list[dict], dict]:
    nw = sum(c.get("word_count") or 0 for c in cards) or len(cand_text.split())
    per100 = lambda n: 100 * n / nw if nw else None  # noqa: E731
    bd = word_breakdown(cand_text)
    n_fill = sum(bd["fillers"].values())
    n_hedge = sum(c.get("hedges") or 0 for c in cards)
    n_rep = sum((c.get("repetitions") or 0) + (c.get("restarts") or 0) for c in cards)
    n_audio = sum((c.get("prolongations") or 0) + (c.get("blocks") or 0) for c in cards)
    minutes = sum(c.get("answer_duration_s") or 0 for c in cards) / 60
    hes = _mean([c.get("hesitation_index") for c in cards])
    top = ", ".join(f"“{w}” ×{n}" for w, n in list(bd["fillers"].items())[:5])
    rows = [
        _row("fillers", "Filler words", n_fill, "", "3 or fewer per 100 words", _status(per100(n_fill), 3, 6),
             f"{_r(per100(n_fill))} per 100 words" + (f": {top}" if top else ""), extra={"breakdown": bd["fillers"]}),
        _row("crutch", "Overused words", len(bd["crutch_words"]), "", "none",
             "good" if not bd["crutch_words"] else "watch",
             ", ".join(f"“{w}” ×{n}" for w, n in bd["crutch_words"].items()) or "No word repeated unusually often.",
             extra={"words": bd["crutch_words"]}),
        _row("hedges", "Hedging phrases", n_hedge, "", "2 or fewer per 100 words", _status(per100(n_hedge), 2, 4),
             ", ".join(f"“{w}” ×{n}" for w, n in bd["hedges"].items()) or "“I think”, “maybe”, “not sure”, ...",
             extra={"breakdown": bd["hedges"]}),
        _row("repetitions", "Repeats and restarts", n_rep, "", "2 or fewer per 100 words", _status(per100(n_rep), 2, 4),
             "Repeated words (“I I”) and abandoned sentence starts."),
        _row("audio_disfluency", "Stretched sounds and blocks", n_audio, "", "2 or fewer per minute",
             _status(n_audio / minutes if minutes else None, 2, 5),
             "Prolongations and blocks heard by the disfluency model (not visible in the transcript)."),
        _row("hesitation", "Hesitation index", _r(hes, 0), "/100", "under 40", _status(hes, 40, 65),
             "Learned combination of pauses, fillers and disfluencies (behaviour model)."),
    ]
    return rows, {"filler_rate": per100(n_fill), "hedge_rate": per100(n_hedge), "rep_rate": per100(n_rep),
                  "audio_pm": n_audio / minutes if minutes else None, "hes": hes, "breakdown": bd}


def _pauses(cards: list[dict]) -> tuple[list[dict], dict]:
    lat = [(c["response_latency_s"], c) for c in cards if _ok(c.get("response_latency_s"))]
    mean_lat = _mean([v for v, _ in lat])
    worst_lat = max(lat, key=lambda t: t[0]) if lat else None
    minutes = sum(c.get("answer_duration_s") or 0 for c in cards) / 60
    longs = sum(c.get("long_pauses") or 0 for c in cards)
    vlongs = sum(c.get("very_long_pauses") or 0 for c in cards)
    lp = [c for c in cards if _ok(c.get("longest_pause_s"))]
    longest = max(lp, key=lambda c: c["longest_pause_s"]) if lp else None
    ratio = _mean([c.get("pause_ratio") for c in cards])
    rows = [
        _row("response_delay", "Response delay", _r(mean_lat), "s", "2 s or less", _status(mean_lat, 2, 4),
             f"Longest {worst_lat[0]:.1f} s, on “{(worst_lat[1].get('question') or '')[:60]}”." if worst_lat else "",
             at=worst_lat[1].get("ask_s") if worst_lat else None),
        _row("long_pauses", "Long pauses (> 1 s)", longs, "", "3 or fewer per minute of answering",
             _status(longs / minutes if minutes else None, 3, 6),
             f"{longs / minutes:.1f} per minute of answering." if minutes else ""),
        _row("very_long_pauses", "Very long pauses (> 3 s)", vlongs, "", "none", _status(vlongs, 0, 2), ""),
        _row("longest_pause", "Longest silence mid-answer", _r(longest["longest_pause_s"]) if longest else None, "s",
             "2.5 s or less", _status(longest["longest_pause_s"] if longest else None, 2.5, 4),
             f"In the answer to “{(longest.get('question') or '')[:60]}”." if longest else "",
             at=longest.get("longest_pause_at_s") if longest else None),
        _row("pause_ratio", "Silence while answering", _r(ratio * 100 if ratio is not None else None, 0), "%", "25 % or less",
             _status(ratio * 100 if ratio is not None else None, 25, 40), ""),
    ]
    return rows, {"mean_lat": mean_lat, "longs_pm": longs / minutes if minutes else None, "vlongs": vlongs,
                  "longest": longest, "worst_lat": worst_lat, "ratio": ratio}


def behaviour_sections(cards: list[dict], windows: list[dict], cand_text: str) -> tuple[list[dict], dict]:
    eyes, re = _eyes(windows)
    voice, rv = _voice(cards)
    flu, rf = _fluency(cards, cand_text)
    pau, rp = _pauses(cards)
    sections = [
        {"key": "eyes", "title": "Eyes and attention", "rows": eyes,
         "series": [{"t_s": w["end_s"], "blink": _r((w.get("features") or {}).get("blink_rate_per_min")),
                     "on_screen": _r((w.get("on_screen_ratio") or 0) * 100)} for w in windows]},
        {"key": "voice", "title": "Voice", "rows": voice},
        {"key": "fluency", "title": "Fluency and word choice", "rows": flu},
        {"key": "pauses", "title": "Pauses and timing", "rows": pau},
    ]
    return sections, {"eyes": re, "voice": rv, "fluency": rf, "pauses": rp}


# --------------------------------------------------------------------------------------------- scorecard
WEIGHTS = {"content": 0.35, "fluency": 0.15, "composure": 0.15, "delivery": 0.15, "eye_contact": 0.10,
           "responsiveness": 0.10}
GRADES = [(85, "A", "Excellent"), (70, "B", "Strong"), (55, "C", "Adequate"), (40, "D", "Needs work"), (0, "E", "Weak")]


def scorecard(cards: list[dict], raw: dict) -> dict:
    re, rv, rf, rp = raw["eyes"], raw["voice"], raw["fluency"], raw["pauses"]
    eff = _mean([c.get("effectiveness") for c in cards])
    rel = _mean([(c.get("answer_model") or {}).get("relevance") for c in cards])
    qual = _mean([(c.get("answer_model") or {}).get("quality") for c in cards])
    content = eff if eff is not None else _mean([x * 100 for x in (rel, qual) if x is not None])
    blink_score = _band(re["blink"], 8, 30, 0, 50)
    dims = {
        "content": (content, "Fusion effectiveness of every answer (relevance, completeness, structure)."
                    if eff is not None else "Answer-quality model: relevance and quality."),
        "fluency": (_mean([_lin(rf["filler_rate"], 2, 10), _lin(rp["longs_pm"], 2, 8), _lin(rf["rep_rate"], 1, 8),
                           _lin(rf["audio_pm"], 1, 6), 100 - rf["hes"] if rf["hes"] is not None else None]),
                    "Fillers, long pauses, repeats and restarts, stretched sounds, hesitation index."),
        "composure": (_mean([blink_score, _lin(rv["break_rate"], 3, 25), _lin(rv["jitter"], 1.5, 3.5),
                             _lin(rf["hedge_rate"], 1, 8)]),
                      "Blink rate, voice breaks, voice steadiness (jitter), hedging."),
        "delivery": (_mean([_band(rv["wpm"], 120, 160, 70, 230), rv["expr"], rv["vol"] * 100 if rv["vol"] is not None else None]),
                     "Pace, pitch expressiveness, volume consistency."),
        "eye_contact": (_mean([_lin(re["on_pct"], 85, 40), _lin(re["glances_pm"], 3, 12),
                               _lin(re["reading_s"], 0, 30) if windows_present(re) else None]),
                        "Time on screen, glances away, reading pattern."),
        "responsiveness": (_mean([_lin(rp["mean_lat"], 1.5, 6), _lin(rp["vlongs"], 0, 4)]),
                           "Delay before answering, very long silences."),
    }
    out = {k: {"score": _r(v, 0), "weight": WEIGHTS[k], "basis": basis} for k, (v, basis) in dims.items()}
    have = {k: d for k, d in out.items() if d["score"] is not None}
    wsum = sum(d["weight"] for d in have.values())
    overall = sum(d["score"] * d["weight"] for d in have.values()) / wsum if wsum else None
    grade = next(((g, word) for cut, g, word in GRADES if overall is not None and overall >= cut), (None, None))
    best = max(have, key=lambda k: have[k]["score"]) if have else None
    worst = min(have, key=lambda k: have[k]["score"]) if have else None
    return {"dimensions": out, "overall": _r(overall, 0), "grade": grade[0], "grade_word": grade[1],
            "strongest": best, "weakest": worst if worst != best else None,
            "method": "Weighted mean of the available dimensions (weights renormalised when one is missing). "
                      "A 85+, B 70+, C 55+, D 40+, E below 40. Decision support only."}


def windows_present(re: dict) -> bool:
    return re.get("on_pct") is not None


# --------------------------------------------------------------------------------------------- extras
QUADRANTS = {
    (True, True): ("confident_on_point", "Confident and on point"),
    (True, False): ("unsure_but_right", "Knows it, sounds unsure"),
    (False, True): ("sure_but_off", "Sounds sure, but off target"),
    (False, False): ("struggled", "Struggled"),
}


def confidence_matrix(cards: list[dict]) -> dict:
    pts = []
    for c in cards:
        am, sub = c.get("answer_model") or {}, c.get("subscores") or {}
        content = sub.get("relevance") if _ok(sub.get("relevance")) else (am["relevance"] * 100 if _ok(am.get("relevance")) else None)
        conf = sub.get("confidence") if _ok(sub.get("confidence")) else (
            100 - c["hesitation_index"] if _ok(c.get("hesitation_index")) else None)
        if content is None or conf is None:
            continue
        key, label = QUADRANTS[(content >= 55, conf >= 55)]
        c["quadrant"] = label
        pts.append({"question_ref": c["question_ref"], "order": c.get("order"), "question": (c.get("question") or "")[:80],
                    "content": _r(content, 0), "confidence": _r(conf, 0), "quadrant": key, "label": label})
    counts = {k: sum(p["quadrant"] == k for p in pts) for k, _ in QUADRANTS.values()}
    note = None
    if counts["sure_but_off"]:
        note = (f"{counts['sure_but_off']} answer(s) sounded confident but missed the question: worth probing "
                "with a follow-up for depth.")
    elif counts["unsure_but_right"]:
        note = (f"{counts['unsure_but_right']} answer(s) were correct but delivered hesitantly: the candidate may know "
                "more than they show under pressure.")
    return {"points": pts, "counts": counts, "note": note,
            "axes": {"x": "content (relevance)", "y": "vocal confidence"}, "cut": 55}


def pressure_response(cards: list[dict], difficulty: dict[int, float]) -> dict | None:
    rated = [c for c in cards if c["question_ref"] in difficulty]
    if len(rated) < 4 or len({difficulty[c["question_ref"]] for c in rated}) < 2:
        return None
    med = float(np.median([difficulty[c["question_ref"]] for c in rated]))
    hard = [c for c in rated if difficulty[c["question_ref"]] > med]
    easy = [c for c in rated if difficulty[c["question_ref"]] <= med]
    if not hard or not easy:
        return None
    metrics = [("response_latency_s", "Response delay", "s", False), ("hesitation_index", "Hesitation index", "", False),
               ("filler_rate", "Fillers per 100 words", "", False), ("voice_break_rate", "Voice breaks per minute", "", False),
               ("pitch_jitter_pct", "Voice jitter", "%", False), ("effectiveness", "Answer score", "", True)]
    rows, worse = [], []
    for key, label, unit, up in metrics:
        e, h = _mean([c.get(key) for c in easy]), _mean([c.get(key) for c in hard])
        if e is None or h is None:
            continue
        d = h - e
        rows.append({"key": key, "label": label, "unit": unit, "easier": _r(e, 2 if unit == "%" else 1), "harder": _r(h, 2 if unit == "%" else 1),
                     "change": _r(d, 2 if unit == "%" else 1),
                     "worse": bool(d < 0 if up else d > 0)})
    for key, label, unit, up in [("blink_rate", "Blink rate", "/min", False), ("on_screen_pct", "Eye contact", "%", True)]:
        e = _mean([(c.get("eyes") or {}).get(key) for c in easy])
        h = _mean([(c.get("eyes") or {}).get(key) for c in hard])
        if e is not None and h is not None:
            d = h - e
            rows.append({"key": key, "label": label, "unit": unit, "easier": _r(e, 1), "harder": _r(h, 1), "change": _r(d, 1),
                         "worse": bool(d < 0 if up else d > 0)})
    # a change counts when it is large relative to the easier level (or 0.5 s / 5 points in absolute terms)
    big = lambda r: abs(r["change"]) >= max(0.25 * abs(r["easier"] or 0), 0.5 if r["unit"] == "s" else 5 if r["key"] in (  # noqa: E731
        "hesitation_index", "effectiveness", "on_screen_pct", "blink_rate") else 1)
    worse = [r for r in rows if r["worse"] and big(r)]
    summary = ("Under harder questions: " + ", ".join(f"{r['label'].lower()} {r['change']:+g}{r['unit']}" for r in worse[:3]) + "."
               if worse else "Delivery stayed steady when the questions got harder.")
    return {"rows": rows, "summary": summary, "easier_n": len(easy), "harder_n": len(hard), "split_difficulty": med}


def key_moments(cards: list[dict], windows: list[dict], raw: dict) -> list[dict]:
    out = []
    scored = [c for c in cards if _ok(c.get("effectiveness"))]
    if len(scored) >= 2:
        b = max(scored, key=lambda c: c["effectiveness"])
        w = min(scored, key=lambda c: c["effectiveness"])
        out.append({"t_s": b["ask_s"], "kind": "best", "label": "Strongest answer",
                    "detail": f"{b['effectiveness']:.0f}/100: “{(b.get('question') or '')[:70]}”"})
        out.append({"t_s": w["ask_s"], "kind": "weakest", "label": "Weakest answer",
                    "detail": f"{w['effectiveness']:.0f}/100: “{(w.get('question') or '')[:70]}”"})
    rp, rv, re = raw["pauses"], raw["voice"], raw["eyes"]
    if rp["worst_lat"] and rp["worst_lat"][0] > 3:
        v, c = rp["worst_lat"]
        out.append({"t_s": c["ask_s"], "kind": "delay", "label": "Slowest start", "detail": f"{v:.1f} s before answering"})
    if rp["longest"] and (rp["longest"].get("longest_pause_s") or 0) > 2.5 and _ok(rp["longest"].get("longest_pause_at_s")):
        c = rp["longest"]
        out.append({"t_s": c["longest_pause_at_s"], "kind": "silence", "label": "Longest silence",
                    "detail": f"{c['longest_pause_s']:.1f} s mid-answer"})
    if rv["break_times"]:
        ts = rv["break_times"]
        # densest 20 s
        best = max(ts, key=lambda t: sum(t <= x < t + 20 for x in ts))
        n = sum(best <= x < best + 20 for x in ts)
        out.append({"t_s": best, "kind": "voice", "label": "Voice breaks", "detail": f"{n} break(s) within 20 s"})
    att = [w for w in windows if _ok(w.get("attention"))]
    if att:
        lo = min(att, key=lambda w: w["attention"])
        if lo["attention"] < 50:
            out.append({"t_s": lo["start_s"], "kind": "attention", "label": "Attention dip",
                        "detail": f"attention {lo['attention']:.0f}/100"})
    if re["first_reading_s"] is not None:
        out.append({"t_s": re["first_reading_s"], "kind": "reading", "label": "Reading pattern",
                    "detail": "eye movements looked like reading"})
    for o in out:
        o["time"] = _mmss(o["t_s"])
    return sorted(out, key=lambda o: o["t_s"] or 0)
