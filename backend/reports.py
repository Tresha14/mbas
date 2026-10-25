"""Post-interview reports from stored model outputs (templates only, no LLM).

interviewer_view: per-question cards (transcript, scores, top-3 reasons), sub-score radar, attention / engagement
                  timeline, follow-up list, conversation summary, models used.
candidate_view:   strengths, 3 improvement areas with timestamps and concrete tips, closing note. The interviewer
                  reviews / edits it and releases it; the candidate never sees raw attention or engagement numbers.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

SUB = ("clarity", "confidence", "engagement", "relevance", "fluency")
TIPS = {
    "response_latency": "Take a short breath, then start with a one-line summary of your answer; it keeps the pause "
                        "under two seconds while you organise the details.",
    "fillers": "Replace fillers (um, basically, matlab) with a short silent pause. Recording yourself answering two "
               "practice questions a day makes them noticeable and easier to drop.",
    "long_pauses": "Outline answers as 'point - example - result' before you speak, so you do not stop mid-sentence "
                   "to search for the next idea.",
    "structure": "For behavioural questions use STAR: one sentence each for Situation and Task, most of the time on "
                 "your Actions, and finish with a measurable Result.",
    "relevance": "Repeat the key words of the question in your first sentence and check at the end that you answered "
                 "every part of it.",
    "eye_contact": "Look at the camera lens (not your own video) when you make your main point; place notes behind the "
                   "camera rather than to the side.",
    "pace": "Aim for about 120-150 words per minute: slow down on key facts and numbers.",
    "monotone": "Stress one key word per sentence and vary your pitch when you move to a new point.",
    "detail": "Give one concrete example with numbers (users, time saved, accuracy) for every claim.",
}


def _f(v, d=1):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), d)


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def build_reports(data: dict) -> tuple[dict, dict]:
    qs = {q["id"]: q for q in data["questions"]}
    answers = sorted(data["answers"], key=lambda a: qs.get(a["question_ref"], {}).get("ask_s", 0))
    fus = defaultdict(list)
    for f in data["follow_ups"]:
        fus[f["question_ref"]].append(f)

    cards = []
    for a in answers:
        q = qs.get(a["question_ref"], {})
        beh = a.get("behaviour") or {}
        cards.append({
            "question_ref": a["question_ref"], "order": q.get("order"), "question": q.get("text"), "category": q.get("category"),
            "ask_s": q.get("ask_s"), "end_s": q.get("end_s"), "transcript": a.get("transcript", ""),
            "effectiveness": _f(a.get("effectiveness")), "subscores": a.get("subscores") or {},
            "answer_model": a.get("answer_model") or {}, "reasons": a.get("reasons") or [],
            "modality_weights": a.get("modality_weights") or {},
            "speaking_pattern": (beh.get("speaking_pattern") or {}).get("label"),
            "hesitation_index": (beh.get("hesitation_index") or {}).get("value"),
            "response_latency_s": (beh.get("timing") or {}).get("response_latency_s"),
            "wpm": (beh.get("prosody") or {}).get("speaking_rate_wpm"),
            "fillers": (beh.get("text_hesitation") or {}).get("filler_count"),
            "follow_ups": [{"reason": f["reason"], "text": f["text"], "source": f["source"]} for f in fus[a["question_ref"]]],
        })
    radar = {k: _f(np.nanmean([c["subscores"].get(k, np.nan) for c in cards])) if cards else None for k in SUB}
    eff = [c["effectiveness"] for c in cards if c["effectiveness"] is not None]
    windows = sorted(data["windows"], key=lambda w: w["start_s"])
    timeline = [{"t_s": w["end_s"], "attention": _f(w.get("attention")), "engagement": w.get("engagement")} for w in windows]
    eng = [w.get("engagement") for w in windows if w.get("engagement")]
    interviewer = {
        "summary": {"questions": len(cards), "mean_effectiveness": _f(np.mean(eff)) if eff else None,
                    "mean_attention": _f(np.nanmean([w["attention"] for w in windows if w.get("attention") is not None]))
                    if windows else None,
                    "engaged_share": round(eng.count("engaged") / len(eng), 3) if eng else None,
                    "follow_ups": len(data["follow_ups"])},
        "radar": radar, "cards": cards, "timeline": timeline,
        "follow_ups": [{**f, "time": _mmss(f["t_s"])} for f in sorted(data["follow_ups"], key=lambda f: f["t_s"])],
        "note": "Scores are decision support for the interviewer, never an automatic decision.",
    }
    return interviewer, candidate_feedback(cards, windows)


def candidate_feedback(cards: list[dict], windows: list[dict]) -> dict:
    """Strengths + 3 improvement areas with timestamps and tips, from simple, explainable rules over our outputs."""
    issues, strengths = [], []
    by_cat = defaultdict(list)
    for c in cards:
        by_cat[c.get("category") or "general"].append(c)

    lat = [(c["response_latency_s"], c) for c in cards if c.get("response_latency_s") is not None]
    if lat:
        for cat, cs in by_cat.items():
            v = [c["response_latency_s"] for c in cs if c.get("response_latency_s") is not None]
            if v and np.mean(v) > 3.0:
                worst = max(cs, key=lambda c: c.get("response_latency_s") or 0)
                issues.append((np.mean(v) - 3.0, "response_latency",
                               f"Average response delay {np.mean(v):.1f} s on {cat} questions "
                               f"(longest at {_mmss(worst['ask_s'] or 0)}).", worst))
        if np.mean([v for v, _ in lat]) <= 2.0:
            strengths.append("You started your answers promptly after each question.")
    fil = [(c["fillers"] / max(len((c["transcript"] or "").split()), 1) * 100, c) for c in cards if c.get("fillers")]
    if fil:
        rate, c = max(fil, key=lambda t: t[0])
        avg = np.mean([r for r, _ in fil])
        if avg > 4:
            issues.append((avg / 4, "fillers", f"About {avg:.0f} filler words per 100 words (most in the answer at "
                                               f"{_mmss(c['ask_s'] or 0)}).", c))
    elif cards:
        strengths.append("Your answers had very few filler words.")
    for c in cards:
        if c.get("speaking_pattern") == "monotone":
            issues.append((0.8, "monotone", f"Your voice sounded flat in the answer at {_mmss(c['ask_s'] or 0)}.", c))
            break
    for c in cards:
        am = c.get("answer_model") or {}
        if c.get("category") == "behavioural" and am.get("star_count") is not None and am["star_count"] < 3:
            missing = [k for k, v in (am.get("star") or {}).items() if v < 0.5]
            issues.append((1.2, "structure", f"The answer at {_mmss(c['ask_s'] or 0)} did not clearly cover "
                                             f"{', '.join(missing)} (STAR).", c))
            break
    low_rel = [c for c in cards if (c.get("answer_model") or {}).get("relevance", 1) < 0.5]
    if low_rel:
        c = low_rel[0]
        issues.append((1.5, "relevance", f"Part of the question at {_mmss(c['ask_s'] or 0)} ('{(c['question'] or '')[:60]}') "
                                         "was not addressed.", c))
    on = [w["features"].get("on_screen_ratio") for w in windows if w.get("features") and w["features"].get("on_screen_ratio") is not None]
    if on and np.mean(on) < 0.6:
        issues.append((1.0, "eye_contact", f"You looked away from the screen for about {100 - 100 * np.mean(on):.0f}% "
                                           "of the interview.", None))
    elif on:
        strengths.append("You kept good eye contact with the screen.")
    wpm = [c["wpm"] for c in cards if c.get("wpm")]
    if wpm and np.mean(wpm) > 175:
        issues.append((0.9, "pace", f"You spoke quickly (about {np.mean(wpm):.0f} words per minute).", None))
    good = [c for c in cards if (c.get("effectiveness") or 0) >= 70]
    if good:
        best = max(good, key=lambda c: c["effectiveness"])
        strengths.append(f"Your answer to '{(best['question'] or '')[:60]}' was clear and well organised.")
    rel = [c["answer_model"].get("relevance") for c in cards if (c.get("answer_model") or {}).get("relevance") is not None]
    if rel and np.mean(rel) >= 0.7:
        strengths.append("You stayed on topic and answered what was asked.")

    issues.sort(key=lambda t: -t[0])
    seen, areas = set(), []
    for _, kind, text, c in issues:
        if kind in seen:
            continue
        seen.add(kind)
        areas.append({"area": kind.replace("_", " "), "observation": text, "tip": TIPS[kind],
                      "timestamp": _mmss(c["ask_s"]) if c and c.get("ask_s") is not None else None})
        if len(areas) == 3:
            break
    if len(areas) < 3 and "detail" not in seen and cards:
        areas.append({"area": "detail", "observation": "Adding concrete numbers would make your examples stronger.",
                      "tip": TIPS["detail"], "timestamp": None})
    return {"strengths": strengths[:4] or ["You completed the full interview and answered every question."],
            "improvements": areas[:3],
            "closing": "This feedback was prepared from your mock interview and reviewed by your interviewer. "
                       "It describes how the conversation went, not your worth as a candidate."}
