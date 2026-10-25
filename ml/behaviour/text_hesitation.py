"""Text-side hesitation from a transcript (ours or our ASR's): fillers, repetitions, restarts, hedges,
self-corrections. Rule-based (no learned weights).
"""
from __future__ import annotations

import re

FILLERS = {"um", "uh", "umm", "uhh", "er", "erm", "ah", "hmm", "hm", "mm", "matlab", "basically", "actually",
           "like", "achha", "acha", "toh", "na"}
MULTI_FILLERS = ["you know", "i mean", "sort of", "kind of", "what to say"]
HEDGES = ["i think", "maybe", "not sure", "i guess", "probably", "perhaps", "i believe", "might be", "kind of",
          "sort of", "i am not sure", "i'm not sure", "not completely sure", "it depends"]
CORRECTIONS = ["i mean", "sorry", "no wait", "rather", "or rather", "let me rephrase", "actually no"]
_WORD = re.compile(r"[a-z']+")


def words(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def count_phrases(text: str, phrases: list[str]) -> int:
    t = " " + " ".join(words(text)) + " "
    return sum(t.count(f" {p} ") for p in phrases)


def filler_count(ws: list[str], text: str) -> int:
    """Single-word fillers; 'like'/'actually'/'basically'/'so' count only when not used as content words."""
    n = 0
    for i, w in enumerate(ws):
        if w in ("um", "uh", "umm", "uhh", "er", "erm", "ah", "hmm", "hm", "mm", "matlab", "achha", "acha"):
            n += 1
        elif w in ("basically", "actually") and (i == 0 or ws[i - 1] in FILLERS or i + 1 < len(ws) and ws[i + 1] in ("i", "we", "it", "the")):
            n += 1
        elif w == "like" and i > 0 and ws[i - 1] not in ("would", "i", "you", "we", "they", "looks", "feel", "something"):
            n += 1
    return n + count_phrases(text, ["you know"])


def repetition_count(ws: list[str]) -> int:
    """Immediate repeats of a word or a 2-word phrase ('i i', 'the the', 'it was it was')."""
    n, i = 0, 1
    while i < len(ws):
        if ws[i] == ws[i - 1] and ws[i] not in ("very", "really", "no", "bye"):
            n += 1
        elif i >= 3 and ws[i - 1:i + 1] == ws[i - 3:i - 1]:
            n += 1
        i += 1
    return n


def restart_count(text: str) -> int:
    """Abandoned sentence starts: dashes, '...' followed by a new start, 'I mean' corrections."""
    t = str(text)
    n = len(re.findall(r"\w\s*(—|--|–)\s*\w", t))
    n += len(re.findall(r"\.\.\.\s*(so|i|we|what|the|it)\b", t.lower()))
    n += count_phrases(t, ["i mean", "let me rephrase", "what i mean is"])
    return n


def text_hesitation(text: str) -> dict:
    ws = words(text)
    n = max(len(ws), 1)
    f = filler_count(ws, text)
    r = repetition_count(ws)
    s = restart_count(text)
    h = count_phrases(text, HEDGES)
    c = count_phrases(text, CORRECTIONS)
    return {"word_count": len(ws), "filler_count": f, "repetition_count": r, "restart_count": s, "hedge_count": h,
            "self_correction_count": c, "filler_rate_per_100w": round(100 * f / n, 2),
            "disfluency_rate_per_100w": round(100 * (f + r + s) / n, 2), "hedge_rate_per_100w": round(100 * h / n, 2)}
