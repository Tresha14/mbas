"""Text-side hesitation from a transcript (ours or our ASR's): fillers, repetitions, restarts, hedges,
self-corrections. Rule-based (no learned weights).
"""
from __future__ import annotations

import re

FILLERS = {"um", "uh", "umm", "uhh", "er", "erm", "ah", "hmm", "hm", "mm", "matlab", "basically", "actually",
           "like", "achha", "acha", "toh", "na"}
MULTI_FILLERS = ["you know", "i mean", "sort of", "kind of", "what to say"]
HEDGES = ["i think", "maybe", "not sure", "i guess", "probably", "perhaps", "i believe", "might be", "kind of",
          "sort of", "i am not sure", "i'm not sure", "not completely sure", "it depends",
          "i suppose", "i feel like", "i would say", "something like that", "or something", "if i am not wrong",
          "if i'm not wrong", "if i'm not mistaken", "to some extent", "more or less"]
CORRECTIONS = ["i mean", "sorry", "no wait", "rather", "or rather", "let me rephrase", "actually no"]
_WORD = re.compile(r"[a-z']+")


def words(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def count_phrases(text: str, phrases: list[str]) -> int:
    t = " " + " ".join(words(text)) + " "
    return sum(t.count(f" {p} ") for p in phrases)


def filler_tokens(ws: list[str]) -> list[str]:
    """Single-word fillers in order; 'like'/'actually'/'basically' count only when not used as content words."""
    out = []
    for i, w in enumerate(ws):
        if w in ("um", "uh", "umm", "uhh", "er", "erm", "ah", "hmm", "hm", "mm", "matlab", "achha", "acha"):
            out.append(w)
        elif w in ("basically", "actually") and (i == 0 or ws[i - 1] in FILLERS or i + 1 < len(ws) and ws[i + 1] in ("i", "we", "it", "the")):
            out.append(w)
        elif w == "like" and i > 0 and ws[i - 1] not in ("would", "i", "you", "we", "they", "looks", "feel", "something"):
            out.append(w)
    return out


def filler_count(ws: list[str], text: str) -> int:
    return len(filler_tokens(ws)) + count_phrases(text, ["you know"])


# spelling variants counted as one filler in the breakdown
_FILLER_FORM = {"umm": "um", "uhh": "uh", "erm": "er", "hm": "hmm", "mm": "hmm", "acha": "achha"}
# function words never reported as crutch words
STOP_WORDS = set("""a an the and or but if so of to in on at by for with from as is are was were be been being am i me my
we our you your he she it its they them their this that these those there here what which who whom when where why how
not no yes do does did done have has had will would can could should shall may might must just also very really then
than too about into over under up down out off again more most some any all each few other such only own same both
because while during before after above below between through until against once lot get got make made like know""".split())


def word_breakdown(text: str, top: int = 5) -> dict:
    """Which fillers and hedges the speaker uses (word -> count), and crutch words: content words repeated far more
    than usual (>= 4 times and >= 2.5 % of all words)."""
    from collections import Counter

    ws = words(text)
    fillers = Counter(_FILLER_FORM.get(w, w) for w in filler_tokens(ws))
    if (k := count_phrases(text, ["you know"])):
        fillers["you know"] = k
    hedges = Counter({h: n for h in HEDGES if (n := count_phrases(text, [h]))})
    n = max(len(ws), 1)
    hedge_words = {w for h in HEDGES for w in h.split()}
    crutch = Counter(w for w in ws if w not in STOP_WORDS and w not in FILLERS and w not in hedge_words and len(w) > 2)
    return {"fillers": dict(fillers.most_common()), "hedges": dict(hedges.most_common(top)),
            "crutch_words": {w: c for w, c in crutch.most_common(top) if c >= 4 and c / n >= 0.025}}


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
