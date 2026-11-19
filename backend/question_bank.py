"""Match a question heard in the interviewer's voice to a question of the bank.

The answer-quality model was trained on the bank's questions and answers, so its relevance score is only meaningful for those
questions. A spoken question that clearly is a bank question ("what's the difference between SQL and NoSQL databases?") is
scored against the bank's own wording; anything else is free-form and gets no relevance score (a made-up "topic not covered"
on a correct answer would be worse than no score).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_WORD = re.compile(r"[a-z0-9']+")
STOP = set("""a an the and or of to in on at by for with from as is are was were be been am i me my we our you your he she it its they
them their this that these those what which who how why when where do does did can could would should will tell about please
okay ok so uh um yes no""".split())
MIN_SHARED, MIN_COVERAGE = 3, 0.7


def _content(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in STOP and len(w) > 1}


@lru_cache(maxsize=4)
def load_bank(path: str) -> tuple[tuple[str, str, frozenset[str]], ...]:
    """((question_id, text, content words), ...) for every bank question."""
    qs = json.loads(Path(path).read_text(encoding="utf-8")).get("questions", [])
    return tuple((q["question_id"], q["text"], frozenset(_content(q["text"]))) for q in qs)


def match_bank(spoken: str, path: str) -> tuple[str, str] | None:
    """(question_id, canonical text) of the bank question that the spoken question clearly is, or None.
    Requires that 70 % of the bank question's content words were said (and at least 3 of them, or all of a shorter question's)."""
    said = _content(spoken)
    best, best_score = None, 0.0
    try:
        bank = load_bank(str(path))
    except Exception:  # noqa: BLE001 - no bank file: everything is free-form
        return None
    for qid, text, words in bank:
        shared = len(said & words)
        if not words or shared < min(MIN_SHARED, len(words)):         # a short question has few content words
            continue
        coverage = shared / len(words)
        score = coverage - 0.01 * max(0, len(said) - len(words))          # slightly prefer the tighter match
        if coverage >= MIN_COVERAGE and score > best_score:
            best, best_score = (qid, text), score
    return best
