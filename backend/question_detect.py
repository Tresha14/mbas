"""Which part of what the interviewer just said is an actual interview question?

Ported from the older project (minor_project/analyzer/questions.py). Judged sentence by sentence, so
"Hello? Hello? How are you? Tell me about your last job." keeps only the last sentence. A sentence counts when it

  * has at least MIN_CONTENT_WORDS meaningful words (fillers, greetings and acknowledgements do not count),
  * is not call chatter or small talk ("can you hear me", "you are on mute", "nice to meet you"), and
  * either contains a '?' (the recogniser adds one to most spoken questions) or uses clear question / prompt phrasing
    ("tell me about", "describe a time", "can you explain").

A bare "what" or "how" is not enough, so a candidate's voice leaking into the interviewer's track ("what I did
was...") does not become a question.

Pure function; no I/O.
"""
from __future__ import annotations

import re

_QUESTION_PHRASES = re.compile(
    r"\b(tell me|walk me through|describe|explain|talk (me )?(about|through)|"
    r"(can|could|would|will) you|do you|did you|have you|are you|what (is|was|are|were|do|did|would)|"
    r"why (did|do|are|would)|how (did|do|would|have|are)|give me an example)\b"
)
_FILLER_WORDS = {
    "ah", "uh", "um", "umm", "uhh", "er", "erm", "hmm", "mm", "mhm", "oh", "okay", "ok", "so", "well",
    "yeah", "yes", "right", "hello", "hi", "hey", "alright", "thanks", "thank", "sorry",
}
_CHATTER = re.compile(
    r"\b(can you (hear|see) me|can you hear|hear me|see my screen|on mute|audio|my mic|microphone|"
    r"camera|connection|testing|sound check|how are you|how's it going|how is it going|"
    r"nice to meet you|good (morning|afternoon|evening))\b"
)
MIN_CONTENT_WORDS = 4


def _content_words(sentence: str) -> int:
    words = [w for w in re.findall(r"[a-z']+", sentence.lower()) if w not in _FILLER_WORDS]
    return sum(1 for i, w in enumerate(words) if i == 0 or w != words[i - 1])     # "maybe, maybe" counts once


def extract_question(text: str) -> str | None:
    kept = []
    for sentence in re.split(r"(?<=[.?!])\s+", (text or "").strip()):
        s = sentence.strip()
        low = s.lower()
        if not s or _CHATTER.search(low) or _content_words(s) < MIN_CONTENT_WORDS:
            continue
        if "?" in s or _QUESTION_PHRASES.search(low):
            kept.append(s)
    return " ".join(kept) if kept else None
