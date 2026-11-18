"""CTC decoding: greedy, prefix beam search, and beam search with word n-gram LM fusion (our own implementation;
no learned weights). Word boundaries come from SentencePiece pieces that start with '▁'.

score(prefix) = log P_ctc(prefix) + alpha * log P_lm(words) + beta * n_words
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np

NEG = -1e30


def logsumexp(a: float, b: float) -> float:
    if a == NEG:
        return b
    if b == NEG:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


def greedy(logp: np.ndarray, blank: int = 0) -> list[int]:
    ids = logp.argmax(-1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != blank:
            out.append(int(i))
        prev = i
    return out


def greedy_with_times(logp: np.ndarray, blank: int = 0, frame_s: float = 0.04) -> list[tuple[int, float]]:
    ids = logp.argmax(-1)
    out, prev = [], -1
    for t, i in enumerate(ids):
        if i != prev and i != blank:
            out.append((int(i), t * frame_s))
        prev = i
    return out


class BeamDecoder:
    def __init__(self, pieces: list[str], lm=None, alpha: float = 0.5, beta: float = 1.0, beam: int = 16,
                 prune: float = -8.0, blank: int = 0):
        self.pieces, self.lm, self.alpha, self.beta = pieces, lm, alpha, beta
        self.beam, self.prune, self.blank = beam, prune, blank
        self.word_start = np.array([p.startswith("▁") for p in pieces])

    def _lm(self, words: tuple[str, ...], new_word: str) -> float:
        if self.lm is None or not new_word:
            return 0.0
        return self.alpha * self.lm.score(new_word, ("<s>",) * 3 + words) + self.beta

    def decode(self, logp: np.ndarray, max_s: float | None = None) -> list[int]:
        """logp (T, V) natural log-probs -> best token sequence. Raises TimeoutError after max_s seconds (live use:
        the caller keeps the greedy result rather than stalling the audio loop)."""
        t_start = time.perf_counter()
        # beam entries: prefix tuple -> [p_blank, p_nonblank, lm_score, words tuple, partial word]
        beams = {(): [0.0, NEG, 0.0, (), ""]}
        for t in range(len(logp)):
            if max_s is not None and time.perf_counter() - t_start > max_s:
                raise TimeoutError("beam search over time budget")
            row = logp[t]
            cand = np.where(row > self.prune)[0]
            if len(cand) == 0:
                cand = [int(row.argmax())]
            nxt: dict[tuple, list] = defaultdict(lambda: [NEG, NEG, 0.0, (), ""])
            for prefix, (pb, pnb, lm_s, words, part) in beams.items():
                total = logsumexp(pb, pnb)
                for c in cand:
                    p = float(row[c])
                    if c == self.blank:
                        e = nxt[prefix]
                        e[0] = logsumexp(e[0], total + p)
                        e[2:] = [lm_s, words, part]
                        continue
                    last = prefix[-1] if prefix else None
                    new = prefix + (int(c),)
                    piece = self.pieces[c]
                    if self.word_start[c]:      # a new word starts: score the completed one
                        w_done = part
                        n_words, n_part = (words + (w_done,)) if w_done else words, piece[1:]
                        add = self._lm(words, w_done)
                    else:
                        n_words, n_part, add = words, part + piece, 0.0
                    e = nxt[new]
                    if c == last:   # repeated symbol needs a blank in between
                        e[1] = logsumexp(e[1], pb + p)
                        same = nxt[prefix]
                        same[1] = logsumexp(same[1], pnb + p)
                        same[2:] = [lm_s, words, part]
                    else:
                        e[1] = logsumexp(e[1], total + p)
                    e[2], e[3], e[4] = lm_s + add, n_words, n_part
            beams = dict(sorted(nxt.items(), key=lambda kv: -(logsumexp(kv[1][0], kv[1][1]) + kv[1][2]))[: self.beam])
        def final(kv):
            (pb, pnb, lm_s, words, part) = kv[1]
            return logsumexp(pb, pnb) + lm_s + self._lm(words, part)
        return list(max(beams.items(), key=final)[0])


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance in words, reference length)."""
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)], len(r)


def live_decoder(pieces: list[str], beam: int = 8):  # prune -6: tighter than offline eval (-8) to keep it fast
    """Beam search + our 4-gram LM for the live path; alpha/beta come from the last ml.speech.asr.eval run.
    None when the LM file is missing (the live path then stays greedy)."""
    import json

    from ml.common.config import METRICS
    from ml.speech.lm.ngram import LM_DIR, ArpaLM

    p = LM_DIR / "interview_4gram.arpa"
    if not p.exists():
        return None
    alpha, beta = 0.5, 1.0
    tuned = METRICS / "asr.json"
    if tuned.exists():
        t = json.loads(tuned.read_text())
        alpha, beta = t.get("alpha") or alpha, t.get("beta") or beta
    return BeamDecoder(pieces, ArpaLM(p), alpha, beta, beam=beam, prune=-6.0)


def words_from_tokens(pieces: list[str], ids: list[int]) -> list[str]:
    out: list[str] = []
    for i in ids:
        p = pieces[i]
        if p.startswith("▁") or not out:
            out.append(p.lstrip("▁"))
        else:
            out[-1] += p
    return [w for w in out if w]


def retime(new: list[str], old: list[dict]) -> list[dict]:
    """Give the beam-search words timestamps by matching them to the greedy words (same audio, same frames):
    identical or equal-length replaced words copy the time, the rest share the surrounding gap by length."""
    from difflib import SequenceMatcher

    if not old:
        return [{"word": w, "start_s": 0.0, "end_s": 0.0} for w in new]
    out: list[dict | None] = [None] * len(new)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, [w["word"] for w in old], new, autojunk=False).get_opcodes():
        if op in ("equal", "replace") and i2 - i1 == j2 - j1:
            for k in range(j2 - j1):
                out[j1 + k] = {"word": new[j1 + k], "start_s": old[i1 + k]["start_s"], "end_s": old[i1 + k]["end_s"]}
    k = 0
    while k < len(new):
        if out[k] is not None:
            k += 1
            continue
        j = k
        while j < len(new) and out[j] is None:
            j += 1
        lo = out[k - 1]["end_s"] if k > 0 else old[0]["start_s"]
        hi = out[j]["start_s"] if j < len(new) else old[-1]["end_s"]
        hi = max(hi, lo)
        total = sum(len(w) for w in new[k:j]) or 1
        t = lo
        for m in range(k, j):
            d = (hi - lo) * len(new[m]) / total
            out[m] = {"word": new[m], "start_s": round(t, 2), "end_s": round(t + d, 2)}
            t += d
        k = j
    return out  # type: ignore[return-value]
