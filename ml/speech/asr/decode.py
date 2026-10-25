"""CTC decoding: greedy, prefix beam search, and beam search with word n-gram LM fusion (our own implementation;
no learned weights). Word boundaries come from SentencePiece pieces that start with '▁'.

score(prefix) = log P_ctc(prefix) + alpha * log P_lm(words) + beta * n_words
"""
from __future__ import annotations

import math
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

    def decode(self, logp: np.ndarray) -> list[int]:
        """logp (T, V) natural log-probs -> best token sequence."""
        # beam entries: prefix tuple -> [p_blank, p_nonblank, lm_score, words tuple, partial word]
        beams = {(): [0.0, NEG, 0.0, (), ""]}
        for t in range(len(logp)):
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
