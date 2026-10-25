"""Word n-gram language model: interpolated Kneser-Ney, built from our text, saved as ARPA (KenLM-compatible).

No learned neural weights: counts + discounting. Corpus = LibriSpeech train transcripts + interview-style text
we wrote (question bank, key points, our and synthetic answers).

    python -m ml.speech.lm.ngram --order 4          # -> data/processed/lm/interview_4gram.arpa
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ml.common.config import PROCESSED

LM_DIR = PROCESSED / "lm"
BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def build_kn(sentences: list[list[str]], order: int = 4, discount: float = 0.75, min_count: int = 1):
    """-> {n: {ngram tuple: (log10 prob, log10 backoff)}} in ARPA form."""
    counts = [Counter() for _ in range(order + 1)]
    for s in sentences:
        toks = [BOS] * (order - 1) + s + [EOS]
        for n in range(1, order + 1):
            for i in range(len(toks) - n + 1):
                g = tuple(toks[i:i + n])
                if n > 1 and g[-1] == BOS:
                    continue
                counts[n][g] += 1
    # highest order uses raw counts; lower orders use continuation counts N1+(. w)
    cont = [Counter() for _ in range(order + 1)]
    cont[order] = Counter({g: c for g, c in counts[order].items() if c >= min_count})
    for n in range(order - 1, 0, -1):
        for g in counts[n + 1]:
            cont[n][g[1:]] += 1
        for g, c in counts[n].items():   # n-grams that start a sentence have no left context: keep raw count
            if g[0] == BOS and g not in cont[n]:
                cont[n][g] = c
    probs: dict[int, dict[tuple, list[float]]] = {}
    vocab = {g[0] for g in cont[1]} | {UNK}
    # unigrams
    total = sum(c for g, c in cont[1].items() if g != (BOS,))
    uni = {}
    n_types = len(vocab)
    for w in vocab:
        c = cont[1].get((w,), 0)
        p = (max(c - discount, 0) / total) + (discount * len(cont[1]) / total) / n_types
        uni[(w,)] = p
    uni[(BOS,)] = 1e-99
    probs[1] = {g: [p, 0.0] for g, p in uni.items()}
    for n in range(2, order + 1):
        ctx_tot, ctx_types = defaultdict(int), defaultdict(int)
        for g, c in cont[n].items():
            ctx_tot[g[:-1]] += c
            ctx_types[g[:-1]] += 1
        cur = {}
        for g, c in cont[n].items():
            h = g[:-1]
            gamma = discount * ctx_types[h] / ctx_tot[h]
            cur[g] = max(c - discount, 0) / ctx_tot[h] + gamma * _full(probs, g[1:])
        probs[n] = {g: [p, 0.0] for g, p in cur.items()}
        # interpolated KN in ARPA form: back-off weight of context h is exactly its leftover mass gamma(h)
        for h in ctx_tot:
            gamma = discount * ctx_types[h] / ctx_tot[h]
            if h in probs[n - 1]:
                probs[n - 1][h][1] = gamma
            else:   # context never seen as an (n-1)-gram on its own (sentence-start padding): add it
                probs[n - 1][h] = [_full(probs, h), gamma]
    out = {}
    for n, d in probs.items():
        out[n] = {g: (math.log10(max(p, 1e-99)), math.log10(b) if b > 0 else 0.0) for g, (p, b) in d.items()}
    return out


def _full(probs, g: tuple) -> float:
    """Interpolated probability of the last word of g given its history, backing off with the weights so far."""
    n = len(g)
    if g in probs.get(n, {}):
        return probs[n][g][0]
    if n == 1:
        return probs[1].get((UNK,), [1e-7])[0]
    bow = probs.get(n - 1, {}).get(g[:-1], [0.0, 1.0])[1] or 1.0
    return bow * _full(probs, g[1:])


def write_arpa(model: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        for n in sorted(model):
            f.write(f"ngram {n}={len(model[n])}\n")
        top = max(model)
        for n in sorted(model):
            f.write(f"\n\\{n}-grams:\n")
            for g, (lp, bo) in sorted(model[n].items()):
                if n < top and bo != 0.0:
                    f.write(f"{lp:.6f}\t{' '.join(g)}\t{bo:.6f}\n")
                else:
                    f.write(f"{lp:.6f}\t{' '.join(g)}\n")
        f.write("\n\\end\\\n")
    return path


class ArpaLM:
    """Back-off scoring of an ARPA file (ours or KenLM's). log10 internally; score() returns natural log."""

    def __init__(self, path: str | Path):
        self.p: dict[tuple, float] = {}
        self.b: dict[tuple, float] = {}
        self.order = 0
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("\\") and line.endswith("-grams:"):
                    n = int(line[1:line.index("-")])
                    self.order = max(self.order, n)
                    continue
                if not line or line.startswith("\\") or line.startswith("ngram ") or n == 0:
                    continue
                parts = line.split("\t")
                g = tuple(parts[1].split(" "))
                self.p[g] = float(parts[0])
                if len(parts) > 2:
                    self.b[g] = float(parts[2])
        self.vocab = {g[0] for g in self.p if len(g) == 1}

    def log10_prob(self, word: str, history: tuple[str, ...]) -> float:
        if word not in self.vocab:
            word = UNK
        h = history[-(self.order - 1):] if self.order > 1 else ()
        bo = 0.0
        while True:
            g = h + (word,)
            if g in self.p:
                return self.p[g] + bo
            if not h:
                return self.p.get((UNK,), -7.0) + bo
            bo += self.b.get(h, 0.0)
            h = h[1:]

    def score(self, word: str, history: tuple[str, ...]) -> float:
        return self.log10_prob(word, history) * math.log(10)

    def sentence_log10(self, words: list[str]) -> float:
        h = (BOS,) * (self.order - 1)
        tot = 0.0
        for w in words + [EOS]:
            tot += self.log10_prob(w, h)
            h = (h + (w,))[1:] if self.order > 1 else ()
        return tot

    def perplexity(self, sentences: list[list[str]]) -> float:
        lp = sum(self.sentence_log10(s) for s in sentences)
        n = sum(len(s) + 1 for s in sentences)
        return 10 ** (-lp / max(n, 1))


def corpus_sentences() -> list[list[str]]:
    from ml.common.text import corpus_asr

    return [l.split() for l in corpus_asr() if l]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", type=int, default=4)
    args = ap.parse_args(argv)
    sents = corpus_sentences()
    held = sents[::50]
    train = [s for i, s in enumerate(sents) if i % 50]
    model = build_kn(train, args.order)
    p = write_arpa(model, LM_DIR / f"interview_{args.order}gram.arpa")
    lm = ArpaLM(p)
    ppl = lm.perplexity(held)
    full = write_arpa(build_kn(sents, args.order), p)   # final LM on all text
    print(f"{len(sents)} sentences, vocab {len(lm.vocab)}; held-out perplexity {ppl:.1f}; wrote {full}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
