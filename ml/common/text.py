"""Text normalisation and our SentencePiece unigram tokenizers (trained from scratch).

  asr     vocab 1000  - LibriSpeech train text + our manual transcripts (+ synthetic transcripts)
  answer  vocab 8000  - question bank + answers (synthetic + ours) + LibriSpeech text

    python -m ml.common.text            # trains both into data/processed/tokenizers/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from ml.common.config import LABELS, PROCESSED, PUBLIC, SYNTHETIC

TOK_DIR = PROCESSED / "tokenizers"
_KEEP = re.compile(r"[^a-z' ]+")
_SPACES = re.compile(r"\s+")


def normalize_asr(text: str) -> str:
    """Lowercase, letters + apostrophe + space only (the ASR output alphabet)."""
    t = str(text).lower().replace("-", " ").replace("—", " ")
    t = re.sub(r"\[[^\]]*\]", " ", t)   # [inaudible], [laugh]
    return _SPACES.sub(" ", _KEEP.sub(" ", t)).strip()


def normalize_answer(text: str) -> str:
    return _SPACES.sub(" ", str(text).replace("\n", " ")).strip()


def corpus_asr() -> list[str]:
    lines: list[str] = []
    ls = PUBLIC / "librispeech" / "manifest.parquet"
    if ls.exists():
        m = pd.read_parquet(ls, columns=["text", "split_official"])
        lines += m[m.split_official == "train"].text.map(normalize_asr).tolist()
    cv = PUBLIC / "common_voice_in" / "manifest.parquet"
    if cv.exists():
        m = pd.read_parquet(cv, columns=["text", "split_official"])
        lines += m[m.split_official != "test"].text.map(normalize_asr).tolist()
    for src in ("recorded", "synthetic"):
        p = LABELS / src / "answers.parquet"
        if p.exists():
            lines += [normalize_asr(t) for t in pd.read_parquet(p, columns=["transcript"]).transcript.dropna()]
    bank = SYNTHETIC / "reference" / "question_bank.csv"
    if bank.exists():
        lines += pd.read_csv(bank).text.map(normalize_asr).tolist()
    return [l for l in lines if l]


def corpus_answer() -> list[str]:
    lines: list[str] = []
    bank = SYNTHETIC / "reference" / "question_bank.csv"
    if bank.exists():
        b = pd.read_csv(bank)
        lines += b.text.tolist()
        for c in ("key_point_1", "key_point_2", "key_point_3", "key_point_4"):
            lines += b[c].dropna().tolist()
    for src in ("recorded", "synthetic"):
        p = LABELS / src / "answers.parquet"
        if p.exists():
            lines += pd.read_parquet(p, columns=["transcript"]).transcript.dropna().tolist()
    ls = PUBLIC / "librispeech" / "manifest.parquet"
    if ls.exists():
        m = pd.read_parquet(ls, columns=["text", "split_official"])
        lines += m[m.split_official == "train"].text.str.lower().tolist()
    return [normalize_answer(l) for l in lines if str(l).strip()]


def train_tokenizer(name: str, lines: list[str], vocab: int, out_dir: Path = TOK_DIR, **extra) -> Path:
    import sentencepiece as spm

    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / f"{name}_corpus.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # unigram needs enough distinct pieces; shrink the vocab for a tiny corpus (tests / first run)
    n_chars = sum(len(l) for l in lines)
    vocab = int(min(vocab, max(50, n_chars // 40)))
    spm.SentencePieceTrainer.train(
        input=str(txt), model_prefix=str(out_dir / name), vocab_size=vocab, model_type="unigram",
        character_coverage=1.0, unk_id=1, bos_id=-1, eos_id=-1, pad_id=0, input_sentence_size=2_000_000,
        shuffle_input_sentence=True, num_threads=8, minloglevel=2, hard_vocab_limit=False, **extra)
    return out_dir / f"{name}.model"


class Tokenizer:
    """Thin wrapper; id 0 = pad (also the CTC blank for ASR), 1 = unk."""

    def __init__(self, path: str | Path):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(path))

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([i for i in ids if i > 1])

    def pieces(self) -> list[str]:
        return [self.sp.id_to_piece(i) for i in range(len(self))]


def load(name: str) -> Tokenizer:
    p = TOK_DIR / f"{name}.model"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run python -m ml.common.text")
    return Tokenizer(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr-vocab", type=int, default=1000)
    ap.add_argument("--answer-vocab", type=int, default=8000)
    ap.add_argument("--force", action="store_true",
                    help="rebuild tokenizers that already exist (the trained ASR / answer checkpoints then need retraining)")
    args = ap.parse_args(argv)
    # The checkpoints are tied to the exact tokenizer they were trained with (the corpus decides the vocabulary
    # size), so an existing tokenizer is never overwritten by accident. The committed answer tokenizer has 1,130
    # pieces: it was built from the interview answers before LibriSpeech was added to the corpus.
    for name, corpus, vocab in (("asr", corpus_asr, args.asr_vocab), ("answer", corpus_answer, args.answer_vocab)):
        if (TOK_DIR / f"{name}.model").exists() and not args.force:
            print(f"{name} tokenizer exists ({len(load(name))} pieces), kept; use --force to rebuild")
            continue
        lines = corpus()
        p = train_tokenizer(name, lines, vocab)
        print(f"{name} tokenizer: {len(Tokenizer(p))} pieces from {len(lines)} lines -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
