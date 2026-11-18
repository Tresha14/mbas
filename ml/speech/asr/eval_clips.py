"""Measure transcript quality on your own clips, the way the live system actually transcribes.

Put audio and the correct text side by side in one folder:

    my_clips/
        clip01.wav   clip01.txt      (what was really said, typed by hand)
        clip02.wav   clip02.txt
        ...

    python -m ml.speech.asr.eval_clips my_clips
    python -m ml.speech.asr.eval_clips my_clips --modes live greedy beam_lm --out docs/metrics/asr_clips.json

Modes (all use the same trained checkpoint):
    live     640 ms chunks through StreamingASR (VAD-gated, beam + LM on finals) - exactly what the interview sees
    greedy   whole clip at once, greedy decoding
    beam_lm  whole clip at once, beam search + our 4-gram LM (alpha/beta from docs/metrics/asr.json if present)

Reports WER and CER per clip and overall, how many errors are substitutions / deletions / insertions, and the
words that are wrong most often. Text is normalised the same way as the training transcripts (lower case, letters
and apostrophes only), so punctuation and capitals in your .txt files do not matter.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ml.common.text import normalize_asr

SR = 16000


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Minimum-edit alignment as (op, ref_word, hyp_word); op is ok / sub / del / ins."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def score(ref_text: str, hyp_text: str) -> dict:
    ref, hyp = normalize_asr(ref_text).split(), normalize_asr(hyp_text).split()
    ops = align(ref, hyp)
    c = Counter(o[0] for o in ops)
    errors = c["sub"] + c["del"] + c["ins"]
    cref, chyp = list(" ".join(ref)), list(" ".join(hyp))
    cer_errors = sum(1 for o in align(cref, chyp) if o[0] != "ok")
    return {"ref_words": len(ref), "hyp_words": len(hyp), "errors": errors, "sub": c["sub"], "del": c["del"], "ins": c["ins"],
            "wer": round(errors / max(len(ref), 1), 4), "cer": round(cer_errors / max(len(cref), 1), 4), "ops": ops}


def summarise(rows: list[dict]) -> dict:
    """Corpus-level numbers: errors summed over all clips, divided by all reference words."""
    words = sum(r["ref_words"] for r in rows)
    errs = sum(r["errors"] for r in rows)
    wrong = Counter()
    for r in rows:
        for op, a, b in r["ops"]:
            if op == "sub":
                wrong[f"{a} -> {b}"] += 1
            elif op == "del":
                wrong[f"{a} -> (missed)"] += 1
            elif op == "ins":
                wrong[f"(extra) -> {b}"] += 1
    return {"clips": len(rows), "ref_words": words, "wer": round(errs / max(words, 1), 4),
            "sub": sum(r["sub"] for r in rows), "del": sum(r["del"] for r in rows), "ins": sum(r["ins"] for r in rows),
            "mean_clip_cer": round(float(np.mean([r["cer"] for r in rows])), 4) if rows else None,
            "most_common_errors": wrong.most_common(15)}


def find_clips(folder: Path) -> list[tuple[Path, str]]:
    out = []
    for wav in sorted(list(folder.glob("*.wav")) + list(folder.glob("*.flac"))):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"skipping {wav.name}: no {txt.name} next to it")
            continue
        ref = txt.read_text(encoding="utf-8").strip()
        if not normalize_asr(ref):
            print(f"skipping {wav.name}: {txt.name} is empty")
            continue
        out.append((wav, ref))
    return out


def transcribe_live(model, tok, audio: np.ndarray, dev: str, vad_fn=None, decoder=None) -> str:
    """Same path as backend/workers/speech_worker.py: StreamingASR with the VAD gate, 640 ms chunks, greedy decoding."""
    from backend.workers.speech_worker import ChunkVAD
    from ml.speech.asr.streaming import StreamingASR

    s = StreamingASR(model, tok, ChunkVAD(vad_fn), device=dev, decoder=decoder)
    finals = []
    step = int(0.64 * SR)
    for i in range(0, len(audio), step):
        finals += [r for r in s.accept(audio[i:i + step]) if r.final]
    finals += [r for r in s.flush() if r.final]
    return " ".join(r.text for r in finals if r.text)


def transcribe_offline(model, tok, audio: np.ndarray, dev: str, decoder=None) -> str:
    import torch

    from ml.speech.asr.decode import greedy

    with torch.no_grad():
        lp, ol = model(torch.from_numpy(audio)[None].to(dev), torch.tensor([len(audio)], device=dev))
    lp = lp[0, : int(ol[0])].cpu().numpy()
    return tok.decode(decoder.decode(lp) if decoder else greedy(lp))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--modes", nargs="+", default=["live", "greedy", "beam_lm"], choices=["live", "greedy", "beam_lm"])
    ap.add_argument("--no-live-lm", action="store_true", help="live mode: greedy only (before this change)")
    ap.add_argument("--out", type=Path, default=Path("docs/metrics/asr_clips.json"))
    ap.add_argument("--show", type=int, default=3, help="print this many worst clips per mode")
    args = ap.parse_args(argv)

    clips = find_clips(args.folder)
    if not clips:
        print(f"No wav/txt pairs found in {args.folder}. Each clip needs name.wav and name.txt with the correct text.")
        return 1

    import torch

    from backend.inference import Models
    from ml.common import text
    from ml.common.audio_features import load_audio
    from ml.common.config import METRICS
    from ml.speech.asr.decode import BeamDecoder
    from ml.speech.asr.eval import load
    from ml.speech.lm.ngram import LM_DIR, ArpaLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, _ = load(dev)
    except FileNotFoundError:
        print("No ASR checkpoint found in checkpoints/asr. Train the model first (python -m ml.speech.asr.train).")
        return 1
    tok = text.load("asr")
    decoder = None
    if "beam_lm" in args.modes:
        lm_p = LM_DIR / "interview_4gram.arpa"
        alpha, beta = 0.5, 1.0
        tuned = METRICS / "asr.json"
        if tuned.exists():
            t = json.loads(tuned.read_text())
            alpha, beta = t.get("alpha") or alpha, t.get("beta") or beta
        decoder = BeamDecoder(tok.pieces(), ArpaLM(lm_p) if lm_p.exists() else None, alpha, beta)

    vad_fn = Models().vad if "live" in args.modes else None
    from ml.speech.asr.decode import live_decoder

    live_dec = None if args.no_live_lm else live_decoder(tok.pieces())
    audio = {w: load_audio(w).astype(np.float32) for w, _ in clips}
    report: dict = {"folder": str(args.folder), "clips": len(clips), "modes": {}}
    for mode in args.modes:
        rows = []
        for wav, ref in clips:
            a = audio[wav]
            hyp = (transcribe_live(model, tok, a, dev, vad_fn, live_dec) if mode == "live"
                   else transcribe_offline(model, tok, a, dev, decoder if mode == "beam_lm" else None))
            r = score(ref, hyp)
            rows.append({"clip": wav.name, "seconds": round(len(a) / SR, 1), "ref": normalize_asr(ref), "hyp": hyp, **r})
        s = summarise(rows)
        print(f"\n=== {mode}: WER {s['wer']:.1%}  (sub {s['sub']}, missed {s['del']}, extra {s['ins']} over {s['ref_words']} words, "
              f"{s['clips']} clips)")
        for r in sorted(rows, key=lambda r: -r["wer"])[: args.show]:
            print(f"  {r['clip']}  WER {r['wer']:.0%}\n    said: {r['ref']}\n    got:  {r['hyp']}")
        print("  most common errors:", "; ".join(f"{k} x{v}" for k, v in s["most_common_errors"][:8]))
        report["modes"][mode] = {**s, "per_clip": [{k: v for k, v in r.items() if k != "ops"} for r in rows]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
