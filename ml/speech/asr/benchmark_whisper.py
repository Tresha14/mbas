"""COMPARISON ONLY - never used by the product.

Runs OpenAI Whisper-small on the same LibriSpeech test-clean subset so the report can show the gap to a large
pre-trained model (the plan's benchmark row). Requires a separate install that is NOT in requirements.txt:

    pip install openai-whisper
    python -m ml.speech.asr.benchmark_whisper --limit 300
Writes docs/metrics/whisper_benchmark.json, which ml/speech/asr/eval.py copies into the WER table.
"""
from __future__ import annotations

import argparse
import sys

from ml.common import text
from ml.common.config import METRICS, abspath, save_json
from ml.common.datasets import ASRDataset
from ml.common.text import normalize_asr
from ml.speech.asr.decode import wer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--model", default="small")
    args = ap.parse_args(argv)
    try:
        import whisper  # noqa: PLC0415 - optional, benchmark only
    except ImportError:
        raise SystemExit("pip install openai-whisper first (benchmark only; not part of MBAS)")
    m = whisper.load_model(args.model)
    ds = ASRDataset("test", text.load("asr"), ("librispeech",), 35.0, limit=args.limit)
    e = n = 0
    for r in ds.m.itertuples():
        hyp = normalize_asr(m.transcribe(str(abspath(r.path)), language="en", fp16=False)["text"])
        a, b = wer(r.norm, hyp)
        e, n = e + a, n + b
    res = {"model": f"whisper-{args.model}", "set": "librispeech_test_clean", "utterances": len(ds), "wer": round(e / n, 4),
           "note": "comparison only; 680k h of training audio vs our ~100 h"}
    save_json(res, METRICS / "whisper_benchmark.json")
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
