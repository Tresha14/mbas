"""`make data`: raw sessions + labels + public datasets -> model-ready tables in data/processed/.

Steps (each is skipped with a note when its input is missing):
  1. synthetic labels -> data/processed/labels/synthetic            (tools.synthetic_to_labels)
  2. person-wise splits: recorded, synthetic, public datasets        (tools.make_splits)
  3. 5 s windows with labels + hesitation counts -> data/processed/windows/
  4. SentencePiece tokenizers (asr 1000, answer 8000) -> data/processed/tokenizers/
  5. dataset shape check: one item from every Dataset class that has data
  6. dataset card -> docs/dataset_card.md                             (tools.dataset_stats)

    python -m ml.common.make_data
"""
from __future__ import annotations

import argparse
import sys
import time

from ml.common.config import LABELS, PROCESSED, PUBLIC, SYNTHETIC

SPLIT_PUBLIC = ["mpiigaze", "daisee", "sep28k", "common_voice_in", "ami", "columbia_gaze"]


def step(name: str):
    print(f"\n== {name}", flush=True)
    return time.time()


def check_datasets() -> dict[str, str]:
    """Instantiate each Dataset that has data and report one item's shapes."""
    from ml.common import datasets as D
    from ml.common import text

    out = {}

    def shapes(item) -> str:
        if isinstance(item, dict):
            return ", ".join(f"{k}{tuple(v.shape)}" for k, v in item.items() if hasattr(v, "shape"))
        return ", ".join(str(tuple(v.shape)) for v in item)

    tries = {
        "FaceDataset": lambda: D.FaceDataset("val", limit=4),
        "GazeDataset": lambda: D.GazeDataset("val", limit=8),
        "EngagementClipDataset": lambda: D.EngagementClipDataset("val", limit=2),
        "VADDataset": lambda: D.VADDataset("val", limit=2),
        "ASRDataset": lambda: D.ASRDataset("val", text.load("asr"), limit=4),
        "DisfluencyDataset": lambda: D.DisfluencyDataset("val", limit=4),
        "AnswerDataset": lambda: D.AnswerDataset("val", text.load("answer")),
        "FusionDataset": lambda: D.FusionDataset("val"),
    }
    for name, make in tries.items():
        try:
            ds = make()
            out[name] = f"{len(ds)} items; {shapes(ds[0])}" if len(ds) else "0 items"
        except (FileNotFoundError, ValueError, IndexError, KeyError) as e:
            out[name] = f"skipped ({str(e).splitlines()[0][:90]})"
        print(f"  {name:22s} {out[name]}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tokenizers", action="store_true")
    args = ap.parse_args(argv)
    from tools import dataset_stats, make_splits, synthetic_to_labels

    t = step("1 synthetic labels")
    if SYNTHETIC.exists():
        synthetic_to_labels.main([])
    else:
        print("  synthetic dataset not found, skipped")

    step("2 splits")
    sources = ["recorded"] + (["synthetic"] if SYNTHETIC.exists() else [])
    sources += [f"public:{n}" for n in SPLIT_PUBLIC if (PUBLIC / n / "manifest.parquet").exists()]
    make_splits.main(["--source", *sources])

    step("3 windows")
    from ml.common import windowing

    for src in ("recorded", "synthetic"):
        if src == "synthetic" and not SYNTHETIC.exists():
            continue
        p = windowing.build(src, PROCESSED / "windows")
        import pandas as pd

        print(f"  {src}: {len(pd.read_parquet(p))} windows -> {p.relative_to(PROCESSED.parent.parent)}")

    if not args.skip_tokenizers:
        step("4 tokenizers")
        from ml.common import text

        text.main([])

    step("5 dataset check")
    check_datasets()

    step("6 dataset card")
    dataset_stats.main([])
    print(f"\nmake data finished in {time.time() - t:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
