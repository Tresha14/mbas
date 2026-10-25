"""LibriSpeech -> utterance manifest (VAD + ASR training, tokenizer/LM text; Stage 3 and 6).

Downloads train-clean-100 (6.4 GB) plus dev-clean and test-clean from OpenSLR.
Audio stays as 16 kHz FLAC (readable by soundfile / torchaudio).

    python -m tools.datasets.librispeech
    python -m tools.datasets.librispeech --subsets dev-clean          # quick test (337 MB)
Output: data/processed/public/librispeech/{manifest.parquet, text_corpus.txt}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

from tools.datasets.common import dirs, download, extract, rel, write_outputs

NAME = "librispeech"
BASE = "https://openslr.trmal.net/resources/12/"  # OpenSLR mirror, ~4x faster from India than www.openslr.org
SIZES = {"train-clean-100": 6387309499, "dev-clean": 337926286, "test-clean": 346663984}
SPLIT = {"train-clean-100": "train", "train-clean-360": "train", "dev-clean": "val", "test-clean": "test"}


def speakers(root: Path) -> dict[str, str]:
    f = next(root.rglob("SPEAKERS.TXT"), None)
    out = {}
    if f:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(";") or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            out[parts[0]] = parts[1]
    return out


def convert(root: Path, subsets: list[str]) -> pd.DataFrame:
    sex = speakers(root)
    rows = []
    for subset in subsets:
        sdir = next((p for p in root.rglob(subset) if p.is_dir()), None)
        if sdir is None:
            print(f"  {subset} not found, skipped")
            continue
        for trans in sorted(sdir.rglob("*.trans.txt")):
            for line in trans.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                utt, text = line.split(" ", 1)
                flac = trans.parent / f"{utt}.flac"
                spk, chapter, _ = utt.split("-")
                rows.append({"item_id": utt, "subject_id": spk, "split_official": SPLIT.get(subset, subset),
                             "subset": subset, "chapter": chapter, "path": rel(flac),
                             "duration_s": round(sf.info(str(flac)).duration, 3),
                             "text": text.strip(), "gender": sex.get(spk)})
        print(f"  {subset}: {sum(r['subset'] == subset for r in rows)} utterances")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", nargs="+", default=list(SIZES))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args(argv)
    downloads, extracted, proc = dirs(NAME)
    if not args.skip_download:
        for s in args.subsets:
            extract(download(f"{BASE}{s}.tar.gz", downloads / f"{s}.tar.gz", SIZES.get(s)), extracted)
    m = convert(extracted, args.subsets)
    (proc / "text_corpus.txt").write_text("\n".join(m[m.split_official == "train"].text.str.lower()) + "\n",
                                          encoding="utf-8")
    write_outputs(NAME, m, {
        "title": "LibriSpeech", "used_for": "VAD, ASR, tokenizer + LM text (Stages 3, 6)", "license": "CC BY 4.0",
        "source": "https://www.openslr.org/12", "subsets": args.subsets,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
