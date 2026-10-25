"""Common Voice English, Indian-accent subset -> 16 kHz WAV + manifest (ASR, Stage 6).

Manual access: download the English corpus (cv-corpus-<version>-en.tar.gz) from
https://datacollective.mozillafoundation.org (Mozilla Data Collective, free account).
Keeps validated clips whose `accents` (or older `accent`) field mentions India /
South Asia, and converts their MP3s to 16 kHz mono WAV.

    python -m tools.datasets.common_voice --archive D:\\cv-corpus-21.0-en.tar.gz
    python -m tools.datasets.common_voice --source D:\\cv-corpus-21.0\\en        # already extracted
Output: data/processed/public/common_voice_in/{wav/<clip>.wav, manifest.parquet}

A tar archive is read twice (tsv files, then the selected clips) without
unpacking the whole ~80 GB corpus.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import soundfile as sf

from tools.datasets.common import dirs, fetch_or_copy, rel, to_wav16k, write_outputs

NAME = "common_voice_in"
INDIAN = re.compile(r"india|south asia", re.I)
SPLIT_TSVS = {"train.tsv": "train", "dev.tsv": "val", "test.tsv": "test"}
HELP = ("Download the English Common Voice corpus from https://datacollective.mozillafoundation.org , then run\n"
        "  python -m tools.datasets.common_voice --archive <cv-corpus-XX-en.tar.gz>")


def select(validated: pd.DataFrame) -> pd.DataFrame:
    col = "accents" if "accents" in validated else "accent"
    sel = validated[validated[col].fillna("").astype(str).str.contains(INDIAN)].copy()
    return sel.rename(columns={col: "accents"})


def _read_tsv(data: bytes | Path) -> pd.DataFrame:
    src = io.BytesIO(data) if isinstance(data, bytes) else data
    return pd.read_csv(src, sep="\t", quoting=3, dtype=str, on_bad_lines="skip")


def from_dir(src: Path) -> tuple[pd.DataFrame, dict[str, str], Path]:
    validated = _read_tsv(next(src.rglob("validated.tsv")))
    split = {}
    for f, s in SPLIT_TSVS.items():
        p = next(src.rglob(f), None)
        if p is not None:
            split.update(dict.fromkeys(_read_tsv(p).path, s))
    clips = next(p for p in src.rglob("clips") if p.is_dir())
    return validated, split, clips


def from_tar(archive: Path, tmp: Path) -> tuple[pd.DataFrame, dict[str, str], Path]:
    tsvs: dict[str, pd.DataFrame] = {}
    with tarfile.open(archive) as t:
        for m in t:
            base = m.name.rsplit("/", 1)[-1]
            if m.isfile() and base in ("validated.tsv", *SPLIT_TSVS):
                tsvs[base] = _read_tsv(t.extractfile(m).read())
    validated = tsvs["validated.tsv"]
    split = {}
    for f, s in SPLIT_TSVS.items():
        if f in tsvs:
            split.update(dict.fromkeys(tsvs[f].path, s))
    wanted = set(select(validated).path)
    clips = tmp / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        for m in t:
            base = m.name.rsplit("/", 1)[-1]
            if m.isfile() and "/clips/" in m.name and base in wanted:
                (clips / base).write_bytes(t.extractfile(m).read())
    return validated, split, clips


def convert(validated: pd.DataFrame, split: dict[str, str], clips: Path, out: Path, workers: int = 8) -> pd.DataFrame:
    sel = select(validated)
    wav_dir = out / "wav"

    def one(r) -> dict | None:
        src = clips / r.path
        dest = wav_dir / (Path(r.path).stem + ".wav")
        if not src.exists():
            return None
        if not dest.exists():
            try:
                to_wav16k(src, dest)
            except RuntimeError:
                return None
        return {"item_id": Path(r.path).stem, "subject_id": r.client_id, "split_official": split.get(r.path, "other"),
                "path": rel(dest), "duration_s": round(sf.info(str(dest)).duration, 3), "text": r.sentence,
                "accents": r.accents, "gender": getattr(r, "gender", None), "age": getattr(r, "age", None)}

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(one, sel.itertuples(index=False)) if r]
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", help="cv-corpus-XX-en.tar.gz (manual download)")
    ap.add_argument("--source", help="already extracted .../en folder")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    downloads, _, proc = dirs(NAME)
    if args.source:
        validated, split, clips = from_dir(Path(args.source))
        m = convert(validated, split, clips, proc, args.workers)
    else:
        archive = fetch_or_copy(NAME, None, args.archive, manual_help=HELP)
        with tempfile.TemporaryDirectory(dir=downloads) as tmp:
            validated, split, clips = from_tar(archive, Path(tmp))
            m = convert(validated, split, clips, proc, args.workers)
    if m.empty:
        raise SystemExit("no Indian-accent clips found")
    write_outputs(NAME, m, {
        "title": "Common Voice EN (Indian accent)", "used_for": "ASR accent coverage (Stage 6)", "license": "CC0",
        "source": "https://datacollective.mozillafoundation.org", "filter": INDIAN.pattern,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
