"""Shared helpers for the public dataset scripts (raw data only, never model weights).

Folder format used by every script:
    data/raw/public/<name>/downloads/     archives as downloaded (or copied from --archive)
    data/raw/public/<name>/extracted/     unpacked archives
    data/processed/public/<name>/
        manifest.parquet                  one row per item; always has item_id, subject_id, split_official, path
        info.json                         title, licence, source URL, counts, hours, status
        ...                               converted media when the raw format is not usable directly
Paths in manifests are relative to the repo root, with forward slashes.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PUBLIC = ROOT / "data" / "raw" / "public"
PROC_PUBLIC = ROOT / "data" / "processed" / "public"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def dirs(name: str) -> tuple[Path, Path, Path]:
    """(downloads, extracted, processed) for a dataset, created if missing."""
    d = RAW_PUBLIC / name / "downloads"
    x = RAW_PUBLIC / name / "extracted"
    p = PROC_PUBLIC / name
    for f in (d, x, p):
        f.mkdir(parents=True, exist_ok=True)
    return d, x, p


def rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def download(url: str, dest: Path, expected_size: int | None = None, retries: int = 30) -> Path:
    """Resumable HTTP download (Range requests). Skips files that are already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
                if r.status_code == 416:  # already complete
                    break
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code} for {url}")
                if r.status_code == 200:
                    have = 0
                total = int(r.headers.get("content-length", 0)) + have
                mode = "ab" if r.status_code == 206 else "wb"
                last = time.time()
                with part.open(mode) as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 15:
                            pct = f"{100 * have / total:.0f}%" if total else ""
                            print(f"  {dest.name}: {have / 1e6:.0f} MB {pct}", flush=True)
                            last = time.time()
            break
        except (httpx.HTTPError, RuntimeError) as e:
            if attempt == retries:
                raise
            print(f"  retry {attempt} for {dest.name}: {e}")
            time.sleep(min(3 * attempt, 30))
    part.replace(dest)
    if expected_size is not None and dest.stat().st_size != expected_size:
        raise RuntimeError(f"{dest.name}: {dest.stat().st_size} bytes, expected {expected_size}")
    return dest


def fetch_or_copy(name: str, url: str | None, archive: str | None, filename: str | None = None,
                  expected_size: int | None = None, manual_help: str = "") -> Path:
    """Use --archive if given (manual-access datasets), else download url into downloads/."""
    downloads, _, _ = dirs(name)
    if archive:
        src = Path(archive)
        if not src.exists():
            raise SystemExit(f"{src} not found")
        return src
    if not url:
        raise SystemExit(f"{name} cannot be downloaded automatically.\n{manual_help}")
    return download(url, downloads / (filename or url.rsplit("/", 1)[-1].split("?")[0]), expected_size)


def extract(archive: Path, dest: Path, members: Callable[[str], bool] | None = None) -> Path:
    """Unpack zip / tar(.gz); skipped if it was already unpacked (marker file)."""
    marker = dest / f".extracted_{archive.name}"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} ...", flush=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if members is None or members(m.filename):
                    z.extract(m, dest)
    else:
        with tarfile.open(archive) as t:
            for m in t:
                if (m.isfile() or m.isdir()) and (members is None or members(m.name)):
                    t.extract(m, dest, filter="data")
    marker.write_text(dt.datetime.now().isoformat(), encoding="utf-8")
    return dest


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def to_wav16k(src: Path, dest: Path, start_s: float | None = None, dur_s: float | None = None) -> None:
    """Any audio -> 16 kHz mono PCM16 WAV."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg(), "-y", "-loglevel", "error"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(src)]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg could not convert {src.name}: {p.stderr.strip()[-300:]}")


def write_outputs(name: str, manifest: pd.DataFrame, info: dict, extra: dict[str, pd.DataFrame] | None = None) -> dict:
    """Save manifest.parquet (+ extra tables) and info.json; returns info."""
    _, _, proc = dirs(name)
    for col in ("item_id", "subject_id", "split_official", "path"):
        if col not in manifest:
            raise ValueError(f"{name} manifest is missing column {col}")
    if manifest.item_id.duplicated().any():
        raise ValueError(f"{name} manifest has duplicate item_id values")
    manifest.to_parquet(proc / "manifest.parquet", index=False)
    for fname, df in (extra or {}).items():
        df.to_parquet(proc / fname, index=False)
    info = {
        "name": name, "status": "converted", "converted_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_items": int(len(manifest)), "n_subjects": int(manifest.subject_id.nunique()),
        "splits_official": {str(k): int(v) for k, v in manifest.split_official.value_counts().items()},
        **info,
    }
    if "duration_s" in manifest:
        info["hours"] = round(float(manifest.duration_s.sum()) / 3600, 2)
    (proc / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{name}: {info['n_items']} items, {info['n_subjects']} subjects"
          + (f", {info['hours']} h" if "hours" in info else "") + f" -> {rel(proc)}")
    return info


def find_one(root: Path, patterns: Iterable[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]
    return None
