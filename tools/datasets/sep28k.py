"""SEP-28k -> 3 s 16 kHz clips + stutter / disfluency labels (disfluency CRNN, Stage 7).

Downloads the label and episode lists from github.com/apple/ml-stuttering-events-dataset,
then each podcast episode from its original URL (some hosts are gone; those
episodes are skipped and listed in info.json), and cuts the labelled 3 s clips.

Labels: three annotators count each event; an event is present when >= 2 marked it.
Mapped to our hesitation types: Interjection -> filler, WordRep/SoundRep -> repetition,
Prolongation -> prolongation, Block -> block.

    python -m tools.datasets.sep28k                     # all episodes
    python -m tools.datasets.sep28k --max-episodes 5    # quick test
Output: data/processed/public/sep28k/{clips/<show>_<ep>_<clip>.wav, manifest.parquet}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from tools.datasets.common import dirs, download, rel, to_wav16k, write_outputs

NAME = "sep28k"
GH = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/"
SR = 16000
EVENTS = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection", "NoStutteredWords"]
OURS = {"filler": ["Interjection"], "repetition": ["WordRep", "SoundRep"], "prolongation": ["Prolongation"],
        "block": ["Block"]}


def read_lists(downloads: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = pd.read_csv(download(GH + "SEP-28k_labels.csv", downloads / "SEP-28k_labels.csv"), skipinitialspace=True)
    ep = pd.read_csv(download(GH + "SEP-28k_episodes.csv", downloads / "SEP-28k_episodes.csv"), header=None,
                     skipinitialspace=True, names=["show_name", "episode_name", "url", "show", "ep_id"])
    lab.columns = [c.strip() for c in lab.columns]
    lab["Show"] = lab["Show"].astype(str).str.strip()
    ep["show"] = ep["show"].astype(str).str.strip()
    ep["url"] = ep["url"].astype(str).str.strip()
    return lab, ep


def label_row(r) -> dict:
    out = {f"n_{e.lower()}": int(getattr(r, e)) for e in EVENTS}
    for ours, cols in OURS.items():
        out[ours] = int(any(getattr(r, c) >= 2 for c in cols))
    out["fluent"] = int(r.NoStutteredWords >= 2)
    out["unusable"] = int(r.PoorAudioQuality >= 2 or r.Music >= 2 or r.NoSpeech >= 2 or r.DifficultToUnderstand >= 2)
    return out


def _row(r, dest: Path) -> dict:
    a, b = int(r.Start), int(r.Stop)
    return {"item_id": dest.stem, "subject_id": f"{r.Show}_{int(r.EpId)}", "show": r.Show,
            "split_official": "all", "path": rel(dest), "duration_s": round((b - a) / SR, 3),
            "episode_start_s": round(a / SR, 3), **label_row(r)}


def _dest(r, out: Path) -> Path:
    return out / f"{r.Show}_{int(r.EpId)}_{int(r.ClipId)}.wav"


def cache_rows(labels: pd.DataFrame, out: Path) -> list[dict] | None:
    """Rows for an episode whose clips all exist already, else None."""
    rs = list(labels.itertuples(index=False))
    if rs and all(_dest(r, out).exists() for r in rs):
        return [_row(r, _dest(r, out)) for r in rs]
    return None


def cut_clips(episode_wav: Path, labels: pd.DataFrame, out: Path) -> list[dict]:
    audio, sr = sf.read(str(episode_wav), dtype="int16")
    assert sr == SR
    rows = []
    for r in labels.itertuples(index=False):
        a, b = int(r.Start), int(r.Stop)  # sample offsets at 16 kHz
        if b > len(audio) or b <= a:
            continue
        dest = _dest(r, out)
        if not dest.exists():
            sf.write(str(dest), audio[a:b], SR, subtype="PCM_16")
        rows.append(_row(r, dest))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--keep-episodes", action="store_true", help="keep the full-episode WAVs after cutting")
    args = ap.parse_args(argv)
    downloads, extracted, proc = dirs(NAME)
    lab, ep = read_lists(downloads)
    keys = lab[["Show", "EpId"]].drop_duplicates().itertuples(index=False)
    keys = list(keys)[: args.max_episodes] if args.max_episodes else list(keys)
    clips = proc / "clips"
    clips.mkdir(exist_ok=True)
    rows, failed = [], []
    for show, ep_id in keys:
        hit = ep[(ep.show == show) & (ep.ep_id == ep_id)]
        if hit.empty:
            failed.append(f"{show}_{ep_id}: not in episode list")
            continue
        url = hit.url.iloc[0]
        ext = Path(url.split("?")[0]).suffix or ".mp3"
        raw = downloads / "episodes" / f"{show}_{ep_id}{ext}"
        wav16 = extracted / f"{show}_{ep_id}.wav"
        ep_labels = lab[(lab.Show == show) & (lab.EpId == ep_id)]
        done = cache_rows(ep_labels, clips)
        if done is not None:  # every clip already cut on an earlier run
            rows += done
            continue
        try:
            if not wav16.exists():
                download(url, raw, retries=2)
                to_wav16k(raw, wav16)
            rows += cut_clips(wav16, ep_labels, clips)
            print(f"  {show}_{ep_id}: ok")
        except Exception as e:  # noqa: BLE001 - dead podcast hosts are expected
            failed.append(f"{show}_{ep_id}: {str(e)[:120]}")
            print(f"  {show}_{ep_id}: FAILED ({str(e)[:80]})")
        if not args.keep_episodes:  # full episodes are large; the 3 s clips are what we keep
            raw.unlink(missing_ok=True)
            wav16.unlink(missing_ok=True)
    m = pd.DataFrame(rows)
    if m.empty:
        raise SystemExit("no SEP-28k clips could be built (all episode downloads failed)")
    write_outputs(NAME, m, {
        "title": "SEP-28k", "used_for": "disfluency CRNN (Stage 7)", "license": "CC BY-NC 4.0 (labels); audio from public podcasts",
        "source": "https://github.com/apple/ml-stuttering-events-dataset",
        "episodes_ok": int(m.subject_id.nunique()), "episodes_failed": failed,
        "positives": {k: int(m[k].sum()) for k in [*OURS, "fluent", "unusable"]},
        "clip_label_rule": ">= 2 of 3 annotators", "random_baseline_positive_rate":
            {k: round(float(np.mean(m[k])), 4) for k in OURS},
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
