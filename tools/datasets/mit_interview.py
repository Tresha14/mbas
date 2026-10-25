"""MIT Interview Dataset -> per-interview manifest with ratings and transcripts (Stage 8).

Manual access: request it from the authors (Naim, Tanveer, Gildea, Hoque,
"Automated Analysis and Prediction of Job Interview Performance", IEEE TAC 2018;
https://roc-hci.com/past-projects/automated-prediction-of-job-interview-performance/).
138 mock interviews of 69 MIT students (two rounds each, PX and PPX), with
Mechanical Turk ratings (Overall, RecommendHiring, Engaged, EyeContact, ...).

The converter looks for, anywhere under the extracted folder:
  * a ratings CSV with columns Participant + Overall (e.g. turker_scores_full_interview.csv);
    rows from worker "AGGR" are used when present, otherwise workers are averaged;
  * a transcripts CSV with a participant id column and a transcript column
    (e.g. interview_transcripts_by_turkers.csv, "|"-separated turns);
  * media files whose name is the participant id (P1.avi, PP1.wav, ...).

    python -m tools.datasets.mit_interview --archive D:\\MITInterview.zip
    python -m tools.datasets.mit_interview --source D:\\MITInterview
Output: data/processed/public/mit_interview/manifest.parquet
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from tools.datasets.common import dirs, extract, fetch_or_copy, rel, write_outputs

NAME = "mit_interview"
PID = re.compile(r"^(PP?\d+)$", re.I)
HELP = ("Request the MIT Interview Dataset from the authors (see this script's docstring), then run\n"
        "  python -m tools.datasets.mit_interview --archive <zip>   or   --source <folder>")


def _norm_pid(x) -> str | None:
    m = PID.match(str(x).strip())
    return m.group(1).upper() if m else None


def find_ratings(root: Path) -> pd.DataFrame:
    for f in sorted(root.rglob("*.csv")):
        df = pd.read_csv(f, encoding_errors="replace")
        cols = {c.strip().lower(): c for c in df.columns}
        if "participant" in cols and "overall" in cols:
            df = df.rename(columns={v: k for k, v in cols.items()})
            df["participant"] = df["participant"].map(_norm_pid)
            df = df.dropna(subset=["participant"])
            if "worker" in df and (df["worker"].astype(str).str.upper() == "AGGR").any():
                df = df[df["worker"].astype(str).str.upper() == "AGGR"]
            num = df.drop(columns=[c for c in ("worker",) if c in df]).groupby("participant").mean(numeric_only=True)
            return num.add_prefix("rating_").reset_index()
    raise SystemExit(f"no ratings CSV (columns Participant, Overall) under {root}")


def find_transcripts(root: Path) -> pd.DataFrame:
    for f in sorted(root.rglob("*.csv")):
        if "transcript" not in f.name.lower():
            continue
        df = pd.read_csv(f, encoding_errors="replace", header=None)
        if df.shape[1] < 2:
            continue
        df = df.iloc[:, :2]
        df.columns = ["participant", "transcript"]
        df["participant"] = df["participant"].map(_norm_pid)
        df = df.dropna(subset=["participant"])
        if len(df):
            df["transcript"] = df["transcript"].astype(str)
            df["n_turns"] = df["transcript"].str.count(r"\|") + 1
            return df.drop_duplicates("participant")
    return pd.DataFrame(columns=["participant", "transcript", "n_turns"])


def find_media(root: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for f in root.rglob("*"):
        pid = _norm_pid(f.stem)
        if pid and f.suffix.lower() in (".avi", ".mp4", ".mov", ".wav", ".mp3"):
            kind = "video" if f.suffix.lower() in (".avi", ".mp4", ".mov") else "audio"
            out.setdefault(pid, {})[kind] = rel(f)
    return out


def convert(root: Path) -> pd.DataFrame:
    r = find_ratings(root)
    t = find_transcripts(root)
    media = find_media(root)
    m = r.merge(t, on="participant", how="left")
    m["video_path"] = m.participant.map(lambda p: media.get(p, {}).get("video"))
    m["audio_path"] = m.participant.map(lambda p: media.get(p, {}).get("audio"))
    m["path"] = m.video_path.fillna(m.audio_path).fillna("")
    # P12 (round 1) and PP12 (round 2) are the same person
    m["subject_id"] = m.participant.str.replace(r"^PP", "P", regex=True)
    m["round"] = m.participant.str.startswith("PP").map({True: 2, False: 1})
    return m.rename(columns={"participant": "item_id"}).assign(split_official="all")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive")
    ap.add_argument("--source")
    args = ap.parse_args(argv)
    _, extracted, _ = dirs(NAME)
    src = Path(args.source) if args.source else extract(fetch_or_copy(NAME, None, args.archive, manual_help=HELP), extracted)
    m = convert(src)
    write_outputs(NAME, m, {
        "title": "MIT Interview", "used_for": "answer-quality + fusion ratings (Stage 8)", "license": "research only, on request",
        "source": "https://roc-hci.com/past-projects/automated-prediction-of-job-interview-performance/",
        "with_transcript": int(m.transcript.notna().sum()), "with_media": int((m.path != "").sum()),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
