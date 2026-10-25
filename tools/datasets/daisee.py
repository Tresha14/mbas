"""DAiSEE -> clip manifest with engagement / boredom / confusion / frustration (Stage 5).

Manual access: request DAiSEE at https://people.iith.ac.in/vineethnb/resources/daisee/index.html
and download DAiSEE.zip. Layout: DataSet/{Train,Validation,Test}/<subject>/<clip>/<clip>.avi
and Labels/AllLabels.csv (ClipID, Boredom, Engagement, Confusion, Frustration; levels 0-3).

Our 3 engagement classes: level 0-1 -> disengaged, 2 -> neutral, 3 -> engaged
(raw levels are kept so Stage 5 can choose another mapping).

    python -m tools.datasets.daisee --archive D:\\Downloads\\DAiSEE.zip
    python -m tools.datasets.daisee --source D:\\DAiSEE          # already extracted
Output: data/processed/public/daisee/manifest.parquet (videos stay in data/raw/public/daisee)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tools.datasets.common import dirs, extract, fetch_or_copy, rel, write_outputs

NAME = "daisee"
SPLIT = {"train": "train", "validation": "val", "test": "test"}
ENGAGEMENT_3 = {0: "disengaged", 1: "disengaged", 2: "neutral", 3: "engaged"}
HELP = ("Request DAiSEE at https://people.iith.ac.in/vineethnb/resources/daisee/index.html , then run\n"
        "  python -m tools.datasets.daisee --archive <path to DAiSEE.zip>")


def load_labels(root: Path) -> pd.DataFrame:
    # AllLabels.csv covers every clip; otherwise combine Train/Validation/TestLabels.csv
    files = sorted(root.rglob("AllLabels.csv"))[:1] or sorted(root.rglob("*Labels.csv"))
    if not files:
        raise SystemExit(f"no DAiSEE label CSVs under {root}")
    lab = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    lab.columns = [c.strip().lower() for c in lab.columns]
    lab["clip"] = lab["clipid"].astype(str).str.strip().str.rsplit(".", n=1).str[0]
    return lab.drop_duplicates("clip").set_index("clip")


def convert(root: Path) -> pd.DataFrame:
    lab = load_labels(root)
    rows = []
    for v in sorted(p for p in root.rglob("*") if p.suffix.lower() in (".avi", ".mp4")):
        clip = v.stem
        if clip not in lab.index:
            continue
        split = next((SPLIT[p.lower()] for p in v.parts if p.lower() in SPLIT), "unknown")
        subject = v.parent.parent.name
        r = lab.loc[clip]
        eng = int(r["engagement"])
        rows.append({
            "item_id": clip, "subject_id": subject, "split_official": split, "path": rel(v),
            "engagement_level": eng, "boredom_level": int(r["boredom"]), "confusion_level": int(r["confusion"]),
            "frustration_level": int(r["frustration"]), "engagement_label": ENGAGEMENT_3[eng],
            "duration_s": 10.0,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", help="DAiSEE.zip (manual download)")
    ap.add_argument("--source", help="already extracted DAiSEE folder")
    args = ap.parse_args(argv)
    _, extracted, _ = dirs(NAME)
    src = Path(args.source) if args.source else extract(fetch_or_copy(NAME, None, args.archive, manual_help=HELP), extracted)
    m = convert(src)
    if m.empty:
        raise SystemExit(f"no labelled DAiSEE clips found under {src}")
    write_outputs(NAME, m, {
        "title": "DAiSEE", "used_for": "engagement CNN-GRU pre-training (Stage 5)", "license": "research only",
        "source": "https://people.iith.ac.in/vineethnb/resources/daisee/index.html",
        "engagement_3class": {k: int(v) for k, v in m.engagement_label.value_counts().items()},
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
