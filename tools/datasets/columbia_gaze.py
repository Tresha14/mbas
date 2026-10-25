"""Columbia Gaze Data Set -> resized images + manifest (gaze on/off-camera, Stage 4).

Manual access: request it at https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/
and download "Columbia Gaze Data Set.zip". 56 people x 5 head poses x 21 gaze
directions. Filenames encode the labels: 0001_2m_-15P_10V_-5H.jpg =
subject 1, 2 m, head pose -15 deg, gaze vertical 10 deg, horizontal -5 deg.

    python -m tools.datasets.columbia_gaze --archive "D:\\Downloads\\Columbia Gaze Data Set.zip"
Output: data/processed/public/columbia_gaze/{images/<subject>/<file>.jpg, manifest.parquet}
looking_at_camera = gaze 0V 0H (the dataset's eye-contact condition).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import pandas as pd

from tools.datasets.common import dirs, extract, fetch_or_copy, rel, write_outputs

NAME = "columbia_gaze"
PATTERN = re.compile(r"(\d{4})_(\d+)m_(-?\d+)P_(-?\d+)V_(-?\d+)H", re.I)
HELP = ("Request access at https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/ , download the zip and run\n"
        "  python -m tools.datasets.columbia_gaze --archive <path to Columbia Gaze Data Set.zip>")


def parse_name(name: str) -> dict | None:
    m = PATTERN.search(name)
    if not m:
        return None
    s, dist, p, v, h = m.groups()
    return {"subject_id": s, "distance_m": int(dist), "head_pose_deg": int(p), "gaze_v_deg": int(v),
            "gaze_h_deg": int(h)}


def convert(extracted: Path, out: Path, max_width: int = 1280) -> pd.DataFrame:
    rows = []
    for img in sorted(extracted.rglob("*.jpg")):
        lab = parse_name(img.name)
        if lab is None:
            continue
        dest = out / "images" / lab["subject_id"] / img.name
        if not dest.exists():
            im = cv2.imread(str(img))
            if im is None:
                continue
            if im.shape[1] > max_width:
                im = cv2.resize(im, (max_width, round(im.shape[0] * max_width / im.shape[1])), interpolation=cv2.INTER_AREA)
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"item_id": img.stem, "split_official": "all", "path": rel(dest), **lab,
                     "looking_at_camera": lab["gaze_v_deg"] == 0 and lab["gaze_h_deg"] == 0})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", help="Columbia Gaze Data Set.zip (manual download)")
    ap.add_argument("--source", help="already extracted folder instead of --archive")
    ap.add_argument("--max-width", type=int, default=1280, help="originals are 5184 px wide")
    args = ap.parse_args(argv)
    _, extracted, proc = dirs(NAME)
    src = Path(args.source) if args.source else extract(fetch_or_copy(NAME, None, args.archive, manual_help=HELP), extracted)
    m = convert(src, proc, args.max_width)
    if m.empty:
        raise SystemExit(f"no Columbia Gaze images found under {src}")
    write_outputs(NAME, m, {
        "title": "Columbia Gaze", "used_for": "gaze / eye-contact CNN (Stage 4)", "license": "research only",
        "source": "https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/",
        "looking_at_camera": int(m.looking_at_camera.sum()),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
