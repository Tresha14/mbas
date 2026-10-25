"""WIDER FACE -> face boxes manifest (face detector pre-training, Stage 4).

Downloads WIDER_train.zip, WIDER_val.zip and wider_face_split.zip (raw images +
box annotations; the test split has no public boxes and is skipped).

    python -m tools.datasets.wider_face
Output: data/processed/public/wider_face/manifest.parquet (one row per image;
boxes = JSON [[x, y, w, h, blur, expression, illumination, invalid, occlusion, pose], ...])
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from tools.datasets.common import dirs, download, extract, rel, write_outputs

NAME = "wider_face"
BASE = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/"
FILES = {"WIDER_train.zip": 1465602149, "WIDER_val.zip": 362752168, "wider_face_split.zip": 3591642}


def parse_gt(path: Path) -> list[dict]:
    """wider_face_<split>_bbx_gt.txt: name, count, then max(count, 1) lines of 10 ints."""
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    out, i = [], 0
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        name, n = lines[i], int(lines[i + 1])
        rows = lines[i + 2: i + 2 + max(n, 1)]
        boxes = [[int(v) for v in r.split()[:10]] for r in rows[:n]]
        out.append({"name": name, "boxes": boxes})
        i += 2 + max(n, 1)
    return out


def convert(extracted: Path) -> pd.DataFrame:
    rows = []
    for split, folder in (("train", "WIDER_train"), ("val", "WIDER_val")):
        gt = next(extracted.rglob(f"wider_face_{split}_bbx_gt.txt"))
        img_root = next(p for p in extracted.rglob(folder) if (p / "images").is_dir()) / "images"
        for e in parse_gt(gt):
            img = img_root / e["name"]
            valid = [b for b in e["boxes"] if b[7] == 0 and b[2] > 0 and b[3] > 0]
            rows.append({
                "item_id": f"{split}/{e['name']}", "subject_id": e["name"].split("/")[0],  # event category
                "split_official": split, "path": rel(img), "event": e["name"].split("/")[0],
                "n_faces": len(e["boxes"]), "n_valid_faces": len(valid), "boxes": json.dumps(e["boxes"]),
                "exists": img.exists(),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true", help="only convert what is already extracted")
    args = ap.parse_args(argv)
    downloads, extracted, _ = dirs(NAME)
    if not args.skip_download:
        for f, size in FILES.items():
            extract(download(BASE + f, downloads / f, size), extracted)
    m = convert(extracted)
    write_outputs(NAME, m, {
        "title": "WIDER FACE", "used_for": "face detector (Stage 4)", "license": "research only (CC BY-NC-ND 4.0)",
        "source": "http://shuoyang1213.me/WIDERFACE/ (mirror: huggingface.co/datasets/CUHK-CSE/wider_face)",
        "n_boxes": int(m.n_faces.sum()), "missing_images": int((~m.exists).sum()),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
