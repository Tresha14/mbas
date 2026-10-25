"""MPIIGaze (normalised eye images) -> per-person .npz + manifest (gaze CNN, Stage 4).

Downloads MPIIGaze.tar.gz (2.1 GB) and extracts only Data/Normalized and the
Evaluation Subset. Each Data/Normalized/pXX/dayYY.mat holds 36x60 grayscale
left/right eye crops, 3-D gaze vectors and head-pose rotation vectors in the
normalised camera space. Converted to pitch/yaw angles in radians
(pitch = asin(-y), yaw = atan2(-x, -z)).

    python -m tools.datasets.mpiigaze
Output: data/processed/public/mpiigaze/{pXX.npz, manifest.parquet}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io

from tools.datasets.common import dirs, download, extract, rel, write_outputs

NAME = "mpiigaze"
URL = "http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz"
SIZE = 2102164307


def gaze_to_angles(v: np.ndarray) -> np.ndarray:
    """(N, 3) gaze vectors -> (N, 2) [pitch, yaw] radians."""
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.stack([np.arcsin(-v[:, 1]), np.arctan2(-v[:, 0], -v[:, 2])], axis=1).astype(np.float32)


def pose_to_angles(r: np.ndarray) -> np.ndarray:
    """(N, 3) rotation vectors -> (N, 2) head [pitch, yaw] radians."""
    out = np.zeros((len(r), 2), dtype=np.float32)
    for i, rv in enumerate(r):
        m, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64).reshape(3, 1))
        z = m[:, 2]
        out[i] = [np.arcsin(z[1]), np.arctan2(z[0], z[2])]
    return out


def _side(data, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = getattr(data, side)
    img = np.asarray(s.image, dtype=np.uint8)
    gaze = np.atleast_2d(np.asarray(s.gaze, dtype=np.float64))
    pose = np.atleast_2d(np.asarray(s.pose, dtype=np.float64))
    if img.ndim == 2:  # a day with a single sample
        img = img[None]
    return img, gaze, pose


def convert(extracted: Path, out: Path) -> pd.DataFrame:
    norm = next(extracted.rglob("Normalized"))
    rows = []
    for pdir in sorted(p for p in norm.iterdir() if p.is_dir()):
        parts: dict[str, list] = {k: [] for k in ("li", "ri", "lg", "rg", "lp", "rp", "day", "idx")}
        for mat in sorted(pdir.glob("day*.mat")):
            d = scipy.io.loadmat(str(mat), squeeze_me=True, struct_as_record=False)["data"]
            li, lg, lp = _side(d, "left")
            ri, rg, rp = _side(d, "right")
            n = len(li)
            parts["li"].append(li), parts["ri"].append(ri)
            parts["lg"].append(gaze_to_angles(lg)), parts["rg"].append(gaze_to_angles(rg))
            parts["lp"].append(pose_to_angles(lp)), parts["rp"].append(pose_to_angles(rp))
            parts["day"] += [mat.stem] * n
            parts["idx"] += list(range(n))
        if not parts["li"]:
            continue
        npz = out / f"{pdir.name}.npz"
        arr = {k: np.concatenate(parts[k]) for k in ("li", "ri", "lg", "rg", "lp", "rp")}
        np.savez_compressed(npz, left_image=arr["li"], right_image=arr["ri"], left_gaze=arr["lg"],
                            right_gaze=arr["rg"], left_head=arr["lp"], right_head=arr["rp"],
                            day=np.array(parts["day"]), index=np.array(parts["idx"]))
        for row, (day, idx) in enumerate(zip(parts["day"], parts["idx"])):
            rows.append({
                "item_id": f"{pdir.name}/{day}/{idx:04d}", "subject_id": pdir.name, "split_official": "all",
                "path": rel(npz), "row": row, "day": day,
                "gaze_pitch_deg": float(np.degrees(arr["lg"][row, 0])), "gaze_yaw_deg": float(np.degrees(arr["lg"][row, 1])),
                "head_pitch_deg": float(np.degrees(arr["lp"][row, 0])), "head_yaw_deg": float(np.degrees(arr["lp"][row, 1])),
            })
        print(f"  {pdir.name}: {len(parts['idx'])} samples")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", help="use an already downloaded MPIIGaze.tar.gz")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args(argv)
    downloads, extracted, proc = dirs(NAME)
    if not args.skip_download:
        arc = Path(args.archive) if args.archive else download(URL, downloads / "MPIIGaze.tar.gz", SIZE)
        extract(arc, extracted, members=lambda n: "/Data/Normalized" in "/" + n or "Evaluation Subset" in n)
    m = convert(extracted, proc)
    write_outputs(NAME, m, {
        "title": "MPIIGaze", "used_for": "gaze CNN pre-training (Stage 4)",
        "license": "CC BY-NC-SA 4.0", "source": URL,
        "note": "all samples look at the laptop screen (on_screen); leave-one-person-out evaluation is standard",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
