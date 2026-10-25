"""Head-pose CNN (random init): 64x64 face crop -> yaw, pitch, roll (degrees).

Labels: geometric solvePnP estimate from the 5 landmarks (headpose/geometry.py) on WIDER FACE faces,
plus our scripted segments (look at camera = 0, 0, 0) once recorded. The CNN learns a smoother,
landmark-free estimate that also works when a landmark is occluded.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ml.common.config import PROCESSED, abspath
from ml.common.datasets import _hash01, manifest
from ml.common.video_features import Augment
from ml.vision.headpose.geometry import head_pose

LABELS = PROCESSED / "headpose" / "wider_labels.parquet"
CROP = 64


class HeadPoseNet(nn.Module):
    def __init__(self, width: int = 32, dropout: float = 0.2):
        super().__init__()

        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(3, width), blk(width, 2 * width), blk(2 * width, 4 * width),
                                 blk(4 * width, 4 * width), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width, 64), nn.ReLU(inplace=True), nn.Linear(64, 3))

    def forward(self, x):
        """x (B, 3, 64, 64) in [-1, 1] -> (B, 3) yaw, pitch, roll in degrees."""
        return self.head(self.net(x)) * 30.0


def face_crop(img: np.ndarray, box_xywh, scale: float = 1.25, size: int = CROP) -> np.ndarray:
    """Square crop around the face box (scale x the longer side), resized to size x size."""
    x, y, w, h = box_xywh
    s = max(w, h) * scale
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    x1, y1 = int(round(cx + s / 2)), int(round(cy + s / 2))
    H, W = img.shape[:2]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = img[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def build_labels(min_face: int = 40, max_abs: float = 75.0) -> pd.DataFrame:
    """Geometric yaw/pitch/roll for every WIDER train face with landmarks and >= min_face px."""
    m = manifest("wider_face")
    m = m[m.split_official == "train"]
    rows = []
    for r in m.itertuples(index=False):
        faces = json.loads(r.landmarks)
        if not faces:
            continue
        size = None
        for k, f in enumerate(faces):
            if f[14] <= 0 or min(f[2], f[3]) < min_face:
                continue
            if size is None:
                im = cv2.imread(str(abspath(r.path)))
                size = (im.shape[1], im.shape[0])
            kp = np.array(f[4:14], np.float64).reshape(5, 2)
            pose = head_pose(kp, size)
            if pose is None or max(abs(v) for v in pose) > max_abs:
                continue
            rows.append({"item_id": f"{r.item_id}#{k}", "path": r.path, "x": f[0], "y": f[1], "w": f[2], "h": f[3],
                         "yaw": pose[0], "pitch": pose[1], "roll": pose[2], "event": r.event})
    df = pd.DataFrame(rows)
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LABELS, index=False)
    return df


class HeadPoseDataset(Dataset):
    """WIDER faces with geometric pose labels; split by image (90/5/5 by hash, same image never in two splits)."""

    def __init__(self, split: str, augment: bool = False, limit: int | None = None):
        df = pd.read_parquet(LABELS) if LABELS.exists() else build_labels()
        h = df.item_id.str.split("#").str[0].map(lambda k: _hash01(k, "pose"))
        rng = {"train": (0, 0.9), "val": (0.9, 0.95), "test": (0.95, 1.01)}[split]
        self.df = df[(h >= rng[0]) & (h < rng[1])].reset_index(drop=True)
        if limit:
            self.df = self.df.iloc[:limit]
        self.augment = augment
        self.aug = Augment(rotate_deg=0.0, blur_p=0.3)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(abspath(r.path))), cv2.COLOR_BGR2RGB)
        box = [r.x, r.y, r.w, r.h]
        y = np.array([r.yaw, r.pitch, r.roll], np.float32)
        if self.augment:  # box jitter like a detector's output
            j = self.aug.rng.uniform(-0.08, 0.08, 4) * max(r.w, r.h)
            box = [r.x + j[0], r.y + j[1], r.w + j[2], r.h + j[3]]
        crop = face_crop(img, box)
        if self.augment:
            crop = self.aug.photometric(crop)
            if self.aug.flip():
                crop = np.ascontiguousarray(crop[:, ::-1])
                y[0], y[2] = -y[0], -y[2]
        x = crop.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        return torch.from_numpy(x), torch.from_numpy(y)


def load(path: Path | None = None, dev="cpu") -> HeadPoseNet:
    from ml.common.config import CHECKPOINTS

    st = torch.load(path or CHECKPOINTS / "headpose" / "best.pt", map_location="cpu", weights_only=False)
    m = HeadPoseNet(st["cfg"]["model"]["width"], st["cfg"]["model"]["dropout"])
    m.load_state_dict(st["model"])
    return m.to(dev).eval()
