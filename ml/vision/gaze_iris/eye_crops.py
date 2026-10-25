"""Eye-crop extractor + classical iris / openness measures.

From a frame and the detector's 5 landmarks: rotate so the eyes are level (roll correction), crop
each eye with width 0.75 x inter-ocular distance at the 60:36 aspect ratio, convert to 36x60 grayscale
and histogram-equalise (the same appearance as MPIIGaze's normalised crops).

`iris_centroid` (dark-pupil centroid) and `eye_openness` are signal processing, used as iris targets'
calibration and for blink detection.
"""
from __future__ import annotations

import cv2
import numpy as np

EYE_W, EYE_H = 60, 36


def eye_crops(frame_rgb: np.ndarray, kp: np.ndarray, scale: float = 0.75) -> tuple[np.ndarray, np.ndarray, dict]:
    """frame (H, W, 3) uint8, kp (5, 2) pixel landmarks -> left, right (36, 60) uint8 + geometry info.

    'left' is the eye on the image left (the person's right eye), matching MPIIGaze's left/right images
    after its mirroring convention is accounted for by training on both orders (flip augmentation).
    """
    le, re = kp[0].astype(np.float64), kp[1].astype(np.float64)
    d = re - le
    iod = float(np.hypot(*d))
    roll = float(np.degrees(np.arctan2(d[1], d[0])))
    centre = (le + re) / 2
    m = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), roll, 1.0)
    rot = cv2.warpAffine(frame_rgb, m, (frame_rgb.shape[1], frame_rgb.shape[0]), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(rot, cv2.COLOR_RGB2GRAY)
    w = max(iod * scale, 8.0)
    h = w * EYE_H / EYE_W
    out = []
    for e in (le, re):
        ex, ey = m @ np.array([e[0], e[1], 1.0])
        x0, y0 = int(round(ex - w / 2)), int(round(ey - h / 2))
        x1, y1 = int(round(ex + w / 2)), int(round(ey + h / 2))
        pad = max(0, -x0, -y0, x1 - gray.shape[1], y1 - gray.shape[0])
        g = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else gray
        crop = g[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
        crop = cv2.resize(crop, (EYE_W, EYE_H), interpolation=cv2.INTER_AREA)
        out.append(cv2.equalizeHist(crop))
    return out[0], out[1], {"iod_px": iod, "roll_deg": roll, "crop_w_px": w}


def iris_centroid(eye: np.ndarray) -> tuple[float, float, float]:
    """Dark-pupil centroid in a 36x60 crop -> (x, y) in 0-1 and a confidence (dark-blob compactness)."""
    g = cv2.GaussianBlur(eye, (5, 5), 0).astype(np.float32)
    inner = g[4:-4, 6:-6]
    thr = np.percentile(inner, 12)
    mask = (g <= thr).astype(np.uint8)
    mask[:3], mask[-3:], mask[:, :4], mask[:, -4:] = 0, 0, 0, 0
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return 0.5, 0.5, 0.0
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[i, cv2.CC_STAT_AREA]
    bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    conf = float(area / max(bw * bh, 1)) * float(min(1.0, area / 40))
    return float(cent[i][0] / eye.shape[1]), float(cent[i][1] / eye.shape[0]), conf


def eye_openness(eye: np.ndarray) -> float:
    """Vertical extent of the dark iris/lash region relative to crop height (drops sharply during a blink)."""
    g = cv2.GaussianBlur(eye, (3, 3), 0).astype(np.float32)
    col = g[:, 15:45]
    thr = np.percentile(col, 20)
    rows = (col <= thr).mean(axis=1) > 0.15
    return float(rows.sum() / eye.shape[0])
