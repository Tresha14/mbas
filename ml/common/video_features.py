"""Video front end: decode at 15 fps, resize, normalise, and train-time augmentation.

Augmentations: brightness, contrast, blur, small rotation, horizontal flip. A flip
mirrors gaze/head yaw (and swaps left/right eyes) so labels stay correct.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

FPS = 15


def _ffmpeg() -> str:
    from tools.check_session import _ffmpeg as f

    return f()


def read_frames(path: str | Path, fps: float = FPS, size: tuple[int, int] | None = None,
                start_s: float = 0.0, duration_s: float | None = None, gray: bool = False) -> np.ndarray:
    """(N, H, W, 3) uint8 RGB (or (N, H, W) gray) frames resampled to `fps` with ffmpeg."""
    w, h = size or probe_size(path)
    cmd = [_ffmpeg(), "-loglevel", "error"]
    if start_s:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(path)]
    if duration_s:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    c = 1 if gray else 3
    n = len(raw) // (w * h * c)
    arr = np.frombuffer(raw[: n * w * h * c], np.uint8)
    return arr.reshape(n, h, w) if gray else arr.reshape(n, h, w, 3)


def probe_size(path: str | Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not w or not h:
        raise RuntimeError(f"cannot read video size of {path}")
    return w, h


def sample_clip(path: str | Path, n_frames: int = 16, size: tuple[int, int] = (112, 112),
                start_s: float = 0.0, duration_s: float | None = None) -> np.ndarray:
    """n_frames evenly spaced RGB frames from a clip (OpenCV seeking; works for avi/mp4/webm)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(fps * (duration_s or 10))
    a = int(start_s * fps)
    b = min(total, a + int((duration_s or total / fps) * fps)) if duration_s else total
    idx = np.linspace(a, max(a, b - 1), n_frames).round().astype(int)
    out = []
    last = np.zeros((size[1], size[0], 3), np.uint8)
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            last = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA)
        out.append(last)
    cap.release()
    return np.stack(out)


def normalize(frames: np.ndarray) -> np.ndarray:
    """uint8 (..., H, W, C) -> float32 (..., C, H, W) in [-1, 1]."""
    x = frames.astype(np.float32) / 127.5 - 1.0
    if x.ndim >= 3 and x.shape[-1] in (1, 3):
        x = np.moveaxis(x, -1, -3)
    return x


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size x size. Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    py, px = (size - nh) // 2, (size - nw) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, s, (px, py)


class Augment:
    """Photometric + geometric augmentation for frames and eye crops (numpy uint8, HWC or HW)."""

    def __init__(self, rng: np.random.Generator | None = None, brightness: float = 0.3, contrast: float = 0.3,
                 blur_p: float = 0.2, rotate_deg: float = 8.0, flip_p: float = 0.5):
        self.rng = rng or np.random.default_rng()
        self.brightness, self.contrast, self.blur_p = brightness, contrast, blur_p
        self.rotate_deg, self.flip_p = rotate_deg, flip_p

    def photometric(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        x = (x - x.mean()) * (1 + self.rng.uniform(-self.contrast, self.contrast)) + x.mean()
        x = x * (1 + self.rng.uniform(-self.brightness, self.brightness))
        x = np.clip(x, 0, 255).astype(np.uint8)
        if self.rng.random() < self.blur_p:
            k = int(self.rng.choice([3, 5]))
            x = cv2.GaussianBlur(x, (k, k), 0)
        return x

    def rotate(self, img: np.ndarray, deg: float | None = None) -> tuple[np.ndarray, float]:
        deg = self.rng.uniform(-self.rotate_deg, self.rotate_deg) if deg is None else deg
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT), deg

    def flip(self) -> bool:
        return bool(self.rng.random() < self.flip_p)

    def frames(self, clip: np.ndarray) -> np.ndarray:
        """Same augmentation for every frame of a clip (T, H, W, C)."""
        seed = int(self.rng.integers(1 << 31))
        out = []
        do_flip = self.flip()
        deg = self.rng.uniform(-self.rotate_deg, self.rotate_deg)
        for f in clip:
            sub = Augment(np.random.default_rng(seed), self.brightness, self.contrast, self.blur_p, self.rotate_deg, 0)
            g = sub.photometric(f)
            g, _ = self.rotate(g, deg)
            out.append(g[:, ::-1] if do_flip else g)
        return np.ascontiguousarray(np.stack(out))


def flip_gaze(pitch_yaw: np.ndarray) -> np.ndarray:
    """Horizontal flip mirrors yaw; pitch is unchanged."""
    out = np.array(pitch_yaw, dtype=np.float32, copy=True)
    out[..., 1] = -out[..., 1]
    return out


def flip_boxes(boxes: np.ndarray, width: float) -> np.ndarray:
    """x, y, w, h boxes mirrored horizontally."""
    b = np.array(boxes, dtype=np.float32, copy=True)
    if len(b):
        b[:, 0] = width - b[:, 0] - b[:, 2]
    return b
