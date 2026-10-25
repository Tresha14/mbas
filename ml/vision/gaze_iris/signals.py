"""Per-window attention signals from per-frame gaze (Stage 4) and the on-screen calibration.

Per 5 s window (15 fps -> 75 frames):
  off_screen_pct     share of frames whose gaze falls outside the calibrated screen region
  glance_away_count  off-screen runs >= 0.3 s
  saccade_rate_hz    rapid eye movements (angular speed > 60 deg/s) per second
  blink_rate_per_min eye-openness dips below 45% of the window median
  reading_flag       >= 3 left-to-right sweeps with fast returns (reading from another screen/paper)
  iris_movement_std  std of iris position
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScreenCalibration:
    """Gaze-angle box of the screen, from the 60 s moving-dot calibration (or defaults for a laptop)."""

    yaw_min: float = -0.42      # radians, about +-24 deg horizontally for a 14" screen at 60 cm
    yaw_max: float = 0.42
    pitch_min: float = -0.45    # looking down at the screen centre is negative pitch
    pitch_max: float = 0.20
    margin: float = 0.06

    def on_screen(self, gaze: np.ndarray) -> np.ndarray:
        p, y = gaze[:, 0], gaze[:, 1]
        m = self.margin
        return (y >= self.yaw_min - m) & (y <= self.yaw_max + m) & (p >= self.pitch_min - m) & (p <= self.pitch_max + m)

    @classmethod
    def fit(cls, gaze_on_dot: np.ndarray, q: float = 2.0) -> "ScreenCalibration":
        """Robust box around the gaze predicted while the candidate followed the calibration dot."""
        lo = np.percentile(gaze_on_dot, q, axis=0)
        hi = np.percentile(gaze_on_dot, 100 - q, axis=0)
        return cls(yaw_min=float(lo[1]), yaw_max=float(hi[1]), pitch_min=float(lo[0]), pitch_max=float(hi[0]))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def window_signals(gaze: np.ndarray, on_screen: np.ndarray, openness: np.ndarray | None = None,
                   iris: np.ndarray | None = None, fps: float = 15.0) -> dict:
    """gaze (T, 2) radians, on_screen (T,) bool, openness (T,), iris (T, 2) -> window features."""
    T = len(gaze)
    dur = T / fps if T else 1.0
    off = ~on_screen.astype(bool)
    glances = [r for r in _runs(off) if (r[1] - r[0]) / fps >= 0.3]
    speed = np.degrees(np.linalg.norm(np.diff(gaze, axis=0), axis=1)) * fps if T > 1 else np.zeros(0)
    sacc = len(_runs(speed > 60.0))
    blinks = 0
    if openness is not None and len(openness) > 2:
        med = np.median(openness)
        blinks = len([r for r in _runs(openness < 0.45 * med) if (r[1] - r[0]) / fps <= 0.5]) if med > 0 else 0
    return {
        "off_screen_pct": round(float(off.mean()) * 100, 2) if T else 0.0,
        "glance_away_count": len(glances),
        "saccade_rate_hz": round(sacc / dur, 3),
        "blink_rate_per_min": round(blinks / dur * 60, 2),
        "reading_flag": int(reading_pattern(gaze[:, 1], fps)),
        "gaze_yaw_std_deg": round(float(np.degrees(gaze[:, 1].std())), 3) if T else 0.0,
        "gaze_pitch_std_deg": round(float(np.degrees(gaze[:, 0].std())), 3) if T else 0.0,
        "iris_movement_std": round(float(iris.std(axis=0).mean()), 4) if iris is not None and len(iris) else 0.0,
    }


def reading_pattern(yaw: np.ndarray, fps: float = 15.0, min_sweeps: int = 3) -> bool:
    """Reading = slow sweeps in one direction followed by fast returns (saw-tooth yaw), repeated."""
    if len(yaw) < fps:
        return False
    y = np.convolve(yaw, np.ones(3) / 3, mode="same")
    v = np.diff(y) * fps                          # rad/s
    slow = (v > 0.05) & (v < 0.8)
    fast_back = v < -1.2
    sweeps = 0
    i = 0
    while i < len(v):
        j = i
        while j < len(v) and slow[j]:
            j += 1
        if (j - i) / fps >= 0.25 and j < len(v) and fast_back[j:j + 3].any():
            sweeps += 1
            i = j + 3
        else:
            i = max(j, i + 1)
    return sweeps >= min_sweeps
