"""Geometric head pose from the 5 detector landmarks (solvePnP on a generic 3-D face), no learning.

Used (a) to make yaw/pitch/roll training labels for the head-pose CNN from WIDER FACE landmarks and
(b) as a fallback at run time. Angles in degrees: yaw + = face turned to the image right,
pitch + = face tilted up, roll + = clockwise in the image.
"""
from __future__ import annotations

import cv2
import numpy as np

# OpenCV camera convention (x right, y down, z away from the camera), millimetres, nose tip at the origin.
# Order = detector order: image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner.
FACE_3D = np.array([[-32.0, -34.0, 30.0], [32.0, -34.0, 30.0], [0.0, 0.0, 0.0],
                    [-25.0, 32.0, 22.0], [25.0, 32.0, 22.0]], dtype=np.float64)


def head_pose(kp: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float, float] | None:
    """kp (5, 2) pixel landmarks, image (w, h) -> (yaw, pitch, roll) degrees, or None if PnP fails."""
    w, h = image_size
    f = float(max(w, h))
    cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(FACE_3D, kp.astype(np.float64).reshape(5, 1, 2), cam, None,
                                  flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(FACE_3D, kp.astype(np.float64).reshape(5, 1, 2), cam, None, rvec, tvec,
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:
        return None
    r, _ = cv2.Rodrigues(rvec)
    # face forward axis in camera coords is -z of the model; express rotation as yaw (y), pitch (x), roll (z)
    yaw = float(np.degrees(np.arctan2(-r[2, 0], np.hypot(r[2, 1], r[2, 2]))))
    pitch = float(np.degrees(np.arctan2(r[2, 1], r[2, 2])))
    roll = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    pitch = ((pitch + 180) % 360) - 180
    return -yaw, -pitch, roll


def roll_from_eyes(kp: np.ndarray) -> float:
    d = kp[1] - kp[0]
    return float(np.degrees(np.arctan2(d[1], d[0])))
