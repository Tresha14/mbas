"""Optional pre-trained face tracking: MediaPipe FaceLandmarker behind the same interface as VisionPipeline.

This is a PRE-TRAINED model (Google MediaPipe, Apache-2.0), not one we trained. It is off by default (MBAS_FACE=own keeps
our own face / head-pose / gaze models); select it with MBAS_FACE=mediapipe. Why: our small face detector loses the face
often (about 41-58 % of frames on a real clip of a candidate wearing glasses; AP 0.857 / recall 0.735 on WIDER) and our
eye-openness estimate loses blinks behind glasses. MediaPipe found the face in 100 % of the same frames and gives
blink scores (blendshapes) and a 3-D head pose.

It produces the same FrameResult as VisionPipeline (face, box, head pose, gaze, iris, on-screen, openness, motion), so the
5 s window features, blink / glance / reading signals and the engagement / attention models work unchanged:

  face            a face found (the largest of up to two is the candidate)
  head            yaw, pitch, roll in degrees from the face's 3-D transformation matrix
  gaze            (pitch, yaw) radians = head direction + where the eyes look (eyeLook* blendshapes)
                  yaw is positive towards the subject's left, pitch is negative when looking down (the convention
                  ScreenCalibration uses)
  openness        1 - 2.2 * blink score, so a blink behind glasses (score about 0.3-0.5) still dips well below 45 % of
                  the window median, which is how window_signals counts blinks
  iris            position of each iris inside its eye (0..1), for the iris-movement signal

Needs the model file checkpoints/mediapipe/face_landmarker.task (3.7 MB) and `pip install --no-deps mediapipe matplotlib`
(--no-deps: mediapipe's metadata asks for opencv-contrib-python, which would sit on top of the OpenCV already installed and
can break it; the face-landmark API does not use OpenCV).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ml.common.config import CHECKPOINTS
from ml.vision.gaze_iris.signals import ScreenCalibration
from ml.vision.pipeline import FrameResult, VisionPipeline

MODEL = CHECKPOINTS / "mediapipe" / "face_landmarker.task"
YAW_EYE_RAD, PITCH_EYE_RAD = 0.60, 0.45          # a blendshape of 1.0 is about 35 / 25 degrees of eye rotation
BLINK_GAIN = 2.2
# landmark indices of MediaPipe's 478-point mesh
IRIS_A, IRIS_B = 468, 473                          # iris centres (subject's right eye first)
EYE_A = {"outer": 33, "inner": 133, "up": 159, "down": 145}
EYE_B = {"outer": 263, "inner": 362, "up": 386, "down": 374}
NOSE, MOUTH_L, MOUTH_R = 1, 61, 291


def available() -> tuple[bool, str]:
    """(True, '') when MediaPipe and its model file can be used, else (False, why)."""
    if not MODEL.exists():
        return False, f"{MODEL} is missing (copy face_landmarker.task there)"
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"mediapipe is not installed ({type(e).__name__}); pip install --no-deps mediapipe matplotlib"
    return True, ""


def _bs(blendshapes) -> dict[str, float]:
    return {c.category_name: float(c.score) for c in blendshapes}


def _eye_ratio(L, iris: int, eye: dict) -> tuple[float, float]:
    """Where the iris sits inside its eye opening: (horizontal, vertical), each 0..1 (0.5 = centred)."""
    xs = sorted((L[eye["outer"]].x, L[eye["inner"]].x))
    ys = sorted((L[eye["up"]].y, L[eye["down"]].y))
    w, h = max(xs[1] - xs[0], 1e-6), max(ys[1] - ys[0], 1e-6)
    return float(np.clip((L[iris].x - xs[0]) / w, 0, 1)), float(np.clip((L[iris].y - ys[0]) / h, 0, 1))


class MediaPipePipeline(VisionPipeline):
    """Drop-in for VisionPipeline: same process() and window_features(), different per-frame detector."""

    def __init__(self, calibration: ScreenCalibration | None = None, fps: float = 15.0):
        ok, why = available()
        if not ok:
            raise RuntimeError(why)
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO, num_faces=2,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))
        self.cal = calibration or ScreenCalibration()
        self.fps = fps
        self.prev_pts: np.ndarray | None = None
        self.window: deque[FrameResult] = deque(maxlen=int(5 * fps))
        self._window_models = None
        self._last_ms = -1

    def process(self, frame_rgb: np.ndarray, t_s: float) -> FrameResult:
        from scipy.spatial.transform import Rotation

        H, W = frame_rgb.shape[:2]
        self._last_ms = max(self._last_ms + 1, int(t_s * 1000))                 # VIDEO mode needs strictly increasing times
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        res = self._lm.detect_for_video(img, self._last_ms)
        if not res.face_landmarks:
            r = FrameResult(t_s, False)
            self.window.append(r)
            self.prev_pts = None
            return r
        areas = []
        for L in res.face_landmarks:
            xs, ys = [p.x for p in L], [p.y for p in L]
            areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
        i = int(np.argmax(areas))                                                # largest face = the candidate
        L = res.face_landmarks[i]
        px = lambda k: np.array([L[k].x * W, L[k].y * H])  # noqa: E731
        xs, ys = np.array([p.x for p in L]) * W, np.array([p.y for p in L]) * H
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        kp = np.stack([px(IRIS_A), px(IRIS_B), px(NOSE), px(MOUTH_L), px(MOUTH_R)])

        yaw, pitch, roll = Rotation.from_matrix(np.asarray(res.facial_transformation_matrixes[i])[:3, :3]).as_euler("yxz", degrees=True)
        head = np.array([yaw, pitch, roll], np.float32)                          # yaw + = subject's left, pitch + = looking down

        b = _bs(res.face_blendshapes[i]) if res.face_blendshapes else {}
        eye_yaw = (np.mean([b.get("eyeLookOutLeft", 0.0), b.get("eyeLookInRight", 0.0)])
                   - np.mean([b.get("eyeLookInLeft", 0.0), b.get("eyeLookOutRight", 0.0)]))
        eye_pitch = (np.mean([b.get("eyeLookUpLeft", 0.0), b.get("eyeLookUpRight", 0.0)])
                     - np.mean([b.get("eyeLookDownLeft", 0.0), b.get("eyeLookDownRight", 0.0)]))
        gaze = np.array([-np.radians(pitch) + PITCH_EYE_RAD * eye_pitch, np.radians(yaw) + YAW_EYE_RAD * eye_yaw], np.float32)

        ia, ib = _eye_ratio(L, IRIS_A, EYE_A), _eye_ratio(L, IRIS_B, EYE_B)
        iris = np.array([ia[0], ia[1], ib[0], ib[1]], np.float32)
        blink = float(np.mean([b.get("eyeBlinkLeft", 0.0), b.get("eyeBlinkRight", 0.0)]))
        openness = float(np.clip(1.0 - BLINK_GAIN * blink, 0.0, 1.0))
        face_w = max(box[2] - box[0], 1.0)
        motion = float(np.clip(np.abs(kp - self.prev_pts).mean() / face_w * 20, 0, 1)) if self.prev_pts is not None else 0.0
        self.prev_pts = kp
        r = FrameResult(t_s, True, box, kp, 1.0, head, gaze, iris, [tuple(kp[0]), tuple(kp[1])],
                        bool(self.cal.on_screen(gaze[None])[0]), openness, motion)
        self.window.append(r)
        return r
