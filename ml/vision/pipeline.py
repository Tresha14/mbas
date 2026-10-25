"""Per-frame vision pipeline + 5 s window aggregation (used by demo_webcam.py and the Stage 9 workers).

frame -> face detector (box + 5 landmarks) -> head-pose CNN (64x64 crop) -> eye crops -> gaze/iris CNN
      -> on/off-screen (calibration box), eye openness, face motion
window (5 s) -> attention signals (gaze_iris/signals.py) -> engagement MLP + attention score model

Runs on PyTorch checkpoints or the ONNX exports (backend="onnx", used in production).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from ml.common.config import CHECKPOINTS, ROOT
from ml.common.datasets import ENGAGEMENT_CLASSES
from ml.common.video_features import letterbox
from ml.vision.gaze_iris.eye_crops import eye_crops, eye_openness
from ml.vision.gaze_iris.signals import ScreenCalibration, window_signals
from ml.vision.headpose.geometry import head_pose
from ml.vision.headpose.model import face_crop

ONNX_DIR = ROOT / "ml" / "export" / "onnx"


@dataclass
class FrameResult:
    t_s: float
    face: bool
    box: np.ndarray | None = None          # pixel x1 y1 x2 y2
    landmarks: np.ndarray | None = None    # (5, 2) pixels
    score: float = 0.0
    head: np.ndarray | None = None         # yaw, pitch, roll degrees
    gaze: np.ndarray | None = None         # pitch, yaw radians
    iris: np.ndarray | None = None         # (4,) lx ly rx ry in eye crops
    iris_px: list = field(default_factory=list)
    on_screen: bool = False
    openness: float = 0.0
    motion: float = 0.0


class _Torch:
    def __init__(self, dev):
        from ml.vision.face_detector.eval import load_model as load_face
        from ml.vision.gaze_iris.eval import load_model as load_gaze
        from ml.vision.headpose.model import load as load_pose

        self.dev = dev
        self.face = load_face(dev=dev)
        self.gaze, _ = load_gaze(dev)
        self.pose = load_pose(dev=dev)

    @torch.no_grad()
    def detect(self, x):
        from ml.vision.face_detector.model import postprocess

        out = self.face(torch.from_numpy(x).to(self.dev))
        b, s, k = postprocess(*out, self.face.anchors, score_thr=0.5)[0]
        return b.cpu().numpy(), s.cpu().numpy(), k.cpu().numpy()

    @torch.no_grad()
    def headpose(self, crop):
        return self.pose(torch.from_numpy(crop).to(self.dev)).cpu().numpy()[0]

    @torch.no_grad()
    def gaze_iris(self, l, r, h):
        g, iris, _ = self.gaze(*(torch.from_numpy(a).to(self.dev) for a in (l, r, h)))
        return g.cpu().numpy()[0], iris.cpu().numpy()[0]


class _Onnx:
    def __init__(self):
        import onnxruntime as ort

        from ml.vision.face_detector.model import make_anchors

        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 2
        mk = lambda n: ort.InferenceSession(str(ONNX_DIR / f"{n}.onnx"), opt, providers=["CPUExecutionProvider"])  # noqa: E731
        self.face, self.pose, self.gaze = mk("face_detector"), mk("headpose"), mk("gaze_iris")
        self.anchors = make_anchors()

    def detect(self, x):
        from ml.vision.face_detector.model import postprocess

        lg, d, k = self.face.run(None, {"image": x})
        b, s, kp = postprocess(torch.from_numpy(lg), torch.from_numpy(d), torch.from_numpy(k), self.anchors, 0.5)[0]
        return b.numpy(), s.numpy(), kp.numpy()

    def headpose(self, crop):
        return self.pose.run(None, {"face": crop})[0][0]

    def gaze_iris(self, l, r, h):
        g, iris, _ = self.gaze.run(None, {"left": l, "right": r, "head": h})
        return g[0], iris[0]


class VisionPipeline:
    def __init__(self, backend: str = "torch", device: str | None = None, calibration: ScreenCalibration | None = None,
                 fps: float = 15.0):
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = _Onnx() if backend == "onnx" else _Torch(dev)
        self.cal = calibration or ScreenCalibration()
        self.fps = fps
        self.prev_face: np.ndarray | None = None
        self.window: deque[FrameResult] = deque(maxlen=int(5 * fps))
        self._window_models = None

    def process(self, frame_rgb: np.ndarray, t_s: float) -> FrameResult:
        H, W = frame_rgb.shape[:2]
        lb, s, (px, py) = letterbox(frame_rgb, 128)
        x = (lb.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        boxes, scores, kps = self.m.detect(x)
        if not len(boxes):
            r = FrameResult(t_s, False)
            self.window.append(r)
            self.prev_face = None
            return r
        i = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))   # largest face = candidate
        to_px = lambda v, o: (v * 128 - o) / s  # noqa: E731
        box = np.array([to_px(boxes[i, 0], px), to_px(boxes[i, 1], py), to_px(boxes[i, 2], px), to_px(boxes[i, 3], py)])
        kp = np.stack([to_px(kps[i, :, 0], px), to_px(kps[i, :, 1], py)], 1)
        xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        crop = face_crop(frame_rgb, xywh)
        head = self.m.headpose((crop.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1)[None])
        geo = head_pose(kp, (W, H))
        left, right, info = eye_crops(frame_rgb, kp)
        f = lambda e: (e.astype(np.float32) / 127.5 - 1)[None, None]  # noqa: E731
        hp = np.radians(np.array([[head[1], head[0]]], np.float32))   # (pitch, yaw) radians like MPIIGaze
        gaze, iris = self.m.gaze_iris(f(left), f(right), hp)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        motion = float(np.abs(gray - self.prev_face).mean() * 4) if self.prev_face is not None else 0.0
        self.prev_face = gray
        # iris points back to frame pixels (approximate: crop centred on each eye landmark, roll ignored)
        w = info["crop_w_px"]
        h = w * 36 / 60
        iris_px = [(kp[0, 0] + (iris[0] - 0.5) * w, kp[0, 1] + (iris[1] - 0.5) * h),
                   (kp[1, 0] + (iris[2] - 0.5) * w, kp[1, 1] + (iris[3] - 0.5) * h)]
        r = FrameResult(t_s, True, box, kp, float(scores[i]), np.asarray(head if geo is None else head), gaze, iris,
                        iris_px, bool(self.cal.on_screen(gaze[None])[0]), (eye_openness(left) + eye_openness(right)) / 2,
                        min(motion, 1.0))
        self.window.append(r)
        return r

    def window_features(self, vad_speech_ratio: float = 0.0, interviewer_speech_ratio: float = 0.0,
                        state: str = "answering") -> dict:
        frames = list(self.window)
        faces = [f for f in frames if f.face]
        n = max(len(frames), 1)
        feats = {"face_present_ratio": len(faces) / n}
        if faces:
            gaze = np.stack([f.gaze for f in faces])
            on = np.array([f.on_screen for f in faces])
            sig = window_signals(gaze, on, np.array([f.openness for f in faces]),
                                 np.stack([f.iris[:2] for f in faces]), self.fps)
            head = np.stack([f.head for f in faces])
            feats.update(on_screen_ratio=float(on.mean()) * len(faces) / n, glance_away_count=sig["glance_away_count"],
                         saccade_rate_hz=sig["saccade_rate_hz"], fixation_duration_mean_s=0.0,
                         blink_rate_per_min=sig["blink_rate_per_min"], reading_pattern_flag=sig["reading_flag"],
                         gaze_yaw_std_deg=sig["gaze_yaw_std_deg"], gaze_pitch_std_deg=sig["gaze_pitch_std_deg"],
                         iris_movement_std=sig["iris_movement_std"], head_yaw_mean_deg=float(head[:, 0].mean()),
                         head_pitch_mean_deg=float(head[:, 1].mean()), head_yaw_std_deg=float(head[:, 0].std()),
                         head_pitch_std_deg=float(head[:, 1].std()),
                         face_motion_energy=float(np.mean([f.motion for f in faces])), mouth_open_ratio=0.0)
        else:
            feats.update(on_screen_ratio=0.0, glance_away_count=0, saccade_rate_hz=0.0, fixation_duration_mean_s=0.0,
                         blink_rate_per_min=0.0, reading_pattern_flag=0, gaze_yaw_std_deg=0.0, gaze_pitch_std_deg=0.0,
                         iris_movement_std=0.0, head_yaw_mean_deg=0.0, head_pitch_mean_deg=0.0, head_yaw_std_deg=0.0,
                         head_pitch_std_deg=0.0, face_motion_energy=0.0, mouth_open_ratio=0.0)
        feats.update(vad_speech_ratio=vad_speech_ratio, interviewer_speech_ratio=interviewer_speech_ratio,
                     candidate_state=state)
        feats.update(self.engagement_attention(feats))
        return feats

    def engagement_attention(self, feats: dict) -> dict:
        from ml.vision.engagement.window_models import MLP, prepare
        import pandas as pd

        if self._window_models is None:
            p = CHECKPOINTS / "window_models" / "best.pt"
            if not p.exists():
                return {}
            st = torch.load(p, map_location="cpu", weights_only=False)
            eng = MLP(len(st["engagement_features"]), 3)
            eng.load_state_dict(st["engagement"])
            att = MLP(len(st["attention_features"]), 2, layers=0 if st["attention_kind"] == "linear" else 2)
            att.load_state_dict(st["attention"])
            self._window_models = (st, eng.eval(), att.eval())
        st, eng, att = self._window_models
        row = prepare(pd.DataFrame([feats]))
        std = lambda x, s: np.nan_to_num((x - np.array(s["mu"])) / np.array(s["sd"])).astype(np.float32)  # noqa: E731
        with torch.no_grad():
            pe = torch.softmax(eng(torch.from_numpy(std(row[st["engagement_features"]].to_numpy(float), st["engagement_std"]))), 1)[0].numpy()
            ai = np.concatenate([row[st["attention_features"][:-3]].to_numpy(float)[0], pe])[None]
            o = att(torch.from_numpy(std(ai, st["attention_std"])))[0].numpy()
        return {"engagement_label": ENGAGEMENT_CLASSES[int(pe.argmax())],
                "engagement_probs": dict(zip(ENGAGEMENT_CLASSES, np.round(pe, 3).tolist())),
                "attention_score": round(float(100 / (1 + np.exp(-o[0]))), 1),
                "attention_on_screen_p": round(float(1 / (1 + np.exp(-o[1]))), 3)}
