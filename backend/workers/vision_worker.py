"""Vision worker: frames:{id} -> per-frame gaze/iris (5 Hz to the UI) + 5 s window scores every 2.5 s.

Publishes to analytics:{id}:
  gaze    {t_s, yaw, pitch, iris, on_screen, face}            (iris/gaze mini-plot)
  window  {start_s, end_s, attention, engagement, on_screen_ratio, glances, blink, reading, ...}
  alert   {kind: off_screen | no_face | reading, t_s, text}
and stores WindowScore rows. Latency: per-frame pipeline time and capture->publish (e2e_vision).

    python -m backend.workers.vision_worker
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from backend.bus import make_bus
from backend.config import settings
from backend.config import settings
from backend.db import Database, WindowScore, clean_json
from backend.flags import raise_flag

HOP_S, WIN_S = 2.5, 5.0


class InterviewVision:
    def __init__(self, interview_id: int, bus, db):
        from ml.vision.pipeline import VisionPipeline

        backend = "onnx" if all((__import__("backend.inference", fromlist=["ONNX"]).ONNX / f"{n}.onnx").exists()
                                for n in ("face_detector", "headpose", "gaze_iris")) else "torch"
        self.pretrained = False                  # True when the pre-trained MediaPipe face tracker is in use
        self.pipe = None
        if settings().face_backend == "mediapipe":
            try:
                from ml.vision.mediapipe_pipeline import MediaPipePipeline

                self.pipe, self.pretrained = MediaPipePipeline(fps=10.0), True
            except Exception as e:  # noqa: BLE001 - never leave the candidate without face tracking
                print(f"[vision] MediaPipe unavailable ({type(e).__name__}: {e}); using our own face models", flush=True)
        if self.pipe is None:
            self.pipe = VisionPipeline(backend, fps=10.0)
        self.id, self.bus, self.db = interview_id, bus, db
        self.next_window = WIN_S
        self.last_gaze_pub = 0.0
        self.off_since: float | None = None

    def frame(self, msg: dict, jpg: bytes):
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        r = self.pipe.process(img, msg["t_s"])
        self.bus.set_metric("vision_frame", (time.perf_counter() - t0) * 1000)
        t = msg["t_s"]
        if t - self.last_gaze_pub >= 0.2:
            self.last_gaze_pub = t
            self.bus.publish(f"analytics:{self.id}", {
                "kind": "gaze", "t_s": t, "face": r.face, "on_screen": r.on_screen,
                "yaw": float(np.degrees(r.gaze[1])) if r.face else None, "pitch": float(np.degrees(r.gaze[0])) if r.face else None,
                "iris": [round(float(v), 3) for v in r.iris] if r.face else None,
                "head": [round(float(v), 1) for v in r.head] if r.face else None})
            self.bus.set_metric("e2e_vision", (time.time() - msg["captured_at"]) * 1000)
        # off-screen alert after 3 s continuously away
        if r.face and not r.on_screen or not r.face:
            self.off_since = self.off_since if self.off_since is not None else t
            if t - self.off_since >= 3.0:
                raise_flag(self.bus, self.db, self.id, "off_screen" if r.face else "no_face", t,
                           "Looking away from the screen for 3 s" if r.face else "Face not visible for 3 s")
                self.off_since = t + 1000      # one alert per episode
        else:
            self.off_since = None
        if t >= self.next_window:
            self.window(t)
            self.next_window += HOP_S

    def window(self, t: float):
        vad = self.bus_state("vad")
        f = self.pipe.window_features(vad.get("candidate", 0.0), vad.get("interviewer", 0.0), vad.get("state", "answering"))
        clean = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in f.items()
                 if k not in ("engagement_probs",)}
        with self.db.session() as s:
            s.add(WindowScore(interview_id=self.id, start_s=round(t - WIN_S, 2), end_s=round(t, 2),
                              attention=f.get("attention_score"), engagement=f.get("engagement_label"),
                              on_screen_ratio=f.get("on_screen_ratio"), features=clean_json(clean)))
            s.commit()
        self.bus.publish(f"analytics:{self.id}", {"kind": "window", "start_s": round(t - WIN_S, 2), "end_s": round(t, 2),
                                                 "attention": f.get("attention_score"), "engagement": f.get("engagement_label"),
                                                 "engagement_probs": f.get("engagement_probs"),
                                                 "on_screen_ratio": f.get("on_screen_ratio"), "glances": f.get("glance_away_count"),
                                                 "blink_rate": f.get("blink_rate_per_min"), "reading": f.get("reading_pattern_flag"),
                                                 "saccade_rate": f.get("saccade_rate_hz")})
        if f.get("reading_pattern_flag"):
            raise_flag(self.bus, self.db, self.id, "reading", t, "Eye movements look like reading from another screen")

    def bus_state(self, key: str) -> dict:
        return self.bus.get_state(f"{self.id}:{key}")


def run(bus=None, db=None, stop=lambda: False):
    cfg = settings()
    bus = bus or make_bus(cfg.redis_url)
    db = db or Database(cfg.database_url)
    live: dict[int, InterviewVision] = {}
    cursors = {"control": bus.last_id("control")}
    print("[vision] ready", flush=True)
    while not stop():
        for name, mid, msg, binary in bus.read(dict(cursors), 500, 50):
            cursors[name] = mid
            if name == "control":
                if msg["type"] == "start":
                    iid = msg["interview_id"]
                    try:
                        live[iid] = InterviewVision(iid, bus, db)
                        # frames from the start message on (ids are time-based), not from when the models finished
                        # loading: frames sent meanwhile would otherwise be skipped
                        cursors[f"frames:{iid}"] = mid
                        bus.publish(f"analytics:{iid}", {"kind": "status", "models": {"vision": True, "face_pretrained": live[iid].pretrained}})
                    except Exception as e:  # noqa: BLE001 - missing / untrained vision models must not kill the worker
                        print(f"[vision] disabled for interview {iid}: {e}", flush=True)
                        bus.publish(f"analytics:{iid}", {"kind": "status", "models": {"vision": False},
                                                         "note": f"vision models unavailable: {type(e).__name__}"})
                elif msg["type"] == "stop":
                    live.pop(msg["interview_id"], None)
                    cursors.pop(f"frames:{msg['interview_id']}", None)
            elif name.startswith("frames:") and binary is not None:
                iid = int(name.split(":")[1])
                if iid in live:
                    try:
                        live[iid].frame(msg, binary)
                    except Exception:  # noqa: BLE001 - keep serving the other frames
                        import traceback

                        traceback.print_exc()


if __name__ == "__main__":
    run()
