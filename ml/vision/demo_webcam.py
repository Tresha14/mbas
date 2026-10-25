"""Live webcam demo: face box, 5 landmarks, iris points, gaze arrow, on/off-screen, head pose,
engagement label and attention gauge (updated every 2.5 s over the last 5 s).

    python -m ml.vision.demo_webcam              # PyTorch checkpoints
    python -m ml.vision.demo_webcam --onnx       # ONNX exports (ml/export/onnx)
    python -m ml.vision.demo_webcam --video file.webm --out demo.mp4
Keys: q = quit, c = 5 s calibration (look at the four screen corners and the centre).
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from ml.vision.gaze_iris.signals import ScreenCalibration
from ml.vision.pipeline import VisionPipeline


def draw(frame_bgr, r, win: dict, fps: float):
    out = frame_bgr
    if r.face:
        x1, y1, x2, y2 = r.box.astype(int)
        col = (80, 200, 80) if r.on_screen else (60, 60, 230)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        for p in r.landmarks:
            cv2.circle(out, tuple(int(v) for v in p), 2, (255, 200, 0), -1)
        for p in r.iris_px:
            cv2.circle(out, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
        c = r.landmarks[:2].mean(0)
        L = (x2 - x1) * 0.8
        end = (int(c[0] - L * np.sin(r.gaze[1]) * np.cos(r.gaze[0])), int(c[1] - L * np.sin(r.gaze[0])))
        cv2.arrowedLine(out, tuple(c.astype(int)), end, (0, 220, 255), 2, tipLength=0.2)
        cv2.putText(out, "ON SCREEN" if r.on_screen else "OFF SCREEN", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(out, f"yaw {r.head[0]:+.0f} pitch {r.head[1]:+.0f} roll {r.head[2]:+.0f}", (10, out.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    else:
        cv2.putText(out, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 230), 2)
    if win.get("attention_score") is not None:
        a = win["attention_score"]
        cv2.rectangle(out, (10, 10), (210, 30), (60, 60, 60), -1)
        cv2.rectangle(out, (10, 10), (10 + int(2 * a), 30), (80, 200, 80) if a >= 60 else (40, 160, 230), -1)
        cv2.putText(out, f"attention {a:.0f}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out, f"engagement: {win['engagement_label']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, f"glances {win['glance_away_count']}  blinks/min {win['blink_rate_per_min']:.0f}"
                         f"{'  READING' if win['reading_pattern_flag'] else ''}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    cv2.putText(out, f"{fps:.0f} fps", (out.shape[1] - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--out")
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args(argv)
    pipe = VisionPipeline("onnx" if args.onnx else "torch")
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("cannot open camera/video")
    writer = None
    win: dict = {}
    t0 = last_win = time.time()
    n, fps, cal_until, cal_gaze = 0, 0.0, 0.0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time() - t0
        tic = time.time()
        r = pipe.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t)
        fps = 0.9 * fps + 0.1 / max(time.time() - tic, 1e-3) if n else 1 / max(time.time() - tic, 1e-3)
        if time.time() < cal_until and r.face:
            cal_gaze.append(r.gaze)
        elif cal_gaze:
            pipe.cal = ScreenCalibration.fit(np.stack(cal_gaze))
            cal_gaze = []
        if time.time() - last_win >= 2.5:
            win, last_win = pipe.window_features(), time.time()
        vis = draw(frame, r, win, fps)
        if args.out:
            writer = writer or cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        if not args.video:
            cv2.imshow("MBAS vision demo", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                cal_until = time.time() + 5
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"{n} frames, last window: {win}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
