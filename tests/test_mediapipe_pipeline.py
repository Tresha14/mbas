"""MediaPipe face tracking: availability / fallback and the geometry helpers. The live model is exercised only when the
package and the model file are present."""
from types import SimpleNamespace as P

import numpy as np
import pytest

from ml.vision import mediapipe_pipeline as mp_pipe


def test_availability_explains_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(mp_pipe, "MODEL", tmp_path / "nope.task")
    ok, why = mp_pipe.available()
    assert not ok and "face_landmarker.task" in why
    with pytest.raises(RuntimeError):
        mp_pipe.MediaPipePipeline()                       # the worker catches this and falls back to our own models


def test_iris_ratio_is_centred_when_the_iris_is_in_the_middle_of_the_eye():
    L = {}
    L[33], L[133] = P(x=0.40, y=0.50), P(x=0.50, y=0.50)             # outer / inner corner
    L[159], L[145] = P(x=0.45, y=0.48), P(x=0.45, y=0.52)           # upper / lower lid
    L[468] = P(x=0.45, y=0.50)
    h, v = mp_pipe._eye_ratio(L, 468, mp_pipe.EYE_A)
    assert abs(h - 0.5) < 1e-6 and abs(v - 0.5) < 1e-6
    L[468] = P(x=0.49, y=0.51)                                       # towards the image-right corner and down
    h, v = mp_pipe._eye_ratio(L, 468, mp_pipe.EYE_A)
    assert h > 0.8 and v > 0.5
    L[468] = P(x=0.9, y=0.9)                                         # outside the eye: clamped, never outside 0..1
    assert mp_pipe._eye_ratio(L, 468, mp_pipe.EYE_A) == (1.0, 1.0)


def test_a_blink_behind_glasses_still_counts_as_a_blink():
    """Blink scores behind glasses peak around 0.3-0.5; openness must then dip below 45 % of the open-eye median, which
    is the rule window_signals uses to count a blink."""
    from ml.vision.gaze_iris.signals import window_signals

    open_eye = float(np.clip(1 - mp_pipe.BLINK_GAIN * 0.07, 0, 1))        # typical resting blink score 0.07
    blink = float(np.clip(1 - mp_pipe.BLINK_GAIN * 0.40, 0, 1))
    openness = np.array(([open_eye] * 10 + [blink] * 2 + [open_eye] * 10) * 3, float)        # 3 blinks in 66 frames
    n = len(openness)
    sig = window_signals(np.zeros((n, 2)), np.ones(n, bool), openness, np.zeros((n, 2)), fps=15.0)
    assert sig["blink_rate_per_min"] > 30                                  # 3 blinks in 4.4 s


def test_the_pipeline_finds_a_face_in_a_real_frame_when_available():
    ok, _ = mp_pipe.available()
    clip = mp_pipe.CHECKPOINTS.parent / "data" / "raw" / "R20261102-232703-7SZTJP" / "candidate_webcam.webm"
    if not ok or not clip.exists():
        pytest.skip("MediaPipe or the sample recording is not on this machine")
    import cv2

    cap = cv2.VideoCapture(str(clip))
    pipe = mp_pipe.MediaPipePipeline(fps=15.0)
    seen = 0
    for k in range(30):
        okf, f = cap.read()
        if not okf:
            break
        r = pipe.process(cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2RGB), k / 15)
        seen += r.face
        if r.face:
            assert r.head.shape == (3,) and r.gaze.shape == (2,) and r.iris.shape == (4,) and 0 <= r.openness <= 1
    assert seen >= 25                                                       # the face is found in (almost) every frame
