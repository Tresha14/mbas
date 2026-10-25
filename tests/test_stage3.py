"""Stage 3: sync, windowing, features, tokenizer and Dataset shapes."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from ml.common import audio_features as af
from ml.common import datasets as D
from ml.common import text, windowing
from ml.common.sync import SessionClock
from ml.common.video_features import Augment, flip_gaze, letterbox, normalize
from tools.check_session import _ffmpeg

T0 = 1_760_000_000_000.0


def _session(tmp, dur=6.0, cam_off=0.4, mic_off=0.2):
    d = tmp / "R1"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"session_id": "R1", "participants": {
        "candidate": {"participant_id": "P001", "consent": {"agreed": True}}}}))
    (d / "timeline.json").write_text(json.dumps({"recording": {"start_ms": T0, "end_ms": T0 + dur * 1000}, "tracks": {
        "candidate_webcam": {"start_ms": T0 + cam_off * 1000, "end_ms": T0 + dur * 1000},
        "candidate_mic": {"start_ms": T0 + mic_off * 1000, "end_ms": T0 + dur * 1000}}}))
    (d / "question_events.json").write_text(json.dumps({"events": [
        {"type": "ask", "question_id": "Q01", "t_ms": T0 + 500}, {"type": "answer_end", "question_id": "Q01", "t_ms": T0 + 4000}]}))
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=320x240:rate=15:duration={dur - cam_off}", "-c:v", "libvpx", "-deadline", "realtime",
                    str(d / "candidate_webcam.webm")], check=True)
    n = int((dur - mic_off) * 16000)
    sf.write(str(d / "candidate_mic.wav"), np.arange(n, dtype=np.float32) / n * 0.5, 16000, subtype="PCM_16")
    return d


def test_sync_frames_and_audio(tmp_path):
    clk = SessionClock.load(_session(tmp_path))
    t = clk.webcam_times
    assert t[0] == pytest.approx(0.4) and np.diff(t).mean() == pytest.approx(1 / 15, abs=1e-3)
    assert clk.frame_at(0.4) == 0 and clk.frame_at(0.4 + 1.0) == 15 and clk.frame_at(0.0) == 0
    assert len(clk.frames_between(1.0, 2.0)) == 15
    a = clk.audio(0.0, 1.0)                       # mic starts at 0.2 s -> first 0.2 s are zeros
    assert len(a) == 16000 and np.all(a[:3200] == 0) and a[3300] >= 0
    b = clk.audio(1.0, 1.5)
    expected = (1.0 - 0.2) * 16000 / ((6.0 - 0.2) * 16000) * 0.5
    assert b[0] == pytest.approx(expected, abs=2e-3)
    assert clk.events()[0]["ask_s"] == 0.5


def test_window_bounds_and_event_counts():
    b = windowing.window_bounds(12.0)
    assert b.tolist() == [[0, 5], [2.5, 7.5], [5, 10], [7.5, 12.5]][:3]
    assert len(windowing.window_bounds(4.9)) == 0
    w = pd.DataFrame({"session_id": ["S"] * 3, "start_s": b[:, 0], "end_s": b[:, 1]})
    ev = pd.DataFrame({"session_id": ["S", "S", "X"], "event_type": ["filler", "filler", "filler"],
                       "start_s": [4.8, 9.0, 1.0], "end_s": [5.2, 9.5, 2.0]})
    c = windowing.count_events(w, ev)
    assert c.filler_count.tolist() == [1, 1, 2] and c.restart_count.sum() == 0


def test_synthetic_windows_have_splits_and_counts():
    if not windowing.SYNTHETIC.exists():
        pytest.skip("synthetic dataset missing")
    w = windowing.synthetic_windows()
    assert len(w) == 46234 and set(w.split.dropna()) == {"train", "val", "test"}
    assert "filler_count" in w and w.filler_count.sum() > 0
    per = w.groupby("participant_id").split.nunique()
    assert (per == 1).all()


def test_audio_features():
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    m = af.log_mel(x)
    assert m.shape == (80, 101)
    assert af.mfcc(x).shape == (40, 101)
    t = np.arange(16000) / 16000
    f0 = af.pitch_yin((0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
    assert np.nanmedian(f0) == pytest.approx(200, rel=0.03)
    sa = af.SpecAugment().train()
    y = sa(m)
    assert y.shape == m.shape and not torch.equal(y, m)
    assert torch.equal(af.SpecAugment().eval()(m), m)
    assert len(af.speed_perturb(x, 1.1)) == pytest.approx(16000 / 1.1, abs=2)
    mask = af.speech_mask(np.concatenate([np.zeros(8000, np.float32), x, np.zeros(8000, np.float32)]))
    assert not mask[:40].any() and mask[60:140].all()


def test_video_helpers():
    img = np.full((60, 120, 3), 200, np.uint8)
    lb, s, (px, py) = letterbox(img, 128)
    assert lb.shape == (128, 128, 3) and s == pytest.approx(128 / 120) and py > 0 and px == 0
    x = normalize(lb)
    assert x.shape == (3, 128, 128) and x.max() <= 1 and x.min() >= -1
    assert flip_gaze(np.array([0.1, 0.3])).tolist() == pytest.approx([0.1, -0.3])
    clip = np.random.default_rng(0).integers(0, 255, (4, 32, 32, 3)).astype(np.uint8)
    assert Augment(np.random.default_rng(1)).frames(clip).shape == clip.shape


def test_text_normalisation_and_tokenizer(tmp_path):
    assert text.normalize_asr("Um, I—I worked on [inaudible] SQL-DBs!") == "um i i worked on sql dbs"
    lines = [f"the quick brown fox number {i} jumps over the lazy dog" for i in range(300)]
    p = text.train_tokenizer("t", lines, vocab=60, out_dir=tmp_path)
    tok = text.Tokenizer(p)
    ids = tok.encode("the lazy dog")
    assert all(i > 1 for i in ids) and tok.decode(ids) == "the lazy dog"


def test_gaze_dataset_shapes(tmp_path, monkeypatch):
    root = tmp_path / "mpiigaze"
    root.mkdir()
    for p in ("p00", "p01"):
        np.savez(root / f"{p}.npz", left_image=np.zeros((5, 36, 60), np.uint8), right_image=np.ones((5, 36, 60), np.uint8),
                 left_gaze=np.full((5, 2), 0.1, np.float32), right_gaze=np.full((5, 2), 0.3, np.float32),
                 left_head=np.zeros((5, 2), np.float32), right_head=np.zeros((5, 2), np.float32))
    monkeypatch.setattr(D, "PUBLIC", tmp_path)
    ds = D.GazeDataset("train", persons=["p00", "p01"], augment=True)
    it = ds[0]
    assert len(ds) == 10 and it["left"].shape == (1, 36, 60) and it["gaze"].shape == (2,)
    assert it["gaze"][0] == pytest.approx(0.2) and abs(float(it["gaze"][1])) == pytest.approx(0.2)


def test_vad_and_asr_datasets(tmp_path, monkeypatch):
    root = tmp_path / "librispeech"
    root.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3):
        wav = tmp_path / f"u{i}.wav"
        sf.write(str(wav), (0.3 * rng.standard_normal(16000)).astype(np.float32), 16000)
        rows.append({"item_id": f"u{i}", "subject_id": "s", "path": str(wav), "duration_s": 1.0,
                     "text": "HELLO WORLD", "split_official": "val"})
    pd.DataFrame(rows).to_parquet(root / "manifest.parquet")
    monkeypatch.setattr(D, "PUBLIC", tmp_path)
    v = D.VADDataset("val", seconds=4.0, limit=2)
    it = v[0]
    assert it["wav"].shape == (64000,) and it["labels"].shape == (401,) and 0 < it["labels"].mean() < 1
    assert torch.equal(v[1]["wav"], v[1]["wav"])           # val items are deterministic
    tok = text.Tokenizer(text.train_tokenizer("a", ["hello world"] * 200, vocab=30, out_dir=tmp_path))
    a = D.ASRDataset("val", tok, sources=("librispeech",))
    b = D.asr_collate([a[0], a[1]])
    assert b["wav"].shape == (2, 16000) and b["tokens"].shape[0] == 2 and b["text"][0] == "hello world"


def test_answer_and_fusion_datasets():
    try:
        tok = text.load("answer")
        ad = D.AnswerDataset("train", tok)
    except FileNotFoundError:
        pytest.skip("run make data first")
    b = D.answer_collate([ad[0], ad[1], ad[2]])
    assert b["ids"].shape[0] == 3 and b["star"].shape == (3, 4) and b["mask"].dtype == torch.bool
    assert int(b["ids"][0, 0]) == ad.cls
    fd = D.FusionDataset("test")
    it = fd[0]
    assert {k: tuple(v.shape) for k, v in it.items() if k in D.FUSION_FEATURES} == {
        m: (len(c),) for m, c in D.FUSION_FEATURES.items()}
    assert it["scores"].shape == (6,) and 0 <= float(it["scores"][0]) <= 1
    tr = D.FusionDataset("train")
    assert abs(float(np.mean(tr.x["prosody"]))) < 0.05          # standardised on train


def test_face_dataset_targets(tmp_path, monkeypatch):
    import cv2

    root = tmp_path / "wider_face"
    root.mkdir()
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((200, 400, 3), np.uint8))
    lm = [[100, 50, 40, 40, 110, 60, 130, 60, 120, 70, 112, 80, 128, 80, 1.0]]
    pd.DataFrame([{"item_id": "val/a.jpg", "subject_id": "e", "split_official": "val", "path": str(img),
                   "boxes": json.dumps([[100, 50, 40, 40, 0, 0, 0, 0, 0, 0], [0, 0, 5, 5, 0, 0, 0, 0, 0, 0]]),
                   "landmarks": "[]"},
                  {"item_id": "train/b.jpg", "subject_id": "e", "split_official": "train", "path": str(img),
                   "boxes": json.dumps([[100, 50, 40, 40, 0, 0, 0, 0, 0, 0]]), "landmarks": json.dumps(lm)}]
                 ).to_parquet(root / "manifest.parquet")
    monkeypatch.setattr(D, "PUBLIC", tmp_path)
    ds = D.FaceDataset("test", size=128)
    x, boxes, kp, mask = ds[0]
    assert x.shape == (3, 128, 128) and boxes.shape == (1, 4)      # 5 px face dropped (min_face 10)
    assert kp.shape == (1, 5, 2) and mask.tolist() == [0.0]         # val images have no landmarks
    # 400 px wide -> scale 0.32, pad_y = (128 - 64) / 2 = 32
    assert boxes[0].tolist() == pytest.approx([100 * .32 / 128, (50 * .32 + 32) / 128, 140 * .32 / 128, (90 * .32 + 32) / 128], abs=1e-3)
    tr = D.FaceDataset("train", size=128, augment=True)
    tr.items = [it for it in tr.items if it[0] == str(img)][:1]
    for _ in range(5):  # crop + flip keep landmarks inside their face box
        x, boxes, kp, mask = tr[0]
        if mask[0] > 0:
            b = boxes[0]
            assert (kp[0, :, 0] >= b[0] - 1e-3).all() and (kp[0, :, 0] <= b[2] + 1e-3).all()
            assert kp[0, 0, 0] < kp[0, 1, 0]          # left eye stays left after a flip
