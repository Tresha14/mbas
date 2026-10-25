"""Stages 4-8: model shapes, losses, decoders, LM, behaviour features, conversation analyser, explanations."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ml.behaviour.conversation import analyse, turns_from_segments
from ml.behaviour.pauses import answer_pauses, question_end, segments
from ml.behaviour.prosody import prosody
from ml.behaviour.text_hesitation import text_hesitation
from ml.vision.face_detector.model import FaceDetector, decode, detection_loss, encode, make_anchors, match, postprocess
from ml.vision.gaze_iris.model import GazeIrisNet, angular_error_deg
from ml.vision.gaze_iris.signals import ScreenCalibration, reading_pattern, window_signals
from ml.vision.headpose.geometry import FACE_3D, head_pose


def test_face_detector_encode_decode_roundtrip():
    a = make_anchors()
    gt = torch.tensor([[0.30, 0.25, 0.62, 0.70]]).repeat(len(a), 1)
    kp = torch.rand(len(a), 5, 2)
    d, k = encode(gt, kp, a)
    b, k2 = decode(d, k, a)
    assert torch.allclose(b, gt, atol=1e-5) and torch.allclose(k2, kp, atol=1e-5)
    idx, lab = match(torch.tensor([[0.30, 0.25, 0.62, 0.70]]), a)
    assert (lab == 1).sum() >= 1 and (idx[lab == 1] == 0).all()


def test_face_detector_forward_loss_and_postprocess():
    m = FaceDetector()
    lg, d, k = m(torch.randn(2, 3, 128, 128))
    assert lg.shape == (2, 896) and d.shape == (2, 896, 4) and k.shape == (2, 896, 5, 2)
    lc, lb, lk = detection_loss(lg, d, k, [torch.tensor([[0.3, 0.3, 0.6, 0.7]]), torch.zeros(0, 4)],
                                [torch.rand(1, 5, 2), torch.zeros(0, 5, 2)], [torch.ones(1), torch.zeros(0)], m.anchors)
    assert all(torch.isfinite(x) for x in (lc, lb, lk))
    (lc + lb + lk).backward()
    out = postprocess(lg.detach(), d.detach(), k.detach(), m.anchors, score_thr=0.0, max_det=5)
    assert len(out) == 2 and out[0][0].shape[1] == 4


def test_gaze_model_and_angular_error():
    m = GazeIrisNet()
    g, iris, on = m(torch.zeros(3, 1, 36, 60), torch.zeros(3, 1, 36, 60), torch.zeros(3, 2))
    assert g.shape == (3, 2) and iris.shape == (3, 4) and on.shape == (3,)
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[0.0, math.radians(10)]])
    assert float(angular_error_deg(a, b)) == pytest.approx(10.0, abs=1e-3)


def test_head_pose_geometry_recovers_yaw():
    import cv2

    cam = np.array([[640, 0, 320], [0, 640, 320], [0, 0, 1.0]])
    r = cv2.Rodrigues(np.array([0, math.radians(20), 0.0]))[0]
    p = (r @ FACE_3D.T).T + [0, 0, 600]
    uv = (cam @ p.T).T
    kp = uv[:, :2] / uv[:, 2:]
    yaw, pitch, roll = head_pose(kp, (640, 640))
    assert abs(abs(yaw) - 20) < 0.5 and abs(pitch) < 0.5 and abs(roll) < 0.5


def test_window_signals():
    fps = 15
    gaze = np.zeros((75, 2))
    gaze[30:45, 1] = 0.9                                     # 1 s look away to the side
    cal = ScreenCalibration()
    on = cal.on_screen(gaze)
    s = window_signals(gaze, on, np.ones(75), np.full((75, 2), 0.5), fps)
    assert s["glance_away_count"] == 1 and s["off_screen_pct"] == pytest.approx(20.0)
    t = np.arange(75) / fps
    saw = (t % 1.0) * 0.5 - 0.25                              # slow sweep right, fast return, 5 times
    assert reading_pattern(saw, fps) and not reading_pattern(np.zeros(75), fps)


def test_text_hesitation_rules():
    h = text_hesitation("Um so I I worked on a — we built a portal, I think maybe it was matlab basically good")
    assert h["filler_count"] >= 2 and h["repetition_count"] == 1 and h["restart_count"] == 1
    assert h["hedge_count"] >= 2 and h["word_count"] > 10


def test_pause_model():
    sp = np.zeros(1000, bool)
    sp[200:400] = sp[550:900] = True                          # 2.0-4.0 s and 5.5-9.0 s at 10 ms
    segs = segments(sp)
    assert segs == [(2.0, 4.0), (5.5, 9.0)]
    qe = question_end([(0.2, 1.4)], ask_s=0.0, answer_end_s=10)
    p = answer_pauses(segs, qe, 9.5)
    assert p["response_latency_s"] == pytest.approx(0.6) and p["long_pause_count"] == 1
    assert p["long_pause_total_s"] == pytest.approx(1.5) and p["pause_ratio"] == pytest.approx(1 - 5.5 / 7.0, abs=1e-3)


def test_prosody_monotone_vs_varied():
    t = np.arange(16000 * 3) / 16000
    flat = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    varied = (0.3 * np.sin(2 * np.pi * (150 * t + 40 * np.sin(2 * np.pi * 0.7 * t) / (2 * np.pi * 0.7)))).astype(np.float32)
    assert prosody(flat, 20)["monotony_index"] > prosody(varied, 20)["monotony_index"]
    assert prosody(flat, 20)["speaking_rate_wpm"] == pytest.approx(400, rel=0.01)


def test_conversation_hand_checked_session():
    """Hand-checked: interviewer asks 0-4 s, candidate answers 5-15 s, interviewer cuts in at 14 s (1 s overlap),
    candidate answers again 20-30 s. Expected: candidate talk 20 s of 30 s (ratio 0.667), 2 interviewer + 2 candidate
    turns, 1 interruption by the interviewer, overlap 1 s, talk share 20 / 26."""
    segs = {"interviewer": [(0, 4), (14, 18)], "candidate": [(5, 9.5), (10, 15), (20, 30)]}
    r = analyse(segs, 30.0, [{"answer_id": "A1", "ask_s": 0.0, "end_s": 15}, {"answer_id": "A2", "ask_s": 14, "end_s": 30}])
    c, i = r["participants"]["candidate"], r["participants"]["interviewer"]
    assert c["talk_time_s"] == pytest.approx(19.5) and c["talk_time_ratio"] == pytest.approx(0.65)
    assert c["turns"] == 2 and i["turns"] == 2 and c["avg_turn_s"] == pytest.approx(10.0)
    assert i["interruptions_made"] == 1 and c["interruptions_made"] == 0 and r["overlap_s"] == pytest.approx(1.0)
    assert c["talk_share"] == pytest.approx(19.5 / 27.5, abs=1e-3)
    assert [p["first_reply_s"] for p in r["question_answer_pairs"]] == [5.0, 20.0]
    assert len(turns_from_segments({"x": [(0, 1), (1.5, 2)]})) == 1


def test_lm_normalises_and_decoder_uses_it(tmp_path):
    from ml.speech.asr.decode import BeamDecoder, greedy, wer
    from ml.speech.lm.ngram import ArpaLM, build_kn, write_arpa

    sents = [s.split() for s in ["the cat sat", "the cat ran", "a dog sat"] * 5]
    lm = ArpaLM(write_arpa(build_kn(sents, 3), tmp_path / "t.arpa"))
    vocab = [w for w in lm.vocab if w != "<s>"]
    for h in [("<s>", "the"), ("the", "cat"), ()]:
        assert sum(10 ** lm.log10_prob(w, h) for w in vocab) == pytest.approx(1.0, abs=0.01)
    pieces = ["<pad>", "<unk>", "▁the", "▁cat", "▁sat", "▁mat"]
    lp = np.full((6, 6), -9.0)
    for t, i in enumerate([2, 0, 3, 0, 4, 0]):
        lp[t, i] = -0.1
    lp[4, 5] = -0.12                                            # acoustically 'mat' is almost as likely as 'sat'
    assert greedy(lp) == [2, 3, 4]
    assert BeamDecoder(pieces, lm, alpha=1.0, beta=0.0).decode(lp) == [2, 3, 4]
    assert wer("a b c", "a x c") == (1, 3) and wer("a b", "") == (2, 2)


def test_speech_models_shapes():
    from ml.behaviour.disfluency.model import DisfluencyCRNN
    from ml.speech.asr.model import ConformerCTC, chunk_mask
    from ml.speech.vad.model import VADNet, smooth

    m = ConformerCTC(vocab=50, d=32, heads=2, layers=2)
    lp, ol = m(torch.randn(2, 16000 * 2), torch.tensor([32000, 16000]))
    assert lp.shape[0] == 2 and lp.shape[2] == 50 and ol[0] > ol[1]
    cm = chunk_mask(8, 2, 1, "cpu")
    assert not cm[3, 0] and cm[5, 0] and cm[0, 2]            # left context 1 chunk, no look into the future chunk
    assert VADNet()(torch.zeros(1, 16000)).shape == (1, 101)
    assert DisfluencyCRNN()(torch.zeros(2, 48000)).shape == (2, 4)
    p = np.r_[np.zeros(10), np.ones(30), np.zeros(40)]
    s = smooth(p, hangover=5, min_speech=5)
    assert s[10:40].all() and s[43] and not s[60]          # hangover keeps 4 frames after the drop


def test_fusion_model_masking_and_explanations():
    from ml.common.datasets import FUSION_FEATURES
    from ml.fusion.explain import shapley, top_reasons
    from ml.fusion.model import FusionNet

    torch.manual_seed(0)
    m = FusionNet().eval()
    x = {k: torch.randn(4, len(c)) for k, c in FUSION_FEATURES.items()}
    o = m(x)
    assert o["scores"].shape == (4, 6) and o["modality_weights"].shape == (4, 5)
    assert torch.allclose(o["modality_weights"].sum(1), torch.ones(4), atol=1e-5)
    o2 = m(x, active=["vision"])
    assert float(o2["modality_weights"][0, 0].detach()) == pytest.approx(1.0)
    flat = np.concatenate([x[k][0].numpy() for k in FUSION_FEATURES])
    phi = shapley(m, flat, n_perm=8)
    with torch.no_grad():
        full = float(m({k: x[k][:1] for k in FUSION_FEATURES})["scores"][0, 0]) * 100
        base = float(m({k: torch.zeros(1, len(c)) for k, c in FUSION_FEATURES.items()})["scores"][0, 0]) * 100
    assert phi.sum() == pytest.approx(full - base, abs=1e-3)   # Shapley efficiency
    raw = {c: 1.0 for cols in FUSION_FEATURES.values() for c in cols}
    reasons = top_reasons(m, flat, raw, 3)
    assert len(reasons) == 3 and all(r["text"] for r in reasons)
