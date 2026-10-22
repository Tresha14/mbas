"""End-to-end test of the recorder API + check_session with synthetic media.

Simulates one 6-second session: consent, clock, recording window, four tracks
uploaded in chunks, question events, calibration, finalize -> check passes.
"""
from __future__ import annotations

import json
import subprocess

import jwt
import numpy as np
import pytest
from fastapi.testclient import TestClient

from recorder.backend import wav
from recorder.backend.main import create_app
from tools.check_session import _ffmpeg, check_session

DUR_S = 6.0
T0 = 1_760_000_000_000.0  # fixed fake server-clock start (ms)


def make_webm(path, seconds=DUR_S, size="1280x720", fps=15):
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}",
         "-c:v", "libvpx", "-b:v", "500k", "-deadline", "realtime", str(path)],
        check=True,
    )
    return path.read_bytes()


def pcm(seconds, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "raw")
    return TestClient(app)


def _consent(c, code):
    r = c.post(f"/api/sessions/{code}/participants", json={
        "role": "candidate", "participant_id": "P001", "age_band": "18-24", "gender": "female",
        "glasses": True, "accent": "south_indian", "consent": {"agreed": True}})
    assert r.status_code == 200, r.text
    r = c.post(f"/api/sessions/{code}/participants", json={
        "role": "interviewer", "participant_id": "P900", "consent": {"agreed": True}})
    assert r.status_code == 200, r.text


def _upload(c, code, track, data, start_ms, end_ms, chunk=64_000):
    assert c.post(f"/api/sessions/{code}/tracks/{track}/start", json={"start_ms": start_ms, "sample_rate": 16000}).status_code == 200
    n = 0
    for i in range(0, len(data), chunk):
        r = c.put(f"/api/sessions/{code}/tracks/{track}/chunks/{n}", content=data[i:i + chunk])
        assert r.status_code == 200, r.text
        n += 1
    assert c.post(f"/api/sessions/{code}/tracks/{track}/stop", json={"end_ms": end_ms, "total_chunks": n}).status_code == 200


def _record_session(c, tmp_path, mic_seconds=DUR_S, with_screen=True):
    code = c.post("/api/sessions").json()["room_code"]
    _consent(c, code)
    assert c.post(f"/api/sessions/{code}/recording", json={"action": "start", "t_ms": T0}).status_code == 200
    end = T0 + DUR_S * 1000
    webm = make_webm(tmp_path / "cam.webm")
    _upload(c, code, "candidate_webcam", webm, T0 + 40, end + 40)
    if with_screen:
        _upload(c, code, "candidate_screen", make_webm(tmp_path / "scr.webm", size="640x360"), T0 + 500, T0 + 500 + DUR_S * 1000)
    _upload(c, code, "candidate_mic", pcm(mic_seconds), T0 + 30, end + 30)
    _upload(c, code, "interviewer_mic", pcm(DUR_S), T0 + 5, end + 5)
    for typ, t, q in [("ask", 500, "Q01"), ("answer_end", 3000, "Q01"), ("ask", 3500, "Q25"), ("answer_end", 5800, "Q25")]:
        assert c.post(f"/api/sessions/{code}/events", json={"type": typ, "question_id": q, "t_ms": T0 + t}).status_code == 200
    # calibration: 60 Hz samples, but session is short so it is checked for rate/bounds only
    samples = [{"t_ms": T0 + 100 + i * 1000 / 60, "x": 1.0, "y": 1.0, "x_norm": 0.5, "y_norm": 0.5, "phase": "fixation", "target": 0}
               for i in range(60 * 5)]
    r = c.post(f"/api/sessions/{code}/calibration", json={"start_ms": T0 + 100, "end_ms": T0 + 5100, "samples": samples})
    assert r.status_code == 200
    assert c.post(f"/api/sessions/{code}/recording", json={"action": "stop", "t_ms": end}).status_code == 200
    return code


def test_full_session_passes(client, tmp_path):
    code = _record_session(client, tmp_path)
    r = client.post(f"/api/sessions/{code}/finalize")
    assert r.status_code == 200, r.text
    rep = r.json()["check"]
    assert rep["passed"], rep["errors"]
    assert rep["info"]["answered_questions"] == 2
    t = rep["tracks"]
    assert t["candidate_mic"]["sample_rate"] == 16000
    assert t["candidate_webcam"]["width"] == 1280 and t["candidate_webcam"]["frames"] == 90
    assert all(v["drift_ms"] < 100 for v in t.values())
    # short calibration only warns
    assert any("calibration lasted" in w for w in rep["warnings"])

    d = tmp_path / "raw" / r.json()["session_id"]
    meta = json.loads((d / "meta.json").read_text())
    assert meta["participants"]["candidate"]["glasses"] is True
    assert wav.info(d / "candidate_mic.wav")["duration_s"] == pytest.approx(DUR_S)


def test_audio_drift_fails(client, tmp_path):
    # mic delivered 0.5 s less audio than the clock says -> drift error
    code = _record_session(client, tmp_path, mic_seconds=DUR_S - 0.5)
    rep = client.post(f"/api/sessions/{code}/finalize").json()["check"]
    assert not rep["passed"]
    assert any("candidate_mic: sync drift" in e for e in rep["errors"])


def test_missing_screen_lenient(client, tmp_path):
    code = _record_session(client, tmp_path, with_screen=False)
    sid = client.post(f"/api/sessions/{code}/finalize").json()["session_id"]
    d = tmp_path / "raw" / sid
    assert not check_session(d)["passed"]
    assert check_session(d, lenient=True)["passed"]


def test_consent_required(client):
    code = client.post("/api/sessions").json()["room_code"]
    r = client.post(f"/api/sessions/{code}/recording", json={"action": "start", "t_ms": T0})
    assert r.status_code == 409 and "consent" in r.json()["detail"]
    r = client.post(f"/api/sessions/{code}/token", json={"role": "candidate", "identity": "x"})
    assert r.status_code == 403


def test_chunk_order_and_duplicates(client):
    code = client.post("/api/sessions").json()["room_code"]
    _consent(client, code)
    client.post(f"/api/sessions/{code}/recording", json={"action": "start", "t_ms": T0})
    client.post(f"/api/sessions/{code}/tracks/candidate_mic/start", json={"start_ms": T0, "sample_rate": 16000})
    url = f"/api/sessions/{code}/tracks/candidate_mic/chunks"
    assert client.put(f"{url}/1", content=b"\0\0").status_code == 409
    assert client.put(f"{url}/0", content=b"\0\0").json() == {"duplicate": False, "next_seq": 1}
    assert client.put(f"{url}/0", content=b"\0\0").json()["duplicate"] is True
    assert client.put(f"{url}/1", content=b"\0").status_code == 422  # odd PCM length


def test_livekit_token(client):
    code = client.post("/api/sessions").json()["room_code"]
    _consent(client, code)
    tok = client.post(f"/api/sessions/{code}/token", json={"role": "candidate", "identity": "candidate-P001"}).json()["token"]
    claims = jwt.decode(tok, options={"verify_signature": False})
    assert claims["video"]["room"] == code and claims["sub"] == "candidate-P001"


def test_questions_file():
    from recorder.backend.main import QUESTIONS_PATH

    qs = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    assert len(qs) == 40
    assert {q["category"] for q in qs} == {"hr", "behavioural", "technical"}
    assert len({q["question_id"] for q in qs}) == 40
