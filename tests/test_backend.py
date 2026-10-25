"""Stage 9-10 backend: auth + roles, interviewer-only analytics channel, events, worker answer analysis, reports."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.bus import MemoryBus
from backend.config import Settings
from backend.db import AnswerScore, Database, FollowUp, Question, WindowScore, select
from backend.main import create_app


@pytest.fixture()
def env(tmp_path):
    cfg = Settings(database_url=f"sqlite:///{tmp_path / 'mbas.db'}", redis_url="memory://")
    db, bus = Database(cfg.database_url), MemoryBus()
    return TestClient(create_app(cfg, db, bus)), db, bus


def _host(c, email="iv@x.com", role="interviewer"):
    r = c.post("/api/auth/register", json={"email": email, "name": "Iv", "password": "password123", "role": role})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _interview(c, h):
    r = c.post("/api/interviews", json={"title": "SDE intern", "candidate_name": "A"}, headers=h)
    assert r.status_code == 200
    return r.json()


def test_auth_roles_and_candidate_join(env):
    c, _, _ = env
    assert c.post("/api/auth/register", json={"email": "c@x.com", "name": "C", "password": "password123",
                                              "role": "candidate"}).status_code == 422
    h = _host(c)
    assert c.post("/api/auth/login", json={"email": "iv@x.com", "password": "wrong-pass"}).status_code == 401
    iv = _interview(c, h)
    assert iv["candidate_link"].endswith(iv["invite_token"]) and iv["mode"] == "interview"
    invite = iv["invite_token"]
    assert c.post(f"/api/join/{invite}", json={"name": "Asha", "consent": False}).status_code == 409
    j = c.post(f"/api/join/{invite}", json={"name": "Asha", "consent": True}).json()
    ch = {"Authorization": f"Bearer {j['token']}"}
    assert c.get(f"/api/interviews/{iv['id']}", headers=ch).json().keys() == {"id", "title", "mode", "status", "room"}
    assert c.get("/api/interviews", headers=ch).status_code == 403            # candidates cannot list
    assert c.get(f"/api/interviews/{iv['id']}/data", headers=ch).status_code == 403
    assert c.post(f"/api/interviews/{iv['id']}/events", json={"type": "ask", "question_id": "Q01"}, headers=ch).status_code == 403
    tok = c.post(f"/api/interviews/{iv['id']}/livekit-token", headers=ch).json()
    assert tok["identity"].startswith("candidate-")
    other = _host(c, "other@x.com")
    assert c.get(f"/api/interviews/{iv['id']}", headers=other).status_code == 403
    m = c.post("/api/interviews", json={"title": "standup"}, headers=_host(c, "host@x.com", "meeting_host")).json()
    assert m["mode"] == "meeting"


def test_analytics_channel_rejects_candidate(env):
    c, _, bus = env
    h = _host(c)
    iv = _interview(c, h)
    j = c.post(f"/api/join/{iv['invite_token']}", json={"name": "A", "consent": True}).json()
    for token in (j["token"], "garbage"):
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect(f"/ws/analytics/{iv['id']}?token={token}") as ws:
                ws.receive_json()
        assert e.value.code in (4401, 4403)
    other = c.post("/api/auth/register", json={"email": "o@x.com", "name": "O", "password": "password123"}).json()["token"]
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect(f"/ws/analytics/{iv['id']}?token={other}") as ws:
            ws.receive_json()
    assert e.value.code == 4403
    bus.publish(f"analytics:{iv['id']}", {"kind": "window", "attention": 80})
    tok = h["Authorization"].split()[1]
    with c.websocket_connect(f"/ws/analytics/{iv['id']}?token={tok}") as ws:
        assert ws.receive_json()["attention"] == 80                          # backlog for the interviewer


def test_question_events_and_metrics(env):
    c, db, bus = env
    h = _host(c)
    iv = _interview(c, h)
    assert c.post(f"/api/interviews/{iv['id']}/start", headers=h).json()["status"] == "live"
    a = c.post(f"/api/interviews/{iv['id']}/events", json={"type": "ask", "question_id": "Q01", "text": "Tell me about yourself.",
                                                           "category": "hr"}, headers=h).json()
    e = c.post(f"/api/interviews/{iv['id']}/events", json={"type": "answer_end", "question_id": "Q01"}, headers=h).json()
    assert e["question_ref"] == a["question_ref"] and e["end_s"] >= e["ask_s"]
    assert c.post(f"/api/interviews/{iv['id']}/events", json={"type": "answer_end", "question_id": "Q09"}, headers=h).status_code == 409
    kinds = [m["type"] for m in bus.history("control")]
    assert kinds == ["start", "ask", "answer_end"]
    bus.set_metric("asr_chunk", 120.0)
    assert c.get("/metrics").json()["models"]["asr_chunk"]["ms"] == 120.0
    assert len(c.get("/api/questions", headers=h).json()) == 40


def test_speech_worker_answer_analysis_and_report(env, monkeypatch):
    """Feed 12 s of synthetic candidate/interviewer audio through the speech worker, end an answer, check the
    stored answer score, follow-ups, then build + edit + release the report."""
    from backend.inference import Models
    from backend.workers.speech_worker import InterviewSpeech

    c, db, bus = env
    h = _host(c)
    iv = _interview(c, h)
    c.post(f"/api/interviews/{iv['id']}/start", headers=h)
    q = c.post(f"/api/interviews/{iv['id']}/events", json={"type": "ask", "question_id": "Q01", "text": "Tell me about yourself.",
                                                           "category": "hr"}, headers=h).json()
    models = Models(bus)
    sp = InterviewSpeech(iv["id"], "interview", bus, db, models)
    rng = np.random.default_rng(0)
    t = np.arange(10240) / 16000
    for k in range(19):                                    # 19 x 640 ms = 12.2 s
        ts = k * 0.64
        cand = (0.3 * np.sin(2 * np.pi * 180 * t) * (ts > 2.5)).astype(np.float32) + 0.001 * rng.standard_normal(10240)
        intr = (0.3 * np.sin(2 * np.pi * 120 * t) * (ts < 1.9)).astype(np.float32) + 0.001 * rng.standard_normal(10240)
        for name, x in (("candidate", cand), ("interviewer", intr)):
            sp.audio(name, {"t_s": ts, "captured_at": 0.0}, (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    with db.session() as s:
        s.add(WindowScore(interview_id=iv["id"], start_s=2.5, end_s=7.5, attention=35.0, engagement="disengaged",
                          on_screen_ratio=0.3, features={"on_screen_ratio": 0.3, "reading_pattern_flag": 1}))
        s.commit()
    ev = {**q, "end_s": 12.0}
    sp.answer_end(ev)
    with db.session() as s:
        a = s.scalars(select(AnswerScore)).one()
        fus = s.scalars(select(FollowUp)).all()
    assert a.behaviour["timing"]["response_latency_s"] == pytest.approx(0.6, abs=0.35)   # interviewer stops ~1.9 s
    assert {f.reason for f in fus} >= {"attention_drop"}
    kinds = {m.get("kind") for m in bus.history(f"analytics:{iv['id']}")}
    assert {"status", "answer", "follow_up", "speaking", "conversation"} <= kinds
    rep = c.post(f"/api/interviews/{iv['id']}/report", headers=h).json()
    assert rep["interviewer_view"]["cards"][0]["question"] == "Tell me about yourself."
    cv = rep["candidate_view"]
    assert cv["improvements"] and cv["strengths"]
    cv["closing"] = "Edited by the interviewer."
    assert c.put(f"/api/interviews/{iv['id']}/report", json={"candidate_view": cv}, headers=h).status_code == 200
    rel = c.post(f"/api/interviews/{iv['id']}/report/release", headers=h).json()
    fb = c.get(f"/api/feedback/{rel['release_token']}").json()
    assert fb["closing"] == "Edited by the interviewer." and "interviewer_view" not in fb and "cards" not in fb
    assert c.get("/api/feedback/not-a-token").status_code == 404


def test_candidate_feedback_rules():
    from backend.reports import candidate_feedback

    cards = [{"question": "Tell me about a time you led a team", "category": "behavioural", "ask_s": 65.0,
              "response_latency_s": 4.5, "fillers": 12, "transcript": " ".join(["word"] * 100), "speaking_pattern": "monotone",
              "answer_model": {"star_count": 2, "star": {"situation": 0.9, "task": 0.8, "action": 0.2, "result": 0.1},
                               "relevance": 0.8}, "effectiveness": 55.0, "wpm": 130},
             {"question": "Why us", "category": "hr", "ask_s": 200.0, "response_latency_s": 1.0, "fillers": 0,
              "transcript": "x", "answer_model": {"relevance": 0.9}, "effectiveness": 80.0, "wpm": 120}]
    fb = candidate_feedback(cards, [])
    assert len(fb["improvements"]) == 3
    text = " ".join(i["observation"] for i in fb["improvements"])
    assert "4.5 s on behavioural" in text and "01:05" in text
    assert any("Why us" in s for s in fb["strengths"])
