"""Stage 9-10 backend: auth + roles, interviewer-only analytics channel, events, worker answer analysis, reports."""
from __future__ import annotations

import json

import jwt
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
    # external call capture (Meet / Zoom / Teams): only the host gets a token, and it joins as the candidate so the
    # bot analyses it
    assert c.post(f"/api/interviews/{iv['id']}/capture-token", headers=ch).status_code == 403
    cap = c.post(f"/api/interviews/{iv['id']}/capture-token", headers=h).json()
    assert cap["identity"] == f"call-capture-{iv['id']}"
    assert c.post(f"/api/interviews/{iv['id']}/capture-token?platform=zoom", headers=h).status_code == 200
    assert c.post(f"/api/interviews/{iv['id']}/capture-token?platform=skype", headers=h).status_code == 422
    assert json.loads(jwt.decode(cap["token"], options={"verify_signature": False})["metadata"])["role"] == "candidate"
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


def test_report_metrics_scorecard_matrix_pressure_moments():
    """Six answers getting worse as the questions get harder: grade, statuses, the confidence matrix, the pressure
    response and key moments all follow from the stored behaviour JSON and window scores."""
    from backend.report_metrics import (behaviour_sections, card_details, confidence_matrix, key_moments,
                                        pressure_response, scorecard)

    cards, windows = [], []
    for i in range(6):
        ask = i * 60.0
        for s in np.arange(ask, ask + 50, 2.5):
            windows.append({"start_s": float(s), "end_s": float(s + 5), "attention": 85.0 - 10 * i, "on_screen_ratio": 0.9 - 0.08 * i,
                            "features": {"blink_rate_per_min": 15 + 5 * i, "glance_away_count": 0.2 * i,
                                         "reading_pattern_flag": int(i == 5 and s == ask)}})
        ans = {"behaviour": {"timing": {"response_latency_s": 1.0 + i, "long_pause_count": i, "longest_pause_s": 1.0 + i,
                                        "longest_pause_at_s": ask + 20, "very_long_pause_count": int(i >= 4),
                                        "pause_ratio": 0.2, "answer_duration_s": 45},
                             "prosody": {"voice_break_count": i, "voice_break_times_s": [ask + 10 + k for k in range(i)],
                                         "voice_break_rate_per_min": 3.0 * i, "pitch_jitter_pct": 1.2 + 0.3 * i,
                                         "monotony_index": 0.4, "volume_consistency": 0.7},
                             "text_hesitation": {"word_count": 100, "hedge_count": 1}, "audio_hesitation": {}}}
        c = {"question_ref": i + 1, "question": f"Q{i}", "ask_s": ask, "end_s": ask + 50, "effectiveness": 85.0 - 12 * i,
             "subscores": {"relevance": 85 - 13 * i, "confidence": 80 - 4 * i if i < 3 else 30}, "answer_model": {},
             "response_latency_s": 1.0 + i, "hesitation_index": 30 + 8 * i, "wpm": 140, "filler_rate": 1 + i,
             "transcript": "um " * i + "the api handles requests " * 5}
        c.update(card_details(c, ans, windows))
        cards.append(c)
    sections, raw = behaviour_sections(cards, windows, " ".join(c["transcript"] for c in cards))
    rows = {r["key"]: r for s in sections for r in s["rows"]}
    assert rows["voice_breaks"]["value"] == 15 and rows["voice_breaks"]["at"] == "01:10"
    assert rows["longest_pause"]["value"] == 6.0 and rows["longest_pause"]["status"] == "concern"
    assert rows["fillers"]["extra"]["breakdown"] == {"um": 15} and rows["reading"]["value"] == 2.5
    sc = scorecard(cards, raw)
    assert sc["grade"] in "ABCDE" and 0 <= sc["overall"] <= 100 and set(sc["dimensions"]) == {
        "content", "fluency", "composure", "delivery", "eye_contact", "responsiveness"}
    m = confidence_matrix(cards)
    assert m["counts"]["confident_on_point"] >= 2 and m["counts"]["struggled"] >= 1 and cards[0]["quadrant"]
    pr = pressure_response(cards, {c["question_ref"]: 0.2 + 0.1 * i for i, c in enumerate(cards)})
    assert pr["harder_n"] == 3 and "response delay +3" in pr["summary"]
    kinds = [m["kind"] for m in key_moments(cards, windows, raw)]
    assert {"best", "weakest", "delay", "silence", "voice", "attention", "reading"} <= set(kinds)


def test_speech_worker_reads_audio_from_the_start_message(env, monkeypatch):
    """Loading the models for a new interview can take ~20 s; audio published meanwhile must still be analysed,
    so the worker reads each interview's audio from the start message's id, not from 'now'."""
    from backend.workers import speech_worker

    _, db, bus = env
    monkeypatch.setattr(speech_worker, "InterviewSpeech", lambda *a, **k: object())
    start_id = bus.publish("control", {"type": "start", "interview_id": 7})
    early = bus.publish("audio:7:candidate", {"t_s": 0.0, "captured_at": 0.0}, b"\0\0")   # sent during model loading
    cursors: dict = {}
    speech_worker.handle("control", {"type": "start", "interview_id": 7}, None, bus, db, None, {}, cursors, start_id)
    got = bus.read({"audio:7:candidate": cursors["audio:7:candidate"]}, 0)
    assert [mid for _, mid, _, _ in got] == [early]


def test_session_platform_and_column_added_to_old_database(tmp_path):
    """Sessions remember where the call happens (our room, Meet, Zoom, Teams); a database created before the column
    existed gets it on startup."""
    import sqlite3

    path = tmp_path / "old.db"
    Database(f"sqlite:///{path}")
    con = sqlite3.connect(path)
    con.execute("ALTER TABLE interviews DROP COLUMN platform")       # an old database without the column
    con.commit()
    con.close()
    db = Database(f"sqlite:///{path}")
    c = TestClient(create_app(Settings(database_url=f"sqlite:///{path}", redis_url="memory://"), db, MemoryBus()))
    h = _host(c)
    assert c.post("/api/interviews", json={"title": "x"}, headers=h).json()["platform"] == "interlens"
    assert c.post("/api/interviews", json={"title": "y", "platform": "zoom_app"}, headers=h).json()["platform"] == "zoom_app"
    assert c.post("/api/interviews", json={"title": "z", "platform": "skype"}, headers=h).status_code == 422


# ---------------------------------------------------------------------------- reports built from little or no data
def _data(questions=(), answers=(), windows=()):
    return {"questions": list(questions), "answers": list(answers), "follow_ups": [], "windows": list(windows),
            "transcript": [], "speech": [], "flags": []}


def _win(t, face=1.0, blink=18.0):
    return {"start_s": float(t), "end_s": float(t + 5), "attention": 80.0, "engagement": "engaged", "on_screen_ratio": 0.9 * face,
            "features": {"face_present_ratio": face, "blink_rate_per_min": blink if face else 0.0,
                         "gaze_yaw_std_deg": 2.0 if face else 0.0, "glance_away_count": 0}}


def test_empty_interview_is_not_rated():
    """Nobody joined: no grade, no invented 100, no invented 'good' rows and no praise in the candidate feedback."""
    from backend.reports import build_reports

    iv, cv = build_reports(_data())
    sc = iv["scorecard"]
    assert sc["overall"] is None and sc["grade"] is None and sc["note"]
    assert all(d["score"] is None for d in sc["dimensions"].values())
    rows = {r["key"]: r for s in iv["behaviour"] for r in s["rows"]}
    for k in ("reading", "voice_breaks", "fillers", "crutch", "hedges", "long_pauses", "very_long_pauses"):
        assert rows[k]["value"] is None and rows[k]["status"] == "neutral", k
    assert cv["strengths"] == [] and cv["improvements"] == []


def test_video_only_or_silent_answers_are_not_rated():
    from backend.reports import build_reports

    q = [{"id": 1, "order": 1, "question_id": "Q1", "text": "Tell me about yourself", "category": "hr", "ask_s": 5.0, "end_s": 40.0}]
    silent = [{"question_ref": 1, "effectiveness": None, "subscores": {}, "answer_model": {}, "behaviour": {}, "reasons": [],
               "modality_weights": {}, "transcript": ""}]
    wins = [_win(t) for t in range(0, 50, 5)]
    for d in (_data(windows=wins), _data(q, silent), _data(q, silent, wins)):
        iv, cv = build_reports(d)
        assert iv["scorecard"]["overall"] is None and iv["scorecard"]["grade"] is None
        assert cv["strengths"] == [] and cv["improvements"] == []      # nothing was said, so nothing to praise or fix


def test_windows_without_a_face_are_not_averaged_in_as_zeros():
    from backend.report_metrics import behaviour_sections

    wins = [_win(t) for t in range(0, 25, 5)] + [_win(t, face=0.0) for t in range(25, 50, 5)]
    sections, _ = behaviour_sections([], wins, "")
    rows = {r["key"]: r for s in sections for r in s["rows"]}
    assert rows["blink_rate"]["value"] == 18.0            # not 9.0 (half zeros)
    assert rows["gaze_stability"]["value"] == 2.0
    assert rows["face_visible"]["value"] == 50.0          # visibility itself still counts the empty windows


def test_zero_blinks_over_a_whole_interview_is_not_rated():
    from backend.report_metrics import behaviour_sections

    wins = [_win(t, blink=0.0) for t in range(0, 50, 5)]       # face seen, but the tracker never saw a blink
    sections, raw = behaviour_sections([], wins, "")
    row = next(r for s in sections for r in s["rows"] if r["key"] == "blink_rate")
    assert row["status"] == "neutral" and "could not be tracked" in row["note"] and raw["eyes"]["blink"] is None


# ---------------------------------------------------------------------------- combined flags (ported from the older project)
def _calm_interview(n=24):
    """Four minutes of steady delivery: windows every 10 s and speaking stats every 10 s."""
    wins = [{"start_s": t - 5.0, "end_s": float(t), "on_screen_ratio": 0.9,
             "features": {"face_present_ratio": 1.0, "head_yaw_std_deg": 2.0}} for t in range(10, 10 * n + 1, 10)]
    sp = [{"t_s": float(t), "wpm": 140.0, "pause_ratio": 0.10, "fillers_per_100w": 0.5, "hedges_per_100w": 0.5} for t in range(10, 10 * n + 1, 10)]
    return wins, sp


def test_combined_flags_need_several_signals_together():
    from backend.combined_flags import find_combined_flags

    wins, sp = _calm_interview()
    assert find_combined_flags(wins, sp, []) == []                      # a calm interview flags nothing
    # one signal getting bad on its own (more pausing) is not enough
    one = [dict(s, pause_ratio=0.6) if 90 <= s["t_s"] <= 130 else s for s in sp]
    assert find_combined_flags(wins, one, []) == []
    # pausing + hedging + looking away together around 90-130 s is flagged, once
    bad_w = [dict(w, on_screen_ratio=0.1) if 90 <= w["end_s"] <= 130 else w for w in wins]
    bad_s = [dict(s, pause_ratio=0.6, hedges_per_100w=12.0) if 90 <= s["t_s"] <= 130 else s for s in sp]
    cards = [{"order": 3, "question_ref": 7, "ask_s": 60.0, "end_s": 150.0}]
    flags = find_combined_flags(bad_w, bad_s, [{"kind": "long_pause", "t_s": 100.0}], cards)
    assert len(flags) == 1 and {"pause_rate", "hedge_rate", "camera_facing", "long_pause"} <= set(flags[0]["signals"])
    assert flags[0]["question_order"] == 3 and "together" in flags[0]["text"] and flags[0]["t_start"] <= 100 <= flags[0]["t_end"]


def test_combined_flags_are_relative_to_the_speaker_and_ignore_missing_data():
    from backend.combined_flags import find_combined_flags

    assert find_combined_flags([], [], []) == []
    wins, sp = _calm_interview()
    # uniformly weak delivery has no baseline to deviate from: no combined flags (the scorecard covers it)
    weak_w = [dict(w, on_screen_ratio=0.2) for w in wins]
    weak_s = [dict(s, pause_ratio=0.6, hedges_per_100w=12.0) for s in sp]
    assert find_combined_flags(weak_w, weak_s, []) == []
    # windows without a face are not "looking away": they are skipped for the camera signals
    nofac = [dict(w, on_screen_ratio=0.0, features={"face_present_ratio": 0.0, "head_yaw_std_deg": 0.0}) if 90 <= w["end_s"] <= 130 else w for w in wins]
    assert find_combined_flags(nofac, sp, []) == []


def test_report_includes_combined_flags_list():
    from backend.reports import build_reports

    iv, _ = build_reports(_data())
    assert iv["combined_flags"] == []


# ---------------------------------------------------------------------------- human-rating validation (ported)
def test_rating_statistics_match_hand_calculated_values():
    from tools.validate_ratings import cohens_kappa, correlate, pearson, rater_agreement, spearman

    assert pearson([1, 2, 3], [2, 4, 6]) == 1.0 and pearson([1, 2, 3], [3, 2, 1]) == -1.0
    assert pearson([1, 2, 3], [5, 5, 5]) is None and pearson([1], [1]) is None          # undefined, not 0
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == 1.0                                 # monotonic but not linear
    assert abs(spearman([1, 2, 2, 3], [1, 2, 3, 4]) - 0.9486832980505138) < 1e-9        # ties share a mean rank
    # observed 3/4, expected .5*.25 + .5*.75 = .5, so kappa = (.75-.5)/.5
    assert abs(cohens_kappa([1, 1, 2, 2], [1, 2, 2, 2]) - 0.5) < 1e-12
    assert cohens_kappa([3, 3, 3], [3, 3, 3]) is None and cohens_kappa([], []) is None   # no disagreement to measure
    assert rater_agreement({"A": {1: 1, 2: 1, 3: 2, 4: 2}, "B": {1: 1, 2: 2, 3: 2, 4: 2}})["kappa"] == 0.5
    c = correlate([60, 70, 80], [3, 4, 5])
    assert c["pearson_r"] == 1.0 and "preliminary" in c["note"]                          # tiny sample is flagged, not hidden


def test_validate_ratings_reads_csv_and_skips_interviews_without_a_score(tmp_path):
    from tools.validate_ratings import analyse, read_ratings, write_template

    t = tmp_path / "t.csv"
    write_template(t)
    try:
        read_ratings(t)
        raise AssertionError("the example row must not count as data")
    except SystemExit as e:
        assert "no ratings" in str(e)
    p = tmp_path / "r.csv"
    p.write_text("interview_id,rater_id,overall,content\n1,A,2,2\n1,B,2,3\n2,A,4,4\n2,B,5,4\n3,A,5,5\n3,B,5,5\n", encoding="utf-8")
    res = analyse(read_ratings(p), {1: {"overall": 40.0, "content": 35.0}, 2: {"overall": 75.0, "content": 70.0}, 3: {"overall": None, "content": 90.0}})
    assert res["overall"]["system_vs_human"]["n"] == 2          # interview 3 has no overall score: skipped, not counted as 0
    assert res["content"]["system_vs_human"]["n"] == 3 and res["content"]["system_vs_human"]["pearson_r"] > 0.9
