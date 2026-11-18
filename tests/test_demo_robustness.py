import threading
import time

from backend.bus import MemoryBus
from backend.main import catch_up


def test_catch_up_drops_partials_and_old_gaze_keeps_the_rest():
    hist = [{"kind": "transcript", "final": False, "text": "par"}, {"kind": "transcript", "final": True, "text": "done"},
            {"kind": "alert", "t_s": 1}] + [{"kind": "gaze", "t_s": i} for i in range(100)] + [{"kind": "window", "end_s": 5}]
    out = catch_up(hist, keep_gaze=10)
    assert [m["kind"] for m in out].count("gaze") == 10
    assert [m["t_s"] for m in out if m["kind"] == "gaze"] == list(range(90, 100))      # the newest ones, in order
    assert not any(m["kind"] == "transcript" and not m["final"] for m in out)
    assert {"alert", "window"} <= {m["kind"] for m in out}
    assert out[0]["text"] == "done" and out[-1]["kind"] == "window"                    # original order kept


def test_history_stops_at_the_given_id_so_live_messages_are_not_sent_twice():
    bus = MemoryBus()
    for i in range(5):
        bus.publish("analytics:1", {"i": i})
    mid = bus.last_id("analytics:1")
    bus.publish("analytics:1", {"i": 5})
    assert [m["i"] for m in bus.history("analytics:1", upto=mid)] == [0, 1, 2, 3, 4]
    assert [m["i"] for m in bus.read({"analytics:1": mid}, 0)[0][2:3]] == [5]


def test_vision_worker_survives_missing_models_and_reports_it(monkeypatch):
    from backend.workers import vision_worker

    class Boom:
        def __init__(self, *a, **k):
            raise FileNotFoundError("checkpoints/face_detector/best.pt")

    monkeypatch.setattr(vision_worker, "InterviewVision", Boom)
    bus, stop = MemoryBus(), threading.Event()
    th = threading.Thread(target=vision_worker.run, kwargs={"bus": bus, "db": object(), "stop": stop.is_set}, daemon=True)
    th.start()
    time.sleep(0.3)
    bus.publish("control", {"type": "start", "interview_id": 7, "room": "r", "started_at": 0})
    for _ in range(40):
        if bus.history("analytics:7"):
            break
        time.sleep(0.1)
    assert th.is_alive()                                                     # the loop did not die
    msgs = bus.history("analytics:7")
    assert msgs and msgs[0]["kind"] == "status" and msgs[0]["models"] == {"vision": False}
    stop.set()
    th.join(3)


def test_report_includes_flags_and_delivery_stats():
    from backend.reports import build_reports

    data = {"questions": [], "answers": [], "follow_ups": [], "transcript": [],
            "windows": [{"start_s": 0, "end_s": 5, "attention": 80, "engagement": "engaged", "on_screen_ratio": 0.9, "features": {"blink_rate_per_min": 16}},
                        {"start_s": 5, "end_s": 10, "attention": None, "engagement": None, "on_screen_ratio": 0.5, "features": {}}],
            "speech": [{"t_s": 5, "wpm": 120, "pause_ratio": 0.2, "fillers_per_100w": 2, "hedges_per_100w": None},
                       {"t_s": 10, "wpm": 140, "pause_ratio": 0.4, "fillers_per_100w": 4, "hedges_per_100w": None}],
            "flags": [{"t_s": 70, "kind": "long_pause", "text": "x"}, {"t_s": 45, "kind": "off_screen", "text": "y"},
                      {"t_s": 80, "kind": "long_pause", "text": "z"}]}
    v, _ = build_reports(data)
    assert [f["time"] for f in v["flags"]] == ["00:45", "01:10", "01:20"]
    d = v["delivery"]
    assert d["wpm"] == 130 and d["pause_pct"] == 30 and d["fillers_per_100w"] == 3 and d["hedges_per_100w"] is None
    assert d["camera_facing_pct"] == 70 and d["blinks_per_min"] == 16
    assert d["flag_counts"] == {"long_pause": 2, "off_screen": 1}
    assert len(v["speech_timeline"]) == 2 and v["summary"]["mean_attention"] == 80


def test_report_still_builds_for_an_old_interview_without_flags_or_speech():
    from backend.reports import build_reports

    v, _ = build_reports({"questions": [], "answers": [], "follow_ups": [], "windows": [], "transcript": []})
    assert v["flags"] == [] and v["delivery"]["wpm"] is None and v["delivery"]["flag_counts"] == {}


def test_second_start_keeps_the_clock_and_ended_cannot_restart(tmp_path):
    from fastapi.testclient import TestClient

    from backend.config import Settings
    from backend.db import Database
    from backend.main import create_app

    cfg = Settings(database_url=f"sqlite:///{tmp_path / 'm.db'}", redis_url="memory://")
    bus = MemoryBus()
    c = TestClient(create_app(cfg, Database(cfg.database_url), bus))
    tok = c.post("/api/auth/register", json={"email": "a@b.com", "name": "A", "password": "password123", "role": "interviewer"}).json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    iv = c.post("/api/interviews", json={"title": "t", "candidate_name": "c"}, headers=h).json()
    first = c.post(f"/api/interviews/{iv['id']}/start", headers=h).json()
    n = len(bus.history("control"))
    second = c.post(f"/api/interviews/{iv['id']}/start", headers=h).json()
    assert second["started_at"][:19] == first["started_at"][:19] and len(bus.history("control")) == n     # no second start event
    c.post(f"/api/interviews/{iv['id']}/stop", headers=h)
    assert c.post(f"/api/interviews/{iv['id']}/start", headers=h).status_code == 409
    again = c.post(f"/api/interviews/{iv['id']}/stop", headers=h).json()
    assert again["status"] == "ended"


def test_speaking_stats_report_no_pace_when_there_is_no_transcript():
    import numpy as np

    from backend.db import Database
    from backend.workers.speech_worker import InterviewSpeech, Track

    class Models:
        vad = asr = disfluency = answer = fusion = None

    bus = MemoryBus()
    sp = InterviewSpeech(1, "interview", bus, Database("sqlite://"), Models())
    tr = Track("candidate")
    tr.speech = np.ones(60 * 100, bool)           # a minute of detected speech, but no recognised words
    sp.speaking(tr, 60.0)
    msg = [m for m in bus.history("analytics:1") if m["kind"] == "speaking"][-1]
    assert msg["wpm"] is None and msg["fillers_per_100w"] is None and msg["hedges_per_100w"] is None
