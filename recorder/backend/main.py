"""Recorder API (FastAPI).

Run from the repo root:
    uvicorn recorder.backend.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .livekit_token import make_token
from .storage import ROLES, SessionStore, StoreError, now_ms

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = REPO_ROOT / "recorder" / "questions.json"


def _load_env(path: Path) -> None:
    """Minimal .env loader so `uvicorn` works without extra tooling."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env(REPO_ROOT / ".env")


class Consent(BaseModel):
    agreed: bool
    version: str = "v1"


class ParticipantIn(BaseModel):
    role: str
    participant_id: str = Field(min_length=1, max_length=32)
    age_band: str | None = None
    gender: str | None = None
    glasses: bool | None = None
    accent: str | None = None
    consent: Consent
    user_agent: str | None = None


class ClockIn(BaseModel):
    role: str
    offset_ms: float
    rtt_ms: float


class TokenIn(BaseModel):
    role: str
    identity: str
    name: str | None = None


class RecordingIn(BaseModel):
    action: str
    t_ms: float


class TrackStartIn(BaseModel):
    start_ms: float
    mime: str | None = None
    sample_rate: int | None = None
    settings: dict = {}


class TrackStopIn(BaseModel):
    end_ms: float
    total_chunks: int | None = None


class EventIn(BaseModel):
    type: str
    question_id: str
    t_ms: float
    question_text: str | None = None
    category: str | None = None


class CalibrationIn(BaseModel):
    start_ms: float
    end_ms: float
    screen: dict = {}
    pattern: str = ""
    samples: list[dict]


def create_app(raw_dir: Path | None = None) -> FastAPI:
    raw = Path(raw_dir or os.environ.get("MBAS_RAW_DIR", "data/raw"))
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    store = SessionStore(raw)
    api_key = os.environ.get("LIVEKIT_API_KEY", "mbasdevkey")
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "mbas_dev_secret_change_me_0123456789abcdef")

    app = FastAPI(title="MBAS recorder", version="1.0")
    app.state.store = store

    @app.exception_handler(StoreError)
    async def _store_error(_: Request, exc: StoreError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    @app.get("/api/health")
    def health():
        return {"ok": True, "raw_dir": str(raw)}

    @app.get("/api/time")
    def server_time():
        return {"t_ms": now_ms()}

    @app.get("/api/questions")
    def questions():
        return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    @app.post("/api/sessions")
    def create_session():
        s = store.create()
        return store.summary(s)

    @app.get("/api/sessions/{code}")
    def get_session(code: str):
        return store.summary(store.get(code))

    @app.post("/api/sessions/{code}/participants")
    def register(code: str, body: ParticipantIn):
        s = store.get(code)
        store.register_participant(s, body.model_dump())
        return store.summary(s)

    @app.post("/api/sessions/{code}/clock")
    def clock(code: str, body: ClockIn):
        if body.role not in ROLES:
            raise HTTPException(422, "bad role")
        store.set_clock(store.get(code), body.role, body.offset_ms, body.rtt_ms)
        return {"ok": True}

    @app.post("/api/sessions/{code}/token")
    def token(code: str, body: TokenIn):
        s = store.get(code)
        if body.role not in ROLES:
            raise HTTPException(422, "bad role")
        if body.role not in s.state["participants"]:
            raise HTTPException(403, "give consent before joining the room")
        tok = make_token(api_key, api_secret, s.code, body.identity, body.name or body.identity, body.role)
        return {"token": tok, "room": s.code}

    @app.post("/api/sessions/{code}/recording")
    def recording(code: str, body: RecordingIn):
        s = store.get(code)
        store.recording(s, body.action, body.t_ms)
        return store.summary(s)

    @app.post("/api/sessions/{code}/tracks/{track}/start")
    def track_start(code: str, track: str, body: TrackStartIn):
        s = store.get(code)
        store.start_track(s, track, body.model_dump())
        return {"ok": True}

    @app.put("/api/sessions/{code}/tracks/{track}/chunks/{seq}")
    async def track_chunk(code: str, track: str, seq: int, request: Request):
        s = store.get(code)
        data = await request.body()
        return store.append_chunk(s, track, seq, data)

    @app.post("/api/sessions/{code}/tracks/{track}/stop")
    def track_stop(code: str, track: str, body: TrackStopIn):
        s = store.get(code)
        store.stop_track(s, track, body.model_dump())
        return {"ok": True}

    @app.post("/api/sessions/{code}/events")
    def event(code: str, body: EventIn):
        return store.add_event(store.get(code), body.model_dump(exclude_none=True))

    @app.post("/api/sessions/{code}/calibration")
    def calibration(code: str, body: CalibrationIn):
        n = store.save_calibration(store.get(code), body.model_dump())
        return {"ok": True, "samples": n}

    @app.post("/api/sessions/{code}/finalize")
    def finalize(code: str):
        from tools.check_session import check_session

        s = store.get(code)
        store.finalize(s)
        report = check_session(s.dir)
        return {"session_id": s.session_id, "dir": str(s.dir), "check": report}

    return app


app = create_app()
