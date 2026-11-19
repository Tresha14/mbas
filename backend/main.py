"""MBAS product API (FastAPI): auth + roles, interviews / meetings, LiveKit tokens, question events,
interviewer-only analytics WebSocket, reports, /metrics.

    uvicorn backend.main:app --host 0.0.0.0 --port 8020
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import secrets
import time
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.auth import HOST_ROLES, hash_password, make_jwt, read_jwt, verify_password
from backend.bus import make_bus
from backend.config import Settings, settings
from backend.db import (AnswerScore, Database, Flag, FollowUp, Interview, Participant, Question, Report, SpeechStat,
                        TranscriptSegment,
                        User, WindowScore, row_dict, select)
from recorder.backend.livekit_token import make_token as livekit_token


class RegisterIn(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=8, max_length=200)
    role: str = "interviewer"


class LoginIn(BaseModel):
    email: str
    password: str


class InterviewIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    mode: str = "interview"
    candidate_name: str | None = None
    scheduled_at: dt.datetime | None = None
    platform: Literal["interlens", "meet", "zoom_web", "zoom_app", "teams"] = "interlens"


class JoinIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    consent: bool


class EventIn(BaseModel):
    type: str
    question_id: str
    text: str = ""
    category: str = ""


class NotesIn(BaseModel):
    notes: str


class ReportEditIn(BaseModel):
    candidate_view: dict[str, Any]


def catch_up(history: list[dict], keep_gaze: int = 60) -> list[dict]:
    """What a (re)connecting cockpit needs: everything except in-progress transcript lines and old gaze points
    (5 per second, only the recent trail is drawn)."""
    gaze_seen = 0
    out = []
    for m in reversed(history):
        k = m.get("kind")
        if k == "gaze":
            gaze_seen += 1
            if gaze_seen > keep_gaze:
                continue
        elif k == "transcript" and not m.get("final"):
            continue
        out.append(m)
    return out[::-1]


def create_app(cfg: Settings | None = None, db: Database | None = None, bus=None) -> FastAPI:
    cfg = cfg or settings()
    db = db or Database(cfg.database_url)
    bus = bus or make_bus(cfg.redis_url)
    app = FastAPI(title="MBAS API", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.db, app.state.bus, app.state.cfg = db, bus, cfg
    questions = json.loads(cfg.questions_path.read_text(encoding="utf-8"))["questions"]

    # ------------------------------------------------------------------ auth helpers
    def claims(authorization: str | None = Header(default=None)) -> dict:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        try:
            return read_jwt(cfg.jwt_secret, authorization.split(" ", 1)[1])
        except Exception as e:  # noqa: BLE001
            raise HTTPException(401, f"invalid token: {e}") from e

    def host(c: dict = Depends(claims)) -> dict:
        if c["role"] not in HOST_ROLES:
            raise HTTPException(403, "interviewer / meeting host only")
        return c

    def own(s, interview_id: int, c: dict) -> Interview:
        iv = s.get(Interview, interview_id)
        if iv is None:
            raise HTTPException(404, "interview not found")
        if c["role"] in HOST_ROLES:
            if iv.owner_id != int(c["sub"]):
                raise HTTPException(403, "not your interview")
        elif c.get("interview_id") != iv.id:
            raise HTTPException(403, "token is for another interview")
        return iv

    def t_rel(iv: Interview) -> float:
        if not iv.started_at:
            return 0.0
        start = iv.started_at if iv.started_at.tzinfo else iv.started_at.replace(tzinfo=dt.timezone.utc)
        return round((dt.datetime.now(dt.timezone.utc) - start).total_seconds(), 3)

    # ------------------------------------------------------------------ accounts
    @app.post("/api/auth/register")
    def register(body: RegisterIn):
        if body.role not in HOST_ROLES:
            raise HTTPException(422, "accounts are interviewers or meeting hosts; candidates join by invite link")
        with db.session() as s:
            if s.scalar(select(User).where(User.email == body.email.lower())):
                raise HTTPException(409, "email already registered")
            u = User(email=body.email.lower(), name=body.name, role=body.role, password_hash=hash_password(body.password))
            s.add(u)
            s.commit()
            return {"token": make_jwt(cfg.jwt_secret, str(u.id), u.role, cfg.jwt_ttl_s, name=u.name), "user": _user(u)}

    @app.post("/api/auth/login")
    def login(body: LoginIn):
        with db.session() as s:
            u = s.scalar(select(User).where(User.email == body.email.lower()))
            if not u or not verify_password(body.password, u.password_hash):
                raise HTTPException(401, "wrong email or password")
            return {"token": make_jwt(cfg.jwt_secret, str(u.id), u.role, cfg.jwt_ttl_s, name=u.name), "user": _user(u)}

    @app.get("/api/me")
    def me(c: dict = Depends(claims)):
        return c

    # ------------------------------------------------------------------ interviews
    @app.post("/api/interviews")
    def create_interview(body: InterviewIn, c: dict = Depends(host)):
        mode = "meeting" if c["role"] == "meeting_host" or body.mode == "meeting" else "interview"
        with db.session() as s:
            iv = Interview(owner_id=int(c["sub"]), title=body.title, mode=mode, room=f"mbas-{secrets.token_hex(6)}",
                           invite_token=secrets.token_urlsafe(24), candidate_name=body.candidate_name,
                           scheduled_at=body.scheduled_at,
                           # a captured Meet / Zoom tab is one mixed stream, so meetings (per-person analysis) stay in our room
                           platform=body.platform if mode == "interview" else "interlens")
            s.add(iv)
            s.commit()
            return _iv(iv, cfg)

    @app.get("/api/interviews")
    def list_interviews(c: dict = Depends(host)):
        with db.session() as s:
            rows = s.scalars(select(Interview).where(Interview.owner_id == int(c["sub"])).order_by(Interview.id.desc()))
            return [_iv(iv, cfg) for iv in rows]

    @app.get("/api/interviews/{interview_id}")
    def get_interview(interview_id: int, c: dict = Depends(claims)):
        with db.session() as s:
            iv = own(s, interview_id, c)
            d = _iv(iv, cfg)
            if c["role"] not in HOST_ROLES:   # candidates see the call details only
                d = {k: d[k] for k in ("id", "title", "mode", "status", "room")}
            return d

    @app.get("/api/questions")
    def get_questions(c: dict = Depends(host)):
        return questions

    @app.post("/api/join/{invite}")
    def join(invite: str, body: JoinIn):
        """Candidate / participant entry from the invite link. Consent is required; no account needed."""
        if not body.consent:
            raise HTTPException(409, "consent is required to join")
        with db.session() as s:
            iv = s.scalar(select(Interview).where(Interview.invite_token == invite))
            if iv is None:
                raise HTTPException(404, "invalid link")
            role = "participant" if iv.mode == "meeting" else "candidate"
            identity = f"{role}-{secrets.token_hex(3)}"
            s.add(Participant(interview_id=iv.id, role=role, identity=identity, display_name=body.name,
                              consent_at=dt.datetime.now(dt.timezone.utc)))
            s.commit()
            tok = make_jwt(cfg.jwt_secret, identity, role, cfg.jwt_ttl_s, interview_id=iv.id, name=body.name)
            return {"token": tok, "interview": {"id": iv.id, "title": iv.title, "mode": iv.mode, "room": iv.room},
                    "identity": identity}

    @app.post("/api/interviews/{interview_id}/livekit-token")
    def lk_token(interview_id: int, c: dict = Depends(claims)):
        with db.session() as s:
            iv = own(s, interview_id, c)
            identity = c["sub"] if c["role"] not in HOST_ROLES else f"{c['role']}-{c['sub']}"
            return {"token": livekit_token(cfg.livekit_key, cfg.livekit_secret, iv.room, identity, c.get("name", identity),
                                           c["role"]), "url": cfg.livekit_url, "identity": identity}

    @app.post("/api/interviews/{interview_id}/capture-token")
    def capture_token(interview_id: int, platform: str = "meet", c: dict = Depends(host)):
        """Token for the interviewer's browser to publish a captured external call (a Google Meet / Zoom / Teams tab,
        or the Zoom desktop app's window with system audio) as the candidate's camera + mic, so the bot and workers
        analyse a call held elsewhere exactly like one held in our own room."""
        names = {"meet": "Google Meet", "zoom": "Zoom", "teams": "Microsoft Teams"}
        if platform not in names:
            raise HTTPException(422, f"platform must be one of {sorted(names)}")
        with db.session() as s:
            iv = own(s, interview_id, c)
            role = "candidate" if iv.mode == "interview" else "participant"
            identity = f"call-capture-{iv.id}"
            return {"token": livekit_token(cfg.livekit_key, cfg.livekit_secret, iv.room, identity,
                                           iv.candidate_name or names[platform], role),
                    "url": cfg.livekit_url, "identity": identity}

    @app.post("/api/interviews/{interview_id}/start")
    def start(interview_id: int, c: dict = Depends(host)):
        with db.session() as s:
            iv = own(s, interview_id, c)
            if iv.status == "live":                 # a second Start must not move the clock the workers already use
                return _iv(iv, cfg)
            if iv.status == "ended":
                raise HTTPException(409, "this interview has already ended")
            iv.status, iv.started_at = "live", dt.datetime.now(dt.timezone.utc)
            s.commit()
            bus.publish("control", {"type": "start", "interview_id": iv.id, "room": iv.room, "mode": iv.mode,
                                    "platform": getattr(iv, "platform", "interlens"), "started_at": iv.started_at.timestamp()})
            return _iv(iv, cfg)

    @app.post("/api/interviews/{interview_id}/stop")
    def stop(interview_id: int, c: dict = Depends(host)):
        with db.session() as s:
            iv = own(s, interview_id, c)
            if iv.status == "ended":
                return _iv(iv, cfg)
            iv.status, iv.ended_at = "ended", dt.datetime.now(dt.timezone.utc)
            s.commit()
            bus.publish("control", {"type": "stop", "interview_id": iv.id, "t_s": t_rel(iv)})
            return _iv(iv, cfg)

    @app.post("/api/interviews/{interview_id}/events")
    def question_event(interview_id: int, body: EventIn, c: dict = Depends(host)):
        if body.type not in ("ask", "answer_end"):
            raise HTTPException(422, "type must be ask or answer_end")
        with db.session() as s:
            iv = own(s, interview_id, c)
            t = t_rel(iv)
            if body.type == "ask":
                n = len(s.scalars(select(Question).where(Question.interview_id == iv.id)).all())
                q = Question(interview_id=iv.id, order=n + 1, question_id=body.question_id, text=body.text,
                             category=body.category, ask_s=t)
                s.add(q)
            else:
                q = s.scalars(select(Question).where(Question.interview_id == iv.id, Question.question_id == body.question_id,
                                                     Question.end_s.is_(None)).order_by(Question.id.desc())).first()
                if q is None:
                    raise HTTPException(409, "answer_end without ask")
                q.end_s = t
            s.commit()
            msg = {"type": body.type, "interview_id": iv.id, "question_ref": q.id, "question_id": q.question_id,
                   "text": q.text, "category": q.category, "ask_s": q.ask_s, "end_s": q.end_s, "t_s": t}
            bus.publish("control", msg)
            bus.publish(f"analytics:{iv.id}", {"kind": "question", **msg})
            return msg

    @app.put("/api/interviews/{interview_id}/notes")
    def notes(interview_id: int, body: NotesIn, c: dict = Depends(host)):
        with db.session() as s:
            iv = own(s, interview_id, c)
            iv.notes = body.notes
            s.commit()
            return {"ok": True}

    @app.get("/api/interviews/{interview_id}/data")
    def interview_data(interview_id: int, c: dict = Depends(host)):
        """Everything stored for the post-interview report (interviewer only)."""
        with db.session() as s:
            iv = own(s, interview_id, c)
            q = lambda model: [row_dict(r) for r in s.scalars(select(model).where(model.interview_id == iv.id))]  # noqa: E731
            return {"interview": _iv(iv, cfg), "questions": q(Question), "transcript": q(TranscriptSegment),
                    "windows": q(WindowScore), "answers": q(AnswerScore), "follow_ups": q(FollowUp),
                    "participants": q(Participant), "flags": q(Flag), "speech": q(SpeechStat)}

    # ------------------------------------------------------------------ reports (Stage 10)
    @app.post("/api/interviews/{interview_id}/report")
    def build_report(interview_id: int, c: dict = Depends(host)):
        from backend.reports import build_reports

        interview_data(interview_id, c)                       # ownership check first
        if not cfg.redis_url.startswith("memory://"):         # (unit tests have no worker to wait for)
            _wait_for_answers(interview_id)
        data = interview_data(interview_id, c)
        iv_view, cand_view = build_reports(data)
        with db.session() as s:
            r = s.scalar(select(Report).where(Report.interview_id == interview_id)) or Report(interview_id=interview_id)
            r.interviewer_view = iv_view
            if not r.released:
                r.candidate_view = cand_view
            s.add(r)
            s.commit()
            return row_dict(r)

    def _wait_for_answers(interview_id: int, timeout: float = 30.0):
        """Analysing a finished answer takes about 10 s on the worker. Opening the report right after the interview ends
        would miss the last answers, so wait (up to `timeout`) until every ended question has its analysis."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with db.session() as s:
                ended = {q.id for q in s.scalars(select(Question).where(Question.interview_id == interview_id, Question.end_s.is_not(None)))}
                done = {a.question_ref for a in s.scalars(select(AnswerScore).where(AnswerScore.interview_id == interview_id))}
            if ended <= done:
                return
            time.sleep(1.0)

    @app.get("/api/interviews/{interview_id}/report")
    def get_report(interview_id: int, c: dict = Depends(host)):
        with db.session() as s:
            own(s, interview_id, c)
            r = s.scalar(select(Report).where(Report.interview_id == interview_id))
            if r is None:
                raise HTTPException(404, "no report yet: POST to build it")
            return row_dict(r)

    @app.get("/api/interviews/{interview_id}/report.html")
    def download_report_html(interview_id: int, c: dict = Depends(host)):
        """The interviewer report as one self-contained HTML file."""
        from fastapi.responses import HTMLResponse

        from backend.report_html import render_report_html

        with db.session() as s:
            iv = own(s, interview_id, c)
            r = s.scalar(select(Report).where(Report.interview_id == interview_id))
            if r is None:
                raise HTTPException(404, "no report yet: build it first")
            page = render_report_html({"title": iv.title, "candidate_name": iv.candidate_name}, r.interviewer_view or {})
        return HTMLResponse(page, headers={"Content-Disposition": f'attachment; filename="interlens-report-{interview_id}.html"'})

    @app.put("/api/interviews/{interview_id}/report")
    def edit_report(interview_id: int, body: ReportEditIn, c: dict = Depends(host)):
        with db.session() as s:
            own(s, interview_id, c)
            r = s.scalar(select(Report).where(Report.interview_id == interview_id))
            if r is None:
                raise HTTPException(404, "no report yet")
            r.candidate_view = body.candidate_view
            s.commit()
            return row_dict(r)

    @app.post("/api/interviews/{interview_id}/report/release")
    def release(interview_id: int, c: dict = Depends(host)):
        with db.session() as s:
            own(s, interview_id, c)
            r = s.scalar(select(Report).where(Report.interview_id == interview_id))
            if r is None:
                raise HTTPException(404, "no report yet")
            r.released, r.released_at = True, dt.datetime.now(dt.timezone.utc)
            r.release_token = r.release_token or secrets.token_urlsafe(24)
            s.commit()
            return {"release_token": r.release_token, "url": f"{cfg.public_url}/feedback/{r.release_token}"}

    @app.get("/api/feedback/{token}")
    def feedback(token: str):
        """Candidate's released feedback (no analytics beyond what the interviewer released)."""
        with db.session() as s:
            r = s.scalar(select(Report).where(Report.release_token == token, Report.released.is_(True)))
            if r is None:
                raise HTTPException(404, "feedback not found or not released")
            iv = s.get(Interview, r.interview_id)
            return {"title": iv.title, "released_at": r.released_at.isoformat(), **r.candidate_view}

    # ------------------------------------------------------------------ analytics channel (interviewer only)
    @app.websocket("/ws/analytics/{interview_id}")
    async def analytics(ws: WebSocket, interview_id: int, token: str = ""):
        try:
            c = read_jwt(cfg.jwt_secret, token)
        except Exception:  # noqa: BLE001
            await ws.close(code=4401)
            return
        if c.get("role") not in HOST_ROLES:          # candidate / participant tokens are rejected server-side
            await ws.close(code=4403)
            return
        with db.session() as s:
            iv = s.get(Interview, interview_id)
            if iv is None or iv.owner_id != int(c["sub"]):
                await ws.close(code=4403)
                return
        await ws.accept()
        stream = f"analytics:{interview_id}"
        try:
            last = bus.last_id(stream)
            for m in catch_up(bus.history(stream, upto=last)):   # rebuild the cockpit state; live messages continue after `last`
                await ws.send_json(m)
            while True:
                items = await asyncio.to_thread(bus.read, {stream: last}, 1000, 200)
                for _, mid, msg, _ in items:
                    last = mid
                    msg["server_sent_at"] = time.time()
                    await ws.send_json(msg)
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.get("/metrics")
    def metrics():
        m = {k: v for k, v in bus.metrics().items() if not k.startswith("_")}
        return {"models": {k: v for k, v in m.items() if not k.startswith("e2e")},
                "end_to_end": {k: v for k, v in m.items() if k.startswith("e2e")}, "budget_ms": 2000}

    @app.get("/api/health")
    def health():
        return {"ok": True, "time": time.time()}

    return app


def _user(u: User) -> dict:
    return {"id": u.id, "email": u.email, "name": u.name, "role": u.role}


def _iv(iv: Interview, cfg: Settings) -> dict:
    d = row_dict(iv)
    d["candidate_link"] = f"{cfg.public_url}/join/{iv.invite_token}"
    return d


_app = None


def __getattr__(name):  # lazy app for `uvicorn backend.main:app` (tests import create_app without a database)
    global _app
    if name == "app":
        _app = _app or create_app()
        return _app
    raise AttributeError(name)
