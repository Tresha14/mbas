"""PostgreSQL schema (SQLAlchemy 2): users, interviews, participants, questions, transcript_segments,
window_scores, answer_scores, follow_ups, reports. SQLite works too (tests, quick demos)."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event,
                        select)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20))                  # interviewer | meeting_host
    password_hash: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)


class Interview(Base):
    __tablename__ = "interviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    mode: Mapped[str] = mapped_column(String(20), default="interview")     # interview | meeting
    room: Mapped[str] = mapped_column(String(40), unique=True)
    invite_token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    candidate_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    scheduled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="scheduled")    # scheduled | live | ended
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)
    participants: Mapped[list["Participant"]] = relationship(back_populates="interview", cascade="all, delete-orphan")


class Participant(Base):
    __tablename__ = "participants"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))      # interviewer | candidate | meeting_host | participant
    identity: Mapped[str] = mapped_column(String(100))
    display_name: Mapped[str] = mapped_column(String(200))
    consent_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    joined_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)
    interview: Mapped[Interview] = relationship(back_populates="participants")


class Question(Base):
    """Questions asked in an interview (Ask / Answer ended timestamps, seconds from the interview start)."""
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    order: Mapped[int] = mapped_column(Integer)
    question_id: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(30), default="")
    ask_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float | None] = mapped_column(Float, nullable=True)


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    speaker: Mapped[str] = mapped_column(String(100))
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    words: Mapped[list[Any]] = mapped_column(JSON, default=list)
    fillers: Mapped[list[Any]] = mapped_column(JSON, default=list)


class WindowScore(Base):
    __tablename__ = "window_scores"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    participant: Mapped[str] = mapped_column(String(100), default="candidate")
    start_s: Mapped[float] = mapped_column(Float, index=True)
    end_s: Mapped[float] = mapped_column(Float)
    attention: Mapped[float | None] = mapped_column(Float, nullable=True)
    engagement: Mapped[str | None] = mapped_column(String(20), nullable=True)
    on_screen_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class AnswerScore(Base):
    __tablename__ = "answer_scores"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    question_ref: Mapped[int] = mapped_column(ForeignKey("questions.id"))
    effectiveness: Mapped[float | None] = mapped_column(Float, nullable=True)   # None until fusion is trained
    subscores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    answer_model: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    behaviour: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    modality_weights: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    transcript: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)


class FollowUp(Base):
    __tablename__ = "follow_ups"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), index=True)
    question_ref: Mapped[int | None] = mapped_column(ForeignKey("questions.id"), nullable=True)
    t_s: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(String(40))
    probability: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20), default="model")      # model | rule
    text: Mapped[str] = mapped_column(Text, default="")


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), unique=True)
    interviewer_view: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    candidate_view: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    released: Mapped[bool] = mapped_column(Boolean, default=False)
    release_token: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    released_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class Database:
    def __init__(self, url: str):
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        self.engine = create_engine(url, **kw)
        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", lambda c, _: c.execute("pragma foreign_keys=on"))
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self.Session()


def row_dict(obj) -> dict:
    out = {}
    for c in obj.__table__.columns:
        v = getattr(obj, c.name)
        out[c.name] = v.isoformat() if isinstance(v, dt.datetime) else v
    return out


__all__ = ["Database", "User", "Interview", "Participant", "Question", "TranscriptSegment", "WindowScore",
           "AnswerScore", "FollowUp", "Report", "row_dict", "select", "json"]
