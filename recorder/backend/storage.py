"""File-based session storage for the recorder.

One folder per session under data/raw/<session_id>/. Every timestamp is in
milliseconds on the recorder server's clock (the shared clock); browsers
estimate their offset to it with /api/time before recording.

Files written per session:
    meta.json              session info, participants, consent, clock offsets
    candidate_webcam.webm  candidate camera (720p, 15 fps target)
    candidate_screen.webm  candidate screen share
    candidate_mic.wav      candidate microphone, 16 kHz mono 16-bit
    interviewer_mic.wav    interviewer microphone, 16 kHz mono 16-bit
    question_events.json   "ask" / "answer_end" button presses
    calibration.json       dot position per animation frame during calibration
    timeline.json          start/end of the recording and of every track
    state.json             recorder bookkeeping (chunk counters)
"""
from __future__ import annotations

import json
import secrets
import string
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import wav

AUDIO_SR = 16000

TRACKS: dict[str, dict[str, str]] = {
    "candidate_webcam": {"role": "candidate", "kind": "video", "file": "candidate_webcam.webm"},
    "candidate_screen": {"role": "candidate", "kind": "video", "file": "candidate_screen.webm"},
    "candidate_mic": {"role": "candidate", "kind": "audio", "file": "candidate_mic.wav"},
    "interviewer_mic": {"role": "interviewer", "kind": "audio", "file": "interviewer_mic.wav"},
}
ROLES = ("candidate", "interviewer")
EVENT_TYPES = ("ask", "answer_end")
CONSENT_VERSION = "v1"


class StoreError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def now_ms() -> float:
    return time.time() * 1000.0


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def _room_code(n: int = 6) -> str:
    alphabet = "".join(c for c in string.ascii_uppercase + string.digits if c not in "O0I1")
    return "".join(secrets.choice(alphabet) for _ in range(n))


@dataclass
class Session:
    dir: Path
    state: dict
    lock: threading.Lock

    @property
    def code(self) -> str:
        return self.state["room_code"]

    @property
    def session_id(self) -> str:
        return self.state["session_id"]

    def save(self) -> None:
        _write_json(self.dir / "state.json", self.state)


class SessionStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        for state_file in self.root.glob("*/state.json"):
            state = json.loads(state_file.read_text(encoding="utf-8"))
            self._sessions[state["room_code"]] = Session(state_file.parent, state, threading.Lock())

    # ---------- sessions ----------
    def create(self) -> Session:
        with self._lock:
            code = _room_code()
            while code in self._sessions:
                code = _room_code()
            sid = "R" + time.strftime("%Y%m%d-%H%M%S") + "-" + code
            d = self.root / sid
            d.mkdir(parents=True)
            state = {
                "session_id": sid,
                "room_code": code,
                "created_ms": now_ms(),
                "status": "created",  # created -> recording -> stopped -> finalized
                "recording": {"start_ms": None, "end_ms": None},
                "participants": {},
                "clock": {},
                "tracks": {},
                "calibration_saved": False,
            }
            s = Session(d, state, threading.Lock())
            s.save()
            _write_json(d / "question_events.json", {"session_id": sid, "clock": "server_epoch_ms", "events": []})
            self._sessions[code] = s
            return s

    def get(self, code: str) -> Session:
        s = self._sessions.get(code.upper())
        if s is None:
            raise StoreError(404, f"room {code} not found")
        return s

    def summary(self, s: Session) -> dict:
        st = s.state
        return {
            "session_id": st["session_id"],
            "room_code": st["room_code"],
            "status": st["status"],
            "recording": st["recording"],
            "participants": {r: {"participant_id": p["participant_id"]} for r, p in st["participants"].items()},
            "tracks": {
                k: {kk: v.get(kk) for kk in ("start_ms", "end_ms", "chunks", "bytes")}
                for k, v in st["tracks"].items()
            },
            "calibration_saved": st["calibration_saved"],
        }

    # ---------- participants / consent ----------
    def register_participant(self, s: Session, info: dict) -> None:
        role = info["role"]
        if role not in ROLES:
            raise StoreError(422, f"role must be one of {ROLES}")
        if not info.get("consent", {}).get("agreed"):
            raise StoreError(422, "consent must be given before joining")
        with s.lock:
            s.state["participants"][role] = {
                "participant_id": info["participant_id"],
                "age_band": info.get("age_band"),
                "gender": info.get("gender"),
                "glasses": info.get("glasses"),
                "accent": info.get("accent"),
                "consent": {
                    "agreed": True,
                    "version": info["consent"].get("version", CONSENT_VERSION),
                    "t_ms": now_ms(),
                },
                "user_agent": info.get("user_agent"),
            }
            s.save()

    def set_clock(self, s: Session, role: str, offset_ms: float, rtt_ms: float) -> None:
        with s.lock:
            s.state["clock"][role] = {"offset_ms": offset_ms, "rtt_ms": rtt_ms, "t_ms": now_ms()}
            s.save()

    # ---------- recording window ----------
    def recording(self, s: Session, action: str, t_ms: float) -> None:
        with s.lock:
            rec = s.state["recording"]
            if action == "start":
                if s.state["status"] != "created":
                    raise StoreError(409, f"cannot start recording in status {s.state['status']}")
                missing = [r for r in ROLES if r not in s.state["participants"]]
                if missing:
                    raise StoreError(409, f"waiting for consent from: {', '.join(missing)}")
                rec["start_ms"] = t_ms
                s.state["status"] = "recording"
            elif action == "stop":
                if s.state["status"] != "recording":
                    raise StoreError(409, "not recording")
                rec["end_ms"] = t_ms
                s.state["status"] = "stopped"
            else:
                raise StoreError(422, "action must be start or stop")
            s.save()

    # ---------- tracks ----------
    def _check_track(self, s: Session, track: str) -> dict:
        if track not in TRACKS:
            raise StoreError(404, f"unknown track {track}")
        role = TRACKS[track]["role"]
        if role not in s.state["participants"]:
            raise StoreError(403, f"{role} has not given consent")
        return TRACKS[track]

    def start_track(self, s: Session, track: str, body: dict) -> None:
        spec = self._check_track(s, track)
        with s.lock:
            if s.state["status"] not in ("recording",):
                raise StoreError(409, "session is not recording")
            if track in s.state["tracks"]:
                # Screen share may be stopped and restarted; keep one file per session.
                t = s.state["tracks"][track]
                if t.get("end_ms") is None:
                    raise StoreError(409, f"track {track} already started")
                raise StoreError(409, f"track {track} was already recorded in this session")
            path = s.dir / spec["file"]
            if spec["kind"] == "audio":
                if int(body.get("sample_rate", AUDIO_SR)) != AUDIO_SR:
                    raise StoreError(422, f"audio must be {AUDIO_SR} Hz")
                wav.create(path, AUDIO_SR)
            else:
                path.write_bytes(b"")
            s.state["tracks"][track] = {
                "file": spec["file"],
                "kind": spec["kind"],
                "role": spec["role"],
                "start_ms": float(body["start_ms"]),
                "end_ms": None,
                "mime": body.get("mime"),
                "settings": body.get("settings", {}),
                "chunks": 0,
                "bytes": 0,
            }
            s.save()

    def append_chunk(self, s: Session, track: str, seq: int, data: bytes) -> dict:
        spec = self._check_track(s, track)
        with s.lock:
            t = s.state["tracks"].get(track)
            if t is None:
                raise StoreError(409, f"track {track} not started")
            if t["end_ms"] is not None:
                raise StoreError(409, f"track {track} already stopped")
            expected = t["chunks"]
            if seq < expected:
                return {"duplicate": True, "next_seq": expected}
            if seq > expected:
                raise StoreError(409, f"out of order chunk {seq}, expected {expected}")
            path = s.dir / spec["file"]
            if spec["kind"] == "audio":
                if len(data) % 2:
                    raise StoreError(422, "PCM16 chunk must have an even number of bytes")
                wav.append(path, data, AUDIO_SR)
            else:
                with path.open("ab") as f:
                    f.write(data)
            t["chunks"] += 1
            t["bytes"] += len(data)
            s.save()
            return {"duplicate": False, "next_seq": t["chunks"]}

    def stop_track(self, s: Session, track: str, body: dict) -> None:
        self._check_track(s, track)
        with s.lock:
            t = s.state["tracks"].get(track)
            if t is None:
                raise StoreError(409, f"track {track} not started")
            t["end_ms"] = float(body["end_ms"])
            if "total_chunks" in body and int(body["total_chunks"]) != t["chunks"]:
                t["missing_chunks"] = int(body["total_chunks"]) - t["chunks"]
            s.save()

    # ---------- events / calibration ----------
    def add_event(self, s: Session, ev: dict) -> dict:
        if ev.get("type") not in EVENT_TYPES:
            raise StoreError(422, f"event type must be one of {EVENT_TYPES}")
        with s.lock:
            path = s.dir / "question_events.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            rec = {
                "type": ev["type"],
                "question_id": ev["question_id"],
                "t_ms": float(ev["t_ms"]),
                "received_ms": now_ms(),
            }
            if "question_text" in ev:
                rec["question_text"] = ev["question_text"]
            if "category" in ev:
                rec["category"] = ev["category"]
            doc["events"].append(rec)
            _write_json(path, doc)
            return rec

    def save_calibration(self, s: Session, body: dict) -> int:
        samples = body.get("samples") or []
        if not samples:
            raise StoreError(422, "calibration has no samples")
        with s.lock:
            doc = {
                "session_id": s.session_id,
                "clock": "server_epoch_ms",
                "start_ms": body["start_ms"],
                "end_ms": body["end_ms"],
                "screen": body.get("screen", {}),
                "pattern": body.get("pattern", ""),
                "samples": samples,
            }
            _write_json(s.dir / "calibration.json", doc)
            s.state["calibration_saved"] = True
            s.save()
        return len(samples)

    # ---------- finalize ----------
    def finalize(self, s: Session) -> None:
        with s.lock:
            st = s.state
            if st["status"] == "recording":
                raise StoreError(409, "stop the recording before finalizing")
            meta = {
                "session_id": st["session_id"],
                "room_code": st["room_code"],
                "created_ms": st["created_ms"],
                "clock": "server_epoch_ms",
                "audio_sample_rate": AUDIO_SR,
                "recording": st["recording"],
                "participants": st["participants"],
                "client_clock": st["clock"],
                "files": {k: v["file"] for k, v in st["tracks"].items()},
            }
            _write_json(s.dir / "meta.json", meta)
            timeline = {
                "session_id": st["session_id"],
                "clock": "server_epoch_ms",
                "recording": st["recording"],
                "tracks": {
                    k: {kk: v.get(kk) for kk in ("file", "kind", "role", "start_ms", "end_ms", "mime", "settings", "chunks", "bytes", "missing_chunks")}
                    for k, v in st["tracks"].items()
                },
            }
            _write_json(s.dir / "timeline.json", timeline)
            st["status"] = "finalized"
            s.save()
