"""Read a recorded session folder into windows and answers on one time base.

All times returned here are seconds from the start of the recording window
(timeline.json -> recording.start_ms on the shared server clock). Window and
answer IDs follow the synthetic dataset: <session>_W0000 and <session>_A01.

Used by the Stage 2 annotation tools; Stage 3 windowing builds on the same
definitions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

WINDOW_S = 5.0
HOP_S = 2.5


@dataclass
class Answer:
    answer_id: str
    order: int
    question_id: str
    question_text: str
    category: str
    ask_s: float          # interviewer pressed "Ask" (question starts)
    end_s: float          # interviewer pressed "Answer ended"


@dataclass
class Window:
    window_id: str
    index: int
    start_s: float
    end_s: float
    answer_id: str | None   # answer that overlaps the window most, if any


@dataclass
class SessionIndex:
    session_id: str
    dir: Path
    duration_s: float
    start_ms: float
    candidate: dict
    track_offsets_s: dict[str, float] = field(default_factory=dict)  # track start - recording start
    answers: list[Answer] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)


def make_windows(session_id: str, duration_s: float, answers: list[Answer] | None = None,
                 window_s: float = WINDOW_S, hop_s: float = HOP_S) -> list[Window]:
    """5 s windows with a 2.5 s hop; only windows that fit completely in the recording."""
    out: list[Window] = []
    i = 0
    while i * hop_s + window_s <= duration_s + 1e-6:
        s = round(i * hop_s, 3)
        e = round(s + window_s, 3)
        best, best_ov = None, 0.0
        for a in answers or []:
            ov = min(e, a.end_s) - max(s, a.ask_s)
            if ov > best_ov:
                best, best_ov = a.answer_id, ov
        out.append(Window(f"{session_id}_W{i:04d}", i, s, e, best))
        i += 1
    return out


def answers_from_events(session_id: str, events: list[dict], start_ms: float) -> list[Answer]:
    """Pair every 'ask' with the next 'answer_end' of the same question."""
    out: list[Answer] = []
    open_ev: dict | None = None
    for e in sorted(events, key=lambda e: e["t_ms"]):
        if e["type"] == "ask":
            open_ev = e
        elif e["type"] == "answer_end" and open_ev and open_ev["question_id"] == e["question_id"]:
            order = len(out) + 1
            out.append(Answer(
                answer_id=f"{session_id}_A{order:02d}",
                order=order,
                question_id=e["question_id"],
                question_text=open_ev.get("question_text", ""),
                category=open_ev.get("category", ""),
                ask_s=round((open_ev["t_ms"] - start_ms) / 1000.0, 3),
                end_s=round((e["t_ms"] - start_ms) / 1000.0, 3),
            ))
            open_ev = None
    return out


def load_session(session_dir: str | Path) -> SessionIndex:
    d = Path(session_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    timeline = json.loads((d / "timeline.json").read_text(encoding="utf-8"))
    rec = timeline["recording"]
    start_ms, end_ms = rec["start_ms"], rec["end_ms"]
    events_path = d / "question_events.json"
    events = json.loads(events_path.read_text(encoding="utf-8")).get("events", []) if events_path.exists() else []
    sid = meta.get("session_id", d.name)
    idx = SessionIndex(
        session_id=sid,
        dir=d,
        duration_s=round((end_ms - start_ms) / 1000.0, 3),
        start_ms=start_ms,
        candidate=meta.get("participants", {}).get("candidate", {}) or {},
        track_offsets_s={name: round((t["start_ms"] - start_ms) / 1000.0, 3)
                         for name, t in timeline.get("tracks", {}).items() if t.get("start_ms") is not None},
    )
    idx.answers = answers_from_events(sid, events, start_ms)
    idx.windows = make_windows(sid, idx.duration_s, idx.answers)
    return idx


def find_sessions(raw_dir: str | Path) -> list[Path]:
    """Recorded session folders (have meta.json + timeline.json); skips the synthetic dataset and public data."""
    raw = Path(raw_dir)
    return sorted(p for p in raw.iterdir()
                  if p.is_dir() and (p / "meta.json").exists() and (p / "timeline.json").exists())
