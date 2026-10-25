"""Session log (data/raw/session_log.csv) -> scripted segments and weak window labels.

The interviewer writes the start/end of the scripted segments (docs/data_protocol.md,
section 5). Windows that lie completely inside a segment get labels without
anyone watching the video:

    look_camera  -> attention on_screen,  engagement neutral
    look_phone   -> attention off_screen, engagement disengaged
    read         -> attention on_screen,  engagement neutral   (reading aloud from the screen)

Calibration windows come from calibration.json (dot on screen -> on_screen, neutral).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.session_index import SessionIndex

SEGMENT_LABELS = {
    "look_camera": ("on_screen", "neutral"),
    "look_phone": ("off_screen", "disengaged"),
    "read": ("on_screen", "neutral"),
    "calibration": ("on_screen", "neutral"),
}
LOG_COLUMNS = {"look_camera": ("look_camera_start_s", "look_camera_end_s"),
               "look_phone": ("look_phone_start_s", "look_phone_end_s"),
               "read": ("read_start_s", "read_end_s")}


def read_session_log(path: Path) -> dict[str, list[dict]]:
    """session_id -> [{segment, start_s, end_s}] for every filled-in scripted segment."""
    out: dict[str, list[dict]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("session_id") or "").strip()
            for seg, (a, b) in LOG_COLUMNS.items():
                try:
                    s, e = float(row.get(a) or ""), float(row.get(b) or "")
                except ValueError:
                    continue
                if e > s:
                    out.setdefault(sid, []).append({"segment": seg, "start_s": s, "end_s": e})
    return out


def calibration_segment(s: SessionIndex) -> dict | None:
    p = s.dir / "calibration.json"
    if not p.exists():
        return None
    cal = json.loads(p.read_text(encoding="utf-8"))
    if not cal.get("start_ms") or not cal.get("end_ms"):
        return None
    return {"segment": "calibration", "start_s": round((cal["start_ms"] - s.start_ms) / 1000, 3),
            "end_s": round((cal["end_ms"] - s.start_ms) / 1000, 3)}


def weak_window_labels(s: SessionIndex, segments: list[dict]) -> list[dict]:
    """Window label rows (annotator 'scripted') for windows fully inside a scripted segment."""
    segs = list(segments)
    cal = calibration_segment(s)
    if cal:
        segs.append(cal)
    rows = []
    for w in s.windows:
        for g in segs:
            if g["start_s"] - 1e-6 <= w.start_s and w.end_s <= g["end_s"] + 1e-6:
                att, eng = SEGMENT_LABELS[g["segment"]]
                rows.append({"window_id": w.window_id, "session_id": s.session_id,
                             "participant_id": s.candidate.get("participant_id"), "window_index": w.index,
                             "start_s": w.start_s, "end_s": w.end_s, "answer_id": w.answer_id,
                             "attention_label": att, "engagement_label": eng, "flags": f"scripted:{g['segment']}",
                             "annotator": "scripted", "annotation_rank": 0, "lead_time_s": None})
                break
    return rows
