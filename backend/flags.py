"""Flagged moments: shown live in the cockpit AND saved, so the report can list them after the interview."""
from __future__ import annotations

import traceback

from backend.db import Flag


def raise_flag(bus, db, interview_id: int, kind: str, t_s: float, text: str) -> None:
    bus.publish(f"analytics:{interview_id}", {"kind": "alert", "alert": kind, "t_s": t_s, "text": text})
    try:
        with db.session() as s:
            s.add(Flag(interview_id=interview_id, t_s=round(float(t_s), 2), kind=kind, text=text))
            s.commit()
    except Exception:  # noqa: BLE001 - never let storage break the live analysis
        traceback.print_exc()
