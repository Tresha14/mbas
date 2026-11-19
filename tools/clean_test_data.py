"""Delete the interviews one account created for testing, with everything attached to them. Dry run unless --yes.

    python -m tools.clean_test_data --owner demo.interviewer@mbas.local --keep 1,60          # shows what WOULD go
    python -m tools.clean_test_data --owner demo.interviewer@mbas.local --keep 1,60 --yes    # deletes it

Deleting cannot be undone. Only interviews owned by --owner and not listed in --keep are touched; other accounts' interviews are never
selected. For each interview it removes the questions, transcript, vision windows, speech stats, flags, follow-ups, answer scores,
participants and report, and the Redis streams that hold its audio, frames and live analytics.
"""
from __future__ import annotations

import argparse
import sys


def plan(db, owner_email: str, keep: set[int]) -> list[dict]:
    """What would be deleted: [{id, title, status, rows}] for the owner's interviews that are not in `keep`."""
    from sqlalchemy import func, select

    from backend.db import (AnswerScore, Flag, FollowUp, Interview, Participant, Question, Report, SpeechStat, TranscriptSegment, User,
                            WindowScore)

    tables = {"questions": Question, "transcript": TranscriptSegment, "windows": WindowScore, "speech": SpeechStat, "flags": Flag,
              "follow_ups": FollowUp, "answers": AnswerScore, "participants": Participant, "reports": Report}
    out = []
    with db.session() as s:
        user = s.scalar(select(User).where(User.email == owner_email))
        if user is None:
            raise SystemExit(f"no account with the email {owner_email}")
        for iv in s.scalars(select(Interview).where(Interview.owner_id == user.id).order_by(Interview.id)):
            if iv.id in keep:
                continue
            rows = {k: s.scalar(select(func.count()).select_from(t).where(t.interview_id == iv.id)) or 0 for k, t in tables.items()}
            out.append({"id": iv.id, "title": iv.title, "status": iv.status, "rows": rows})
    return out


def delete(db, bus, ids: list[int]) -> None:
    from sqlalchemy import delete as sql_delete

    from backend.db import (AnswerScore, Flag, FollowUp, Interview, Participant, Question, Report, SpeechStat, TranscriptSegment,
                            WindowScore)

    order = (AnswerScore, FollowUp, Flag, SpeechStat, WindowScore, TranscriptSegment, Question, Participant, Report)   # children first
    with db.session() as s:
        for i in ids:
            for t in order:
                s.execute(sql_delete(t).where(t.interview_id == i))
            s.execute(sql_delete(Interview).where(Interview.id == i))
        s.commit()
    r = getattr(bus, "r", None)                                  # Redis: audio / frames / analytics streams of those interviews
    if r is not None:
        for i in ids:
            for pattern in (f"audio:{i}:*", f"frames:{i}", f"analytics:{i}"):
                for key in r.scan_iter(pattern):
                    r.delete(key)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", required=True, help="email of the account whose test interviews to delete")
    ap.add_argument("--keep", default="", help="comma-separated interview ids to keep")
    ap.add_argument("--yes", action="store_true", help="really delete (without it, only show what would go)")
    args = ap.parse_args(argv)

    from backend.bus import make_bus
    from backend.config import settings
    from backend.db import Database

    cfg = settings()
    db = Database(cfg.database_url)
    keep = {int(x) for x in args.keep.split(",") if x.strip()}
    items = plan(db, args.owner, keep)
    if not items:
        print("nothing to delete")
        return 0
    for it in items:
        rows = ", ".join(f"{k} {v}" for k, v in it["rows"].items() if v)
        print(f"  #{it['id']:<4d} {it['status']:9s} {it['title'][:48]:48s} {rows}")
    print(f"\n{len(items)} interview(s) of {args.owner}; kept: {sorted(keep) or 'none'}")
    if not args.yes:
        print("Dry run: nothing was deleted. Add --yes to delete these (it cannot be undone).")
        return 0
    delete(db, make_bus(cfg.redis_url), [it["id"] for it in items])
    print(f"deleted {len(items)} interview(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
