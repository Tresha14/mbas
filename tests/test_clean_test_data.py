"""tools/clean_test_data: deletes only the named account's interviews that are not kept, with everything attached."""
from sqlalchemy import func, select

from backend.db import Interview, Question, SpeechStat, TranscriptSegment, WindowScore
from tests.test_backend import _host, _interview, env  # noqa: F401
from tools.clean_test_data import delete, plan


def _fill(db, iid):
    with db.session() as s:
        q = Question(interview_id=iid, order=1, question_id="Q1", text="x", category="hr", ask_s=1.0, end_s=9.0)
        s.add(q)
        s.add(WindowScore(interview_id=iid, start_s=0.0, end_s=5.0, features={}))
        s.add(SpeechStat(interview_id=iid, t_s=5.0))
        s.add(TranscriptSegment(interview_id=iid, speaker="candidate", start_s=1.0, end_s=2.0, text="hi"))
        s.commit()


def _count(db, table, iid):
    with db.session() as s:
        return s.scalar(select(func.count()).select_from(table).where(table.interview_id == iid))


def test_only_the_owners_unkept_interviews_are_deleted(env):
    c, db, bus = env
    mine, other = _host(c, "me@x.com"), _host(c, "someone@x.com")
    a, b, keep = _interview(c, mine)["id"], _interview(c, mine)["id"], _interview(c, mine)["id"]
    theirs = _interview(c, other)["id"]
    for i in (a, b, keep, theirs):
        _fill(db, i)
    bus.publish(f"analytics:{a}", {"kind": "status"})
    items = plan(db, "me@x.com", {keep})
    assert [it["id"] for it in items] == [a, b]                                   # not the kept one, not someone else's
    assert items[0]["rows"]["questions"] == 1 and items[0]["rows"]["windows"] == 1
    # planning deletes nothing
    assert _count(db, Question, a) == 1
    delete(db, bus, [a, b])
    for table in (Question, WindowScore, SpeechStat, TranscriptSegment):
        assert _count(db, table, a) == 0 and _count(db, table, b) == 0
        assert _count(db, table, keep) == 1 and _count(db, table, theirs) == 1
    with db.session() as s:
        left = {i.id for i in s.scalars(select(Interview))}
    assert a not in left and b not in left and keep in left and theirs in left
