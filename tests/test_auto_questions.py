"""Questions found in the interviewer's voice (ported from the older project) and how the worker uses them."""
import time

import pytest
from sqlalchemy import select

from backend.db import Question
from backend.question_detect import extract_question
from backend.workers import speech_worker
from backend.workers.speech_worker import InterviewSpeech
from ml.speech.asr.streaming import Result
from tests.test_backend import _host, _interview, env  # noqa: F401  (the fixture and helpers)


def test_only_real_questions_are_kept():
    q = extract_question
    assert q("Tell me about yourself and a project you are proud of.") == "Tell me about yourself and a project you are proud of."
    assert q("Hello? Hello? How are you? Tell me about your last job.") == "Tell me about your last job."    # chatter dropped
    assert q("Can you hear me? You are on mute.") is None                                                     # sound check
    assert q("Okay, so, um, yeah.") is None and q("") is None and q(None) is None                            # nothing to ask
    assert q("Why did you choose that technology for the backend?") is not None                               # '?' and wording
    assert q("What I did was build the backend in Python and then test it.") is None                          # candidate leaking in
    assert q("Good morning. Nice to meet you.") is None
    assert q("Explain how a hash map works") is not None                                                      # prompt wording, no '?'


class _Models:                      # nothing is loaded: only the question logic is under test
    vad = asr = disfluency = answer = fusion = None


def _say(sp, speaker, text, start):
    sp.track(speaker)
    words = [{"word": w, "start_s": start + 0.3 * i, "end_s": start + 0.3 * i + 0.25} for i, w in enumerate(text.split())]
    sp.transcript(speaker, Result(text=text, final=True, words=words, t_s=start + 0.3 * len(words)), {"captured_at": time.time()})


def _control(bus, kind=None):
    return [m for m in bus.history("control") if kind is None or m["type"] == kind]


def test_voice_questions_start_and_close_answers(env):
    c, db, bus = env
    iv = _interview(c, _host(c))
    sp = InterviewSpeech(iv["id"], "interview", bus, db, _Models(), platform="zoom_web")
    assert sp.auto_questions

    _say(sp, "interviewer", "Hello, can you hear me okay?", 2.0)
    assert not _control(bus, "ask") and sp.open_q is None                                  # chatter is not a question
    _say(sp, "interviewer", "Tell me about a project you are proud of.", 5.0)
    assert [m["question_id"] for m in _control(bus, "ask")] == ["AUTO1"] and sp.open_q["ask_s"] == pytest.approx(5.0, abs=0.5)
    _say(sp, "candidate", "Why did you choose that technology for the backend", 20.0)      # the candidate's words are never questions
    assert len(_control(bus, "ask")) == 1
    _say(sp, "interviewer", "Why did you choose that technology for it?", 60.0)
    ends = _control(bus, "answer_end")
    assert len(ends) == 1 and ends[0]["question_id"] == "AUTO1" and 59.0 < ends[0]["end_s"] < 60.0 and ends[0]["auto"]
    assert [m["question_id"] for m in _control(bus, "ask")] == ["AUTO1", "AUTO2"]
    with db.session() as s:
        qs = s.scalars(select(Question).where(Question.interview_id == iv["id"]).order_by(Question.order)).all()
    assert [q.question_id for q in qs] == ["AUTO1", "AUTO2"] and qs[0].end_s == pytest.approx(ends[0]["end_s"]) and qs[1].end_s is None


def test_a_click_suppresses_the_voice_duplicate_and_our_own_room_is_off_by_default(env):
    c, db, bus = env
    iv = _interview(c, _host(c))
    sp = InterviewSpeech(iv["id"], "interview", bus, db, _Models(), platform="teams")
    sp.on_ask({"type": "ask", "question_ref": 99, "question_id": "Q01", "text": "x", "ask_s": 100.0})     # the interviewer clicked Ask
    _say(sp, "interviewer", "Tell me about a time you led a team.", 110.0)                                  # the same question, spoken
    assert not _control(bus, "ask")
    _say(sp, "interviewer", "Describe a conflict with a teammate and how you resolved it.", 160.0)          # later: a new one
    assert [m["question_id"] for m in _control(bus, "ask")] == ["AUTO1"]
    own = InterviewSpeech(iv["id"], "interview", bus, db, _Models(), platform="interlens")
    assert not own.auto_questions                                                                            # the list is used in our room
    meeting = InterviewSpeech(iv["id"], "meeting", bus, db, _Models(), platform="teams")
    assert not meeting.auto_questions


def test_stopping_mid_answer_analyses_the_open_question(env, monkeypatch):
    c, db, bus = env
    iv = _interview(c, _host(c))
    sp = InterviewSpeech(iv["id"], "interview", bus, db, _Models(), platform="meet")
    _say(sp, "interviewer", "Explain the difference between a process and a thread.", 8.0)
    got = []
    monkeypatch.setattr(sp, "answer_end", lambda ev: got.append(ev))
    live = {iv["id"]: sp}
    speech_worker.handle("control", {"type": "stop", "interview_id": iv["id"], "t_s": 70.0}, None, bus, db, None, live, {})
    assert len(got) == 1 and got[0]["question_id"] == "AUTO1" and got[0]["end_s"] == 70.0 and iv["id"] not in live
