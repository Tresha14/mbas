"""Duplicate lines from one voice on two microphones, recogniser context (name, question), per-role Whisper models, the
report waiting for the last answer, and attention that ignores frames with nobody in them."""
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.bus import MemoryBus
from backend.config import Settings
from backend.db import AnswerScore, Database, Question, TranscriptSegment
from backend.main import create_app
from backend.workers.speech_worker import InterviewSpeech
from ml.speech.asr.streaming import Result
from ml.speech.asr.whisper_streaming import FILLER_PROMPT, WhisperStreamingASR
from tests.test_backend import _host, _interview, env  # noqa: F401


class _Models:
    vad = asr = disfluency = answer = fusion = None


def _say(sp, speaker, text, start):
    sp.track(speaker)
    words = [{"word": w, "start_s": start + 0.3 * i, "end_s": start + 0.3 * i + 0.25} for i, w in enumerate(text.split())]
    sp.transcript(speaker, Result(text=text, final=True, words=words, t_s=start + 0.3 * len(words)), {"captured_at": time.time()})


def _sound(sp, speaker, t0, seconds, level):
    """Put `seconds` of noise at `level` into the track's audio buffer, in 0.64 s chunks."""
    tr = sp.track(speaker)
    rng = np.random.default_rng(3)
    for k in range(int(seconds / 0.64) + 1):
        tr.audio.append((t0 + k * 0.64, (level * rng.standard_normal(10240)).astype(np.float32)))


def _texts(db, iid):
    with db.session() as s:
        return sorted((r.speaker, r.text) for r in s.scalars(select(TranscriptSegment).where(TranscriptSegment.interview_id == iid)))


SENT = "hello this is a test of the whole system today"


def test_the_quieter_track_is_the_leak_whichever_arrives_first(env):
    c, db, bus = env
    iv = _interview(c, _host(c))
    # candidate loud, interviewer quiet (the interviewer's mic hears the speakers)
    sp = InterviewSpeech(iv["id"], "interview", bus, db, _Models())
    _sound(sp, "candidate", 8.0, 8.0, 0.3)
    _sound(sp, "interviewer", 8.0, 8.0, 0.01)
    _say(sp, "candidate", SENT, 10.0)
    _say(sp, "interviewer", SENT, 10.2)                                       # the duplicate arrives second: dropped
    assert _texts(db, iv["id"]) == [("candidate", SENT)]
    # the duplicate arrives FIRST (shown on screen), then the louder original: the earlier line is retracted
    sp2 = InterviewSpeech(iv["id"], "interview", bus, db, _Models())
    _sound(sp2, "candidate", 28.0, 8.0, 0.3)
    _sound(sp2, "interviewer", 28.0, 8.0, 0.01)
    _say(sp2, "interviewer", "we can see the face in the camera now", 30.0)
    assert ("interviewer", "we can see the face in the camera now") in _texts(db, iv["id"])
    _say(sp2, "candidate", "we can see the face in the camera now", 30.1)
    assert ("interviewer", "we can see the face in the camera now") not in _texts(db, iv["id"])
    assert ("candidate", "we can see the face in the camera now") in _texts(db, iv["id"])
    retracts = [m for m in bus.history(f"analytics:{iv['id']}") if m.get("kind") == "transcript_retract"]
    assert retracts and retracts[-1]["speaker"] == "interviewer"


def test_different_words_or_a_louder_interviewer_are_handled_correctly(env):
    c, db, bus = env
    iv = _interview(c, _host(c))
    sp = InterviewSpeech(iv["id"], "interview", bus, db, _Models())
    _sound(sp, "candidate", 48.0, 8.0, 0.3)
    _sound(sp, "interviewer", 48.0, 8.0, 0.3)
    _say(sp, "candidate", "my name is Trisha and I study computer science", 50.0)
    _say(sp, "interviewer", "tell me about your final year project please", 50.1)       # overlapping but different: both kept
    assert len(_texts(db, iv["id"])) == 2
    # here the interviewer is the loud one and the candidate mic hears them: the candidate's copy is the leak
    sp2 = InterviewSpeech(iv["id"], "interview", bus, db, _Models())
    _sound(sp2, "interviewer", 78.0, 8.0, 0.3)
    _sound(sp2, "candidate", 78.0, 8.0, 0.01)
    _say(sp2, "interviewer", "why did you choose that database for the backend", 80.0)
    _say(sp2, "candidate", "why did you choose that database for the backend", 80.1)
    assert ("candidate", "why did you choose that database for the backend") not in _texts(db, iv["id"])
    assert ("interviewer", "why did you choose that database for the backend") in _texts(db, iv["id"])


def test_recogniser_context_carries_the_name_and_the_question(env):
    c, db, bus = env
    r = c.post("/api/interviews", json={"title": "x", "candidate_name": "Trisha"}, headers=_host(c, "n@x.com"))
    iid = r.json()["id"]

    class FakeModel:
        pass

    class M(_Models):
        asr_backend = "whisper"
        whisper = FakeModel()
        whisper_pool = None

        def whisper_for(self, role):
            return self.whisper

    sp = InterviewSpeech(iid, "interview", bus, db, M())
    assert sp.candidate_name == "Trisha"
    asr = sp.track("candidate").asr
    assert isinstance(asr, WhisperStreamingASR) and asr.prompt.startswith(FILLER_PROMPT) and "Candidate: Trisha." in asr.prompt
    sp.on_ask({"question_ref": 1, "question_id": "Q1", "text": "Explain the difference between SQL and NoSQL.", "ask_s": 5.0})
    assert "Question: Explain the difference between SQL and NoSQL." in sp.tracks["candidate"].asr.prompt
    assert "Question:" not in (sp.track("interviewer").asr.prompt or "")           # the interviewer's track only gets the name
    sp.end_open(30.0)
    assert "Question:" not in sp.tracks["candidate"].asr.prompt                     # cleared when the answer ends
    # a generic placeholder name is not worth telling the recogniser
    sp2 = InterviewSpeech(iid, "interview", bus, db, M())
    sp2.candidate_name = "Candidate"
    assert sp2.asr_context("candidate") == ""


def test_context_never_changes_the_prompt_echo_guard():
    class Model:
        def transcribe(self, audio, **kw):
            from types import SimpleNamespace as NS
            self.prompt = kw.get("initial_prompt")
            return [NS(text=" Candidate: Trisha. Question: tell me", no_speech_prob=0.0, avg_logprob=-0.2, compression_ratio=1.1, words=[])], None

    asr = WhisperStreamingASR(Model(), lambda ch: np.ones(len(ch) // 160, np.float32), prompt=FILLER_PROMPT)
    asr.set_context("Candidate: Trisha. Question: tell me about yourself")
    assert asr.prompt.endswith("tell me about yourself") and asr._prompt_norm == " " + " ".join(
        "".join(c if c.isalpha() or c in "' " else " " for c in FILLER_PROMPT.lower()).split()) + " "


def test_the_candidate_can_use_a_bigger_whisper(monkeypatch):
    import backend.inference as inf
    import faster_whisper

    loaded = []

    class FakeW:
        def __init__(self, name, **kw):
            loaded.append(name)
            self.name = name

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeW)
    monkeypatch.setenv("MBAS_ASR", "whisper")
    monkeypatch.setenv("MBAS_WHISPER_MODEL", "base.en")
    monkeypatch.setenv("MBAS_WHISPER_MODEL_CANDIDATE", "small.en")
    m = inf.Models(MemoryBus())
    assert m.whisper_for("interviewer").name == "base.en" and m.whisper_for("candidate").name == "small.en"
    monkeypatch.setenv("MBAS_WHISPER_MODEL_CANDIDATE", "")
    m2 = inf.Models(MemoryBus())
    assert m2.whisper_for("candidate") is m2.whisper_for("interviewer")             # unset: one model for everyone


def test_the_report_waits_for_the_answer_that_is_still_being_analysed(tmp_path):
    cfg = Settings(database_url=f"sqlite:///{tmp_path / 'w.db'}", redis_url="redis://unused:1/0")       # a real (non-test) deployment
    db, bus = Database(cfg.database_url), MemoryBus()
    c = TestClient(create_app(cfg, db, bus))
    h = _host(c)
    iv = _interview(c, h)
    with db.session() as s:
        q = Question(interview_id=iv["id"], order=1, question_id="Q1", text="Tell me about yourself", category="hr", ask_s=5.0, end_s=40.0)
        s.add(q)
        s.commit()
        qid = q.id

    def finish_analysis():
        time.sleep(1.6)
        with db.session() as s:
            s.add(AnswerScore(interview_id=iv["id"], question_ref=qid, effectiveness=64.0, subscores={}, answer_model={}, behaviour={},
                              reasons=[], modality_weights={}, transcript="I built an app."))
            s.commit()

    threading.Thread(target=finish_analysis).start()
    t = time.time()
    rep = c.post(f"/api/interviews/{iv['id']}/report", headers=h).json()
    assert 1.0 < time.time() - t < 12.0 and len(rep["interviewer_view"]["cards"]) == 1          # it waited, and the answer is in


def test_attention_ignores_frames_with_nobody_in_them():
    from backend.reports import build_reports

    def win(t, face, att):
        return {"start_s": t - 5.0, "end_s": float(t), "attention": att, "engagement": "engaged", "on_screen_ratio": 0.9 if face else 0.0,
                "features": {"face_present_ratio": 1.0 if face else 0.0, "blink_rate_per_min": 16.0 if face else 0.0}}

    data = {"questions": [], "answers": [], "follow_ups": [], "transcript": [], "speech": [], "flags": [],
            "windows": [win(10, True, 80.0), win(20, True, 80.0), win(30, False, 5.0), win(40, False, 5.0)]}
    iv, _ = build_reports(data)
    assert iv["summary"]["mean_attention"] == 80.0                                  # not 42.5
    assert [p["attention"] for p in iv["timeline"]] == [80.0, 80.0, None, None]


# ---------------------------------------------------------------------------- relevance only where the model can judge it
def test_spoken_questions_are_matched_to_the_bank_only_when_they_clearly_are_bank_questions():
    from backend.question_bank import match_bank

    path = str(Settings().questions_path)
    assert match_bank("What is the difference between SQL and NoSQL databases?", path)[0] == "Q32"
    assert match_bank("Okay, tell me about, uh, will start with the interview first, so tell me about yourself.", path)[0] == "Q01"
    assert match_bank("So can you explain how a hash map works", path)[0] == "Q33"
    for free_form in ("What I wanted to ask was, uh, what is the capital of India?", "Is it getting detected or is it a fake data?",
                      "Tell me about your last job.", "", "Okay, uh, yes."):
        assert match_bank(free_form, path) is None, free_form
    assert match_bank("tell me about yourself", "no/such/file.json") is None                  # no bank: everything is free-form


def test_the_answer_model_abstains_on_free_form_questions(env, monkeypatch):
    from backend.workers import speech_worker

    c, db, bus = env
    iv = _interview(c, _host(c))
    calls = []

    class M(_Models):
        def answer_scores(self, question, transcript):
            calls.append(question)
            return {"relevance": 0.9, "quality": 3.0, "completeness": 0.8, "star": {}, "star_count": 2, "follow_up_p": 0.1}

        def fusion_scores(self, feats):
            return None

    beh = {"timing": {"response_latency_s": 1.0, "long_pause_count": 0}, "prosody": {}, "text_hesitation": {"filler_count": 0}}
    monkeypatch.setattr(speech_worker, "answer_behaviour", lambda *a, **k: beh)
    monkeypatch.setattr(speech_worker, "assemble", lambda *a, **k: {})
    sp = InterviewSpeech(iv["id"], "interview", bus, db, M())
    tr = sp.track("candidate")
    tr.finals.append({"start_s": 11.0, "end_s": 14.0, "text": "the capital of India is Delhi", "words": []})
    with db.session() as s:
        qs = []
        for i, qid in enumerate(("AUTO1", "AUTO2", "Q01")):
            q = Question(interview_id=iv["id"], order=i + 1, question_id=qid, text="x", category="hr", ask_s=10.0, end_s=20.0)
            s.add(q)
            qs.append(q)
        s.commit()
        refs = [q.id for q in qs]
    free = {"ask_s": 10.0, "end_s": 20.0, "question_ref": refs[0], "question_id": "AUTO1", "text": "what is the capital of India?"}
    sp.answer_end(free)
    assert calls == []                                                          # free-form: the model is not asked
    sp.answer_end({**free, "question_ref": refs[1], "question_id": "AUTO2", "text": "so tell me about, uh, yourself", "bank_text": "Tell me about yourself."})
    assert calls == ["Tell me about yourself."]                                 # heard and matched: scored against the bank wording
    sp.answer_end({**free, "question_ref": refs[2], "question_id": "Q01", "text": "Tell me about yourself."})
    assert calls == ["Tell me about yourself.", "Tell me about yourself."]      # clicked from the list: scored
    with db.session() as s:
        stored = {a.question_ref: a.answer_model for a in s.scalars(select(AnswerScore).where(AnswerScore.interview_id == iv["id"]))}
    assert stored[refs[0]] == {} and stored[refs[1]]["relevance"] == 0.9
    assert not [m for m in bus.history(f"analytics:{iv['id']}") if m.get("kind") == "follow_up" and m.get("reason") == "topic_not_covered"]
