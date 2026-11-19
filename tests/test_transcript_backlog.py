"""The live transcript must keep up with the call and lose nothing: Whisper runs in the background, answer analysis does not
block the audio loop, audio still queued when the interview stops is processed, and filler / hedge rates exist."""
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import numpy as np

from backend.bus import MemoryBus
from backend.db import Database
from backend.workers import speech_worker
from backend.workers.speech_worker import InterviewSpeech
from ml.behaviour.text_hesitation import HEDGES, count_phrases
from ml.speech.asr.whisper_streaming import FILLER_PROMPT, WhisperStreamingASR

SR, CH = 16000, 10240


class SlowModel:
    def __init__(self, delay=0.25, text="hello there world"):
        self.delay, self.text, self.prompts = delay, text, []

    def transcribe(self, audio, **kw):
        self.prompts.append(kw.get("initial_prompt"))
        time.sleep(self.delay)
        words = [NS(word=f" {w}", start=0.2 + 0.3 * i, end=0.45 + 0.3 * i) for i, w in enumerate(self.text.split())]
        return [NS(text=f" {self.text}", no_speech_prob=0.01, avg_logprob=-0.3, compression_ratio=1.2, words=words)], None


def loud_vad(chunk):
    return np.full(len(chunk) // 160, 1.0 if np.abs(chunk).max() > 0.05 else 0.0, np.float32)


def sound(seconds, loud):
    rng = np.random.default_rng(1)
    return (0.3 * rng.standard_normal(int(seconds * SR)) if loud else np.zeros(int(seconds * SR))).astype(np.float32)


def test_whisper_in_the_background_never_blocks_the_audio_loop_and_keeps_order():
    pool = ThreadPoolExecutor(max_workers=1)
    model = SlowModel(delay=0.4)
    asr = WhisperStreamingASR(model, loud_vad, pool=pool)
    asr.t0 = 50.0
    audio = np.concatenate([sound(2, True), sound(2.5, False), sound(2, True), sound(2.5, False), sound(2, True), sound(2.5, False)])
    t = time.time()
    early = []
    for k in range(0, len(audio), CH):
        early += asr.accept(audio[k:k + CH])
    assert time.time() - t < 0.5                                    # three 0.4 s transcriptions did not hold the loop up
    rest = asr.flush()                                              # flush waits for the background jobs
    got = early + rest
    assert len(got) == 3 and len(model.prompts) == 3
    assert [r.words[0]["start_s"] for r in got] == sorted(r.words[0]["start_s"] for r in got)      # still in order
    assert all(r.words[0]["start_s"] >= 50.0 for r in got)
    pool.shutdown()


def test_the_filler_prompt_is_sent_and_an_echoed_prompt_is_dropped():
    m = SlowModel(delay=0.0)
    out = WhisperStreamingASR(m, loud_vad).flush()
    asr = WhisperStreamingASR(m, loud_vad)
    for k in range(0, 5 * SR, CH):
        asr.accept(sound(5, True)[k:k + CH])
    asr.flush()
    assert m.prompts and all(p == FILLER_PROMPT for p in m.prompts) and out == []
    echo = SlowModel(delay=0.0, text="Umm, let me think, like, hmm...")           # Whisper repeating the prompt on noise
    a2 = WhisperStreamingASR(echo, loud_vad)
    res = []
    for k in range(0, 4 * SR, CH):
        res += a2.accept(sound(4, True)[k:k + CH])
    assert res + a2.flush() == []
    real = WhisperStreamingASR(SlowModel(delay=0.0, text="um so I think it is like fine"), loud_vad)
    res = []
    for k in range(0, 4 * SR, CH):
        res += real.accept(sound(4, True)[k:k + CH])
    assert len(res + real.flush()) == 1                                           # a genuine filler-rich sentence is kept


def test_a_job_that_cannot_keep_up_is_skipped_instead_of_lagging_for_ever():
    pool = ThreadPoolExecutor(max_workers=1)
    asr = WhisperStreamingASR(SlowModel(delay=0.3), loud_vad, pool=pool, max_pending=2)
    audio = np.concatenate([np.concatenate([sound(2, True), sound(2.5, False)]) for _ in range(5)])
    for k in range(0, len(audio), CH):
        asr.accept(audio[k:k + CH])
    assert len(asr.pending) <= 2
    asr.flush()
    pool.shutdown()


class _Models:
    vad = asr = disfluency = answer = fusion = None


def _pcm(level, k):
    rng = np.random.default_rng(k)
    return (np.clip(level * rng.standard_normal(CH), -1, 1) * 32767).astype(np.int16).tobytes()


def test_audio_still_queued_when_the_stop_arrives_is_processed():
    bus = MemoryBus()
    sp = InterviewSpeech(1, "interview", bus, Database("sqlite://"), _Models())
    live = {1: sp}
    ids = [bus.publish("audio:1:candidate", {"t_s": k * 0.64, "captured_at": 0.0}, _pcm(0.2, k)) for k in range(12)]
    cursors = {"control": "0", "audio:1:candidate": "0", "audio:1:interviewer": "0"}     # the worker has read nothing yet
    speech_worker.handle("control", {"type": "stop", "interview_id": 1, "t_s": 8.0}, None, bus, Database("sqlite://"), None, live, cursors)
    assert len(sp.tracks["candidate"].audio) == 12 and cursors["audio:1:candidate"] == ids[-1] and 1 not in live


def test_a_slow_answer_analysis_does_not_block_the_loop(monkeypatch):
    bus = MemoryBus()
    sp = InterviewSpeech(1, "interview", bus, Database("sqlite://"), _Models())
    done = []
    monkeypatch.setattr(sp, "answer_end", lambda ev: (time.sleep(0.6), done.append(ev["question_ref"])))
    live = {1: sp}
    t = time.time()
    speech_worker.handle("control", {"type": "answer_end", "interview_id": 1, "question_ref": 7, "ask_s": 1.0, "end_s": 20.0}, None, bus, None, None, live, {})
    assert time.time() - t < 0.3 and done == []                    # handled at once; the analysis is still running
    speech_worker.wait_for_analysis()
    assert done == [7]


def _say(sp, text, end_s):
    tr = sp.track("candidate")
    tr.finals.append({"start_s": end_s - 3, "end_s": end_s, "text": text, "words": []})


def test_filler_and_hedge_rates_exist_from_a_few_sentences():
    bus = MemoryBus()
    sp = InterviewSpeech(1, "interview", bus, Database("sqlite://"), _Models())
    tr = sp.track("candidate")
    tr.speech = np.ones(100 * 60, bool)
    _say(sp, "Um, so, like, I think maybe it was, you know, kind of hard.", 20.0)               # 13 words in the last minute
    sp.speaking(tr, 40.0)
    m = [x for x in bus.history("analytics:1") if x.get("kind") == "speaking"][-1]
    assert m["fillers_per_100w"] is not None and m["fillers_per_100w"] > 10 and m["hedges_per_100w"] > 10
    sp2 = InterviewSpeech(2, "interview", MemoryBus(), Database("sqlite://"), _Models())
    t2 = sp2.track("candidate")
    t2.speech = np.ones(100 * 60, bool)
    _say(sp2, "Hello there.", 10.0)                                                              # 2 words: too few to say anything
    sp2.speaking(t2, 40.0)
    m2 = [x for x in sp2.bus.history("analytics:2") if x.get("kind") == "speaking"][-1]
    assert m2["fillers_per_100w"] is None and m2["hedges_per_100w"] is None


def test_common_spoken_hedges_are_counted():
    for t in ("I suppose it is fine.", "It was, something like that.", "If I am not wrong, yes.", "I would say more or less."):
        assert count_phrases(t, HEDGES) >= 1, t


# ---------------------------------------------------------------------------- the filler prompt echoed back on noise
def _guard_result(text):
    from types import SimpleNamespace as NS

    class Model:
        def transcribe(self, audio, **kw):
            return [NS(text=f" {text}", no_speech_prob=0.0, avg_logprob=-0.2, compression_ratio=1.1, words=[])], None

    asr = WhisperStreamingASR(Model(), loud_vad)
    out = []
    for k in range(0, 4 * SR, CH):
        out += asr.accept(sound(4, True)[k:k + CH])
    return out + asr.flush()


def test_a_looped_prompt_is_dropped_but_real_stutters_and_stalls_stay():
    for echo in ("Okay, so, um, I think it's, like, kind of, like, kind of, like, kind of, like,",
                 "Okay, so, uh, I mean, I think it's, like, kind of, like, kind of, like, kind of, like,",
                 "Umm, let me think, like, hmm... Okay, so, uh, I mean, you know, I think maybe it's, like, kind of, um, basically."):
        assert _guard_result(echo) == [], echo
    for real in ("I think I think I think we should go with the second option",
                 "um so like you know I mean",
                 "Um, so, I think it is, like, a good fit for this role because of my project",
                 "I am a final year student and I built a web application for mock interviews"):
        assert len(_guard_result(real)) == 1, real


def test_subtitle_phrases_whisper_invents_are_removed_but_the_rest_is_kept():
    from ml.speech.asr.whisper_streaming import strip_hallucinations

    words = [{"word": w, "start_s": i * 0.3, "end_s": i * 0.3 + 0.25} for i, w in enumerate(["Is", "the", "transcript-", "and", "subscribe."])]
    text, kept = strip_hallucinations("Is the transcript- and subscribe.", words)
    assert text == "Is the transcript-" and [w["word"] for w in kept] == ["Is", "the", "transcript-"]
    assert kept[0]["start_s"] == 0.0 and kept[2]["start_s"] == 0.6                     # the remaining words keep their own times
    t2, _ = strip_hallucinations("Thanks for watching, see you.", [{"word": w, "start_s": 0, "end_s": 0} for w in "Thanks for watching, see you.".split()])
    assert t2 == "see you."
    t3, w3 = strip_hallucinations("I will subscribe to a newsletter about databases", [{"word": w, "start_s": 0, "end_s": 0} for w in "I will subscribe to a newsletter about databases".split()])
    assert "subscribe" not in t3                                                      # the word itself is never said in an interview here
    plain = "My project used SQL and I subscribed nowhere"
    assert strip_hallucinations(plain, [{"word": w, "start_s": 0, "end_s": 0} for w in plain.split()])[0] == plain
