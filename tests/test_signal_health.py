"""Live signal health: the cockpit must be able to tell a dead microphone from a quiet speaker, and 'hearing speech
but no words' from 'nobody is talking'."""
import numpy as np

from backend.bus import MemoryBus
from backend.db import Database
from backend.workers.speech_worker import InterviewSpeech

CH = 10240


class _Models:
    vad = asr = disfluency = answer = fusion = None


def _chunk(level, k):
    rng = np.random.default_rng(k)
    return (np.clip(level * rng.standard_normal(CH), -1, 1) * 32767).astype(np.int16).tobytes()


def _feed(sp, level, start, n):
    for k in range(n):
        sp.audio("candidate", {"t_s": start + k * 0.64, "captured_at": 0.0}, _chunk(level, k))


def _health(bus):
    return [m for m in bus.history(f"analytics:{1}") if m.get("kind") == "health"]


def test_a_silent_microphone_is_reported_with_how_long_it_has_been_silent():
    bus = MemoryBus()
    sp = InterviewSpeech(1, "interview", bus, Database("sqlite://"), _Models())
    _feed(sp, 0.0, 0.0, 7)                                         # 4.5 s of digital silence (muted / not working)
    last = _health(bus)[-1]["mics"]["candidate"]
    assert last["silent_s"] >= 3.0 and last["speech_recent"] is False and last["since_words_s"] is None
    assert last["db"] is not None and last["db"] < -80


def test_a_loud_microphone_is_alive_and_hears_speech():
    bus = MemoryBus()
    sp = InterviewSpeech(1, "interview", bus, Database("sqlite://"), _Models())
    _feed(sp, 0.2, 0.0, 7)
    last = _health(bus)[-1]["mics"]["candidate"]
    assert last["silent_s"] < 1.5 and last["speech_recent"] is True and last["db"] > -30
    assert _health(bus)[-1]["asr"] == "own"                        # which recogniser is running is part of the message


def test_health_is_published_about_once_a_second_not_per_chunk():
    bus = MemoryBus()
    sp = InterviewSpeech(1, "interview", bus, Database("sqlite://"), _Models())
    _feed(sp, 0.2, 0.0, 7)                                         # 4.5 s of audio
    assert 3 <= len(_health(bus)) <= 6
