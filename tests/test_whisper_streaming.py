"""WhisperStreamingASR: utterance splitting, session-time word stamps, and the guards against invented text.
Uses a fake model and VAD, so nothing is downloaded."""
from types import SimpleNamespace as NS

import numpy as np

from ml.speech.asr.whisper_streaming import WhisperStreamingASR

SR = 16000


class FakeModel:
    def __init__(self, text="hello there world", no_speech=0.01, logprob=-0.3, comp=1.2):
        self.calls, self.seg = [], dict(text=text, no_speech=no_speech, logprob=logprob, comp=comp)

    def transcribe(self, audio, **kw):
        self.calls.append(len(audio) / SR)
        s = self.seg
        words = [NS(word=f" {w}", start=0.2 + 0.3 * i, end=0.45 + 0.3 * i) for i, w in enumerate(s["text"].split())]
        return [NS(text=f" {s['text']}", no_speech_prob=s["no_speech"], avg_logprob=s["logprob"], compression_ratio=s["comp"], words=words)], None


def loud_vad(chunk):                 # "speech" wherever the signal is loud
    return np.full(len(chunk) // 160, 1.0 if np.abs(chunk).max() > 0.05 else 0.0, np.float32)


def audio(*parts):                   # ("q", seconds) | ("s", seconds)
    rng = np.random.default_rng(0)
    return np.concatenate([np.zeros(int(sec * SR), np.float32) if k == "q" else (0.3 * rng.standard_normal(int(sec * SR))).astype(np.float32) for k, sec in parts])


def feed(asr, x, step=10240):
    out = []
    for k in range(0, len(x), step):
        out += asr.accept(x[k:k + step])
    return out


def test_one_utterance_becomes_one_final_with_words_in_session_time():
    m = FakeModel()
    asr = WhisperStreamingASR(m, loud_vad)
    asr.t0 = 100.0                                   # the worker anchors the clock to the first chunk's session time
    out = feed(asr, audio(("q", 1.3), ("s", 3.0), ("q", 2.0)))
    assert len(out) == 1 and out[0].final and out[0].text == "hello there world" and len(m.calls) == 1
    ws = out[0].words
    assert [w["word"] for w in ws] == ["hello", "there", "world"]
    assert all(w["start_s"] >= 100.0 for w in ws)                      # not 0-based
    assert 100.4 < ws[0]["start_s"] < 102.0                            # ~1.3 s in, minus one chunk of lead-in


def test_silence_alone_never_calls_the_model():
    m = FakeModel()
    assert feed(WhisperStreamingASR(m, loud_vad), audio(("q", 8.0))) == [] and m.calls == []


def test_two_pauses_make_two_utterances_and_flush_finishes_the_last():
    m = FakeModel()
    asr = WhisperStreamingASR(m, loud_vad)
    out = feed(asr, audio(("s", 2.0), ("q", 2.5), ("s", 2.0)))        # the second one has not paused yet
    assert len(out) == 1
    assert len(asr.flush()) == 1 and len(m.calls) == 2


def test_a_very_long_utterance_is_split():
    m = FakeModel()
    out = feed(WhisperStreamingASR(m, loud_vad, max_utt_s=6.0), audio(("s", 14.0), ("q", 2.5)))
    assert len(out) >= 2


def test_text_whisper_invents_is_dropped():
    # whisper itself doubts there is speech
    assert feed(WhisperStreamingASR(FakeModel("thanks", no_speech=0.9, logprob=-1.5), loud_vad), audio(("s", 2.0), ("q", 2.5))) == []
    # a repetitive loop
    assert feed(WhisperStreamingASR(FakeModel("a a a a a a", comp=3.5), loud_vad), audio(("s", 2.0), ("q", 2.5))) == []
    # a stock phrase on a very short clip
    assert feed(WhisperStreamingASR(FakeModel("Thank you."), loud_vad), audio(("s", 1.0), ("q", 2.5))) == []
    # but a real sentence with the same words inside is kept
    assert len(feed(WhisperStreamingASR(FakeModel("thank you for having me"), loud_vad), audio(("s", 1.0), ("q", 2.5)))) == 1


def test_models_choose_the_backend_from_the_environment(monkeypatch):
    from backend.bus import MemoryBus
    from backend.inference import Models

    monkeypatch.setenv("MBAS_ASR", "own")
    assert Models(MemoryBus()).asr_backend == "own"
    monkeypatch.setenv("MBAS_ASR", "whisper")
    monkeypatch.setattr(Models, "whisper", property(lambda self: None))      # model missing -> fall back to our own ASR
    assert Models(MemoryBus()).asr_backend == "own"
