import numpy as np
import torch

from backend.workers.speech_worker import ChunkVAD
from ml.speech.asr.model import ConformerCTC
from ml.speech.asr.streaming import StreamingASR

SR = 16000


class _Sp:
    def id_to_piece(self, i):
        return "▁a" if i % 2 else "▁b"


class _Tok:
    sp = _Sp()


def _audio(speech_s: list[tuple[float, float]], total_s: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.001, int(total_s * SR)).astype(np.float32)
    for a, b in speech_s:
        x[int(a * SR):int(b * SR)] += rng.normal(0, 0.1, int((b - a) * SR)).astype(np.float32)
    return x


def _run(vad, x):
    torch.manual_seed(0)
    model = ConformerCTC(vocab=20, d=16, heads=2, layers=1, kernel=5)
    s = StreamingASR(model, _Tok(), vad)
    out = []
    for i in range(0, len(x), 5000):
        out += s.accept(x[i:i + 5000])
    return out + s.flush()


def test_vad_closes_utterances_during_the_stream():
    x = _audio([(0.5, 4.0), (8.0, 11.0)], 14.0)
    res = _run(ChunkVAD(None), x)          # energy gate stands in for the trained VAD
    finals = [r for r in res if r.final]
    assert len(finals) == 2, [r.final for r in res]
    # the first utterance is closed well before the second one starts
    assert finals[0].t_s < 8.0


def test_without_a_vad_nothing_closes_until_flush():
    res = _run(None, _audio([(0.5, 4.0), (8.0, 11.0)], 14.0))
    assert [r.final for r in res].count(True) <= 1


def test_window_aligned_to_whole_chunks():
    s = StreamingASR(ConformerCTC(vocab=20, d=16, heads=2, layers=1, kernel=5), _Tok(), None)
    assert s.left % s.chunk == 0 and (s.left // 160) % 16 == 0


def test_running_stats_match_whole_utterance_normalisation():
    torch.manual_seed(0)
    m = ConformerCTC(vocab=20, d=16, heads=2, layers=1, kernel=5).eval()
    x = torch.randn(1, SR * 3) * 0.1
    n = torch.tensor([x.shape[1]])
    mel = m.front(x)
    stats = (mel.mean(-1, keepdim=True), mel.var(-1, unbiased=False, keepdim=True))
    a, _ = m.features(x, n)
    b, _ = m.features(x, n, stats)
    assert torch.allclose(a, b, atol=1e-3)


def test_retime_keeps_times_for_same_length_words_and_interpolates_the_rest():
    from ml.speech.asr.decode import retime

    old = [{"word": "helo", "start_s": 1.0, "end_s": 1.4}, {"word": "wurld", "start_s": 1.5, "end_s": 2.0}]
    same = retime(["hello", "world"], old)
    assert [(w["start_s"], w["end_s"]) for w in same] == [(1.0, 1.4), (1.5, 2.0)]
    more = retime(["hello", "big", "world"], old[:1] + old[1:])
    assert len(more) == 3 and more[0]["word"] == "hello"
    assert all(1.0 <= w["start_s"] <= w["end_s"] <= 2.0 for w in more)
    assert [w["word"] for w in more] == ["hello", "big", "world"]


def test_final_is_redecoded_with_beam_search_and_keeps_timestamps():
    from ml.speech.asr.decode import BeamDecoder

    torch.manual_seed(0)
    model = ConformerCTC(vocab=20, d=16, heads=2, layers=1, kernel=5)
    pieces = ["<blank>", "<unk>"] + [("▁" if i % 2 else "") + chr(97 + i) for i in range(18)]

    class Tok:
        sp = type("S", (), {"id_to_piece": staticmethod(lambda i: pieces[i])})()

        @staticmethod
        def pieces():
            return pieces

    s = StreamingASR(model, Tok(), ChunkVAD(None), decoder=BeamDecoder(pieces, None, beam=4))
    res = []
    x = _audio([(0.5, 4.0)], 7.0)
    for i in range(0, len(x), 5000):
        res += s.accept(x[i:i + 5000])
    res += s.flush()
    finals = [r for r in res if r.final]
    assert len(finals) == 1
    f = finals[0]
    assert f.text == " ".join(w["word"] for w in f.words)
    assert all(a["start_s"] <= b["start_s"] for a, b in zip(f.words, f.words[1:]))


def test_beam_latency_for_a_long_utterance_is_reasonable():
    import time

    from ml.speech.asr.decode import BeamDecoder

    rng = np.random.default_rng(0)
    pieces = ["<blank>", "<unk>"] + [("▁" if i % 3 == 0 else "") + f"p{i}" for i in range(998)]
    logits = rng.normal(0, 1, (375, 1000))                   # 15 s of frames, flat-ish posteriors (worst case)
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    d = BeamDecoder(pieces, None, beam=8, prune=-6.0)
    t = time.time()
    try:
        d.decode(logp, max_s=1.0)
    except TimeoutError:
        pass
    assert time.time() - t < 2.5         # the time budget bounds the worst case; the caller falls back to greedy


def test_streaming_falls_back_to_greedy_when_beam_search_times_out():
    from ml.speech.asr.decode import BeamDecoder

    torch.manual_seed(0)
    model = ConformerCTC(vocab=20, d=16, heads=2, layers=1, kernel=5)
    pieces = ["<blank>", "<unk>"] + [("▁" if i % 2 else "") + chr(97 + i) for i in range(18)]

    class Tok:
        sp = type("S", (), {"id_to_piece": staticmethod(lambda i: pieces[i])})()

        @staticmethod
        def pieces():
            return pieces

    s = StreamingASR(model, Tok(), ChunkVAD(None), decoder=BeamDecoder(pieces, None), beam_budget_s=-1.0)
    x = _audio([(0.5, 4.0)], 7.0)
    res = []
    for i in range(0, len(x), 5000):
        res += s.accept(x[i:i + 5000])
    res += s.flush()
    assert [r.final for r in res].count(True) == 1
