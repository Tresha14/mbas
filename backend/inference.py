"""Model access for the workers: ONNX Runtime sessions (ml/export/onnx) with a PyTorch-checkpoint fallback.

A model that is not trained yet returns None and the UI shows "model not trained" instead of a number.
Every call is timed; workers report the latency to /metrics.
"""
from __future__ import annotations

import time
from functools import cached_property

import numpy as np

from ml.common.config import CHECKPOINTS, ROOT

ONNX = ROOT / "ml" / "export" / "onnx"


class Timer:
    def __init__(self, bus, key: str):
        self.bus, self.key = bus, key

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t) * 1000
        if self.bus is not None:
            self.bus.set_metric(self.key, self.ms)


def _session(name: str):
    p = ONNX / f"{name}.onnx"
    if not p.exists():
        return None
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.intra_op_num_threads = 2
    return ort.InferenceSession(str(p), o, providers=["CPUExecutionProvider"])


class Models:
    def __init__(self, bus=None):
        self.bus = bus

    # ---------------------------------------------------------------- speech
    @cached_property
    def vad(self):
        s = _session("vad")
        if s is not None:
            return lambda wav: 1 / (1 + np.exp(-s.run(None, {"wav": wav[None].astype(np.float32)})[0][0]))
        if (CHECKPOINTS / "vad" / "best.pt").exists():
            import torch

            from ml.speech.vad.train import load

            m = load()
            return lambda wav: torch.sigmoid(m(torch.from_numpy(wav[None].astype(np.float32))))[0].detach().numpy()
        return None

    @cached_property
    def asr(self):
        """torch module (streaming needs the chunk mask) or None."""
        if not ((CHECKPOINTS / "asr" / "averaged.pt").exists() or (CHECKPOINTS / "asr" / "best.pt").exists()):
            return None
        import torch

        from ml.common import text
        from ml.speech.asr.eval import load

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m, _ = load(dev)
        return m, text.load("asr"), dev

    @cached_property
    def disfluency(self):
        s = _session("disfluency")
        if s is not None:
            return lambda x: 1 / (1 + np.exp(-s.run(None, {"wav": x.astype(np.float32)})[0]))
        if (CHECKPOINTS / "disfluency" / "best.pt").exists():
            import torch

            from ml.behaviour.disfluency.eval import load

            m = load()
            return lambda x: torch.sigmoid(m(torch.from_numpy(x.astype(np.float32)))).detach().numpy()
        return None

    @cached_property
    def disfluency_thresholds(self) -> dict:
        import json

        p = ROOT / "docs" / "metrics" / "disfluency.json"
        return json.loads(p.read_text())["thresholds"] if p.exists() else {}

    # ---------------------------------------------------------------- response + fusion
    @cached_property
    def answer(self):
        if not (CHECKPOINTS / "answer_quality" / "best.pt").exists():
            return None
        from ml.common import text
        from ml.response.answer_quality.eval import load

        m, st = load()
        return m, st, text.load("answer")

    def answer_scores(self, question: str, transcript: str) -> dict | None:
        if self.answer is None:
            return None
        import torch

        from ml.common.datasets import FOLLOW_UP_REASONS
        from ml.response.answer_quality.model import AnswerQualityModel

        m, st, tok = self.answer
        ids = ([st["cls"]] + tok.encode(question)[:48] + [st["sep"]] + tok.encode(transcript))[:256]
        with Timer(self.bus, "answer_quality"), torch.no_grad():
            o = m(torch.tensor([ids]), st["sep"])
        star = torch.sigmoid(o["star"])[0].numpy()
        rs = torch.softmax(o["reason"], -1)[0].numpy()
        return {"relevance": float(torch.sigmoid(o["relevance"])[0]),
                "quality": float(AnswerQualityModel.expected_quality(o["quality"])[0]),
                "completeness": float(torch.sigmoid(o["completeness"])[0]),
                "star": dict(zip(["situation", "task", "action", "result"], np.round(star, 3).tolist())),
                "star_count": int((star > 0.5).sum()), "follow_up_p": float(torch.sigmoid(o["follow_up"])[0]),
                "reason_probs": dict(zip(FOLLOW_UP_REASONS, np.round(rs, 4).tolist()))}

    @cached_property
    def fusion(self):
        if not (CHECKPOINTS / "fusion" / "best.pt").exists():
            return None
        import torch

        from ml.fusion.model import FusionNet

        st = torch.load(CHECKPOINTS / "fusion" / "best.pt", map_location="cpu", weights_only=False)
        mc = st["cfg"]["model"]
        m = FusionNet(d=mc["d"], heads=mc["heads"], dropout=mc["dropout"])
        m.load_state_dict(st["model"])
        return m.eval(), st

    def fusion_scores(self, feats: dict) -> dict | None:
        if self.fusion is None:
            return None
        import torch

        from ml.common.datasets import FOLLOW_UP_REASONS, FUSION_FEATURES, FUSION_TARGETS
        from ml.fusion.explain import top_reasons
        from ml.fusion.model import MODALITIES

        m, st = self.fusion
        x = {}
        for mod, cols in FUSION_FEATURES.items():
            mu, sd = (np.array(v) for v in st["stats"][mod])
            v = np.array([feats.get(c, np.nan) for c in cols], float)
            x[mod] = np.nan_to_num((v - mu) / sd).astype(np.float32)
        with Timer(self.bus, "fusion"), torch.no_grad():
            o = m({k: torch.from_numpy(v[None]) for k, v in x.items()})
        flat = np.concatenate([x[k] for k in FUSION_FEATURES])
        with Timer(self.bus, "explain"):
            reasons = top_reasons(m, flat, feats, 3)
        scores = (o["scores"][0].numpy() * 100).round(1)
        return {"effectiveness": float(scores[0]),
                "subscores": dict(zip([t.replace("_score", "") for t in FUSION_TARGETS[1:]], scores[1:].tolist())),
                "follow_up_p": float(torch.sigmoid(o["follow_up"][0])),
                "reason": FOLLOW_UP_REASONS[int(o["reason"][0].argmax())],
                "modality_weights": dict(zip(MODALITIES, np.round(o["modality_weights"][0].numpy(), 3).tolist())),
                "reasons": reasons}
