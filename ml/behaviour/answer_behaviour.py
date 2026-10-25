"""One behaviour JSON per answer (schema: docs/schemas/behaviour.json).

Combines the pause model, prosody, text and audio (disfluency CRNN) hesitation, the speaking-pattern MLP and
the hesitation index. Used by the Stage 9 workers and the Stage 8 feature assembler.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ml.behaviour.pauses import answer_pauses
from ml.behaviour.prosody import prosody
from ml.behaviour.text_hesitation import text_hesitation
from ml.common.config import CHECKPOINTS
from ml.common.datasets import SPEAKING_PATTERNS

SCHEMA_VERSION = "1.0"


@lru_cache(maxsize=1)
def _models():
    from ml.behaviour.answer_models import HesitationIndex
    from ml.vision.engagement.window_models import MLP

    st = torch.load(CHECKPOINTS / "behaviour" / "best.pt", map_location="cpu", weights_only=False)
    sp = MLP(len(st["pattern_features"]), len(SPEAKING_PATTERNS))
    sp.load_state_dict(st["speaking_pattern"])
    hi = HesitationIndex(len(st["hesitation_features"]))
    hi.load_state_dict(st["hesitation_index"])
    return st, sp.eval(), hi.eval()


def _std(x: np.ndarray, s: dict) -> np.ndarray:
    return np.nan_to_num((x - np.array(s["mu"])) / np.array(s["sd"])).astype(np.float32)


def answer_behaviour(answer_id: str, audio: np.ndarray, transcript: str, cand_segments: list[tuple[float, float]],
                     q_end_s: float, answer_end_s: float, clip_t0_s: float, audio_events: dict | None = None) -> dict:
    """audio = candidate mic from clip_t0_s; segments/times in session seconds; audio_events = disfluency CRNN counts."""
    pz = answer_pauses(cand_segments, q_end_s, answer_end_s)
    th = text_hesitation(transcript)
    a0 = int(max((pz["answer_start_s"] or q_end_s) - clip_t0_s, 0) * 16000)
    a1 = int(max(answer_end_s - clip_t0_s, 0) * 16000)
    pr = prosody(audio[a0:a1] if a1 > a0 + 1600 else audio, th["word_count"], pz["speech_s"] or None)
    ev = {"prolongation_count": 0, "block_count": 0, **(audio_events or {})}
    feats = {**pz, **pr, **th, **ev}
    out = {"schema_version": SCHEMA_VERSION, "answer_id": answer_id, "timing": pz, "prosody": pr,
           "text_hesitation": th, "audio_hesitation": ev}
    try:
        st, sp, hi = _models()
        xp = _std(np.array([[feats.get(k, np.nan) for k in st["pattern_features"]]], float), st["pattern_std"])
        xh = _std(np.array([[feats.get(k, np.nan) for k in st["hesitation_features"]]], float), st["hesitation_std"])
        with torch.no_grad():
            probs = torch.softmax(sp(torch.from_numpy(xp)), 1)[0].numpy()
            idx = float(100 * torch.sigmoid(hi(torch.from_numpy(xh)))[0, 0])
        contrib = (xh[0] * torch.nn.functional.softplus(hi.w).detach().numpy())
        top = [st["hesitation_features"][i] for i in np.argsort(-contrib)[:3] if contrib[i] > 0]
        out["speaking_pattern"] = {"label": SPEAKING_PATTERNS[int(probs.argmax())],
                                   "probabilities": dict(zip(SPEAKING_PATTERNS, np.round(probs, 4).tolist()))}
        out["hesitation_index"] = {"value": round(idx, 1), "top_contributors": top}
    except FileNotFoundError:
        out["speaking_pattern"] = None
        out["hesitation_index"] = None
    return out
