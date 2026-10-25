"""Explanations for every fusion score: Shapley values (permutation sampling, our own implementation of SHAP)
+ modality attention weights -> top-3 reasons in plain-English templates (no LLM).

shapley(): for one answer, average marginal contribution of each feature over random feature orderings,
with absent features set to the training mean (0 after standardisation).
"""
from __future__ import annotations

import numpy as np
import torch

from ml.common.datasets import FUSION_FEATURES

TEMPLATES = {
    # feature: (text when it pushes the score up, text when it pushes the score down, unit / format)
    "response_latency_s": ("answered promptly ({v:.1f} s)", "long pause before answering ({v:.1f} s)"),
    "long_pause_count": ("few long pauses ({v:.0f})", "{v:.0f} pauses longer than 1 s inside the answer"),
    "pause_ratio": ("steady flow of speech", "silent for {p:.0f}% of the answer"),
    "filler_rate_per_100w": ("few filler words", "frequent fillers ({v:.1f} per 100 words)"),
    "filler_count": ("few filler words", "{v:.0f} filler words (um, uh, matlab...)"),
    "repetition_count": ("no repeated words", "{v:.0f} word repetitions"),
    "restart_count": ("sentences finished cleanly", "{v:.0f} restarted sentences"),
    "hedge_count": ("stated answers with confidence", "{v:.0f} hedges (I think, maybe, not sure)"),
    "disfluency_rate_per_100w": ("fluent delivery", "many disfluencies ({v:.1f} per 100 words)"),
    "speaking_rate_wpm": ("comfortable speaking pace ({v:.0f} wpm)", "speaking pace of {v:.0f} words per minute"),
    "monotony_index": ("varied, expressive voice", "monotone voice"),
    "pitch_std_hz": ("lively intonation", "little pitch variation"),
    "volume_consistency": ("steady volume", "uneven volume"),
    "energy_std_db": ("good vocal energy", "low vocal energy variation"),
    "mean_on_screen_ratio": ("kept eye contact with the screen ({p:.0f}%)", "looked away from the screen often (on screen {p:.0f}%)"),
    "mean_attention_score": ("attentive throughout ({v:.0f}/100)", "attention dropped ({v:.0f}/100)"),
    "engaged_window_fraction": ("looked engaged ({p:.0f}% of the time)", "rarely looked engaged ({p:.0f}%)"),
    "disengaged_window_fraction": ("rarely disengaged", "looked disengaged {p:.0f}% of the time"),
    "total_glance_away": ("few glances away", "{v:.0f} glances away from the screen"),
    "reading_pattern_detected": ("no sign of reading from another screen", "eye movements look like reading from another screen"),
    "ans_relevance": ("answer stayed on the question (relevance {v:.2f})", "answer drifted from the question (relevance {v:.2f})"),
    "ans_quality": ("well-structured answer ({v:.1f}/5)", "answer quality was low ({v:.1f}/5)"),
    "ans_completeness": ("covered the expected points ({p:.0f}%)", "missed expected points (covered {p:.0f}%)"),
    "ans_star_count": ("used situation-task-action-result structure", "STAR structure incomplete ({v:.0f}/4 parts)"),
    "keyword_overlap": ("used the key terms of the topic", "few key terms of the topic"),
    "word_count": ("detailed answer ({v:.0f} words)", "short answer ({v:.0f} words)"),
    "answer_duration_s": ("answer length was appropriate ({v:.0f} s)", "answer length ({v:.0f} s) worked against it"),
    "ans_reason_vague": ("concrete examples", "answer sounded vague"),
    "ans_reason_contradiction": ("consistent statements", "possible contradiction in the answer"),
    "ans_reason_topic_not_covered": ("addressed what was asked", "part of the question was not covered"),
    "interrupted_by_interviewer": ("no interruptions", "was interrupted"),
    "asr_confidence_mean": ("clear pronunciation", "speech was hard to transcribe clearly"),
}


def flat_names() -> list[tuple[str, str]]:
    return [(m, c) for m, cols in FUSION_FEATURES.items() for c in cols]


def _to_dict(x: np.ndarray, names) -> dict[str, torch.Tensor]:
    out, i = {}, 0
    for m, cols in FUSION_FEATURES.items():
        out[m] = torch.from_numpy(x[:, i:i + len(cols)].astype(np.float32))
        i += len(cols)
    return out


@torch.no_grad()
def shapley(model, x_flat: np.ndarray, output: str = "effectiveness", n_perm: int = 48, seed: int = 0) -> np.ndarray:
    """Shapley values of every flat feature for one sample (x_flat (F,), standardised, baseline 0)."""
    rng = np.random.default_rng(seed)
    F = len(x_flat)
    names = flat_names()

    def f(batch):
        o = model(_to_dict(batch, names))
        return (o["scores"][:, 0] * 100 if output == "effectiveness" else torch.sigmoid(o["follow_up"])).numpy()

    phi = np.zeros(F)
    for _ in range(n_perm):
        order = rng.permutation(F)
        batch = np.zeros((F + 1, F), np.float32)
        for k in range(F):
            batch[k + 1] = batch[k]
            batch[k + 1, order[k]] = x_flat[order[k]]
        v = f(batch)
        phi[order] += np.diff(v)
    return phi / n_perm


def describe(feature: str, raw: float, positive: bool) -> str:
    up, down = TEMPLATES.get(feature, (f"{feature.replace('_', ' ')} helped", f"{feature.replace('_', ' ')} lowered the score"))
    t = up if positive else down
    try:
        return t.format(v=raw, p=raw * 100 if raw <= 1.0 else raw)
    except (ValueError, KeyError):
        return t


def top_reasons(model, x_flat: np.ndarray, raw: dict[str, float], k: int = 3, output: str = "effectiveness") -> list[dict]:
    """Top-k features by |Shapley value| with direction and a template sentence."""
    phi = shapley(model, x_flat, output)
    names = flat_names()
    order = np.argsort(-np.abs(phi))
    out = []
    for i in order:
        m, c = names[i]
        if abs(phi[i]) < 1e-3 or c.startswith("mfcc"):
            continue
        sign = phi[i] > 0 if output == "effectiveness" else phi[i] < 0   # for follow-up, lowering risk is 'good'
        out.append({"feature": c, "modality": m, "contribution": round(float(phi[i]), 3), "raw_value": raw.get(c),
                    "text": describe(c, float(raw.get(c, np.nan)), bool(sign))})
        if len(out) == k:
            break
    return out
