"""Export every trained model to ONNX (ml/export/onnx/<name>.onnx) and check ONNX Runtime matches PyTorch.

    python -m ml.export.export_onnx              # all models whose checkpoint exists
    python -m ml.export.export_onnx face_detector gaze_iris
Then: python -m ml.export.benchmark
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from ml.common.config import CHECKPOINTS, save_json

OUT = Path(__file__).resolve().parent / "onnx"
OPSET = 17


def _face():
    from ml.vision.face_detector.eval import load_model

    m = load_model()
    return m, (torch.zeros(1, 3, 128, 128),), ["image"], ["logits", "boxes", "keypoints"], None


def _gaze():
    from ml.vision.gaze_iris.eval import load_model

    m, _ = load_model()
    z = torch.zeros(1, 1, 36, 60)
    return m, (z, z, torch.zeros(1, 2)), ["left", "right", "head"], ["gaze", "iris", "onscreen"], \
        {"left": {0: "b"}, "right": {0: "b"}, "head": {0: "b"}}


def _pose():
    from ml.vision.headpose.model import load

    return load(), (torch.zeros(1, 3, 64, 64),), ["face"], ["pose"], {"face": {0: "b"}}


def _engagement():
    from ml.vision.engagement.eval import load

    m, _ = load("engagement", "cpu")
    if m is None:
        m, _ = load("engagement_daisee", "cpu")
    if m is None:
        raise FileNotFoundError("no engagement checkpoint")
    return m, (torch.zeros(1, 16, 3, 112, 112),), ["frames"], ["engagement", "boredom", "confusion"], None


def _window():
    from ml.vision.engagement.window_models import MLP

    st = torch.load(CHECKPOINTS / "window_models" / "best.pt", map_location="cpu", weights_only=False)
    eng = MLP(len(st["engagement_features"]), 3)
    eng.load_state_dict(st["engagement"])
    return eng.eval(), (torch.zeros(1, len(st["engagement_features"])),), ["features"], ["engagement"], {"features": {0: "b"}}


def _attention():
    from ml.vision.engagement.window_models import MLP

    st = torch.load(CHECKPOINTS / "window_models" / "best.pt", map_location="cpu", weights_only=False)
    m = MLP(len(st["attention_features"]), 2, layers=0 if st["attention_kind"] == "linear" else 2)
    m.load_state_dict(st["attention"])
    return m.eval(), (torch.zeros(1, len(st["attention_features"])),), ["features"], ["attention"], {"features": {0: "b"}}


def _vad():
    from ml.speech.vad.train import load

    return load(), (torch.zeros(1, 16000),), ["wav"], ["logits"], {"wav": {1: "samples"}, "logits": {1: "frames"}}


class _ASRWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, wav):
        lp, _ = self.m(wav, torch.full((wav.shape[0],), wav.shape[1], dtype=torch.long))
        return lp


def _asr():
    from ml.speech.asr.eval import load

    m, _ = load()
    return _ASRWrap(m).eval(), (torch.zeros(1, 79360),), ["wav"], ["log_probs"], {"wav": {1: "samples"}, "log_probs": {1: "frames"}}


def _disfluency():
    from ml.behaviour.disfluency.eval import load

    return load(), (torch.zeros(1, 48000),), ["wav"], ["logits"], {"wav": {0: "b"}}


class _AQWrap(torch.nn.Module):
    def __init__(self, m, sep):
        super().__init__()
        self.m, self.sep = m, sep

    def forward(self, ids):
        o = self.m(ids, self.sep)
        return o["relevance"], o["quality"], o["star"], o["completeness"], o["follow_up"], o["reason"]


def _answer():
    from ml.response.answer_quality.eval import load

    m, st = load()
    return _AQWrap(m, st["sep"]).eval(), (torch.ones(1, 64, dtype=torch.long),), ["ids"], \
        ["relevance", "quality", "star", "completeness", "follow_up", "reason"], {"ids": {0: "b", 1: "len"}}


class _FusionWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, vision, prosody, hesitation, content, conversation):
        o = self.m({"vision": vision, "prosody": prosody, "hesitation": hesitation, "content": content,
                    "conversation": conversation})
        return o["scores"], o["follow_up"], o["reason"], o["modality_weights"]


def _fusion():
    from ml.common.datasets import FUSION_FEATURES
    from ml.fusion.model import FusionNet

    st = torch.load(CHECKPOINTS / "fusion" / "best.pt", map_location="cpu", weights_only=False)
    mc = st["cfg"]["model"]
    m = FusionNet(d=mc["d"], heads=mc["heads"], dropout=mc["dropout"])
    m.load_state_dict(st["model"])
    ins = tuple(torch.zeros(1, len(c)) for c in FUSION_FEATURES.values())
    return _FusionWrap(m.eval()), ins, list(FUSION_FEATURES), ["scores", "follow_up", "reason", "modality_weights"], \
        {k: {0: "b"} for k in FUSION_FEATURES}


def _behaviour():
    from ml.common.datasets import SPEAKING_PATTERNS
    from ml.vision.engagement.window_models import MLP

    st = torch.load(CHECKPOINTS / "behaviour" / "best.pt", map_location="cpu", weights_only=False)
    m = MLP(len(st["pattern_features"]), len(SPEAKING_PATTERNS))
    m.load_state_dict(st["speaking_pattern"])
    return m.eval(), (torch.zeros(1, len(st["pattern_features"])),), ["features"], ["pattern"], {"features": {0: "b"}}


def _meeting():
    from ml.fusion.model import MeetingHead

    st = torch.load(CHECKPOINTS / "meeting_head" / "best.pt", map_location="cpu", weights_only=False)
    m = MeetingHead(len(st["features"]))
    m.load_state_dict(st["model"])
    return m.eval(), (torch.zeros(1, len(st["features"])),), ["features"], ["meeting"], {"features": {0: "b"}}


MODELS = {"face_detector": _face, "gaze_iris": _gaze, "headpose": _pose, "engagement": _engagement,
          "engagement_window": _window, "attention": _attention, "vad": _vad, "asr": _asr, "disfluency": _disfluency,
          "answer_quality": _answer, "fusion": _fusion, "speaking_pattern": _behaviour, "meeting_head": _meeting}


def export(name: str) -> dict:
    import onnxruntime as ort

    model, inputs, in_names, out_names, dyn = MODELS[name]()
    model = model.cpu().eval()
    torch.backends.mha.set_fastpath_enabled(False)   # the fused transformer kernel has no ONNX equivalent
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.onnx"
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, inputs, str(path), input_names=in_names, output_names=out_names,
                          dynamic_axes=dyn, opset_version=OPSET, dynamo=False)
        ref = model(*inputs)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    ref = [r for r in ref if torch.is_tensor(r)]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {n: x.numpy() for n, x in zip(in_names, inputs)})
    err = max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref))
    return {"file": path.name, "mb": round(path.stat().st_size / 1e6, 2), "max_abs_diff_vs_torch": round(err, 6)}


def main(argv: list[str] | None = None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(MODELS)
    report = {}
    for n in names:
        try:
            report[n] = export(n)
            print(f"{n:18s} -> {report[n]}")
        except FileNotFoundError as e:
            report[n] = {"skipped": str(e).splitlines()[0][:120]}
            print(f"{n:18s} skipped ({report[n]['skipped']})")
        except Exception as e:  # noqa: BLE001 - keep exporting the others
            report[n] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}
            print(f"{n:18s} ERROR {report[n]['error']}")
    prev = OUT / "export_report.json"
    old = __import__("json").loads(prev.read_text()) if prev.exists() else {}
    save_json({**old, **report}, prev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
