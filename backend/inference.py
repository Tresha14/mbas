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


def _add_cuda_dlls() -> bool:
    """Make the CUDA libraries from the pip packages nvidia-cublas-cu12 (and friends) findable on Windows.
    True when cuBLAS 12 can be found (CTranslate2 bundles cuDNN itself but not cuBLAS)."""
    import os
    import shutil
    import sysconfig
    from pathlib import Path

    base = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    found = shutil.which("cublas64_12.dll") is not None
    for d in sorted(base.glob("*/bin")) if base.exists() else []:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        found = found or (d / "cublas64_12.dll").exists() or any(d.glob("libcublas.so*"))
    return found or os.name != "nt"


def _cuda_whisper_works() -> tuple[bool, str]:
    """Can faster-whisper transcribe on the GPU? Checked in a child process with a time limit: when a CUDA library is
    missing, CTranslate2 can hang or crash the whole process instead of raising an error."""
    import subprocess
    import sys

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() < 1:
            return False, "no NVIDIA GPU visible"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    if not _add_cuda_dlls():
        return False, "cuBLAS missing: pip install nvidia-cublas-cu12"
    probe = ("import numpy as np; from backend.inference import _add_cuda_dlls; _add_cuda_dlls(); "
             "from faster_whisper import WhisperModel; from backend.config import settings; "
             "m = WhisperModel(settings().whisper_model, device='cuda', compute_type='float16'); "
             "s, _ = m.transcribe(np.zeros(16000, np.float32), language='en', beam_size=1); list(s); print('OK')")
    try:
        r = subprocess.run([sys.executable, "-c", probe], cwd=str(ROOT), capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return False, "the GPU test did not finish in 90 s"
    if r.returncode == 0 and "OK" in r.stdout:
        return True, ""
    return False, ((r.stderr or r.stdout).strip().splitlines() or ["the GPU test failed"])[-1][:200]


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
    def asr_backend(self) -> str:
        """Which recogniser runs: 'whisper' only when asked for AND available, otherwise our own."""
        from backend.config import settings

        return "whisper" if settings().asr_backend == "whisper" and self.whisper is not None else "own"

    @cached_property
    def whisper_device(self) -> str:
        """'cuda' when Whisper can run on the NVIDIA GPU (asked for, or auto and it passes a test), else 'cpu'."""
        from backend.config import settings

        want = settings().whisper_device
        if want == "cpu":
            return "cpu"
        ok, why = _cuda_whisper_works()
        if not ok:
            print(f"[speech] Whisper on the GPU not possible ({why}); using the CPU", flush=True)
        return "cuda" if ok else "cpu"

    def _load_whisper(self, name: str):
        from faster_whisper import WhisperModel

        if self.whisper_device == "cuda":
            try:
                return WhisperModel(name, device="cuda", compute_type="float16")
            except Exception as e:  # noqa: BLE001 - the CPU still works
                print(f"[speech] Whisper {name} failed on the GPU ({type(e).__name__}: {e}); using the CPU", flush=True)
        return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=4)

    @cached_property
    def whisper(self):
        """Pre-trained Whisper (faster-whisper; GPU float16 or CPU int8), or None when the package / model is missing."""
        from backend.config import settings

        if settings().asr_backend != "whisper":
            return None
        try:
            return self._load_whisper(settings().whisper_model)
        except Exception as e:  # noqa: BLE001 - fall back to our own ASR instead of having no transcript
            print(f"[speech] Whisper unavailable ({type(e).__name__}: {e}); using our own ASR", flush=True)
            return None

    @cached_property
    def whisper_candidate(self):
        """The (optionally bigger) Whisper used for the candidate's track; falls back to the standard one."""
        from backend.config import settings

        name = settings().whisper_model_candidate
        if not name or name == settings().whisper_model or self.whisper is None:
            return self.whisper
        try:
            return self._load_whisper(name)
        except Exception as e:  # noqa: BLE001
            print(f"[speech] Whisper {name} unavailable ({type(e).__name__}: {e}); using {settings().whisper_model}", flush=True)
            return self.whisper

    def whisper_for(self, role: str):
        return self.whisper_candidate if role == "candidate" else self.whisper

    @cached_property
    def whisper_pool(self):
        """One background thread that runs every Whisper transcription, so the audio loops never wait for it."""
        from concurrent.futures import ThreadPoolExecutor

        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

    @property
    def asr_ready(self) -> bool:
        return self.asr_backend == "whisper" or self.asr is not None

    @cached_property
    def lm_decoder(self):
        """Beam search + 4-gram LM for finished utterances; None (greedy only) when the ASR or the LM is missing."""
        if self.asr is None:
            return None
        from ml.speech.asr.decode import live_decoder

        return live_decoder(self.asr[1].pieces())

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
