"""ONNX Runtime latency of every exported model on this machine's CPU (FP32, and INT8 dynamic quantisation for
the larger models) -> docs/metrics/onnx_latency.json + docs/metrics/onnx_latency.md.

    python -m ml.export.benchmark
"""
from __future__ import annotations

import platform
import sys

import numpy as np

from ml.common.config import METRICS, save_json
from ml.common.train_utils import benchmark
from ml.export.export_onnx import OUT

INPUTS = {
    "face_detector": {"image": (1, 3, 128, 128)}, "gaze_iris": {"left": (1, 1, 36, 60), "right": (1, 1, 36, 60), "head": (1, 2)},
    "headpose": {"face": (1, 3, 64, 64)}, "engagement": {"frames": (1, 16, 3, 112, 112)},
    "vad": {"wav": (1, int(1.14 * 16000))}, "asr": {"wav": (1, int(4.96 * 16000))}, "disfluency": {"wav": (1, 48000)},
    "answer_quality": {"ids": (1, 128)},
}
BUDGET_MS = {"face_detector": 66, "gaze_iris": 66, "headpose": 66, "engagement": 2500, "vad": 640, "asr": 640,
             "disfluency": 3000, "answer_quality": 2000, "fusion": 2000}
NOTE = {"face_detector": "per frame (10-15 fps)", "gaze_iris": "per frame", "headpose": "per frame",
        "engagement": "per 2.5 s window", "vad": "per 640 ms chunk (with 0.5 s context)",
        "asr": "per 640 ms chunk (4.96 s window)", "disfluency": "per 3 s window", "answer_quality": "per answer",
        "fusion": "per answer"}


def run(path, shapes, threads: int = 2) -> dict:
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
    feed = {}
    for inp in s.get_inputs():
        shp = shapes.get(inp.name) if shapes else None
        if shp is None:
            shp = [d if isinstance(d, int) else 1 for d in inp.shape]
        dt = np.int64 if "int64" in inp.type else np.float32
        feed[inp.name] = (np.ones(shp, dt) if dt == np.int64 else np.random.default_rng(0).standard_normal(shp).astype(dt))
    return benchmark(lambda: s.run(None, feed), warmup=3, iters=30)


def main(argv: list[str] | None = None) -> int:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    res = {"cpu": platform.processor(), "threads": 2, "models": {}}
    for p in sorted(OUT.glob("*.onnx")):
        if p.stem.endswith("_int8"):
            continue
        name = p.stem
        r = {"fp32": run(p, INPUTS.get(name)), "mb": round(p.stat().st_size / 1e6, 2), "unit": NOTE.get(name, "per call")}
        if p.stat().st_size > 2e6:
            q = p.with_name(f"{name}_int8.onnx")
            try:
                quantize_dynamic(str(p), str(q), weight_type=QuantType.QInt8)
                r["int8"] = run(q, INPUTS.get(name))
                r["int8_mb"] = round(q.stat().st_size / 1e6, 2)
            except Exception as e:  # noqa: BLE001 - some ops have no int8 kernel
                r["int8"] = {"error": str(e)[:100]}
        b = BUDGET_MS.get(name)
        r["real_time_budget_ms"] = b
        r["within_budget"] = bool(b is None or r["fp32"]["median_ms"] < b)
        res["models"][name] = r
        print(f"{name:18s} {r['fp32']['median_ms']:8.2f} ms  ({r['unit']})")
    save_json(res, METRICS / "onnx_latency.json")
    lines = ["# ONNX Runtime latency (CPU, 2 threads)", "", f"CPU: {res['cpu']}", "",
             "| Model | Unit | FP32 median ms | INT8 median ms | Size MB | Budget ms | OK |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for n, r in res["models"].items():
        i8 = r.get("int8", {}).get("median_ms", "") if isinstance(r.get("int8"), dict) else ""
        lines.append(f"| {n} | {r['unit']} | {r['fp32']['median_ms']} | {i8} | {r['mb']} | {r['real_time_budget_ms'] or ''} | "
                     f"{'yes' if r['within_budget'] else 'no'} |")
    (METRICS / "onnx_latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
