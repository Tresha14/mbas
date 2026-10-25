"""Viva demo: prove we train our own models by retraining the VAD and the gaze CNN from random init for 2 epochs
in a few minutes, then showing their test numbers next to the full runs.

    python -m ml.live_retrain
Runs under the names vad_live / gaze_iris_live so the production checkpoints are untouched.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from ml.common.config import METRICS, PUBLIC

PY = sys.executable


def run(args: list[str]) -> float:
    t = time.time()
    subprocess.check_call([PY, *args])
    return time.time() - t


def main() -> int:
    print("Live retrain: every weight below starts from random numbers.\n", flush=True)
    rows = []
    if (PUBLIC / "librispeech" / "manifest.parquet").exists():
        s = run(["-m", "ml.speech.vad.train", "--set", "name=vad_live", "train.epochs=2", "data.train_items=1500",
                 "data.workers=2"])
        rows.append(("VAD (2 epochs, 1,500 mixtures)", f"{s / 60:.1f} min"))
    if (PUBLIC / "mpiigaze" / "manifest.parquet").exists():
        s = run(["-m", "ml.vision.gaze_iris.train", "--set", "name=gaze_iris_live", "train.epochs=2", "data.limit=30000",
                 "data.workers=2"])
        rows.append(("Gaze CNN (2 epochs, 30k eye pairs)", f"{s / 60:.1f} min"))
    print("\nTrained live:")
    for r in rows:
        print(f"  {r[0]:40s} {r[1]}")
    for name in ("vad", "gaze_iris"):
        p = METRICS / f"{name}.json"
        if p.exists():
            m = json.loads(p.read_text())
            key = m.get("test", {}).get("frame_f1_smoothed") if name == "vad" else m.get("angular_error_mean_deg")
            print(f"  full run {name}: {key}")
    print("Curves of the live runs: tensorboard --logdir runs (vad_live, gaze_iris_live)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
