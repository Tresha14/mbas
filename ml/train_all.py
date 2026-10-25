"""Train + evaluate every model one after another (low RAM: one job at a time, 2 data-loader workers).

    python -m ml.train_all                    # everything whose data exists
    python -m ml.train_all asr vad            # only these
Each step is skipped (with the reason) when its data is missing; --resume is used where a run already exists.
"""
from __future__ import annotations

import subprocess
import sys
import time

from ml.common.config import CHECKPOINTS, PUBLIC, PROCESSED

PY = sys.executable
STEPS = {
    "face_detector": (["-m", "ml.vision.face_detector.train", "--resume", "--set", "data.workers=2"],
                      ["-m", "ml.vision.face_detector.eval"], PUBLIC / "wider_face" / "manifest.parquet"),
    "headpose": (["-m", "ml.vision.headpose.train", "--set", "data.workers=2"], None, PUBLIC / "wider_face" / "manifest.parquet"),
    "gaze_iris": (["-m", "ml.vision.gaze_iris.train", "--resume", "--set", "data.workers=2"], ["-m", "ml.vision.gaze_iris.eval"],
                  PUBLIC / "mpiigaze" / "manifest.parquet"),
    "engagement_daisee": (["-m", "ml.vision.engagement.train", "--stage", "pretrain", "--set", "data.workers=2"], None,
                          PUBLIC / "daisee" / "manifest.parquet"),
    "engagement": (["-m", "ml.vision.engagement.train", "--stage", "finetune", "--set", "data.workers=2"],
                   ["-m", "ml.vision.engagement.eval"], CHECKPOINTS / "engagement_daisee" / "best.pt"),
    "window_models": (["-m", "ml.vision.engagement.window_models"], None, PROCESSED / "windows" / "windows_synthetic.parquet"),
    "vad": (["-m", "ml.speech.vad.train", "--set", "data.workers=2"], None, PUBLIC / "librispeech" / "manifest.parquet"),
    "lm": (["-m", "ml.speech.lm.ngram"], None, PUBLIC / "librispeech" / "manifest.parquet"),
    "asr": (["-m", "ml.speech.asr.train", "--resume", "--set", "data.workers=2"], ["-m", "ml.speech.asr.eval"],
            PUBLIC / "librispeech" / "manifest.parquet"),
    "disfluency": (["-m", "ml.behaviour.disfluency.train", "--resume", "--set", "data.workers=2"],
                   ["-m", "ml.behaviour.disfluency.eval"], PUBLIC / "sep28k" / "manifest.parquet"),
    "behaviour": (["-m", "ml.behaviour.answer_models"], None, PROCESSED / "labels" / "synthetic" / "answers.parquet"),
    "answer_quality": (["-m", "ml.response.answer_quality.train"], ["-m", "ml.response.answer_quality.eval"],
                       PROCESSED / "tokenizers" / "answer.model"),
    "answer_features": (["-m", "ml.fusion.answer_features"], None, CHECKPOINTS / "answer_quality" / "best.pt"),
    "fusion": (["-m", "ml.fusion.train"], None, PROCESSED / "features" / "answer_model_synthetic.parquet"),
    "meeting_head": (["-m", "ml.fusion.meeting"], None, PUBLIC / "ami" / "manifest.parquet"),
}


def main(argv: list[str] | None = None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(STEPS)
    done = {}
    for n in names:
        train, ev, need = STEPS[n]
        if not need.exists():
            done[n] = f"skipped (missing {need.name})"
            print(f"== {n}: {done[n]}", flush=True)
            continue
        t = time.time()
        print(f"== {n}: training", flush=True)
        rc = subprocess.call([PY, *train])
        if rc == 0 and ev:
            rc = subprocess.call([PY, *ev])
        done[n] = f"{'ok' if rc == 0 else f'failed ({rc})'} in {(time.time() - t) / 60:.1f} min"
        print(f"== {n}: {done[n]}", flush=True)
    print("\n".join(f"{k:18s} {v}" for k, v in done.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
