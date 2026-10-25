"""Live microphone transcription in the terminal with our VAD + streaming Conformer-CTC.

    pip install sounddevice          # microphone capture (plumbing only)
    python -m ml.speech.demo_mic
Partial results overwrite the current line; finals are printed with word timestamps and the delay.
"""
from __future__ import annotations

import queue
import sys
import time

import numpy as np
import torch

from ml.common import text
from ml.speech.asr.eval import load as load_asr
from ml.speech.asr.streaming import StreamingASR
from ml.speech.vad.model import StreamingVAD
from ml.speech.vad.train import load as load_vad


def main() -> int:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("pip install sounddevice")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    asr, _ = load_asr(dev)
    stream = StreamingASR(asr, text.load("asr"), StreamingVAD(load_vad()), device=dev)
    q: queue.Queue = queue.Queue()
    t_start = time.time()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    print("Listening (Ctrl+C to stop) ...")
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=1600, callback=cb):
        try:
            while True:
                audio = q.get()
                for r in stream.accept(audio):
                    delay = (time.time() - t_start) - (r.words[-1]["end_s"] if r.words else r.t_s)
                    if r.final:
                        stamps = " ".join(f"{w['word']}@{w['start_s']:.1f}" for w in r.words)
                        print(f"\r[final  {delay:4.1f} s] {r.text}\n          {stamps}")
                    else:
                        print(f"\r[partial {delay:4.1f} s] {r.text[-100:]:100s}", end="", flush=True)
        except KeyboardInterrupt:
            for r in stream.flush():
                print(f"\r[final] {r.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
