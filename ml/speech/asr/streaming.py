"""Streaming ASR: 640 ms chunks, cached left audio context, limited right context, partial + final transcripts
with word timestamps.

Every chunk: the encoder runs on [left context (4 s) | new chunk | look-ahead (320 ms)] with the chunked attention
the model was trained with; frames of the new chunk are kept (the look-ahead frames are recomputed next time).
The VAD closes an utterance after `end_silence_s` of silence -> a final result; otherwise a partial is emitted.
Latency = chunk (640 ms) + look-ahead (320 ms) + compute (~50-150 ms on the RTX 4060 / ~300 ms CPU).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ml.speech.asr.decode import greedy_with_times

SR = 16000
FRAME_S = 0.04


@dataclass
class Result:
    text: str
    final: bool
    words: list[dict] = field(default_factory=list)   # {"word", "start_s", "end_s"}
    t_s: float = 0.0


class StreamingASR:
    def __init__(self, model, tokenizer, vad=None, chunk_s: float = 0.64, left_s: float = 4.0, right_s: float = 0.32,
                 end_silence_s: float = 0.6, device: str = "cpu"):
        self.model, self.tok, self.vad = model.eval(), tokenizer, vad
        self.chunk, self.left, self.right = int(chunk_s * SR), int(left_s * SR), int(right_s * SR)
        self.end_silence = end_silence_s
        self.dev = device
        self.reset_stream()

    def reset_stream(self):
        self.buf = np.zeros(0, np.float32)        # all audio not yet decoded + context
        self.t0 = 0.0                             # session time of buf[0]
        self.decoded_until = 0                    # samples of buf already turned into frames
        self.frames: list[np.ndarray] = []        # log-prob frames of the current utterance
        self.frame_t0: float | None = None
        self.silence = 0.0

    def _encode(self, audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(audio)[None].to(self.dev)
            lp, ol = self.model(x, torch.tensor([len(audio)], device=self.dev), chunk=16, left_chunks=6)
        return lp[0, : int(ol[0])].cpu().numpy()

    def _words(self, frames: np.ndarray, t0: float) -> list[dict]:
        toks = greedy_with_times(frames, frame_s=FRAME_S)
        words, cur = [], None
        for tid, t in toks:
            piece = self.tok.sp.id_to_piece(tid)
            if piece.startswith("▁") or cur is None:
                if cur:
                    words.append(cur)
                cur = {"word": piece.lstrip("▁"), "start_s": round(t0 + t, 2), "end_s": round(t0 + t + FRAME_S, 2)}
            else:
                cur["word"] += piece
                cur["end_s"] = round(t0 + t + FRAME_S, 2)
        if cur:
            words.append(cur)
        return [w for w in words if w["word"]]

    def _result(self, final: bool, now: float) -> Result | None:
        if not self.frames:
            return None
        f = np.concatenate(self.frames)
        words = self._words(f, self.frame_t0 or 0.0)
        r = Result(" ".join(w["word"] for w in words), final, words, now)
        if final:
            self.frames, self.frame_t0 = [], None
        return r if (r.text or final) else None

    def accept(self, audio: np.ndarray) -> list[Result]:
        """Feed any amount of audio (float32, 16 kHz); returns new partial/final results."""
        out = []
        self.buf = np.concatenate([self.buf, audio.astype(np.float32)])
        while len(self.buf) - self.decoded_until >= self.chunk + self.right:
            s = self.decoded_until
            a = max(0, s - self.left)
            window = self.buf[a: s + self.chunk + self.right]
            lp = self._encode(window)
            k0 = int((s - a) / SR / FRAME_S)
            k1 = k0 + int(self.chunk / SR / FRAME_S)
            chunk_frames = lp[k0:k1]
            now = self.t0 + (s + self.chunk) / SR
            speech = True
            if self.vad is not None:
                p = self.vad(self.buf[s:s + self.chunk])
                speech = bool((p > 0.5).mean() > 0.1)
            if speech:
                if self.frame_t0 is None:
                    self.frame_t0 = self.t0 + s / SR
                self.frames.append(chunk_frames)
                self.silence = 0.0
                r = self._result(False, now)
            else:
                self.silence += self.chunk / SR
                r = self._result(True, now) if self.frames and self.silence >= self.end_silence else None
            if r:
                out.append(r)
            self.decoded_until += self.chunk
            drop = max(0, self.decoded_until - self.left)       # keep only the left context
            if drop:
                self.buf, self.decoded_until, self.t0 = self.buf[drop:], self.decoded_until - drop, self.t0 + drop / SR
        return out

    def flush(self) -> list[Result]:
        pad = np.zeros(self.chunk + self.right, np.float32)
        out = self.accept(pad)
        r = self._result(True, self.t0 + len(self.buf) / SR)
        return out + ([r] if r else [])
