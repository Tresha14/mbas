"""Streaming ASR: 640 ms chunks, cached left audio context, limited right context, partial + final transcripts
with word timestamps.

Every chunk: the encoder runs on [left context (6 chunks) | new chunk | look-ahead (320 ms)] with the chunked attention
the model was trained with (4 left chunks); the left context is a whole number of chunks so the window's chunk
boundaries line up with the new chunk. Features are normalised with running statistics of the current utterance (the
model was trained on whole-utterance normalisation), not with the short window. Frames of the new chunk are kept (the
look-ahead frames are recomputed next time).
Partials use fast greedy decoding. When an utterance ends, the final is re-decoded with beam search + our 4-gram LM
(if a decoder is given); word timestamps are carried over from the greedy pass.
The VAD closes an utterance after `end_silence_s` of silence -> a final result; otherwise a partial is emitted.
Latency = chunk (640 ms) + look-ahead (320 ms) + compute (~50-150 ms on the RTX 4060 / ~300 ms CPU).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ml.speech.asr.decode import greedy_with_times, retime, words_from_tokens

SR = 16000
FRAME_S = 0.04
HOP = 160                   # samples per log-mel frame
TRAIN_LEFT_CHUNKS = 4       # ml/speech/asr/config.yaml: train.left_chunks
MIN_STAT_FRAMES = 200       # 2 s of speech before running statistics replace the window's own


@dataclass
class Result:
    text: str
    final: bool
    words: list[dict] = field(default_factory=list)   # {"word", "start_s", "end_s"}
    t_s: float = 0.0


class StreamingASR:
    def __init__(self, model, tokenizer, vad=None, chunk_s: float = 0.64, left_s: float | None = None, right_s: float = 0.32,
                 end_silence_s: float = 0.6, device: str = "cpu", decoder=None, beam_budget_s: float = 1.5):
        self.model, self.tok, self.vad = model.eval(), tokenizer, vad
        self.chunk, self.right = int(chunk_s * SR), int(right_s * SR)
        self.left = 6 * self.chunk if left_s is None else int(left_s * SR)
        self.end_silence = end_silence_s
        self.dev = device
        self.decoder, self.beam_budget_s = decoder, beam_budget_s
        self.reset_stream()

    def reset_stream(self):
        self.buf = np.zeros(0, np.float32)        # all audio not yet decoded + context
        self.t0 = 0.0                             # session time of buf[0]
        self.decoded_until = 0                    # samples of buf already turned into frames
        self.frames: list[np.ndarray] = []        # log-prob frames of the current utterance
        self.frame_t0: float | None = None
        self.silence = 0.0
        self.stat_n, self.stat_sum, self.stat_sq = 0, None, None   # log-mel statistics of the current utterance

    def _encode(self, audio: np.ndarray, new_frames: tuple[int, int], accumulate: bool) -> np.ndarray:
        """Log-probs for a window; new_frames = (first, last) log-mel frame of the new chunk inside the window."""
        with torch.no_grad():
            x = torch.from_numpy(audio)[None].to(self.dev)
            kw = {}
            if hasattr(self.model, "front"):
                mel = self.model.front(x)[0].double().cpu().numpy()          # (80, F)
                if accumulate:
                    seg = mel[:, new_frames[0]:new_frames[1]]
                    self.stat_n += seg.shape[1]
                    self.stat_sum = seg.sum(1) if self.stat_sum is None else self.stat_sum + seg.sum(1)
                    self.stat_sq = (seg ** 2).sum(1) if self.stat_sq is None else self.stat_sq + (seg ** 2).sum(1)
                if self.stat_n >= MIN_STAT_FRAMES:
                    mean = self.stat_sum / self.stat_n
                    var = np.maximum(self.stat_sq / self.stat_n - mean ** 2, 0.0)
                    kw["stats"] = tuple(torch.from_numpy(v)[None, :, None].float() for v in (mean, var))
            lp, ol = self.model(x, torch.tensor([len(audio)], device=self.dev), chunk=16, left_chunks=TRAIN_LEFT_CHUNKS, **kw)
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
        if final and self.decoder is not None and words:
            try:
                beam = words_from_tokens(self.tok.pieces(), self.decoder.decode(f, max_s=self.beam_budget_s))
            except TimeoutError:
                beam = []                      # keep the greedy words; never stall the audio loop
            if beam:
                words = retime(beam, words)
        r = Result(" ".join(w["word"] for w in words), final, words, now)
        if final:
            self.frames, self.frame_t0 = [], None
            self.stat_n, self.stat_sum, self.stat_sq = 0, None, None
        return r if (r.text or final) else None

    def accept(self, audio: np.ndarray) -> list[Result]:
        """Feed any amount of audio (float32, 16 kHz); returns new partial/final results."""
        out = []
        self.buf = np.concatenate([self.buf, audio.astype(np.float32)])
        while len(self.buf) - self.decoded_until >= self.chunk + self.right:
            s = self.decoded_until
            a = max(0, s - self.left)
            window = self.buf[a: s + self.chunk + self.right]
            speech = True
            if self.vad is not None:
                p = self.vad(self.buf[s:s + self.chunk])
                speech = bool((p > 0.5).mean() > 0.1)
            now = self.t0 + (s + self.chunk) / SR
            if speech:
                k0 = int((s - a) / SR / FRAME_S)
                lp = self._encode(window, ((s - a) // HOP, (s - a + self.chunk) // HOP), True)
                if self.frame_t0 is None:
                    self.frame_t0 = self.t0 + s / SR
                self.frames.append(lp[k0:k0 + int(self.chunk / SR / FRAME_S)])
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
