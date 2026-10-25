"""Voice activity detection (random init): log-mel frames -> speech / non-speech every 10 ms.

Tiny CNN: 1-D convolutions over time on 40-band log-mel (receptive field ~ 330 ms, so each 30 ms decision
sees its context), then per-frame logits. Post-processing: threshold + hangover smoothing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from ml.common.audio_features import LogMel


class VADNet(nn.Module):
    def __init__(self, n_mels: int = 40, ch: int = 64):
        super().__init__()
        self.mel = LogMel(n_mels=n_mels, f_min=60.0, f_max=7600.0)
        layers, c = [], n_mels
        for d in (1, 2, 4, 8):
            layers += [nn.Conv1d(c, ch, 3, padding=d, dilation=d), nn.BatchNorm1d(ch), nn.ReLU()]
            c = ch
        self.net = nn.Sequential(*layers, nn.Conv1d(ch, 1, 1))

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        f = self.mel(wav)
        # utterance-level mean normalisation keeps silence/speech contrast; tolerant of gain changes
        return f - f.mean(dim=(1, 2), keepdim=True)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, T) 16 kHz -> logits (B, frames) at 10 ms."""
        return self.net(self.features(wav))[:, 0]


def smooth(prob: np.ndarray, on: float = 0.5, off: float = 0.35, hangover: int = 20, min_speech: int = 12) -> np.ndarray:
    """Hysteresis threshold + hangover (frames kept 'speech' after the probability drops) + min speech length."""
    out = np.zeros(len(prob), bool)
    active, hold = False, 0
    for i, p in enumerate(prob):
        if p >= on:
            active, hold = True, hangover
        elif active and p < off:
            hold -= 1
            if hold <= 0:
                active = False
        out[i] = active
    # drop very short bursts (clicks, coughs)
    i = 0
    while i < len(out):
        if out[i]:
            j = i
            while j < len(out) and out[j]:
                j += 1
            if j - i < min_speech:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


class StreamingVAD:
    """Chunk-by-chunk VAD for the live system: keeps 0.5 s of audio context between calls."""

    def __init__(self, model: VADNet, context_s: float = 0.5):
        self.model = model.eval()
        self.ctx = np.zeros(int(context_s * 16000), np.float32)

    @torch.no_grad()
    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        x = np.concatenate([self.ctx, chunk])
        p = torch.sigmoid(self.model(torch.from_numpy(x)[None]))[0].numpy()
        self.ctx = x[-len(self.ctx):]
        n = len(chunk) // 160
        return p[-n - 1:-1] if n else p[:0]
