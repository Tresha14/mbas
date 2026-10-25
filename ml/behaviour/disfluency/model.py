"""Disfluency CRNN (random init): 3 s audio -> filler, repetition, prolongation, block (multi-label).

Log-mel front end (inside the model, also exported to ONNX) -> 4 conv blocks (pool over frequency,
mild pool over time) -> BiGRU -> attention pooling -> 4 logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.common.audio_features import LogMel, SpecAugment

LABELS = ("filler", "repetition", "prolongation", "block")


class DisfluencyCRNN(nn.Module):
    def __init__(self, width: int = 32, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.front = LogMel()
        self.specaug = SpecAugment(freq_masks=2, freq_width=12, time_masks=2, time_width=0.08)

        def blk(a, b, pt):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, pt)))
        self.cnn = nn.Sequential(blk(1, width, 2), blk(width, 2 * width, 1), blk(2 * width, 4 * width, 1),
                                 blk(4 * width, 4 * width, 1))                  # 80 mel -> 5, frames / 2
        self.gru = nn.GRU(4 * width * 5, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, len(LABELS))

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        f = self.front(wav)                                         # (B, 80, T)
        f = (f - f.mean((1, 2), keepdim=True)) / (f.std((1, 2), keepdim=True) + 1e-5)
        return self.specaug(f)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, 48000) float32 16 kHz -> logits (B, 4)."""
        x = self.cnn(self.features(wav)[:, None])                   # (B, C, 5, T')
        x = x.permute(0, 3, 1, 2).flatten(2)                        # (B, T', C*5)
        h, _ = self.gru(x)
        a = torch.softmax(self.att(h), 1)
        return self.out(self.drop((a * h).sum(1)))
