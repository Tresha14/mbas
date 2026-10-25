"""Engagement CNN-GRU (random init): 16 frames of a 5 s window -> engaged / neutral / disengaged.

Per-frame CNN encoder (4 conv blocks, 112x112 RGB) -> bidirectional GRU over the 16 frames ->
mean + last state -> heads: engagement (3), auxiliary boredom (4 levels) and confusion (4 levels).
Pre-train on DAiSEE, fine-tune on our labelled windows (weighted cross-entropy for class imbalance).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    def __init__(self, width: int = 32, dim: int = 128):
        super().__init__()

        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, 2, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.Conv2d(b, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.net = nn.Sequential(blk(3, width), blk(width, 2 * width), blk(2 * width, 4 * width),
                                 blk(4 * width, 8 * width), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(8 * width, dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class EngagementCNNGRU(nn.Module):
    def __init__(self, width: int = 32, dim: int = 128, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.enc = FrameEncoder(width, dim)
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.engagement = nn.Linear(4 * hidden, 3)
        self.boredom = nn.Linear(4 * hidden, 4)
        self.confusion = nn.Linear(4 * hidden, 4)

    def forward(self, frames):
        """frames (B, T, 3, H, W) in [-1, 1] -> engagement (B, 3), boredom (B, 4), confusion (B, 4) logits."""
        B, T = frames.shape[:2]
        f = self.enc(frames.flatten(0, 1)).view(B, T, -1)
        h, _ = self.gru(f)
        z = self.drop(torch.cat([h.mean(1), h[:, -1]], 1))
        return self.engagement(z), self.boredom(z), self.confusion(z)
