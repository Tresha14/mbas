"""Conformer-CTC ASR (random init), ~13 M params: 12 layers, d_model 176, 4 heads, conv kernel 31,
SentencePiece vocab 1000 (id 0 = CTC blank / pad).

Front end: 80-band log-mel (inside the model) -> SpecAugment (train) -> 2 x Conv2d stride 2 (40 ms frames).
Streaming: attention can be limited to chunks (640 ms = 16 frames) + a number of left chunks; training mixes
full-context and random chunk sizes ("dynamic chunk training"), so one model serves offline and streaming.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.common.audio_features import LogMel, SpecAugment


class ConvSubsample(nn.Module):
    def __init__(self, n_mels: int, d: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, d, 3, 2), nn.ReLU(), nn.Conv2d(d, d, 3, 2), nn.ReLU())
        f = ((n_mels - 1) // 2 - 1) // 2
        self.out = nn.Linear(d * f, d)

    def forward(self, x, lens):
        x = self.conv(x[:, None])                      # (B, d, F', T')
        B, C, Fq, T = x.shape
        x = self.out(x.permute(0, 3, 1, 2).reshape(B, T, C * Fq))
        lens = ((lens - 1) // 2 - 1) // 2
        return x, lens


class RelPositionalEncoding(nn.Module):
    """Sinusoidal absolute positions added to the input (simple, works with chunked attention)."""

    def __init__(self, d: int, max_len: int = 6000):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, offset: int = 0):
        return x + self.pe[offset:offset + x.shape[1]][None]


class FeedForward(nn.Module):
    def __init__(self, d, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(d * mult, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d, kernel=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        y = self.norm(x).masked_fill(pad_mask[..., None], 0.0).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        y = self.pw2(F.silu(self.bn(self.dw(y))))
        return self.drop(y.transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d, heads, kernel, dropout):
        super().__init__()
        self.ff1, self.ff2 = FeedForward(d, 4, dropout), FeedForward(d, 4, dropout)
        self.att_norm = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.att_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d, kernel, dropout)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x, pad_mask, att_mask):
        x = x + 0.5 * self.ff1(x)
        y = self.att_norm(x)
        y, _ = self.att(y, y, y, key_padding_mask=pad_mask, attn_mask=att_mask, need_weights=False)
        x = x + self.att_drop(y)
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


def chunk_mask(T: int, chunk: int, left_chunks: int, device) -> torch.Tensor:
    """(T, T) bool mask, True = blocked. Frame i sees frames of its own chunk and `left_chunks` chunks before it."""
    idx = torch.arange(T, device=device)
    c = idx // chunk
    allowed = (c[None, :] <= c[:, None]) & (c[None, :] >= c[:, None] - left_chunks)
    return ~allowed


class ConformerCTC(nn.Module):
    def __init__(self, vocab: int = 1000, d: int = 176, heads: int = 4, layers: int = 12, kernel: int = 31,
                 n_mels: int = 80, dropout: float = 0.1):
        super().__init__()
        self.front = LogMel(n_mels=n_mels)
        self.specaug = SpecAugment(freq_masks=2, freq_width=27, time_masks=10, time_width=0.05)
        self.sub = ConvSubsample(n_mels, d)
        self.pos = RelPositionalEncoding(d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, kernel, dropout) for _ in range(layers)])
        self.ctc = nn.Linear(d, vocab)
        self.vocab = vocab

    def features(self, wav, wav_len):
        f = self.front(wav)                                         # (B, 80, T)
        f_len = wav_len // 160 + 1
        mask = torch.arange(f.shape[-1], device=f.device)[None] < f_len[:, None]
        mean = (f * mask[:, None]).sum(-1, keepdim=True) / f_len[:, None, None]
        var = ((f - mean) ** 2 * mask[:, None]).sum(-1, keepdim=True) / f_len[:, None, None]
        f = (f - mean) / torch.sqrt(var + 1e-5) * mask[:, None]
        return self.specaug(f), f_len

    def encode(self, wav, wav_len, chunk: int | None = None, left_chunks: int = 4):
        f, f_len = self.features(wav, wav_len)
        x, lens = self.sub(f, f_len)
        x = self.drop(self.pos(x))
        T = x.shape[1]
        pad = torch.arange(T, device=x.device)[None] >= lens[:, None]
        att = chunk_mask(T, chunk, left_chunks, x.device) if chunk else None
        for b in self.blocks:
            x = b(x, pad, att)
        return x, lens

    def forward(self, wav, wav_len, chunk: int | None = None, left_chunks: int = 4):
        """-> log-probs (B, T', vocab) at 40 ms, output lengths (B,)."""
        x, lens = self.encode(wav, wav_len, chunk, left_chunks)
        return F.log_softmax(self.ctc(x).float(), -1), lens
