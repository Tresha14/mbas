"""Answer-quality transformer (random init): [CLS] question [SEP] answer -> relevance, quality, STAR, completeness,
follow-up and reason. 4 encoder layers, d_model 256, 4 heads, our SentencePiece answer vocabulary.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

N_REASONS = 6


class AnswerQualityModel(nn.Module):
    def __init__(self, vocab: int, d_model: int = 256, heads: int = 4, layers: int = 4, ff: int = 1024,
                 max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.seg = nn.Embedding(2, d_model)          # 0 = question, 1 = answer
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.relevance = nn.Linear(d_model, 1)
        self.quality = nn.Linear(d_model, 5)
        self.star = nn.Linear(d_model, 4)
        self.completeness = nn.Linear(d_model, 1)
        self.follow_up = nn.Linear(d_model, 1)
        self.reason = nn.Linear(d_model, N_REASONS)
        self.d_model = d_model
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, ids: torch.Tensor, sep_id: int):
        """ids (B, L) with 0 = padding -> dict of logits; pooled = mean over non-pad tokens + CLS."""
        B, L = ids.shape
        pad = ids == 0
        seg = (torch.cumsum((ids == sep_id).int(), 1) > 0).long()
        pos = torch.arange(L, device=ids.device)[None].expand(B, L)
        x = self.emb(ids) * math.sqrt(self.d_model) / 4 + self.pos(pos) + self.seg(seg)
        h = self.norm(self.enc(self.drop(x), src_key_padding_mask=pad))
        keep = (~pad).float()[..., None]
        z = self.drop(0.5 * h[:, 0] + 0.5 * (h * keep).sum(1) / keep.sum(1).clamp(min=1))
        return {"relevance": self.relevance(z)[:, 0], "quality": self.quality(z), "star": self.star(z),
                "completeness": self.completeness(z)[:, 0], "follow_up": self.follow_up(z)[:, 0],
                "reason": self.reason(z), "pooled": z}

    @staticmethod
    def expected_quality(logits: torch.Tensor) -> torch.Tensor:
        """Expected score 1-5 from the quality distribution."""
        return (torch.softmax(logits, -1) * torch.arange(1, 6, device=logits.device)).sum(-1)
