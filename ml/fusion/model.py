"""Multimodal fusion network (random init).

Per-modality MLP encoders -> one 64-d token per modality (+ learned modality embedding) ->
attention fusion (a learned query attends over the modality tokens; the attention weights are the
modality importances shown to the interviewer) -> heads:
  a. communication effectiveness 0-100 + clarity, confidence, engagement, relevance, fluency
  b. follow-up yes/no + reason (none, vague, low_confidence, attention_drop, topic_not_covered, contradiction)
A modality can be dropped (ablation / camera off) with `active`; its token is masked out of the attention.

MeetingHead (c): per-participant meeting features -> engagement 0-1 and contribution balance 0-1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.common.datasets import FOLLOW_UP_REASONS, FUSION_FEATURES, FUSION_TARGETS

MODALITIES = list(FUSION_FEATURES)


class FusionNet(nn.Module):
    def __init__(self, dims: dict[str, int] | None = None, d: int = 64, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        dims = dims or {m: len(c) for m, c in FUSION_FEATURES.items()}
        self.mods = list(dims)
        self.enc = nn.ModuleDict({m: nn.Sequential(nn.Linear(n, 2 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d, d),
                                                   nn.LayerNorm(d)) for m, n in dims.items()})
        self.mod_emb = nn.Parameter(torch.zeros(len(self.mods), d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(2 * d, 128), nn.ReLU(), nn.Dropout(dropout))
        self.scores = nn.Linear(128, len(FUSION_TARGETS))
        self.follow_up = nn.Linear(128, 1)
        self.reason = nn.Linear(128, len(FOLLOW_UP_REASONS))

    def forward(self, x: dict[str, torch.Tensor], active: list[str] | None = None):
        active = active or self.mods
        tokens = torch.stack([self.enc[m](x[m]) for m in self.mods], 1) + self.mod_emb       # (B, M, d)
        mask = torch.tensor([m not in active for m in self.mods], device=tokens.device)       # True = ignore
        tokens = tokens.masked_fill(mask[None, :, None], 0.0)
        q = self.query.expand(tokens.shape[0], -1, -1)
        z, w = self.att(q, tokens, tokens, key_padding_mask=mask[None].expand(tokens.shape[0], -1), need_weights=True)
        keep = (~mask).float()[None, :, None]
        pooled = (tokens * keep).sum(1) / keep.sum(1)
        h = self.trunk(torch.cat([z[:, 0], pooled], 1))
        return {"scores": torch.sigmoid(self.scores(h)), "follow_up": self.follow_up(h)[:, 0], "reason": self.reason(h),
                "modality_weights": w[:, 0]}


# Turn features are relative to the meeting (scale-free): absolute turns/min and turn length differ a lot between
# meeting styles (synthetic 12-31 s turns vs AMI 2-8 s), so they did not transfer (docs/metrics/meeting_head.json).
MEETING_FEATURES = ["talk_share", "share_vs_fair", "turn_share_vs_fair", "turn_length_vs_meeting", "on_screen_ratio",
                    "camera_on", "face_motion_energy", "head_yaw_std_deg", "multitasking_share"]
MEETING_FEATURES_ABSOLUTE = ["talk_share", "share_vs_fair", "turns_per_min", "avg_turn_s", "on_screen_ratio", "camera_on",
                             "face_motion_energy", "head_yaw_std_deg", "multitasking_share"]


class MeetingHead(nn.Module):
    """Per participant: features -> (engagement 0-1, contribution balance 0-1). Vision features are zero + camera_on=0
    when the camera is off (AMI has audio only)."""

    def __init__(self, n: int = len(MEETING_FEATURES), hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return torch.sigmoid(self.net(x))
