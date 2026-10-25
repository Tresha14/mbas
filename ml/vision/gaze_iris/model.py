"""Gaze + iris CNN (random init): two 36x60 eye crops + head pose -> gaze, iris centres, on/off-screen.

Shared eye encoder (4 conv blocks) applied to both eyes; features concatenated with the head pose
(pitch, yaw) feed three heads:
  gaze      (pitch, yaw) radians, regression
  iris      (lx, ly, rx, ry) iris centre in each crop, 0-1 (0.5 = crop centre)
  onscreen  logit; trained only on samples with on/off labels (our recordings, Columbia Gaze)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def block(cin: int, cout: int, pool: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class EyeEncoder(nn.Module):
    def __init__(self, width: int = 32, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(block(1, width), block(width, 2 * width), block(2 * width, 4 * width),
                                 block(4 * width, 4 * width, pool=False))       # 36x60 -> 4x7
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(4 * width * 4 * 7, dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fc(self.net(x))


class GazeIrisNet(nn.Module):
    def __init__(self, width: int = 32, dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.enc = EyeEncoder(width, dim)
        self.trunk = nn.Sequential(nn.Linear(2 * dim + 2, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                   nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.gaze = nn.Linear(128, 2)
        self.iris = nn.Linear(128, 4)
        self.onscreen = nn.Linear(128, 1)

    def forward(self, left, right, head):
        """left/right (B, 1, 36, 60) in [-1, 1], head (B, 2) radians -> gaze (B, 2), iris (B, 4), onscreen logit (B,)"""
        h = self.trunk(torch.cat([self.enc(left), self.enc(right), head], 1))
        return self.gaze(h), torch.sigmoid(self.iris(h)), self.onscreen(h)[:, 0]


def angular_error_deg(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Angle (degrees) between 3-D gaze vectors built from (pitch, yaw)."""
    def vec(py):
        p, y = py[:, 0], py[:, 1]
        return torch.stack([-torch.cos(p) * torch.sin(y), -torch.sin(p), -torch.cos(p) * torch.cos(y)], 1)
    a, b = vec(pred), vec(true)
    cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-9)
    return torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7)))
