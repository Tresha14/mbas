"""BlazeFace-style single-shot face + 5-landmark detector (~0.4 M params at width 2, 128x128 input), random init.

Backbone: 5x5 stride-2 stem, BlazeBlocks (depthwise 5x5 + pointwise, residual) down to a
16x16 map (2 anchors / cell) and an 8x8 map (6 anchors / cell) = 896 anchors.
Per anchor: face logit, box offsets (cx, cy, w, h) and 5 key points (x, y) relative to the anchor.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_KP = 5
VAR_XY, VAR_WH = 0.1, 0.2


class BlazeBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.stride, self.pad_c = stride, cout - cin
        self.dw = nn.Conv2d(cin, cin, 5, stride, 2, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        y = self.bn(self.pw(self.dw(x)))
        s = F.max_pool2d(x, 2) if self.stride == 2 else x
        if self.pad_c:
            s = F.pad(s, (0, 0, 0, 0, 0, self.pad_c))
        return F.relu(y + s)


class FaceDetector(nn.Module):
    ANCHORS_PER_CELL = (2, 6)

    def __init__(self, size: int = 128, width: float = 2.0):
        super().__init__()
        self.size = size
        c = lambda n: int(round(n * width / 4)) * 4  # noqa: E731
        chans16 = [24, 24, 28, 32, 36, 42, 48, 56, 64, 72, 80, 88]
        strides16 = [1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
        self.stem = nn.Sequential(nn.Conv2d(3, c(24), 5, 2, 2, bias=False), nn.BatchNorm2d(c(24)), nn.ReLU())
        self.b16 = nn.Sequential(*[BlazeBlock(c(a), c(b), st)       # 64x64 -> 16x16
                                   for a, b, st in zip(chans16[:-1], chans16[1:], strides16)])
        self.b8 = nn.Sequential(BlazeBlock(c(88), c(96), 2), *[BlazeBlock(c(96), c(96)) for _ in range(4)])
        out = 1 + 4 + 2 * N_KP
        self.head16 = nn.Conv2d(c(88), self.ANCHORS_PER_CELL[0] * out, 1)
        self.head8 = nn.Conv2d(c(96), self.ANCHORS_PER_CELL[1] * out, 1)
        nn.init.constant_(self.head16.bias, 0.0)
        nn.init.constant_(self.head8.bias, 0.0)
        with torch.no_grad():  # focal-loss prior: start with p(face) = 0.01
            prior = -math.log((1 - 0.01) / 0.01)
            self.head16.bias.view(self.ANCHORS_PER_CELL[0], out)[:, 0] = prior
            self.head8.bias.view(self.ANCHORS_PER_CELL[1], out)[:, 0] = prior
        self.register_buffer("anchors", make_anchors(size), persistent=False)

    def forward(self, x):
        """x (B, 3, 128, 128) in [-1, 1] -> logits (B, A), box deltas (B, A, 4), kp deltas (B, A, 5, 2)."""
        f16 = self.b16(self.stem(x))
        f8 = self.b8(f16)
        out = 1 + 4 + 2 * N_KP
        p16 = self.head16(f16).permute(0, 2, 3, 1).reshape(x.shape[0], -1, out)
        p8 = self.head8(f8).permute(0, 2, 3, 1).reshape(x.shape[0], -1, out)
        p = torch.cat([p16, p8], 1)
        return p[..., 0], p[..., 1:5], p[..., 5:].reshape(x.shape[0], -1, N_KP, 2)


def make_anchors(size: int = 128) -> torch.Tensor:
    """(896, 4) anchors as normalised (cx, cy, w, h). Square anchors sized for webcam faces."""
    specs = [(16, (0.06, 0.11)), (8, (0.16, 0.24, 0.34, 0.46, 0.62, 0.82))]
    out = []
    for grid, scales in specs:
        for y in range(grid):
            for x in range(grid):
                for s in scales:
                    out.append([(x + 0.5) / grid, (y + 0.5) / grid, s, s])
    return torch.tensor(out, dtype=torch.float32)


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between (N, 4) and (M, 4) xyxy boxes -> (N, M)."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def encode(gt_xyxy: torch.Tensor, gt_kp: torch.Tensor, anchors: torch.Tensor):
    cxcy = (gt_xyxy[:, :2] + gt_xyxy[:, 2:]) / 2
    wh = (gt_xyxy[:, 2:] - gt_xyxy[:, :2]).clamp(min=1e-6)
    d_xy = (cxcy - anchors[:, :2]) / (anchors[:, 2:] * VAR_XY)
    d_wh = torch.log(wh / anchors[:, 2:]) / VAR_WH
    d_kp = (gt_kp - anchors[:, None, :2]) / (anchors[:, None, 2:] * VAR_XY)
    return torch.cat([d_xy, d_wh], -1), d_kp


def decode(deltas: torch.Tensor, kp_deltas: torch.Tensor, anchors: torch.Tensor):
    """-> boxes xyxy (…, A, 4) and key points (…, A, 5, 2), normalised to the input image."""
    cxcy = anchors[..., :2] + deltas[..., :2] * VAR_XY * anchors[..., 2:]
    wh = anchors[..., 2:] * torch.exp(deltas[..., 2:] * VAR_WH)
    kp = anchors[..., None, :2] + kp_deltas * VAR_XY * anchors[..., None, 2:]
    return torch.cat([cxcy - wh / 2, cxcy + wh / 2], -1), kp


def match(gt_boxes: torch.Tensor, anchors: torch.Tensor, pos_iou: float = 0.45, neg_iou: float = 0.3):
    """Per anchor: index of its GT (-1 none) and label 1 pos / 0 neg / -1 ignore."""
    A = anchors.shape[0]
    labels = torch.zeros(A, dtype=torch.long, device=anchors.device)
    idx = torch.full((A,), -1, dtype=torch.long, device=anchors.device)
    if gt_boxes.numel() == 0:
        return idx, labels
    iou = box_iou(cxcywh_to_xyxy(anchors), gt_boxes)       # (A, G)
    best, best_g = iou.max(1)
    labels[(best >= neg_iou) & (best < pos_iou)] = -1
    pos = best >= pos_iou
    best_a = iou.argmax(0)                                   # every GT gets at least its best anchor
    pos[best_a] = True
    best_g[best_a] = torch.arange(gt_boxes.shape[0], device=anchors.device)
    labels[pos] = 1
    idx[pos] = best_g[pos]
    # faces far smaller than the smallest anchor cannot be learned: their anchors are ignored, not negatives
    small = ((gt_boxes[:, 2:] - gt_boxes[:, :2]).max(1).values < 0.03)
    if small.any():
        lab_small = small[best_g] & pos
        labels[lab_small] = -1
        idx[lab_small] = -1
    return idx, labels


def detection_loss(logits, deltas, kp_deltas, gts, kps, kp_masks, anchors, alpha: float = 0.25, gamma: float = 2.0):
    """Focal classification + smooth-L1 box + smooth-L1 key points (faces with landmarks)."""
    B = logits.shape[0]
    tot_cls, tot_box, tot_kp, n_pos_all = 0.0, 0.0, 0.0, 0
    for i in range(B):
        idx, lab = match(gts[i], anchors)
        valid = lab >= 0
        target = (lab == 1).float()
        p = torch.sigmoid(logits[i])
        ce = F.binary_cross_entropy_with_logits(logits[i], target, reduction="none")
        pt = p * target + (1 - p) * (1 - target)
        w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt) ** gamma
        tot_cls = tot_cls + (w * ce)[valid].sum()
        pos = lab == 1
        n_pos = int(pos.sum())
        n_pos_all += n_pos
        if n_pos:
            g = idx[pos]
            t_box, t_kp = encode(gts[i][g], kps[i][g], anchors[pos])
            tot_box = tot_box + F.smooth_l1_loss(deltas[i][pos], t_box, reduction="sum", beta=0.1)
            m = kp_masks[i][g] > 0
            if m.any():
                tot_kp = tot_kp + F.smooth_l1_loss(kp_deltas[i][pos][m], t_kp[m], reduction="sum", beta=0.1) / N_KP
    n = max(n_pos_all, 1)
    return tot_cls / n, tot_box / n, tot_kp / n


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou: float = 0.3) -> torch.Tensor:
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][ious < iou]
    return torch.stack(keep) if keep else torch.zeros(0, dtype=torch.long)


@torch.no_grad()
def postprocess(logits, deltas, kp_deltas, anchors, score_thr: float = 0.5, iou_thr: float = 0.3, max_det: int = 20):
    """-> list of (boxes xyxy, scores, key points) per image, normalised coordinates."""
    boxes, kps = decode(deltas, kp_deltas, anchors)
    scores = torch.sigmoid(logits)
    out = []
    for b, s, k in zip(boxes, scores, kps):
        m = s >= score_thr
        b, s, k = b[m], s[m], k[m]
        if len(s):
            keep = nms(b, s, iou_thr)[:max_det]
            b, s, k = b[keep], s[keep], k[keep]
        out.append((b, s, k))
    return out
