"""Shared training plumbing: run folders, TensorBoard, resumable checkpoints, metrics JSON, model cards, plots."""
from __future__ import annotations

import datetime as dt
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.common.config import CHECKPOINTS, METRICS, MODEL_CARDS, ROOT, RUNS, save_json


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Run:
    """runs/<name>/ (TensorBoard + config + log) and checkpoints/<name>/ (last.pt, best.pt)."""

    def __init__(self, name: str, cfg: dict, resume: bool = False):
        self.name = name
        self.dir = RUNS / name
        self.ckpt_dir = CHECKPOINTS / name
        if not resume and self.dir.exists():
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            shutil.move(str(self.dir), str(RUNS / f"{name}_old_{stamp}"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        from torch.utils.tensorboard import SummaryWriter

        self.tb = SummaryWriter(str(self.dir))
        self.log_path = self.dir / "log.txt"
        self.history: list[dict] = []
        hist = self.dir / "history.json"
        if resume and hist.exists():
            self.history = json.loads(hist.read_text(encoding="utf-8"))

    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def scalars(self, step: int, **values: float) -> None:
        for k, v in values.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                self.tb.add_scalar(k, v, step)

    def epoch_end(self, epoch: int, **values: float) -> None:
        self.history.append({"epoch": epoch, **{k: float(v) for k, v in values.items() if v is not None}})
        (self.dir / "history.json").write_text(json.dumps(self.history, indent=1), encoding="utf-8")
        self.scalars(epoch, **{f"epoch/{k}": v for k, v in values.items()})
        self.tb.flush()

    def save(self, state: dict, best: bool = False) -> None:
        torch.save(state, self.ckpt_dir / "last.pt")
        if best:
            shutil.copy(self.ckpt_dir / "last.pt", self.ckpt_dir / "best.pt")

    def load_last(self) -> dict | None:
        p = self.ckpt_dir / "last.pt"
        return torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None

    def close(self) -> None:
        self.tb.close()


def cosine_lr(step: int, total: int, base: float, warmup: int = 0, floor: float = 0.02) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def noam_lr(step: int, d_model: int, warmup: int, factor: float = 1.0) -> float:
    step = max(step, 1)
    return factor * d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def write_metrics(name: str, metrics: dict) -> Path:
    metrics = {"model": name, "generated": dt.datetime.now().isoformat(timespec="seconds"), **metrics}
    return save_json(metrics, METRICS / f"{name}.json")


def plot_history(name: str, keys: list[str], out: Path | None = None) -> Path | None:
    """Loss / metric curves from runs/<name>/history.json -> docs/metrics/<name>_curves.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist_p = RUNS / name / "history.json"
    if not hist_p.exists():
        return None
    h = json.loads(hist_p.read_text(encoding="utf-8"))
    keys = [k for k in keys if any(k in r for r in h)]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.2), squeeze=False)
    for ax, k in zip(axes[0], keys):
        xs = [r["epoch"] for r in h if k in r]
        ax.plot(xs, [r[k] for r in h if k in r], marker=".")
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.suptitle(name)
    fig.tight_layout()
    out = out or METRICS / f"{name}_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.1 * len(labels) + 1.5))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def model_card(name: str, title: str, sections: dict[str, Any]) -> Path:
    """docs/model_cards/<name>.md from a dict of section -> text / dict / list."""
    lines = [f"# Model card: {title}", "",
             f"_Generated {dt.date.today().isoformat()} by the training/evaluation scripts. "
             "Trained from random initialisation; no pre-trained weights._", ""]
    for head, body in sections.items():
        lines += [f"## {head}", ""]
        if isinstance(body, dict):
            lines += ["| | |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in body.items()]
        elif isinstance(body, list):
            lines += [f"- {x}" for x in body]
        else:
            lines.append(str(body))
        lines.append("")
    p = MODEL_CARDS / f"{name}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def benchmark(fn, *args, warmup: int = 5, iters: int = 50) -> dict:
    """Latency of fn(*args) in ms (median, p90) and throughput."""
    for _ in range(warmup):
        fn(*args)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn(*args)
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"median_ms": round(float(np.median(ts)), 3), "p90_ms": round(float(np.percentile(ts, 90)), 3),
            "fps": round(1000 / float(np.median(ts)), 1)}
