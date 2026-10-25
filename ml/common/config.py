"""Paths, seeds, YAML configs and run directories shared by every model."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
PUBLIC = PROCESSED / "public"
LABELS = PROCESSED / "labels"
RUNS = ROOT / "runs"
CHECKPOINTS = ROOT / "checkpoints"
DOCS = ROOT / "docs"
METRICS = DOCS / "metrics"
MODEL_CARDS = DOCS / "model_cards"
SYNTHETIC = RAW / "mbas_synthetic_dataset"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """YAML config with optional dotted overrides, e.g. {"train.epochs": 2}."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key, value in (overrides or {}).items():
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = value
    return cfg


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """['train.epochs=2', 'data.limit=100'] -> {'train.epochs': 2, 'data.limit': 100}"""
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def rel(path: str | Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def abspath(path: str | Path) -> Path:
    """Manifest paths are relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
