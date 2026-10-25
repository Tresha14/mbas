"""5 s windows with a 2.5 s hop, with labels attached from data/processed/labels/<source>/.

Two sources share one schema:
  recorded   windows built from each session folder; labels from export_labels.py
  synthetic  windows.csv of the synthetic dataset (features + labels already per window)

Every window also gets hesitation-event counts (events overlapping the window)
and the person-wise split from data/splits/<source>/sessions.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.common.config import LABELS, RAW, SPLITS, SYNTHETIC
from tools.session_index import HOP_S, WINDOW_S, find_sessions, load_session, make_windows

HESITATION_TYPES = ("filler", "long_pause", "repetition", "restart")


def window_bounds(duration_s: float, window_s: float = WINDOW_S, hop_s: float = HOP_S) -> np.ndarray:
    """(N, 2) start/end of every complete window."""
    n = int(np.floor((duration_s - window_s) / hop_s + 1e-6)) + 1 if duration_s >= window_s else 0
    s = np.arange(n) * hop_s
    return np.stack([s, s + window_s], axis=1) if n else np.zeros((0, 2))


def count_events(windows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add <type>_count columns: events of each type overlapping each window (by session)."""
    out = windows.copy()
    for t in HESITATION_TYPES:
        out[f"{t}_count"] = 0
    if events is None or events.empty:
        return out
    for sid, ev in events.groupby("session_id"):
        m = out.session_id == sid
        if not m.any():
            continue
        ws, we = out.loc[m, "start_s"].to_numpy(), out.loc[m, "end_s"].to_numpy()
        for t in HESITATION_TYPES:
            e = ev[ev.event_type == t]
            if e.empty:
                continue
            es, ee = e.start_s.to_numpy(), e.end_s.to_numpy()
            ov = (es[None, :] < we[:, None]) & (ee[None, :] > ws[:, None])
            out.loc[m, f"{t}_count"] = ov.sum(axis=1)
    return out


def _attach_split(df: pd.DataFrame, source: str) -> pd.DataFrame:
    p = SPLITS / source / "sessions.csv"
    if not p.exists():
        return df.assign(split=None)
    ses = pd.read_csv(p)[["session_id", "split"]].drop_duplicates("session_id")
    return df.drop(columns=[c for c in ("split",) if c in df]).merge(ses, on="session_id", how="left")


def recorded_windows(raw: Path = RAW, labels_dir: Path = LABELS / "recorded") -> pd.DataFrame:
    rows = []
    for d in find_sessions(raw):
        s = load_session(d)
        for w in s.windows:
            rows.append({"window_id": w.window_id, "session_id": s.session_id,
                         "participant_id": s.candidate.get("participant_id"), "window_index": w.index,
                         "start_s": w.start_s, "end_s": w.end_s, "answer_id": w.answer_id})
    win = pd.DataFrame(rows, columns=["window_id", "session_id", "participant_id", "window_index", "start_s", "end_s",
                                      "answer_id"])
    lab_p = labels_dir / "windows.parquet"
    if lab_p.exists():
        lab = pd.read_parquet(lab_p)[["window_id", "attention_label", "engagement_label", "annotator"]]
        win = win.merge(lab, on="window_id", how="left")
    else:
        win = win.assign(attention_label=None, engagement_label=None, annotator=None)
    hes_p = labels_dir / "hesitation_events.parquet"
    hes = pd.read_parquet(hes_p) if hes_p.exists() else None
    return _attach_split(count_events(win, hes), "recorded")


def synthetic_windows(syn: Path = SYNTHETIC, labels_dir: Path = LABELS / "synthetic") -> pd.DataFrame:
    """Synthetic windows with every feature column (HIDDEN ground truth is never joined)."""
    win = pd.read_csv(syn / "interview" / "windows.csv")
    hes_p = labels_dir / "hesitation_events.parquet"
    hes = pd.read_parquet(hes_p) if hes_p.exists() else pd.read_csv(syn / "interview" / "hesitation_events.csv")
    win["window_index"] = win.window_id.str.split("_W").str[1].astype(int)
    return _attach_split(count_events(win, hes), "synthetic")


def build(source: str, out_dir: Path) -> Path:
    df = recorded_windows() if source == "recorded" else synthetic_windows()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"windows_{source}.parquet"
    df.to_parquet(p, index=False)
    return p


__all__ = ["window_bounds", "count_events", "recorded_windows", "synthetic_windows", "build", "make_windows"]
