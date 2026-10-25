"""Inter-annotator agreement on double-labelled sessions.

Reads *_annotations.parquet written by export_labels.py (or synthetic_to_labels.py)
and compares every pair of annotators on the items both labelled:

  windows : attention (Cohen's kappa), engagement (kappa + linear-weighted kappa)
  answers : quality (quadratic-weighted kappa), relevance (kappa on 5 bins + Pearson r + MAE),
            follow_up (kappa), follow_up_reason (kappa), speaking_pattern (kappa, if present)
  hesitation spans : per event type, frame-level kappa at 10 Hz over each answer clip

    python -m tools.agreement --labels data/processed/labels/recorded
    python -m tools.agreement --labels data/processed/labels/synthetic
Writes docs/metrics/agreement_<source>.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_labels import HESITATION_TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "docs" / "metrics"
FRAME_HZ = 10


def kappa(a, b, weights: str | None = None, labels=None) -> float | None:
    a, b = list(a), list(b)
    if len(a) < 2:
        return None
    if len(set(a) | set(b)) < 2:  # both annotators used one and the same class only: kappa is undefined
        return None
    k = cohen_kappa_score(a, b, weights=weights, labels=labels)
    return None if np.isnan(k) else round(float(k), 4)


def interpret(k: float | None) -> str:
    """Landis & Koch (1977) bands."""
    if k is None:
        return "n/a"
    for lim, name in [(0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if k <= lim:
            return name
    return "almost perfect"


def _pairs(df: pd.DataFrame, key: str, cols: list[str]):
    """Yield (annotator_a, annotator_b, merged frame of items both labelled)."""
    anns = sorted(df["annotator"].dropna().unique())
    for a, b in itertools.combinations(anns, 2):
        m = df[df.annotator == a][[key, "session_id", *cols]].merge(
            df[df.annotator == b][[key, *cols]], on=key, suffixes=("_a", "_b"))
        if len(m):
            yield a, b, m


def window_agreement(df: pd.DataFrame) -> list[dict]:
    out = []
    for a, b, m in _pairs(df, "window_id", ["attention_label", "engagement_label"]):
        m = m.dropna()
        eng_order = {"disengaged": 0, "neutral": 1, "engaged": 2}
        out.append({
            "annotators": [a, b], "n_items": len(m), "n_sessions": int(m.session_id.nunique()),
            "attention_kappa": kappa(m.attention_label_a, m.attention_label_b),
            "attention_agreement": round(float((m.attention_label_a == m.attention_label_b).mean()), 4),
            "engagement_kappa": kappa(m.engagement_label_a, m.engagement_label_b),
            "engagement_kappa_linear": kappa(m.engagement_label_a.map(eng_order), m.engagement_label_b.map(eng_order),
                                             weights="linear", labels=[0, 1, 2]),
            "engagement_agreement": round(float((m.engagement_label_a == m.engagement_label_b).mean()), 4),
        })
    return out


def answer_agreement(df: pd.DataFrame) -> list[dict]:
    cols = ["quality_1to5", "relevance", "follow_up_needed", "follow_up_reason"]
    has_sp = "speaking_pattern" in df.columns and df["speaking_pattern"].notna().any()
    if has_sp:
        cols.append("speaking_pattern")
    out = []
    for a, b, m in _pairs(df, "answer_id", cols):
        m = m.dropna(subset=["quality_1to5_a", "quality_1to5_b", "relevance_a", "relevance_b"])
        if not len(m):
            continue
        ra, rb = m.relevance_a.astype(float), m.relevance_b.astype(float)
        bins = [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01]
        r = {
            "annotators": [a, b], "n_items": len(m), "n_sessions": int(m.session_id.nunique()),
            "quality_kappa_quadratic": kappa(m.quality_1to5_a.astype(int), m.quality_1to5_b.astype(int),
                                             weights="quadratic", labels=[1, 2, 3, 4, 5]),
            "quality_exact_agreement": round(float((m.quality_1to5_a == m.quality_1to5_b).mean()), 4),
            "relevance_kappa_5bins": kappa(pd.cut(ra, bins, labels=False), pd.cut(rb, bins, labels=False),
                                           weights="linear", labels=[0, 1, 2, 3, 4]),
            "relevance_pearson_r": (round(float(np.corrcoef(ra, rb)[0, 1]), 4)
                                    if len(m) > 2 and ra.std() > 0 and rb.std() > 0 else None),
            "relevance_mae": round(float((ra - rb).abs().mean()), 4),
            "follow_up_kappa": kappa(m.follow_up_needed_a.astype(int), m.follow_up_needed_b.astype(int)),
            "follow_up_reason_kappa": kappa(m.follow_up_reason_a.fillna("none"), m.follow_up_reason_b.fillna("none")),
        }
        if has_sp:
            r["speaking_pattern_kappa"] = kappa(m.speaking_pattern_a.fillna(""), m.speaking_pattern_b.fillna(""))
        out.append(r)
    return out


def _frames(spans: pd.DataFrame, t0: float, t1: float) -> np.ndarray:
    n = max(int(np.ceil((t1 - t0) * FRAME_HZ)), 1)
    x = np.zeros(n, dtype=int)
    for s, e in zip(spans.start_s, spans.end_s):
        x[max(int((s - t0) * FRAME_HZ), 0):max(int(np.ceil((e - t0) * FRAME_HZ)), 0)] = 1
    return x


def hesitation_agreement(hes: pd.DataFrame, answers: pd.DataFrame) -> list[dict]:
    """Only answers both annotators submitted count, including ones where an annotator marked no spans."""
    out = []
    span = answers.groupby("answer_id").agg(t0=("ask_s", "min"), t1=("end_s", "max"))
    for a, b, m in _pairs(answers, "answer_id", ["quality_1to5"]):
        res = {"annotators": [a, b], "n_answers": len(m)}
        for typ in HESITATION_TYPES:
            xa, xb = [], []
            for aid in m.answer_id:
                t0, t1 = span.loc[aid, "t0"], span.loc[aid, "t1"] + 0.5
                h = hes[(hes.answer_id == aid) & (hes.event_type == typ)]
                xa.append(_frames(h[h.annotator == a], t0, t1))
                xb.append(_frames(h[h.annotator == b], t0, t1))
            if xa:
                fa, fb = np.concatenate(xa), np.concatenate(xb)
                res[f"{typ}_frame_kappa"] = kappa(fa, fb, labels=[0, 1])
                res[f"{typ}_spans"] = [int(((hes.annotator == x) & (hes.event_type == typ)
                                            & hes.answer_id.isin(m.answer_id)).sum()) for x in (a, b)]
        out.append(res)
    return out


def compute(labels_dir: Path) -> dict:
    res: dict = {"labels_dir": str(labels_dir.relative_to(ROOT) if labels_dir.is_relative_to(ROOT) else labels_dir)}
    wpath, apath, hpath = (labels_dir / f"{k}_annotations.parquet" for k in ("windows", "answers", "hesitation"))
    if wpath.exists():
        w = pd.read_parquet(wpath)
        res["windows"] = window_agreement(w)
        sessions = w.session_id.nunique()
        multi = w.groupby("window_id").annotator.nunique()
        res["double_labelled_sessions"] = int(w[w.window_id.isin(multi[multi > 1].index)].session_id.nunique())
        res["sessions"] = int(sessions)
        res["double_labelled_fraction"] = round(res["double_labelled_sessions"] / max(sessions, 1), 4)
    if apath.exists():
        a = pd.read_parquet(apath)
        res["answers"] = answer_agreement(a)
        if hpath.exists():
            h = pd.read_parquet(hpath)
            if h["annotator"].nunique() > 1:
                res["hesitation"] = hesitation_agreement(h, a)
    return res


def _fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(ROOT / "data" / "processed" / "labels" / "recorded"))
    ap.add_argument("--out", help="metrics JSON (default docs/metrics/agreement_<labels dir name>.json)")
    args = ap.parse_args(argv)
    labels_dir = Path(args.labels).resolve()
    res = compute(labels_dir)
    out = Path(args.out) if args.out else METRICS / f"agreement_{labels_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"Agreement for {res['labels_dir']}  "
          f"(double-labelled sessions: {res.get('double_labelled_sessions', 0)}/{res.get('sessions', 0)})")
    for section in ("windows", "answers", "hesitation"):
        for r in res.get(section, []):
            print(f"  [{section}] {r['annotators'][0]} vs {r['annotators'][1]}")
            for k, v in r.items():
                if k.endswith("kappa") or "kappa_" in k:
                    print(f"      {k:28s} {_fmt(v):>7s}  {interpret(v)}")
    if not any(res.get(s) for s in ("windows", "answers")):
        print("  no items labelled by two annotators yet")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
