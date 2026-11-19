"""Does our score match what people think? Compares the report's scores with ratings given by human raters.

Ported and adapted from the older project (minor_project/evaluation/). Two checks:
  1. Rater agreement: are the human raters consistent with each other (Cohen's kappa, averaged over rater pairs)?
     If they are not, an averaged human rating cannot be trusted, and that is itself a finding (clearer rubric, rater
     training), not something to hide.
  2. System vs human: does our score (per dimension and overall) correlate with the mean human rating of the same
     interview (Pearson and Spearman)? Warns, instead of hiding the number, when fewer than 10 interviews were rated.

Every statistic returns None (never a made-up 0 or 1) when the data cannot support it: fewer than two items, a
constant series, or no disagreement to measure.

    python -m tools.validate_ratings --template ratings.csv     # write an empty template
    python -m tools.validate_ratings ratings.csv                # compare with the reports in the database
    python -m tools.validate_ratings ratings.csv --json out.json

CSV columns: interview_id, rater_id, overall, content, fluency, composure, delivery, eye_contact, responsiveness, notes
(ratings are whole numbers 1-5; leave a cell empty to skip a dimension; every interview needs a built report).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

DIMENSIONS = ["overall", "content", "fluency", "composure", "delivery", "eye_contact", "responsiveness"]
KAPPA_RELIABLE = 0.6          # Landis & Koch rule of thumb for "substantial agreement": a documented starting point
MIN_N = 10                    # fewer rated interviews than this and a correlation is preliminary, not a result


# ----------------------------------------------------------------------------------------------- statistics
def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length, got {len(x)} and {len(y)}")
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    vx, vy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return None                                   # undefined against a constant series, not 0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (ties share the mean rank), the standard convention for Spearman's rho."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length, got {len(x)} and {len(y)}")
    return pearson(_ranks(x), _ranks(y)) if len(x) >= 2 else None


def cohens_kappa(a: list, b: list) -> float | None:
    """Agreement between two raters corrected for chance. None when undefined (no items, or both raters gave one
    identical category to everything, so there was no disagreement to explain)."""
    if len(a) != len(b):
        raise ValueError(f"a and b must be the same length, got {len(a)} and {len(b)}")
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b), key=str)
    observed = sum(1 for p, q in zip(a, b) if p == q) / n
    expected = sum((sum(1 for p in a if p == c) / n) * (sum(1 for q in b if q == c) / n) for c in cats)
    return None if expected >= 1.0 else (observed - expected) / (1 - expected)


# ----------------------------------------------------------------------------------------------- the two checks
def rater_agreement(ratings: dict[str, dict[int, float]]) -> dict:
    """ratings: {rater: {interview_id: rating}} for ONE dimension. Mean pairwise kappa over raters that rated at
    least two of the same interviews."""
    ks, items = [], 0
    for r1, r2 in combinations(sorted(ratings), 2):
        common = sorted(set(ratings[r1]) & set(ratings[r2]))
        if len(common) < 2:
            continue
        k = cohens_kappa([ratings[r1][i] for i in common], [ratings[r2][i] for i in common])
        items = max(items, len(common))
        if k is not None:
            ks.append(k)
    kappa = sum(ks) / len(ks) if ks else None
    return {"kappa": None if kappa is None else round(kappa, 3), "pairs": len(ks), "items": items,
            "reliable": None if kappa is None else kappa >= KAPPA_RELIABLE}


def correlate(system: list[float], human: list[float]) -> dict:
    n = len(system)
    r, rho = pearson(system, human), spearman(system, human)
    note = None if n >= MIN_N else (f"only {n} rated interview(s): below the {MIN_N} this project treats as a trustworthy "
                                    "sample, so treat these numbers as preliminary, not a validated result")
    return {"n": n, "pearson_r": None if r is None else round(r, 3), "spearman_rho": None if rho is None else round(rho, 3), "note": note}


# ----------------------------------------------------------------------------------------------- data
def read_ratings(path: Path) -> dict[str, dict[str, dict[int, float]]]:
    """{dimension: {rater: {interview_id: rating}}} from the CSV."""
    out: dict[str, dict[str, dict[int, float]]] = {d: defaultdict(dict) for d in DIMENSIONS}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("interview_id") and not str(r["interview_id"]).startswith("#")]
    if not rows:
        raise SystemExit(f"{path} has no ratings yet. Fill in the template (python -m tools.validate_ratings --template {path}).")
    missing = [c for c in ("interview_id", "rater_id") if c not in rows[0]]
    if missing:
        raise SystemExit(f"{path} is missing the column(s): {', '.join(missing)}")
    for r in rows:
        for d in DIMENSIONS:
            v = (r.get(d) or "").strip()
            if v:
                try:
                    x = float(v)
                except ValueError:
                    raise SystemExit(f"interview {r['interview_id']}, rater {r['rater_id']}: '{v}' in column {d} is not a number")
                if not 1 <= x <= 5:
                    raise SystemExit(f"interview {r['interview_id']}, rater {r['rater_id']}: {d} must be 1-5, got {v}")
                out[d][r["rater_id"].strip()][int(r["interview_id"])] = x
    return out


def system_scores(interview_ids: list[int]) -> dict[int, dict[str, float | None]]:
    """The stored report scorecard of each interview: {id: {dimension: 0-100 score or None}}."""
    from sqlalchemy import select

    from backend.config import settings
    from backend.db import Database, Report

    db = Database(settings().database_url)
    out: dict[int, dict[str, float | None]] = {}
    with db.session() as s:
        for rep in s.scalars(select(Report).where(Report.interview_id.in_(interview_ids))):
            sc = (rep.interviewer_view or {}).get("scorecard") or {}
            row = {"overall": sc.get("overall")}
            row.update({k: v.get("score") for k, v in (sc.get("dimensions") or {}).items()})
            out[rep.interview_id] = row
    return out


def analyse(ratings: dict, scores: dict[int, dict]) -> dict:
    result = {}
    for d in DIMENSIONS:
        by_rater = ratings[d]
        if not by_rater:
            continue
        ids = sorted({i for r in by_rater.values() for i in r})
        mean_h, sys_s = [], []
        for i in ids:
            hs = [r[i] for r in by_rater.values() if i in r]
            s = (scores.get(i) or {}).get(d)
            if hs and s is not None:                      # an interview without a system score is skipped, not zeroed
                mean_h.append(sum(hs) / len(hs))
                sys_s.append(float(s))
        result[d] = {"agreement": rater_agreement(by_rater), "system_vs_human": correlate(sys_s, mean_h)}
    return result


def show(res: dict) -> None:
    fmt = lambda v: "undefined" if v is None else f"{v:+.2f}"  # noqa: E731
    print(f"{'dimension':16s} {'raters agree (kappa)':26s} {'system vs human mean':34s}")
    for d, r in res.items():
        a, c = r["agreement"], r["system_vs_human"]
        agree = "undefined" if a["kappa"] is None else f"{a['kappa']:.2f} ({'reliable' if a['reliable'] else 'NOT reliable'})"
        corr = f"r {fmt(c['pearson_r'])}  rho {fmt(c['spearman_rho'])}  n={c['n']}"
        print(f"{d:16s} {agree:26s} {corr}")
    notes = {c["system_vs_human"]["note"] for c in res.values() if c["system_vs_human"]["note"]}
    for n in sorted(notes):
        print(f"\nNote: {n}")
    if any(r["agreement"]["reliable"] is False for r in res.values()):
        print("\nNote: where raters do not agree with each other, the human mean is not a trustworthy yardstick; "
              "fix the rubric or train the raters before reading the correlation.")


def write_template(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["interview_id", "rater_id"] + DIMENSIONS + ["notes"])
        w.writerow(["# example (delete this row):", "rater_A", 4, 4, 3, 4, 4, 3, 4, "clear answers, a few long pauses"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ratings", nargs="?", help="CSV of human ratings (see --template)")
    ap.add_argument("--template", metavar="CSV", help="write an empty ratings template to this path and exit")
    ap.add_argument("--json", metavar="PATH", help="also save the results as JSON")
    args = ap.parse_args(argv)
    if args.template:
        write_template(Path(args.template))
        print(f"template written to {args.template}")
        return 0
    if not args.ratings:
        ap.error("give a ratings CSV, or --template to create one")
    ratings = read_ratings(Path(args.ratings))
    ids = sorted({i for d in ratings.values() for r in d.values() for i in r})
    scores = system_scores(ids)
    absent = [i for i in ids if i not in scores]
    if absent:
        print(f"No built report for interview(s) {absent}: build it first (open the report, or press Refresh).\n")
    res = analyse(ratings, scores)
    show(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
