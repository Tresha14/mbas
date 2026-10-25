"""User study kit: SUS (System Usability Scale) + usefulness/fairness survey for 10 interviewers and
10 candidates, and a summary script.

    python -m tools.user_study --template            # writes docs/user_study/responses_template.csv
    python -m tools.user_study docs/user_study/responses.csv
    python -m tools.user_study --demo               # checks the scoring on a clearly-marked example (not saved as results)
Questionnaire text: docs/user_study/README.md. SUS score = ((sum of odd items - 5) + (25 - sum of even items)) * 2.5.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "user_study"
SUS = [f"sus{i}" for i in range(1, 11)]
EXTRA = ["useful_overall", "useful_followups", "trust_scores", "fair_to_candidate", "feedback_actionable", "privacy_ok"]
COLUMNS = ["respondent_id", "role", *SUS, *EXTRA, "comment"]


def sus_score(row) -> float:
    odd = sum(row[f"sus{i}"] for i in (1, 3, 5, 7, 9))
    even = sum(row[f"sus{i}"] for i in (2, 4, 6, 8, 10))
    return ((odd - 5) + (25 - even)) * 2.5


def grade(s: float) -> str:
    return "excellent" if s >= 80.3 else "good" if s >= 68 else "ok" if s >= 51 else "poor"


def summarise(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["sus"] = df.apply(sus_score, axis=1)
    out = {}
    for role, g in df.groupby("role"):
        out[role] = {"n": len(g), "sus_mean": round(float(g.sus.mean()), 1), "sus_sd": round(float(g.sus.std(ddof=1)), 1) if len(g) > 1 else 0.0,
                     "sus_grade": grade(float(g.sus.mean())),
                     **{f"{c}_mean_1to5": round(float(g[c].mean()), 2) for c in EXTRA}}
    out["all"] = {"n": len(df), "sus_mean": round(float(df.sus.mean()), 1), "sus_grade": grade(float(df.sus.mean()))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.template or not (args.csv or args.demo):
        pd.DataFrame(columns=COLUMNS).to_csv(OUT / "responses_template.csv", index=False)
        print(f"wrote {OUT / 'responses_template.csv'} (one row per respondent, items scored 1-5)")
    if args.demo:
        rng = np.random.default_rng(0)
        demo = pd.DataFrame([{"respondent_id": f"X{i}", "role": "interviewer" if i < 10 else "candidate",
                              **{c: int(rng.integers(3, 6)) if int(c[3:]) % 2 else int(rng.integers(1, 3)) for c in SUS},
                              **{c: int(rng.integers(3, 6)) for c in EXTRA}, "comment": ""} for i in range(20)])
        print("EXAMPLE DATA ONLY (scoring check, not study results):", summarise(demo))
    if args.csv:
        res = summarise(pd.read_csv(args.csv))
        from ml.common.config import save_json

        save_json(res, OUT / "summary.json")
        lines = ["# User study results", "", "| group | n | SUS mean | grade |", "| --- | ---: | ---: | --- |"]
        lines += [f"| {k} | {v['n']} | {v['sus_mean']} | {v['sus_grade']} |" for k, v in res.items()]
        (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
