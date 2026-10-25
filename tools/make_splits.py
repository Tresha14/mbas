"""Train / val / test split BY PARTICIPANT (no person in two splits).

Sources:
  recorded   our sessions in data/raw/<session>/meta.json. New 70/15/15 split,
             stratified by gender, seeded. Participants already assigned keep
             their split when more sessions are recorded.
  synthetic  data/raw/mbas_synthetic_dataset. Keeps the dataset's person-wise
             interview split (84/18/18) after verifying it. Meeting rows follow
             each participant's split; a meeting whose people sit in different
             splits is marked "mixed" (the dataset's own meeting-level split
             column is NOT person-wise, so it is not used).
  public:<name>  data/processed/public/<name>/manifest.parquet grouped by subject_id.

    python -m tools.make_splits                    # recorded + synthetic
    python -m tools.make_splits --source public:librispeech
Outputs: data/splits/<source>/{train,val,test}_participants.csv, sessions.csv, summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.session_index import find_sessions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SYN = RAW / "mbas_synthetic_dataset"
SPLITS = ROOT / "data" / "splits"
PUBLIC = ROOT / "data" / "processed" / "public"
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_NAMES = tuple(RATIOS)


def _rank(seed: int, pid: str) -> str:
    return hashlib.sha1(f"{seed}:{pid}".encode()).hexdigest()


def _targets(n: int) -> dict[str, int]:
    """Largest-remainder rounding of 70/15/15; val and test get at least one person once n >= 3."""
    raw = {k: r * n for k, r in RATIOS.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    if n >= 3:
        for k in ("val", "test"):
            if out[k] == 0:
                out[k], out["train"] = 1, out["train"] - 1
    return out


def split_people(people: pd.DataFrame, seed: int = 42, strata: str | None = "gender",
                 previous: dict[str, str] | None = None) -> dict[str, str]:
    """Assign each participant_id to a split. Stratified round-robin keeps groups balanced."""
    previous = previous or {}
    out = {p: s for p, s in previous.items() if p in set(people.participant_id) and s in SPLIT_NAMES}
    todo = people[~people.participant_id.isin(out)].copy()
    target = _targets(len(people))
    need = {k: target[k] - sum(1 for s in out.values() if s == k) for k in SPLIT_NAMES}
    key = todo[strata].fillna("unknown").astype(str) if strata and strata in todo else pd.Series("all", index=todo.index)
    todo["_stratum"] = key
    todo["_rank"] = todo.participant_id.map(lambda p: _rank(seed, p))
    # interleave strata so every split gets a similar mix
    todo = todo.sort_values(["_stratum", "_rank"])
    todo["_pos"] = todo.groupby("_stratum").cumcount() / todo.groupby("_stratum")["_rank"].transform("count")
    order = list(todo.sort_values(["_pos", "_rank"]).participant_id)
    stratum_of = dict(zip(todo.participant_id, todo._stratum))
    size = todo._stratum.value_counts().to_dict()
    taken: dict[tuple[str, str], int] = {}
    left = dict(need)
    for pid in order:
        st = stratum_of[pid]
        open_ = [k for k in SPLIT_NAMES if left[k] > 0] or list(SPLIT_NAMES)
        # split furthest below its share inside this stratum, then globally
        s = max(open_, key=lambda k: (RATIOS[k] * size[st] - taken.get((st, k), 0),
                                      left[k] / max(need[k], 1), k == "train"))
        out[pid] = s
        left[s] -= 1
        taken[(st, s)] = taken.get((st, s), 0) + 1
    return out


def verify(sessions: pd.DataFrame) -> list[str]:
    """Every participant must appear in exactly one split."""
    per = sessions.groupby("participant_id").split.nunique()
    return [f"participant {p} is in {n} splits" for p, n in per.items() if n > 1]


def _write(name: str, people: pd.DataFrame, sessions: pd.DataFrame, extra: dict | None = None) -> dict:
    d = SPLITS / name
    d.mkdir(parents=True, exist_ok=True)
    for s in SPLIT_NAMES:
        people[people.split == s][["participant_id"]].sort_values("participant_id").to_csv(
            d / f"{s}_participants.csv", index=False)
    sessions.sort_values(["split", "session_id"]).to_csv(d / "sessions.csv", index=False)
    problems = verify(sessions)
    summary = {
        "source": name, "seed": 42, "ratios": RATIOS,
        "participants": {s: int((people.split == s).sum()) for s in SPLIT_NAMES},
        "sessions": {s: int((sessions.split == s).sum()) for s in SPLIT_NAMES},
        "leakage_problems": problems, **(extra or {}),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def recorded(raw: Path = RAW, seed: int = 42) -> dict:
    rows = []
    for d in find_sessions(raw):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        c = meta.get("participants", {}).get("candidate") or {}
        if not c.get("participant_id"):
            continue
        rows.append({"session_id": meta.get("session_id", d.name), "participant_id": c["participant_id"],
                     "gender": c.get("gender"), "accent": c.get("accent"), "glasses": c.get("glasses"),
                     "age_band": c.get("age_band")})
    sessions = pd.DataFrame(rows, columns=["session_id", "participant_id", "gender", "accent", "glasses", "age_band"])
    people = sessions.drop_duplicates("participant_id").drop(columns="session_id")
    prev_path = SPLITS / "recorded" / "sessions.csv"
    previous = {}
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        previous = dict(zip(prev.participant_id, prev.split))
    assign = split_people(people, seed, "gender", previous) if len(people) else {}
    people = people.assign(split=people.participant_id.map(assign))
    sessions = sessions.assign(split=sessions.participant_id.map(assign))
    return _write("recorded", people, sessions[["session_id", "participant_id", "split"]])


def synthetic(syn: Path = SYN) -> dict:
    p = pd.read_csv(syn / "interview" / "participants.csv")
    for s in SPLIT_NAMES:
        listed = set(pd.read_csv(syn / "splits" / f"{s}_participants.csv").participant_id)
        if listed != set(p[p.split == s].participant_id):
            raise SystemExit(f"synthetic splits/{s}_participants.csv disagrees with participants.csv")
    assign = dict(zip(p.participant_id, p.split))
    ses = pd.read_csv(syn / "interview" / "sessions.csv")[["session_id", "participant_id"]]
    ses = ses.assign(split=ses.participant_id.map(assign), mode="interview")

    mp = pd.read_csv(syn / "meeting" / "meeting_participant_summary.csv")[["meeting_id", "participant_id"]]
    mp = mp.assign(split=mp.participant_id.map(assign))
    mt = pd.read_csv(syn / "meeting" / "meetings.csv")[["meeting_id", "split"]].rename(columns={"split": "split_provided"})
    per_meeting = mp.groupby("meeting_id").split.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
    mt = mt.assign(split=mt.meeting_id.map(per_meeting))
    d = SPLITS / "synthetic"
    d.mkdir(parents=True, exist_ok=True)
    mp.to_csv(d / "meeting_participants.csv", index=False)
    mt.to_csv(d / "meetings.csv", index=False)
    mp_ses = mp.rename(columns={"meeting_id": "session_id"}).assign(mode="meeting")
    extra = {
        "meetings": mt.split.value_counts().to_dict(),
        "meeting_rows_where_provided_split_is_not_personwise": int(
            (mp.merge(mt[["meeting_id", "split_provided"]], on="meeting_id")
             .pipe(lambda x: (x.split != x.split_provided).sum()))),
        "note": "meeting-level targets: use meetings with split != 'mixed'; per-participant meeting rows follow "
                "the participant's split (meeting_participants.csv)",
    }
    people = p[["participant_id", "split"]]
    return _write("synthetic", people, pd.concat([ses, mp_ses], ignore_index=True), extra)


def public(name: str, seed: int = 42) -> dict:
    man = pd.read_parquet(PUBLIC / name / "manifest.parquet")
    if "subject_id" not in man:
        raise SystemExit(f"{name} manifest has no subject_id column")
    people = man[["subject_id"]].drop_duplicates().rename(columns={"subject_id": "participant_id"})
    assign = split_people(people, seed, strata=None)
    people = people.assign(split=people.participant_id.map(assign))
    items = man[["item_id", "subject_id"]].rename(columns={"item_id": "session_id", "subject_id": "participant_id"})
    return _write(f"public/{name}", people, items.assign(split=items.participant_id.map(assign)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="*", default=["recorded", "synthetic"],
                    help="recorded, synthetic and/or public:<name>")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    bad = False
    for src in args.source:
        if src == "recorded":
            s = recorded(seed=args.seed)
        elif src == "synthetic":
            s = synthetic()
        elif src.startswith("public:"):
            s = public(src.split(":", 1)[1], args.seed)
        else:
            ap.error(f"unknown source {src}")
        print(f"{s['source']}: participants {s['participants']}  sessions/items {s['sessions']}")
        for k in ("meetings", "meeting_rows_where_provided_split_is_not_personwise"):
            if k in s:
                print(f"  {k}: {s[k]}")
        for prob in s["leakage_problems"]:
            print(f"  LEAK {prob}")
            bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
