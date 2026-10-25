"""Label Studio JSON exports -> data/processed/labels/recorded/*.parquet.

Reads the four MBAS projects (see tools/labelstudio_client.py), either from a
running Label Studio (--pull) or from JSON export files saved from the UI.

Outputs (one row per item; times in seconds from the recording start):
    windows.parquet                 primary label per window (attention, engagement, flags)
    answers.parquet                 primary label per answer (transcript, quality, relevance, follow-up)
    hesitation_events.parquet       primary annotator's hesitation spans
    windows_annotations.parquet     every annotator's labels  (input to tools/agreement.py)
    answers_annotations.parquet
    hesitation_annotations.parquet

The primary annotator for a double-labelled item is the first one to submit.

    python -m tools.export_labels --pull
    python -m tools.export_labels --windows exp_w.json exp_wd.json --answers exp_a.json exp_ad.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.labelstudio_client import PROJECTS, LabelStudio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "labels" / "recorded"
EXPORT_DIR = ROOT / "data" / "processed" / "labelstudio" / "exports"

HESITATION_TYPES = ("filler", "long_pause", "repetition", "restart")
REASONS = ("vague", "low_confidence", "attention_drop", "topic_not_covered", "contradiction")


def _annotator(ann: dict, users: dict[int, str]) -> str:
    cb = ann.get("completed_by")
    if isinstance(cb, dict):
        return cb.get("email") or f"user{cb.get('id')}"
    if cb is None:
        return "unknown"
    return users.get(cb, f"user{cb}")


def _results(ann: dict) -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = {}
    for r in ann.get("result", []):
        by_name.setdefault(r.get("from_name"), []).append(r)
    return by_name


def _choice(res: dict, name: str) -> str | None:
    for r in res.get(name, []):
        ch = r.get("value", {}).get("choices") or []
        if ch:
            return ch[0]
    return None


def _choices(res: dict, name: str) -> list[str]:
    out: list[str] = []
    for r in res.get(name, []):
        out += r.get("value", {}).get("choices") or []
    return out


def _submitted(task: dict):
    """Annotations that count: not cancelled (skipped), sorted by creation."""
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    return sorted(anns, key=lambda a: (a.get("created_at") or "", a.get("id") or 0))


def parse_windows(tasks: list[dict], users: dict[int, str] | None = None) -> pd.DataFrame:
    users = users or {}
    rows = []
    for t in tasks:
        d = t["data"]
        for rank, ann in enumerate(_submitted(t)):
            res = _results(ann)
            rows.append({
                "window_id": d["window_id"], "session_id": d["session_id"], "participant_id": d.get("participant_id"),
                "window_index": d.get("window_index"), "start_s": d["t0_s"], "end_s": d["t1_s"],
                "answer_id": d.get("answer_id") or None,
                "attention_label": _choice(res, "attention"), "engagement_label": _choice(res, "engagement"),
                "flags": "|".join(_choices(res, "flags")),
                "annotator": _annotator(ann, users), "annotation_rank": rank, "lead_time_s": ann.get("lead_time"),
            })
    return pd.DataFrame(rows, columns=[
        "window_id", "session_id", "participant_id", "window_index", "start_s", "end_s", "answer_id",
        "attention_label", "engagement_label", "flags", "annotator", "annotation_rank", "lead_time_s"])


def parse_answers(tasks: list[dict], users: dict[int, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = users or {}
    rows, spans = [], []
    for t in tasks:
        d = t["data"]
        for rank, ann in enumerate(_submitted(t)):
            res = _results(ann)
            who = _annotator(ann, users)
            text = ""
            for r in res.get("transcript", []):
                text = " ".join(r.get("value", {}).get("text") or [])
            rating = next((r["value"].get("rating") for r in res.get("quality", [])), None)
            relevance = next((r["value"].get("number") for r in res.get("relevance", [])), None)
            fu = _choice(res, "follow_up")
            reason = _choice(res, "follow_up_reason") if fu == "yes" else "none"
            rows.append({
                "answer_id": d["answer_id"], "session_id": d["session_id"], "participant_id": d.get("participant_id"),
                "question_id": d.get("question_id"), "question_text": d.get("question_text"),
                "category": d.get("category"), "ask_s": d.get("ask_s"), "end_s": d.get("end_s"),
                "transcript": text, "quality_1to5": rating, "relevance": relevance,
                "follow_up_needed": None if fu is None else int(fu == "yes"), "follow_up_reason": reason,
                "annotator": who, "annotation_rank": rank, "lead_time_s": ann.get("lead_time"),
            })
            clip0 = float(d.get("clip_t0_s") or 0.0)
            for r in res.get("hesitation", []):
                v = r.get("value", {})
                for lab in v.get("labels") or []:
                    if lab not in HESITATION_TYPES:
                        continue
                    spans.append({
                        "session_id": d["session_id"], "answer_id": d["answer_id"],
                        "participant_id": d.get("participant_id"), "event_type": lab,
                        "start_s": round(clip0 + float(v["start"]), 3), "end_s": round(clip0 + float(v["end"]), 3),
                        "clip_t0_s": clip0, "clip_t1_s": d.get("clip_t1_s"),
                        "annotator": who, "annotation_rank": rank,
                    })
    a = pd.DataFrame(rows, columns=[
        "answer_id", "session_id", "participant_id", "question_id", "question_text", "category", "ask_s", "end_s",
        "transcript", "quality_1to5", "relevance", "follow_up_needed", "follow_up_reason",
        "annotator", "annotation_rank", "lead_time_s"])
    h = pd.DataFrame(spans, columns=[
        "session_id", "answer_id", "participant_id", "event_type", "start_s", "end_s", "clip_t0_s", "clip_t1_s",
        "annotator", "annotation_rank"])
    return a, h


def primary(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df[df["annotation_rank"] == 0].drop(columns=["annotation_rank"]).sort_values(key).reset_index(drop=True)


def write_all(win: pd.DataFrame, ans: pd.DataFrame, hes: pd.DataFrame, out: Path) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    first_ann = ans[ans["annotation_rank"] == 0][["answer_id", "annotator"]]
    hes_primary = hes.merge(first_ann, on=["answer_id", "annotator"], how="inner")
    files = {
        "windows_annotations.parquet": win, "answers_annotations.parquet": ans, "hesitation_annotations.parquet": hes,
        "windows.parquet": primary(win, "window_id"), "answers.parquet": primary(ans, "answer_id"),
        "hesitation_events.parquet": hes_primary.drop(columns=["annotation_rank"])
                                                 .sort_values(["answer_id", "start_s"]).reset_index(drop=True),
    }
    for name, df in files.items():
        df.to_parquet(out / name, index=False)
    return {k: len(v) for k, v in files.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pull", action="store_true", help="export directly from the running Label Studio")
    ap.add_argument("--windows", nargs="*", default=[], help="JSON exports of the windows projects")
    ap.add_argument("--answers", nargs="*", default=[], help="JSON exports of the answers projects")
    ap.add_argument("--url")
    ap.add_argument("--token", help="personal access token (default: log in with $LABEL_STUDIO_USERNAME/PASSWORD)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    win_tasks: list[dict] = []
    ans_tasks: list[dict] = []
    users: dict[int, str] = {}
    if args.pull:
        ls = LabelStudio(args.url, args.token)
        users = ls.users()
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        for key, (title, _, _) in PROJECTS.items():
            p = ls.project_by_title(title)
            if p is None:
                print(f"project '{title}' not found, skipped")
                continue
            tasks = ls.export_tasks(p["id"])
            (EXPORT_DIR / f"{key}.json").write_text(json.dumps(tasks), encoding="utf-8")
            (win_tasks if key.startswith("windows") else ans_tasks).extend(tasks)
            print(f"pulled '{title}': {len(tasks)} tasks")
    for f in args.windows:
        win_tasks += json.loads(Path(f).read_text(encoding="utf-8"))
    for f in args.answers:
        ans_tasks += json.loads(Path(f).read_text(encoding="utf-8"))
    if not win_tasks and not ans_tasks:
        ap.error("nothing to export: use --pull or pass export files")

    win = parse_windows(win_tasks, users)
    ans, hes = parse_answers(ans_tasks, users)
    counts = write_all(win, ans, hes, Path(args.out))
    for k, v in counts.items():
        print(f"{args.out}/{k}: {v} rows")
    unlabelled = sum(1 for t in win_tasks + ans_tasks if not _submitted(t))
    if unlabelled:
        print(f"note: {unlabelled} tasks have no submitted annotation yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
