"""Label Studio JSON exports -> data/processed/labels/recorded/*.parquet.

Reads the MBAS projects (see tools/labelstudio_client.py), either from a
running Label Studio (--pull) or from JSON export files saved from the UI.
Window labels come from three sources, merged per window:
  * the sessions (timeline) projects: exception spans -> 5 s windows (> 50% rule);
  * the optional per-window projects;
  * weak labels ("scripted" annotator) for windows inside the scripted segments
    of data/raw/session_log.csv and the calibration; humans always rank first.

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
from tools.session_index import load_session, make_windows  # noqa: E402
from tools.session_log import read_session_log, weak_window_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "labels" / "recorded"
EXPORT_DIR = ROOT / "data" / "processed" / "labelstudio" / "exports"

RAW = ROOT / "data" / "raw"
HESITATION_TYPES = ("filler", "long_pause", "repetition", "restart")
WINDOW_COLUMNS = ["window_id", "session_id", "participant_id", "window_index", "start_s", "end_s", "answer_id",
                  "attention_label", "engagement_label", "flags", "annotator", "annotation_rank", "lead_time_s"]
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
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


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


def window_label_from_spans(spans: list[tuple[str, float, float]], start: float, end: float) -> tuple[str, str]:
    """Timeline spans -> (attention, engagement) for one window by the > 50% overlap rule."""
    cover: dict[str, float] = {}
    for lab, s, e in spans:
        cover[lab] = cover.get(lab, 0.0) + max(0.0, min(e, end) - max(s, start))
    half = (end - start) / 2
    att = "off_screen" if cover.get("off_screen", 0.0) > half else "on_screen"
    if cover.get("disengaged", 0.0) > half:
        eng = "disengaged"
    elif cover.get("engaged", 0.0) > half:
        eng = "engaged"
    else:
        eng = "neutral"
    return att, eng


def parse_sessions(tasks: list[dict], users: dict[int, str] | None = None, raw_dir: Path = RAW) -> pd.DataFrame:
    """Whole-session timeline tasks -> one row per 5 s window per annotator (same columns as parse_windows)."""
    users = users or {}
    rows = []
    for t in tasks:
        d = t["data"]
        fps = float(d.get("fps") or 16)
        sdir = raw_dir / d["session_id"]
        try:  # answer ids need the session folder; without it windows still get labels
            idx = load_session(sdir)
            windows, sid = idx.windows, idx.session_id
        except FileNotFoundError:
            windows, sid = make_windows(d["session_id"], float(d["duration_s"])), d["session_id"]
        for rank, ann in enumerate(_submitted(t)):
            res = _results(ann)
            spans = []
            for r in res.get("timeline", []):
                v = r.get("value", {})
                for lab in v.get("timelinelabels") or []:
                    for rg in v.get("ranges") or []:  # 1-based inclusive frame numbers
                        spans.append((lab, (float(rg["start"]) - 1) / fps, float(rg["end"]) / fps))
            flags = "|".join(_choices(res, "flags"))
            for w in windows:
                att, eng = window_label_from_spans(spans, w.start_s, w.end_s)
                rows.append({
                    "window_id": w.window_id, "session_id": sid, "participant_id": d.get("participant_id"),
                    "window_index": w.index, "start_s": w.start_s, "end_s": w.end_s, "answer_id": w.answer_id,
                    "attention_label": att, "engagement_label": eng, "flags": flags,
                    "annotator": _annotator(ann, users), "annotation_rank": rank, "lead_time_s": ann.get("lead_time"),
                })
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def scripted_labels(session_ids: set[str], raw_dir: Path = RAW) -> pd.DataFrame:
    """Weak labels for windows inside scripted segments (session log + calibration)."""
    log = read_session_log(raw_dir / "session_log.csv")
    rows = []
    for sid in sorted(session_ids):
        if (raw_dir / sid / "meta.json").exists():
            rows += weak_window_labels(load_session(raw_dir / sid), log.get(sid, []))
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def rank_windows(win: pd.DataFrame) -> pd.DataFrame:
    """Re-rank per window: human annotators in submission order first, 'scripted' weak labels last."""
    if win.empty:
        return win
    win = win.copy()
    win["_scripted"] = win.annotator == "scripted"
    win = win.sort_values(["window_id", "_scripted", "annotation_rank"], kind="stable")
    win["annotation_rank"] = win.groupby("window_id").cumcount()
    # one human may label a window in both the timeline and the per-window project: keep the first
    win = win.drop_duplicates(["window_id", "annotator"]).drop(columns="_scripted")
    win["annotation_rank"] = win.groupby("window_id").cumcount()
    return win.reset_index(drop=True)


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
    ap.add_argument("--sessions", nargs="*", default=[], help="JSON exports of the sessions (timeline) projects")
    ap.add_argument("--no-scripted", action="store_true", help="do not add weak labels from the session log")
    ap.add_argument("--url")
    ap.add_argument("--token", help="personal access token (default: log in with $LABEL_STUDIO_USERNAME/PASSWORD)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    win_tasks: list[dict] = []
    ans_tasks: list[dict] = []
    ses_tasks: list[dict] = []
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
            {"windows": win_tasks, "answers": ans_tasks, "sessions": ses_tasks}[key.split("_")[0]].extend(tasks)
            print(f"pulled '{title}': {len(tasks)} tasks")
    for f in args.windows:
        win_tasks += json.loads(Path(f).read_text(encoding="utf-8"))
    for f in args.answers:
        ans_tasks += json.loads(Path(f).read_text(encoding="utf-8"))
    for f in args.sessions:
        ses_tasks += json.loads(Path(f).read_text(encoding="utf-8"))
    if not win_tasks and not ans_tasks and not ses_tasks:
        ap.error("nothing to export: use --pull or pass export files")

    parts = [parse_windows(win_tasks, users), parse_sessions(ses_tasks, users)]
    if not args.no_scripted:
        sids = {t["data"]["session_id"] for t in win_tasks + ans_tasks + ses_tasks}
        parts.append(scripted_labels(sids))
    win = rank_windows(pd.concat([p for p in parts if not p.empty], ignore_index=True)
                       if any(not p.empty for p in parts) else parts[0])
    ans, hes = parse_answers(ans_tasks, users)
    counts = write_all(win, ans, hes, Path(args.out))
    for k, v in counts.items():
        print(f"{args.out}/{k}: {v} rows")
    unlabelled = sum(1 for t in win_tasks + ans_tasks + ses_tasks if not _submitted(t))
    if unlabelled:
        print(f"note: {unlabelled} tasks have no submitted annotation yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
