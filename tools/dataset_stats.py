"""Dataset statistics -> docs/dataset_card.md + docs/metrics/dataset_stats.json.

Covers our recorded sessions, the synthetic dataset and the converted public
datasets: participants, hours, label counts, kappa, demographics, splits and
the Stage 2 "done when" checklist.

    python -m tools.dataset_stats
Run after export_labels / synthetic_to_labels, agreement and make_splits.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.session_index import find_sessions, load_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SYN = RAW / "mbas_synthetic_dataset"
LABELS = ROOT / "data" / "processed" / "labels"
PUBLIC = ROOT / "data" / "processed" / "public"
SPLITS = ROOT / "data" / "splits"
METRICS = ROOT / "docs" / "metrics"
CARD = ROOT / "docs" / "dataset_card.md"

TARGET_SESSIONS, TARGET_PEOPLE, TARGET_DOUBLE = 60, 30, 0.20
PUBLIC_DATASETS = ["wider_face", "mpiigaze", "columbia_gaze", "daisee", "librispeech", "common_voice_in",
                   "sep28k", "mit_interview", "ami"]


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _counts(s: pd.Series | None) -> dict:
    if s is None:
        return {}
    return {str(k): int(v) for k, v in s.value_counts(dropna=False).sort_index().items()}


def label_stats(labels_dir: Path) -> dict:
    w = _read(labels_dir / "windows.parquet")
    a = _read(labels_dir / "answers.parquet")
    h = _read(labels_dir / "hesitation_events.parquet")
    out: dict = {"labelled_windows": 0 if w is None else len(w), "labelled_answers": 0 if a is None else len(a)}
    if w is not None:
        out["attention"] = _counts(w.attention_label)
        out["engagement"] = _counts(w.engagement_label)
        out["labelled_sessions"] = int(w.session_id.nunique())
    if a is not None and len(a):
        out["quality_1to5"] = _counts(a.quality_1to5)
        out["follow_up_needed"] = _counts(a.follow_up_needed)
        out["follow_up_reason"] = _counts(a.follow_up_reason)
        out["relevance_mean"] = round(float(a.relevance.mean()), 3)
        out["transcribed_answers"] = int((a.transcript.fillna("").str.len() > 0).sum())
        out["transcript_words"] = int(a.transcript.fillna("").str.split().str.len().sum())
        if "speaking_pattern" in a:
            out["speaking_pattern"] = _counts(a.speaking_pattern)
    if h is not None:
        out["hesitation_events"] = _counts(h.event_type)
    return out


def _agreement(name: str) -> dict:
    p = METRICS / f"agreement_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _splits(name: str) -> dict:
    p = SPLITS / name / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def recorded_stats() -> dict:
    rows = []
    for d in find_sessions(RAW):
        s = load_session(d)
        c = s.candidate
        rows.append({"session_id": s.session_id, "participant_id": c.get("participant_id"),
                     "duration_s": s.duration_s, "answers": len(s.answers), "windows": len(s.windows),
                     "gender": c.get("gender"), "age_band": c.get("age_band"), "accent": c.get("accent"),
                     "glasses": c.get("glasses")})
    df = pd.DataFrame(rows, columns=["session_id", "participant_id", "duration_s", "answers", "windows",
                                     "gender", "age_band", "accent", "glasses"])
    people = df.drop_duplicates("participant_id")
    return {
        "sessions": len(df), "participants": int(df.participant_id.nunique()),
        "hours": round(float(df.duration_s.sum()) / 3600, 3),
        "mean_session_min": round(float(df.duration_s.mean()) / 60, 2) if len(df) else 0.0,
        "answered_questions": int(df.answers.sum()), "windows": int(df.windows.sum()),
        "demographics": {k: _counts(people[k]) for k in ("gender", "age_band", "accent", "glasses")},
        "labels": label_stats(LABELS / "recorded"), "agreement": _agreement("recorded"), "splits": _splits("recorded"),
    }


def synthetic_stats() -> dict:
    if not SYN.exists():
        return {}
    iv = SYN / "interview"
    p = pd.read_csv(iv / "participants.csv")
    s = pd.read_csv(iv / "sessions.csv")
    m = pd.read_csv(SYN / "meeting" / "meetings.csv")
    return {
        "sessions": len(s), "participants": len(p), "meetings": len(m),
        "hours": round(float(s.duration_s.sum()) / 3600, 2),
        "meeting_hours": round(float(m.duration_min.sum()) / 60, 2),
        "demographics": {k: _counts(p[k]) for k in ("gender", "age_band", "accent", "glasses")},
        "labels": label_stats(LABELS / "synthetic"), "agreement": _agreement("synthetic"), "splits": _splits("synthetic"),
    }


def public_stats() -> dict:
    out = {}
    for name in PUBLIC_DATASETS:
        info = PUBLIC / name / "info.json"
        out[name] = json.loads(info.read_text(encoding="utf-8")) if info.exists() else {"status": "not converted"}
    return out


def _kappa_rows(agr: dict) -> list[tuple[str, float | None, int]]:
    rows = []
    for sec in ("windows", "answers", "hesitation"):
        for r in agr.get(sec, []):
            n = r.get("n_items", r.get("n_answers", 0))
            for k, v in r.items():
                if k.endswith("kappa") or "kappa_" in k:
                    rows.append((f"{sec}: {k}", v, n))
    return rows


def _table(d: dict, head: str = "Value") -> str:
    if not d:
        return "_none yet_\n"
    lines = [f"| {head} | Count |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in d.items()]
    return "\n".join(lines) + "\n"


def _source_section(title: str, st: dict, synthetic: bool) -> str:
    if not st:
        return f"## {title}\n\n_not available_\n"
    lab, agr, spl = st.get("labels", {}), st.get("agreement", {}), st.get("splits", {})
    out = [f"## {title}\n"]
    if synthetic:
        out.append("> Simulated tabular data (features + labels, no raw media). Always reported as synthetic.\n")
    out.append("| | |\n| --- | ---: |")
    for k, label in [("participants", "Participants"), ("sessions", "Interview sessions"), ("meetings", "Meetings"),
                     ("hours", "Interview hours"), ("meeting_hours", "Meeting hours"),
                     ("mean_session_min", "Mean session length (min)"), ("answered_questions", "Answered questions"),
                     ("windows", "5 s windows (2.5 s hop)")]:
        if k in st:
            out.append(f"| {label} | {st[k]} |")
    for k, label in [("labelled_sessions", "Labelled sessions"), ("labelled_windows", "Labelled windows"),
                     ("labelled_answers", "Labelled answers"), ("transcribed_answers", "Manually transcribed answers"),
                     ("transcript_words", "Transcript words")]:
        if k in lab:
            out.append(f"| {label} | {lab[k]} |")
    if agr:
        out.append(f"| Double-labelled sessions | {agr.get('double_labelled_sessions', 0)} "
                   f"({agr.get('double_labelled_fraction', 0):.0%}) |")
    out.append("")

    out.append("### Splits (by participant)\n")
    if spl:
        out.append("| Split | Participants | Sessions |\n| --- | ---: | ---: |")
        for s in ("train", "val", "test"):
            out.append(f"| {s} | {spl['participants'].get(s, 0)} | {spl['sessions'].get(s, 0)} |")
        out.append(f"\nLeakage check: {'**FAILED** ' + str(spl['leakage_problems']) if spl['leakage_problems'] else 'no participant in two splits.'}")
        if "meetings" in spl:
            out.append(f"Meetings by split: {spl['meetings']} (`mixed` = participants from more than one split; "
                       "the dataset's own meeting split column is not person-wise and is not used).")
    else:
        out.append("_run `python -m tools.make_splits`_")
    out.append("")

    out.append("### Inter-annotator agreement (Cohen's kappa)\n")
    rows = _kappa_rows(agr)
    if rows:
        out.append("| Label | Kappa | Items |\n| --- | ---: | ---: |")
        out += [f"| {k} | {'n/a' if v is None else f'{v:.3f}'} | {n} |" for k, v, n in rows]
    else:
        out.append("_no double-labelled items yet_")
    out.append("")

    out.append("### Label counts\n")
    for k, label in [("attention", "Attention (per window)"), ("engagement", "Engagement (per window)"),
                     ("hesitation_events", "Hesitation events"), ("quality_1to5", "Answer quality"),
                     ("follow_up_reason", "Follow-up reason"), ("speaking_pattern", "Speaking pattern")]:
        if lab.get(k):
            out.append(f"**{label}**\n\n{_table(lab[k], 'Label')}")
    if "relevance_mean" in lab:
        out.append(f"Mean relevance: {lab['relevance_mean']}\n")

    out.append("### Demographics (participants)\n")
    for k, v in st.get("demographics", {}).items():
        out.append(f"**{k}**\n\n{_table(v, k)}")
    return "\n".join(out) + "\n"


def _check(ok: bool) -> str:
    return "x" if ok else " "


def render(stats: dict) -> str:
    rec, pub = stats["recorded"], stats["public"]
    agr = rec.get("agreement", {})
    labelled = rec["labels"].get("labelled_sessions", 0)
    double_ok = agr.get("double_labelled_fraction", 0) >= TARGET_DOUBLE and bool(_kappa_rows(agr))
    pub_ok = all(v.get("status") == "converted" for v in pub.values())
    spl = rec.get("splits", {})
    lines = [
        "# MBAS dataset card",
        "",
        f"_Generated by `python -m tools.dataset_stats` on {stats['generated']}. Do not edit by hand._",
        "",
        "Multimodal Behavioural Analysis System: mock interviews recorded with our own recorder (`recorder/`), "
        "labelled in Label Studio (`docs/annotation_guide.md`), plus a simulated tabular dataset and public raw "
        "datasets used for pre-training only. Protocol: `docs/data_protocol.md`.",
        "",
        "## Stage 2 checklist",
        "",
        f"- [{_check(rec['sessions'] >= TARGET_SESSIONS and rec['participants'] >= TARGET_PEOPLE)}] "
        f"At least {TARGET_SESSIONS} sessions by {TARGET_PEOPLE}+ participants "
        f"(now {rec['sessions']} sessions, {rec['participants']} participants)",
        f"- [{_check(rec['sessions'] > 0 and labelled == rec['sessions'] and double_ok)}] All sessions labelled; "
        f"20% double-labelled, kappa reported (now {labelled}/{rec['sessions']} labelled, "
        f"{agr.get('double_labelled_fraction', 0):.0%} double-labelled)",
        f"- [{_check(pub_ok)}] Public datasets downloaded and converted "
        f"({sum(v.get('status') == 'converted' for v in pub.values())}/{len(pub)})",
        f"- [{_check(bool(spl) and not spl.get('leakage_problems'))}] Person-wise splits and dataset card in docs/",
        "",
        _source_section("Our recorded sessions", rec, synthetic=False),
        _source_section("Synthetic dataset (mbas_synthetic_dataset v1.0)", stats["synthetic"], synthetic=True),
        "## Public raw datasets (pre-training only; no model weights)",
        "",
        "| Dataset | Used for | Status | Items | Subjects | Hours | Licence |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for name, v in pub.items():
        lines.append(f"| {v.get('title', name)} | {v.get('used_for', '')} | {v.get('status', 'not converted')} | "
                     f"{v.get('n_items', '')} | {v.get('n_subjects', '')} | {v.get('hours', '')} | "
                     f"{v.get('license', '')} |")
    lines += [
        "",
        "Scripts: `tools/datasets/` (see its README). Manual-access datasets need a signed request first.",
        "",
        "## Collection, consent and storage",
        "",
        "- Every recorded participant signed the consent screen (`meta.json -> consent`); sessions without consent "
        "are never imported for labelling.",
        "- Personal data under India's DPDP Act 2023: raw sessions stay on project machines, are git-ignored and are "
        "deleted on request (see `docs/data_protocol.md`, section 6).",
        "- Labels describe observable behaviour in a mock interview. They are not a judgement of the person, and "
        "model outputs trained on them are decision support only.",
        "",
        "## Known limitations",
        "",
        "- Webcam attention and engagement labels are subjective and culture-dependent (eye-contact norms); kappa "
        "is reported so readers can judge label reliability.",
        "- Participants are mostly engineering students aged 18-25: results may not transfer to other groups.",
        "- The synthetic dataset's relationships are designed; high scores on it mostly mean the simulator was learned.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    stats = {"generated": dt.date.today().isoformat(), "recorded": recorded_stats(),
             "synthetic": synthetic_stats(), "public": public_stats()}
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "dataset_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    CARD.write_text(render(stats), encoding="utf-8")
    r = stats["recorded"]
    print(f"recorded: {r['sessions']} sessions, {r['participants']} participants, {r['hours']} h")
    print(f"wrote {CARD.relative_to(ROOT)} and docs/metrics/dataset_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
