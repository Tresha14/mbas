"""Turn recorded sessions into Label Studio tasks (and optionally push them).

For every session in data/raw/ that passes check_session and has consent:
  1. a 360p 16 fps proxy of the candidate webcam with the candidate mic, aligned
     to the recording start      -> one "sessions (timeline)" task: mark exceptions only;
  2. one audio + video clip per answered question -> answers tasks;
  3. with --per-window: one 5 s clip per window (2.5 s hop) for the slower per-window projects.

20% of sessions (at least one) go to the "double-labelled" projects, which
both annotators label completely; that overlap is used for Cohen's kappa.
A third of the sessions (plus every double-labelled one) need a manual transcript;
the rest only get quality / relevance / follow-up labels. Assignments are kept in
data/processed/labelstudio/assignment.json so they never change when sessions are added.
Scripted segments from data/raw/session_log.csv are shown in the task header.

    python -m tools.import_to_labelstudio                # build clips + task files
    python -m tools.import_to_labelstudio --push         # ... and create projects + import
Outputs: data/processed/labelstudio/{media/<session>/..., tasks/<project>.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_session import _ffmpeg, check_session  # noqa: E402
from tools.labelstudio_client import PROJECTS, LabelStudio, media_url  # noqa: E402
from tools.session_index import HOP_S, WINDOW_S, SessionIndex, find_sessions, load_session  # noqa: E402
from tools.session_log import read_session_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "labelstudio"
DOUBLE_FRACTION = 0.20
ANSWER_TAIL_S = 0.5
PROXY_HEIGHT = 360
PROXY_FPS = 16           # 2.5 s hop = exactly 40 frames, so every window starts on a keyframe
TRANSCRIBE_FRACTION = 1 / 3  # manual transcripts for ~20 of 60 sessions (ASR test set); others: labels only
ID_FIELD = {"sessions": "session_id", "windows": "window_id", "answers": "answer_id"}


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:6])} ...\n{p.stderr[-1500:]}")


def _align(stream: str, offset_s: float, kind: str) -> str:
    """Filter that moves a track so t=0 is the recording start (pads or trims)."""
    if kind == "v":
        if offset_s >= 0:
            return f"[{stream}]tpad=start_duration={offset_s:.3f}:start_mode=clone,setpts=PTS-STARTPTS"
        return f"[{stream}]trim=start={-offset_s:.3f},setpts=PTS-STARTPTS"
    if offset_s >= 0:
        return f"[{stream}]adelay={offset_s * 1000:.0f}:all=1,apad"
    return f"[{stream}]atrim=start={-offset_s:.3f},asetpts=PTS-STARTPTS,apad"


def make_proxy(s: SessionIndex, dest: Path) -> Path:
    """Webcam + candidate mic on the recording time base, 360p 16 fps, keyframes on the window grid."""
    if dest.exists() and dest.stat().st_mtime >= (s.dir / "candidate_webcam.webm").stat().st_mtime:
        return dest
    ov = s.track_offsets_s.get("candidate_webcam", 0.0)
    oa = s.track_offsets_s.get("candidate_mic", 0.0)
    fc = (f"{_align('0:v', ov, 'v')},fps={PROXY_FPS},scale=-2:{PROXY_HEIGHT}[v];"
          f"{_align('1:a', oa, 'a')}[a]")
    tmp = dest.with_suffix(".tmp.webm")
    _run([_ffmpeg(), "-y", "-loglevel", "error",
          "-i", str(s.dir / "candidate_webcam.webm"), "-i", str(s.dir / "candidate_mic.wav"),
          "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{s.duration_s:.3f}",
          "-c:v", "libvpx", "-b:v", "600k", "-deadline", "realtime", "-cpu-used", "8",
          "-force_key_frames", f"expr:gte(t,n_forced*{HOP_S})",
          "-c:a", "libopus", "-b:a", "32k", "-ar", "48000", str(tmp)])
    tmp.replace(dest)
    return dest


def cut_copy(src: Path, start_s: float, dur_s: float, dest: Path) -> None:
    """Keyframe-aligned cut without re-encoding (window starts are on the keyframe grid)."""
    # seek a hair past the keyframe so rounding never snaps back to the previous one
    _run([_ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{start_s + 0.01:.3f}", "-i", str(src),
          "-t", f"{dur_s:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero", str(dest)])


def cut_video(src: Path, start_s: float, dur_s: float, dest: Path) -> None:
    _run([_ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(src),
          "-t", f"{dur_s:.3f}", "-an", "-c:v", "libvpx", "-b:v", "300k", "-deadline", "realtime",
          "-cpu-used", "8", str(dest)])


def cut_audio(wav_path: Path, offset_s: float, start_s: float, end_s: float, dest: Path) -> None:
    """Exact sample cut of the candidate mic; start/end on the recording time base."""
    info = sf.info(str(wav_path))
    sr = info.samplerate
    a = int(round((start_s - offset_s) * sr))
    b = int(round((end_s - offset_s) * sr))
    data, _ = sf.read(str(wav_path), start=max(a, 0), stop=max(min(b, info.frames), 0), dtype="int16")
    if a < 0:  # track started after the clip start: pad with silence so clip t=0 is still start_s
        data = np.concatenate([np.zeros(-a, dtype=np.int16), data])
    sf.write(str(dest), data, sr, subtype="PCM_16")


def _context(s: SessionIndex, w) -> str:
    if not w.answer_id:
        return "no question"
    a = next(a for a in s.answers if a.answer_id == w.answer_id)
    return f"{a.question_id} ({a.category})"


def _scripted_text(segments: list[dict]) -> str:
    return ", ".join(f"{g['segment']} {g['start_s']:.0f}-{g['end_s']:.0f} s" for g in segments) or "not logged"


def build_session(s: SessionIndex, out: Path = OUT, force: bool = False, per_window: bool = False,
                  transcribe: bool = True, scripted: list[dict] | None = None) -> dict[str, list[dict]]:
    media = out / "media" / s.session_id
    (media / "answers").mkdir(parents=True, exist_ok=True)
    proxy = make_proxy(s, media / "proxy.webm")
    pid = s.candidate.get("participant_id")

    session = [{"data": {
        "session_id": s.session_id, "participant_id": pid, "duration_s": s.duration_s, "fps": PROXY_FPS,
        "scripted": _scripted_text(scripted or []), "video": media_url(f"{s.session_id}/proxy.webm")}}]

    windows = []
    if per_window:
        (media / "windows").mkdir(exist_ok=True)
        for w in s.windows:
            rel = f"{s.session_id}/windows/W{w.index:04d}.webm"
            clip = out / "media" / rel
            if force or not clip.exists():
                cut_copy(proxy, w.start_s, WINDOW_S, clip)
            windows.append({"data": {
                "window_id": w.window_id, "session_id": s.session_id, "participant_id": pid,
                "window_index": w.index, "t0_s": w.start_s, "t1_s": w.end_s, "answer_id": w.answer_id or "",
                "context": _context(s, w), "video": media_url(rel)}})

    answers = []
    oa = s.track_offsets_s.get("candidate_mic", 0.0)
    for a in s.answers:
        t0, t1 = a.ask_s, min(a.end_s + ANSWER_TAIL_S, s.duration_s)
        rel_a = f"{s.session_id}/answers/A{a.order:02d}.wav"
        rel_v = f"{s.session_id}/answers/A{a.order:02d}.webm"
        if force or not (out / "media" / rel_a).exists():
            cut_audio(s.dir / "candidate_mic.wav", oa, t0, t1, out / "media" / rel_a)
            cut_video(proxy, t0, t1 - t0, out / "media" / rel_v)
        answers.append({"data": {
            "answer_id": a.answer_id, "session_id": s.session_id, "participant_id": pid,
            "question_id": a.question_id, "question_text": a.question_text, "category": a.category,
            "ask_s": a.ask_s, "end_s": a.end_s, "clip_t0_s": t0, "clip_t1_s": t1,
            "transcribe": "REQUIRED" if transcribe else "skip",
            "audio": media_url(rel_a), "video": media_url(rel_v)}})
    return {"sessions": session, "windows": windows, "answers": answers}


def assign_fraction(session_ids: list[str], previous: dict[str, bool], fraction: float,
                    seed: int = 42, salt: str = "") -> dict[str, bool]:
    """Keep earlier assignments; flag new sessions until >= fraction of all sessions are flagged (at least one)."""
    out = {sid: previous[sid] for sid in session_ids if sid in previous}
    new = sorted((sid for sid in session_ids if sid not in out),
                 key=lambda sid: hashlib.sha1(f"{seed}:{salt}{sid}".encode()).hexdigest())
    need = max(1, math.ceil(fraction * len(session_ids))) if session_ids else 0
    have = sum(out.values())
    for sid in new:
        out[sid] = have < need
        have += out[sid]
    return out


def assign_double(session_ids: list[str], previous: dict[str, bool], fraction: float = DOUBLE_FRACTION,
                  seed: int = 42) -> dict[str, bool]:
    return assign_fraction(session_ids, previous, fraction, seed)


def eligible(d: Path, allow_failed: bool) -> tuple[bool, str]:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    parts = meta.get("participants", {})
    if not all(parts.get(r, {}).get("consent", {}).get("agreed") for r in ("candidate", "interviewer")):
        return False, "no consent"
    rep = check_session(d, lenient=True)
    if not rep["passed"] and not allow_failed:
        return False, "check_session failed: " + "; ".join(rep["errors"][:3])
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sessions", nargs="*", help="only these session IDs")
    ap.add_argument("--allow-failed", action="store_true", help="also include sessions that fail check_session")
    ap.add_argument("--force", action="store_true", help="re-cut clips that already exist")
    ap.add_argument("--per-window", action="store_true",
                    help="also cut one clip per 5 s window for the slower per-window projects")
    ap.add_argument("--push", action="store_true", help="create the Label Studio projects and import new tasks")
    ap.add_argument("--url", help="Label Studio URL (default $LABEL_STUDIO_URL or http://localhost:8090)")
    ap.add_argument("--token", help="personal access token (default: log in with $LABEL_STUDIO_USERNAME/PASSWORD)")
    args = ap.parse_args(argv)
    out = Path(args.out)

    dirs = [d for d in find_sessions(args.raw) if not args.sessions or d.name in args.sessions]
    ok_dirs = []
    for d in dirs:
        good, why = eligible(d, args.allow_failed)
        print(f"{d.name}: {why}")
        if good:
            ok_dirs.append(d)
    if not ok_dirs:
        print("no eligible sessions")
        return 1

    assign_path = out / "assignment.json"
    prev = json.loads(assign_path.read_text(encoding="utf-8")) if assign_path.exists() else {}
    all_ids = sorted(set(prev.get("double", {})) | {d.name for d in ok_dirs})
    double = assign_double(all_ids, prev.get("double", {}))
    trans = assign_fraction(all_ids, prev.get("transcribe", {}), TRANSCRIBE_FRACTION, salt="transcribe:")
    for sid, dbl in double.items():  # double-labelled sessions are always transcribed (transcript agreement)
        trans[sid] = trans[sid] or dbl
    out.mkdir(parents=True, exist_ok=True)
    assign_path.write_text(json.dumps({"double_fraction": DOUBLE_FRACTION, "double": double,
                                       "transcribe_fraction": TRANSCRIBE_FRACTION, "transcribe": trans},
                                      indent=2), encoding="utf-8")
    log = read_session_log(Path(args.raw) / "session_log.csv")

    kinds = ["sessions", "answers"] + (["windows"] if args.per_window else [])
    tasks: dict[str, list[dict]] = {k + sfx: [] for k in kinds for sfx in ("", "_double")}
    for d in ok_dirs:
        s = load_session(d)
        t = build_session(s, out, force=args.force, per_window=args.per_window,
                          transcribe=trans[s.session_id], scripted=log.get(s.session_id))
        suffix = "_double" if double[s.session_id] else ""
        for k in kinds:
            tasks[k + suffix] += t[k]
        print(f"{s.session_id}: {len(t['answers'])} answers{' (double-labelled)' if suffix else ''}"
              f"{', transcribe' if trans[s.session_id] else ''}")

    # task files hold every eligible session, so they can be imported by hand in the UI as well
    tdir = out / "tasks"
    tdir.mkdir(exist_ok=True)
    for key, items in tasks.items():
        existing = []
        p = tdir / f"{key}.json"
        if p.exists():
            existing = json.loads(p.read_text(encoding="utf-8"))
        id_field = ID_FIELD[key.split("_")[0]]
        seen = {x["data"][id_field] for x in items}
        merged = [x for x in existing if x["data"][id_field] not in seen] + items
        p.write_text(json.dumps(merged, indent=1), encoding="utf-8")
        print(f"tasks/{key}.json: {len(merged)} tasks")

    if args.push:
        ls = LabelStudio(args.url, args.token)
        for key, items in tasks.items():
            proj = ls.ensure_project(key)
            id_field = ID_FIELD[key.split("_")[0]]
            have = {t["data"].get(id_field) for t in ls.export_tasks(proj["id"])}
            new = [x for x in items if x["data"][id_field] not in have]
            n = ls.import_tasks(proj["id"], new) if new else 0
            print(f"Label Studio '{proj['title']}' (id {proj['id']}): imported {n} new tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
