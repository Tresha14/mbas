"""Verify a recorded session folder.

Checks that all files are present, meta/consent are complete, audio is 16 kHz
mono, every track covers the recording window, and that each track's media
clock agrees with the shared server clock (sync drift < 100 ms).

Usage:
    python -m tools.check_session data/raw/<session_id>
    python -m tools.check_session data/raw/<session_id> --lenient --json report.json
Exit code 0 = pass, 1 = fail.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder.backend import wav  # noqa: E402

MAX_DRIFT_MS = 100.0
MIN_COVERAGE = 0.95          # each required track covers >= 95% of the recording window
MAX_TRACK_DURATION_DIFF_S = 1.0
EXPECTED_FPS = 15.0
EXPECTED_SIZE = (1280, 720)
CALIBRATION_S = 60.0

REQUIRED_FILES = ["meta.json", "timeline.json", "question_events.json"]
REQUIRED_TRACKS = ["candidate_webcam", "candidate_mic", "interviewer_mic"]
OPTIONAL_IN_LENIENT = ["candidate_screen", "calibration"]
META_FIELDS = ["participant_id", "age_band", "gender", "glasses", "accent"]


def _ffmpeg() -> str:
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: Path) -> dict:
    """Frame timestamps, duration, resolution and codec of a webm (stream copy, no decoding).

    Uses ffmpeg's framemd5 muxer, which lists every packet's pts and duration,
    so duration = last pts + last duration - first pts (exact, no frame rounding).
    """
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "framemd5", "-"],
        capture_output=True, text=True, errors="replace",
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else proc.returncode
        raise RuntimeError(f"ffmpeg could not read {path.name}: {tail}")
    out: dict = {}
    m = re.search(r"Video: (\w+)[^\n]*?, (\d{2,5})x(\d{2,5})", proc.stderr)
    if m:
        out["codec"], out["width"], out["height"] = m.group(1), int(m.group(2)), int(m.group(3))
    tb = 1 / 1000
    pts: list[int] = []
    last_dur = 0
    for line in proc.stdout.splitlines():
        if line.startswith("#tb 0:"):
            num, den = line.split(":", 1)[1].strip().split("/")
            tb = int(num) / int(den)
        elif line and not line.startswith("#"):
            f = [x.strip() for x in line.split(",")]
            pts.append(int(f[2]))
            last_dur = int(f[3])
    out["frames"] = len(pts)
    if len(pts) < 2:
        out.update(duration_s=0.0, fps=0.0, max_gap_s=0.0)
        return out
    if last_dur <= 0:
        last_dur = round((pts[-1] - pts[0]) / (len(pts) - 1))
    out["duration_s"] = (pts[-1] + last_dur - pts[0]) * tb
    out["fps"] = len(pts) / out["duration_s"]
    out["max_gap_s"] = max(b - a for a, b in zip(pts, pts[1:])) * tb
    return out


class Report:
    def __init__(self, session_dir: Path):
        self.session = session_dir.name
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.tracks: dict[str, dict] = {}
        self.info: dict = {}

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "session": self.session,
            "passed": not self.errors,
            "errors": self.errors,
            "warnings": self.warnings,
            "tracks": self.tracks,
            "info": self.info,
        }


def _load(path: Path, rep: Report) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        rep.err(f"missing file {path.name}")
    except json.JSONDecodeError as e:
        rep.err(f"{path.name} is not valid JSON: {e}")
    return None


def check_session(session_dir: str | Path, lenient: bool = False) -> dict:
    d = Path(session_dir)
    rep = Report(d)
    if not d.is_dir():
        rep.err(f"{d} is not a directory")
        return rep.to_dict()

    meta = _load(d / "meta.json", rep)
    timeline = _load(d / "timeline.json", rep)
    events_doc = _load(d / "question_events.json", rep)

    # ---- meta + consent ----
    if meta:
        parts = meta.get("participants", {})
        for role in ("candidate", "interviewer"):
            p = parts.get(role)
            if not p:
                rep.err(f"meta.json: no {role} participant")
                continue
            if not p.get("consent", {}).get("agreed"):
                rep.err(f"meta.json: {role} consent missing")
            if role == "candidate":
                missing = [f for f in META_FIELDS if p.get(f) in (None, "")]
                if missing:
                    rep.err(f"meta.json: candidate missing {missing}")
        for role, c in (meta.get("client_clock") or {}).items():
            if c.get("rtt_ms", 0) > 2 * MAX_DRIFT_MS:
                rep.warn(f"{role} clock sync RTT {c['rtt_ms']:.0f} ms (offset uncertainty ±{c['rtt_ms'] / 2:.0f} ms)")

    if not timeline:
        return rep.to_dict()

    rec = timeline.get("recording", {})
    rs, re_ = rec.get("start_ms"), rec.get("end_ms")
    if rs is None or re_ is None:
        rep.err("timeline.json: recording start/end missing")
        return rep.to_dict()
    window_s = (re_ - rs) / 1000.0
    rep.info["recording_s"] = round(window_s, 3)
    if window_s <= 0:
        rep.err("recording end is before start")
        return rep.to_dict()

    # ---- tracks ----
    tracks = timeline.get("tracks", {})
    wanted = REQUIRED_TRACKS + ([] if lenient else ["candidate_screen"])
    for name in wanted + [t for t in tracks if t not in wanted]:
        t = tracks.get(name)
        if t is None:
            (rep.warn if name not in wanted else rep.err)(f"track {name} not recorded")
            continue
        f = d / t["file"]
        res: dict = {"file": t["file"]}
        rep.tracks[name] = res
        if not f.exists() or f.stat().st_size == 0:
            rep.err(f"{t['file']} missing or empty")
            continue
        if t.get("end_ms") is None:
            rep.err(f"track {name} was never stopped")
            continue
        if t.get("missing_chunks"):
            rep.err(f"track {name}: {t['missing_chunks']} chunk(s) never reached the server")
        wall_s = (t["end_ms"] - t["start_ms"]) / 1000.0
        res["wall_s"] = round(wall_s, 3)
        res["start_offset_s"] = round((t["start_ms"] - rs) / 1000.0, 3)
        try:
            if t["kind"] == "audio":
                wi = wav.info(f)
                media_s = wi["duration_s"]
                res.update(sample_rate=wi["sample_rate"], channels=wi["channels"], bits=wi["bits"])
                if wi["sample_rate"] != 16000 or wi["channels"] != 1 or wi["bits"] != 16:
                    rep.err(f"{t['file']}: expected 16 kHz mono 16-bit, got {wi['sample_rate']} Hz {wi['channels']} ch {wi['bits']} bit")
            else:
                vi = probe_video(f)
                media_s = vi["duration_s"]
                res.update({k: vi.get(k) for k in ("codec", "width", "height", "frames")})
                res["fps"] = round(vi["fps"], 2)
                res["max_gap_s"] = round(vi["max_gap_s"], 3)
                if vi["max_gap_s"] > 0.5:
                    rep.warn(f"{name}: {vi['max_gap_s']:.2f} s without frames (camera stall or tab in background)")
                if name == "candidate_webcam":
                    if (vi.get("width"), vi.get("height")) != EXPECTED_SIZE:
                        rep.warn(f"webcam is {vi.get('width')}x{vi.get('height')}, expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}")
                    if abs(vi["fps"] - EXPECTED_FPS) > 1.5:
                        rep.warn(f"webcam averages {vi['fps']:.1f} fps, expected {EXPECTED_FPS:.0f}")
        except Exception as e:  # noqa: BLE001
            rep.err(f"{t['file']}: cannot read media ({e})")
            continue
        res["media_s"] = round(media_s, 3)
        drift_ms = abs(media_s - wall_s) * 1000.0
        res["drift_ms"] = round(drift_ms, 1)
        if drift_ms >= MAX_DRIFT_MS:
            rep.err(f"{name}: sync drift {drift_ms:.0f} ms (media {media_s:.2f} s vs clock {wall_s:.2f} s)")
        if name in REQUIRED_TRACKS:
            overlap = max(0.0, min(t["end_ms"], re_) - max(t["start_ms"], rs)) / 1000.0
            cov = overlap / window_s
            res["coverage"] = round(cov, 3)
            if cov < MIN_COVERAGE:
                rep.err(f"{name} covers only {cov:.0%} of the recording window")

    durs = {k: v["media_s"] for k, v in rep.tracks.items() if k in REQUIRED_TRACKS and "media_s" in v}
    if len(durs) >= 2:
        spread = max(durs.values()) - min(durs.values())
        rep.info["required_duration_spread_s"] = round(spread, 3)
        if spread > MAX_TRACK_DURATION_DIFF_S:
            rep.err(f"required track durations differ by {spread:.2f} s: {durs}")

    # ---- question events ----
    if events_doc is not None:
        evs = sorted(events_doc.get("events", []), key=lambda e: e["t_ms"])
        rep.info["question_events"] = len(evs)
        if not evs:
            rep.warn("no question events recorded")
        open_q: str | None = None
        n_pairs = 0
        for e in evs:
            if not (rs - 1000 <= e["t_ms"] <= re_ + 1000):
                rep.warn(f"event {e['type']} {e['question_id']} at {(e['t_ms'] - rs) / 1000:.1f} s is outside the recording")
            if e["type"] == "ask":
                if open_q is not None:
                    rep.warn(f"question {open_q} asked without 'answer_end' before {e['question_id']}")
                open_q = e["question_id"]
            elif e["type"] == "answer_end":
                if open_q != e["question_id"]:
                    rep.warn(f"'answer_end' for {e['question_id']} without a matching 'ask'")
                else:
                    n_pairs += 1
                open_q = None
        if open_q is not None:
            rep.warn(f"question {open_q} never got 'answer_end'")
        rep.info["answered_questions"] = n_pairs

    # ---- calibration ----
    cal_path = d / "calibration.json"
    if cal_path.exists():
        cal = _load(cal_path, rep) or {}
        samples = cal.get("samples", [])
        dur = (cal.get("end_ms", 0) - cal.get("start_ms", 0)) / 1000.0
        rep.info["calibration"] = {"samples": len(samples), "duration_s": round(dur, 2)}
        if abs(dur - CALIBRATION_S) > 2.0:
            rep.warn(f"calibration lasted {dur:.1f} s, expected {CALIBRATION_S:.0f}")
        if samples:
            rate = len(samples) / max(dur, 1e-6)
            rep.info["calibration"]["rate_hz"] = round(rate, 1)
            if rate < EXPECTED_FPS:
                rep.err(f"calibration logged {rate:.1f} samples/s, below webcam rate {EXPECTED_FPS:.0f}")
            bad = [s for s in samples if not (0 <= s.get("x_norm", -1) <= 1 and 0 <= s.get("y_norm", -1) <= 1)]
            if bad:
                rep.err(f"calibration: {len(bad)} samples with dot outside the screen")
        wc = tracks.get("candidate_webcam")
        if wc and wc.get("end_ms") and samples:
            if not (wc["start_ms"] <= cal["start_ms"] and cal["end_ms"] <= wc["end_ms"]):
                rep.err("calibration is not inside the webcam recording, frames cannot be matched")
    elif lenient:
        rep.warn("calibration.json missing")
    else:
        rep.err("calibration.json missing")

    return rep.to_dict()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("--lenient", action="store_true", help="screen share and calibration only warn when missing")
    ap.add_argument("--json", help="also write the report to this file")
    args = ap.parse_args(argv)
    rep = check_session(args.session_dir, lenient=args.lenient)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"Session {rep['session']}: {'PASS' if rep['passed'] else 'FAIL'}")
    for k, v in rep["info"].items():
        print(f"  {k}: {v}")
    for name, t in rep["tracks"].items():
        fields = ", ".join(f"{k}={v}" for k, v in t.items() if k != "file")
        print(f"  [{name}] {fields}")
    for w in rep["warnings"]:
        print(f"  WARN  {w}")
    for e in rep["errors"]:
        print(f"  ERROR {e}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
