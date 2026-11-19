"""Analyse one finished interview: what was captured, what the system made of it, and where it lagged or failed.

    python -m tools.analyze_interview                      # the newest interview
    python -m tools.analyze_interview 66                   # a specific one
    python -m tools.analyze_interview 66 --retranscribe    # also re-listen to the audio offline and list what the live transcript missed
    python -m tools.analyze_interview 66 --export out_dir  # save both audio tracks as WAV files

It reads the stored interview (database), the audio / frame / analytics streams kept in Redis (about the last 20 minutes), the
worker log (runs/workers.log), the LiveKit and Caddy container logs, and the CPU recording made by `python -m tools.monitor_run`
if one overlaps the interview. Lag is measured as: time a result was published minus the time its audio / frame was captured.

Everything printed is ASCII. Run it after the interview, with the computer idle (--retranscribe uses a lot of CPU for about a minute).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from backend.config import settings  # noqa: E402
from backend.db import (AnswerScore, Database, Flag, FollowUp, Interview, Question, SpeechStat, TranscriptSegment,  # noqa: E402
                        WindowScore)

SR = 16000
IDLE_MS = {"vision_frame": 35, "vad": 8, "asr_chunk": 20, "disfluency": 2500, "e2e_vision": 150, "e2e_transcript": 1500}   # normal on an idle computer


class Report:
    def __init__(self):
        self.problems: list[tuple[int, str]] = []         # (severity 3 = broken, 2 = laggy / degraded, 1 = minor, text)

    def head(self, title: str):
        print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))

    def line(self, text: str = ""):
        print(text)

    def problem(self, severity: int, text: str):
        self.problems.append((severity, text))


def pct(a, q):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    return float(np.percentile(a, q)) if len(a) else float("nan")


def fmt(x, nd=1, unit=""):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}{unit}"


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


# ----------------------------------------------------------------------------------------------- loading
def load_stream(r, key, binary=False):
    out = []
    for mid, f in r.xrange(key):
        ms = int(mid.decode().split("-")[0])
        out.append((ms / 1000.0, json.loads(f[b"json"]), f.get(b"bin") if binary else None))
    return out


class Audio:
    """One participant's audio as kept in Redis: chunk times, capture clock, loudness."""

    def __init__(self, role, items):
        self.role = role
        self.n = len(items)
        self.t = np.array([m["t_s"] for _, m, _ in items], float)
        self.cap = np.array([m.get("captured_at", np.nan) for _, m, _ in items], float)
        self.pub = np.array([p for p, _, _ in items], float)
        self.pcm = [np.frombuffer(b, np.int16) for _, _, b in items]
        self.db = np.array([10 * np.log10(float(((x.astype(np.float32) / 32768) ** 2).mean()) + 1e-10) for x in self.pcm])

    def wav(self):
        return np.concatenate(self.pcm) if self.n else np.zeros(0, np.int16)


# ----------------------------------------------------------------------------------------------- the analysis
def analyse(iid: int | None, retranscribe: bool, export: str | None, monitor: str | None):
    import redis

    cfg = settings()
    db = Database(cfg.database_url)
    r = redis.Redis.from_url(cfg.redis_url)
    rep = Report()
    with db.session() as s:
        if iid is None:
            iid = s.scalar(select(Interview.id).order_by(Interview.id.desc()).limit(1))
        iv = s.get(Interview, iid)
        if iv is None:
            print(f"no interview {iid}")
            return 2
        segs = list(s.scalars(select(TranscriptSegment).where(TranscriptSegment.interview_id == iid).order_by(TranscriptSegment.start_s)))
        wins = list(s.scalars(select(WindowScore).where(WindowScore.interview_id == iid).order_by(WindowScore.end_s)))
        stats = list(s.scalars(select(SpeechStat).where(SpeechStat.interview_id == iid).order_by(SpeechStat.t_s)))
        qs = list(s.scalars(select(Question).where(Question.interview_id == iid).order_by(Question.order)))
        ans = {a.question_ref: a for a in s.scalars(select(AnswerScore).where(AnswerScore.interview_id == iid))}
        fus = list(s.scalars(select(FollowUp).where(FollowUp.interview_id == iid)))
        flags = list(s.scalars(select(Flag).where(Flag.interview_id == iid).order_by(Flag.t_s)))
    started = iv.started_at.timestamp() if iv.started_at else None
    ended = iv.ended_at.timestamp() if iv.ended_at else None
    dur = (ended - started) if started and ended else (max([w.end_s for w in wins] or [0]))

    rep.head(f"INTERVIEW #{iid}: {iv.title}")
    rep.line(f"platform {iv.platform}, mode {iv.mode}, status {iv.status}, candidate '{iv.candidate_name}', length {mmss(dur)} ({dur:.0f} s)")
    rep.line(f"models: speech = {settings().asr_backend} ({settings().whisper_model}"
             f"{', candidate ' + settings().whisper_model_candidate if settings().whisper_model_candidate else ''}), face = {settings().face_backend}")

    # ---- streams
    analytics = load_stream(r, f"analytics:{iid}")
    control = [m for _, m, _ in load_stream(r, "control") if m.get("interview_id") == iid]
    audio = {}
    for role in ("candidate", "interviewer"):
        items = load_stream(r, f"audio:{iid}:{role}", binary=True)
        if items:
            audio[role] = Audio(role, items)
    frames = load_stream(r, f"frames:{iid}")

    ref = audio.get("candidate") or audio.get("interviewer")
    skew = float(np.nanmedian(ref.pub - ref.cap)) if ref is not None else 0.0

    def captured(t):                       # session time -> the host clock time when that moment was captured
        return float(np.interp(t, ref.t, ref.cap)) if ref is not None else float("nan")

    def lag_of(pub, t):                    # seconds between capture and publication of a result about moment t
        return (pub - skew) - captured(t)

    # ---- 1. capture
    rep.head("1. CAPTURE: did the audio and video arrive?")
    for role, a in audio.items():
        span = a.t[-1] - a.t[0] + 0.64 if a.n else 0
        gaps = np.diff(a.t)
        big = gaps[gaps > 0.9]
        quiet = float((a.db < -80).mean()) * 100
        rep.line(f"{role:11s} audio: {a.n} chunks = {a.n * 0.64:.0f} s of {span:.0f} s ({a.n * 0.64 / max(span, 1) * 100:.0f}% present), "
                 f"{len(big)} gaps over 0.9 s (longest {fmt(float(gaps.max()) if len(gaps) else 0, 1)} s), digital silence {quiet:.0f}% of the time")
        rep.line(f"{'':11s}         loudness p10/p50/p90 = {pct(a.db, 10):.0f} / {pct(a.db, 50):.0f} / {pct(a.db, 90):.0f} dBFS (speech is usually -35..-15, room noise -60..-45)")
        delay = (a.pub - a.cap) - np.nanmedian(a.pub - a.cap)
        slope = float(np.polyfit(a.t, a.pub - a.cap, 1)[0]) if a.n > 5 and np.isfinite(a.cap).all() else 0.0
        rep.line(f"{'':11s}         bot->Redis delay jitter p90 {fmt(pct(delay, 90), 2)} s, max {fmt(float(np.nanmax(delay)), 2)} s, drift {slope * 60:+.2f} s per minute")
        if len(big):
            rep.problem(2, f"{role} audio had {len(big)} gaps over 0.9 s (longest {gaps.max():.1f} s): the call or the bot dropped audio")
        if abs(slope * 60) > 1.0:
            rep.problem(3, f"{role} audio reached Redis later and later ({slope * 60:+.1f} s per minute): the bot or the machine could not keep up with capture")
        if role == "candidate" and quiet > 80 and dur > 20:
            rep.problem(3, f"candidate audio was digital silence {quiet:.0f}% of the time: the candidate's microphone was muted / not captured")
        if a.n * 0.64 < 0.6 * dur and dur > 20:
            rep.problem(3, f"{role} audio covers only {a.n * 0.64:.0f} s of a {dur:.0f} s interview")
    if "candidate" not in audio:
        rep.problem(3, "no candidate audio at all was captured")
    if "interviewer" not in audio:
        rep.problem(2, "no interviewer audio was captured (the interviewer's microphone was not published)")
    if frames:
        ft = np.array([m["t_s"] for _, m, _ in frames])
        fps = (len(ft) - 1) / max(ft[-1] - ft[0], 1e-6)
        fg = np.diff(ft)
        rep.line(f"camera frames kept: {len(frames)} (the last {ft[-1] - ft[0]:.0f} s), about {fps:.1f} fps, {int((fg > 0.5).sum())} stalls over 0.5 s (longest {fg.max():.1f} s)")
        if fps < 7:
            rep.problem(2, f"the camera reached the analysis at only {fps:.1f} fps (10 expected): the call's video was throttled or the bot was overloaded")
        if (fg > 1.5).any():
            rep.problem(2, f"camera frames stalled for up to {fg.max():.1f} s")
    else:
        rep.line("camera frames: none kept (the retention is only the last 30 s, or no camera video was captured)")

    # ---- 2. lag
    rep.head("2. LAG: how long after something happened did the result appear?")
    kinds = Counter(m.get("kind") for _, m, _ in analytics)
    rep.line("messages published: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common(8)))

    def lags(kind, key, cond=lambda m: True):
        pts = []
        for pub, m, _ in analytics:
            if m.get("kind") == kind and cond(m) and isinstance(m.get(key), (int, float)):
                pts.append((float(m[key]), lag_of(pub, float(m[key]))))
        return pts

    def lag_block(name, pts, warn, bad, what):
        if not pts:
            rep.line(f"{name:28s} no data")
            return
        v = [x for _, x in pts]
        worst = max(pts, key=lambda p: p[1])
        rep.line(f"{name:28s} median {fmt(pct(v, 50), 1)} s   p90 {fmt(pct(v, 90), 1)} s   worst {fmt(worst[1], 1)} s (at {mmss(worst[0])})")
        buckets = {}
        for t, x in pts:
            buckets.setdefault(int(t // 30), []).append(x)
        trend = "  ".join(f"{mmss(b * 30)}:{np.median(x):.0f}s" for b, x in sorted(buckets.items()))
        rep.line(f"{'':28s} per 30 s (median): {trend}")
        if pct(v, 90) > bad:
            rep.problem(3, f"{what}: typically {pct(v, 90):.0f} s behind (worst {worst[1]:.0f} s at {mmss(worst[0])})")
        elif pct(v, 90) > warn:
            rep.problem(2, f"{what}: up to {pct(v, 90):.1f} s behind")
        if len(buckets) >= 4:
            keys = sorted(buckets)
            first, last = np.median(buckets[keys[0]]), np.median(buckets[keys[-1]])
            if last - first > 5:
                rep.problem(3, f"{what}: the delay kept growing ({first:.0f} s at the start, {last:.0f} s at the end): the workers could not keep up")

    lag_block("speech loop (audio -> worker)", lags("health", "t_s"), 2.5, 8, "the speech worker fell behind the audio")
    lag_block("vision loop (frame -> gaze)", lags("gaze", "t_s"), 1.5, 4, "the vision worker fell behind the camera")
    lag_block("transcript (speech end -> line)", lags("transcript", "end_s", lambda m: m.get("final")), 4, 9, "transcript lines appeared late")
    lag_block("vision windows", lags("window", "end_s"), 4, 10, "vision window scores arrived late")
    ends = {q.id: q.end_s for q in qs}
    ans_pts = []
    for pub, m, _ in analytics:
        if m.get("kind") == "answer" and isinstance(m.get("end_s"), (int, float)):
            ans_pts.append((float(m["end_s"]), lag_of(pub, float(m["end_s"]))))
    lag_block("answer analysis (end -> card)", ans_pts, 15, 45, "answer analysis finished late")

    # ---- 3. transcript
    rep.head("3. TRANSCRIPT: how much of the speech became text, and is it believable?")
    for role in ("candidate", "interviewer"):
        mine = [x for x in segs if x.speaker == role]
        words = sum(len(x.text.split()) for x in mine)
        a = audio.get(role)
        speech_s = None
        if a is not None:
            floor = pct(a.db, 10)
            thr = -50.0 if floor < -70 else max(-50.0, floor + 8.0)
            speechy = a.db > thr
            speech_s = float(speechy.sum()) * 0.64
            covered = np.zeros(a.n, bool)
            for x in mine:
                covered |= (a.t + 0.64 >= x.start_s - 1.0) & (a.t <= x.end_s + 1.0)
            unc = speechy & ~covered
            rep.line(f"{role:11s}: {len(mine)} lines, {words} words; loud audio {speech_s:.0f} s of which {float(unc.sum()) * 0.64:.0f} s has no transcript line nearby "
                     f"(threshold {thr:.0f} dBFS; approximate)")
            stretches, i = [], 0
            while i < a.n:
                if unc[i]:
                    j = i
                    while j + 1 < a.n and unc[j + 1]:
                        j += 1
                    if (j - i + 1) * 0.64 >= 2.5:
                        stretches.append((a.t[i], a.t[j] + 0.64, float(np.mean(a.db[i:j + 1]))))
                    i = j + 1
                else:
                    i += 1
            for t0, t1, d in sorted(stretches, key=lambda x: x[0] - x[1])[:5]:
                rep.line(f"{'':11s}  loud but no text: {mmss(t0)}-{mmss(t1)} ({t1 - t0:.0f} s, {d:.0f} dBFS)")
            if role == "candidate" and speech_s > 20 and float(unc.sum()) * 0.64 > 0.35 * speech_s:
                rep.problem(3, f"{float(unc.sum()) * 0.64:.0f} s of loud candidate audio ({float(unc.sum()) * 0.64 / speech_s * 100:.0f}%) has no transcript")
        else:
            rep.line(f"{role:11s}: {len(mine)} lines, {words} words (audio no longer in Redis)")
    texts = [x.text for x in segs]
    short = sum(1 for t in texts if len(t.split()) <= 2)
    rep.line(f"very short lines (1-2 words): {short} of {len(texts)}")
    bad = [x for x in segs if re.search(r"subscribe|thanks for watching|kind of, like, kind of|like, kind of, like", x.text, re.I)]
    rep.line(f"invented subtitle phrases / looped prompt lines: {len(bad)}" + (f"  e.g. {bad[0].text[:60]!r}" if bad else ""))
    if bad:
        rep.problem(2, f"{len(bad)} transcript lines look invented by the speech model (e.g. {bad[0].text[:50]!r})")
    dups = 0
    for x in segs:
        for y in segs:
            if x.speaker == "candidate" and y.speaker == "interviewer" and abs(x.start_s - y.start_s) <= 4:
                A, B = set(re.findall(r"[a-z']+", x.text.lower())), set(re.findall(r"[a-z']+", y.text.lower()))
                if min(len(A), len(B)) >= 3 and len(A & B) / min(len(A), len(B)) >= 0.7:
                    dups += 1
    retracts = kinds.get("transcript_retract", 0)
    rep.line(f"same words on both tracks at once (left in the transcript): {dups}; lines removed as a leak while live: {retracts}")
    if dups:
        rep.problem(1, f"{dups} lines appear on both microphones (one voice heard twice)")
    cand_fill = [x.fillers_per_100w for x in stats if x.fillers_per_100w is not None]
    cand_hedge = [x.hedges_per_100w for x in stats if x.hedges_per_100w is not None]
    rep.line(f"speech stats: {len(stats)} rows (about {dur / 5:.0f} expected), filler rate present in {len(cand_fill)}, hedge rate in {len(cand_hedge)}; "
             f"max filler/100w {max(cand_fill or [0])}, max hedge/100w {max(cand_hedge or [0])}")
    if stats and len(cand_fill) < 0.3 * len(stats):
        rep.problem(2, f"filler / hedge rates were missing in {100 - len(cand_fill) / len(stats) * 100:.0f}% of the speech-stat rows: too few words recognised")
    if stats and len(stats) < 0.6 * dur / 5 and dur > 30:
        rep.problem(2, f"only {len(stats)} speech-stat rows for {dur:.0f} s (about {dur / 5:.0f} expected): the candidate audio was not processed continuously")

    # ---- 4. face / vision
    rep.head("4. FACE AND CAMERA")
    exp = dur / 2.5 if dur else 0
    present = [(w.features or {}).get("face_present_ratio") for w in wins]
    present = [p for p in present if p is not None]
    blink = [(w.features or {}).get("blink_rate_per_min") for w in wins if ((w.features or {}).get("face_present_ratio") or 1) >= 0.5]
    rep.line(f"vision windows: {len(wins)} (about {exp:.0f} expected for {dur:.0f} s)")
    if present:
        nof = sum(1 for p in present if p < 0.5)
        rep.line(f"face present: mean {np.mean(present) * 100:.0f}%, windows with no face {nof} ({nof / len(present) * 100:.0f}%)")
        rep.line(f"blink rate (windows with a face): mean {fmt(float(np.mean([b for b in blink if b is not None] or [float('nan')])))} /min, "
                 f"zero in {sum(1 for b in blink if not b)} of {len(blink)} windows")
        if nof / len(present) > 0.25:
            rep.problem(3 if nof / len(present) > 0.5 else 2, f"no face was found in {nof / len(present) * 100:.0f}% of the windows: the candidate was out of frame, the camera was off or the video was too poor")
        if blink and sum(1 for b in blink if not b) / len(blink) > 0.7:
            rep.problem(2, "blinks were not detected in most windows (glasses, glare or low frame rate)")
    if exp and len(wins) < 0.7 * exp and dur > 20:
        rep.problem(3, f"only {len(wins)} vision windows for {dur:.0f} s (about {exp:.0f} expected): the vision worker was not receiving frames")
    gz = [m for _, m, _ in analytics if m.get("kind") == "gaze"]
    if gz:
        span = gz[-1]["t_s"] - gz[0]["t_s"]
        rep.line(f"live gaze messages: {len(gz)} = {len(gz) / max(span, 1):.1f} per second (5 expected), with a face in {np.mean([bool(m.get('face')) for m in gz]) * 100:.0f}%")
    fk = Counter(f.kind for f in flags)
    rep.line("alerts raised: " + (", ".join(f"{k} {v}" for k, v in fk.most_common()) or "none"))
    if fk.get("no_face", 0) >= 3:
        rep.problem(2, f"'face not in frame' alert fired {fk['no_face']} times")

    # ---- 5. microphone health (what the cockpit's health row said)
    rep.head("5. WHAT THE COCKPIT'S HEALTH ROW REPORTED")
    hs = [m for _, m, _ in analytics if m.get("kind") == "health"]
    for role in ("candidate", "interviewer"):
        rows = [m["mics"][role] for m in hs if role in m.get("mics", {})]
        if rows:
            maxsil = max(x["silent_s"] for x in rows)
            nowords = sum(1 for x in rows if x["speech_recent"] and (x["since_words_s"] is None or x["since_words_s"] > 15))
            rep.line(f"{role:11s}: loudest silence {maxsil:.0f} s, {nowords} of {len(rows)} seconds 'hearing speech but no words for 15 s+'")
            if role == "candidate" and nowords > 10:
                rep.problem(2, f"the candidate was heard speaking but no words appeared for 15 s or more, {nowords} times")

    # ---- 6. questions and answers
    rep.head("6. QUESTIONS AND ANSWERS")
    if not qs:
        rep.line("no questions were asked or detected")
    for q in qs:
        a = ans.get(q.id)
        lag = None
        if a is not None and started and q.end_s is not None:
            lag = a.created_at.timestamp() - (started + q.end_s)
        rel = (a.answer_model or {}).get("relevance") if a else None
        rep.line(f"{q.question_id:6s} {mmss(q.ask_s or 0)}-{mmss(q.end_s or 0)} | analysed={'YES' if a else 'NO '} (took {fmt(lag, 0)} s after the answer ended) "
                 f"| effectiveness {fmt(a.effectiveness if a else None, 0)} | relevance {'abstained' if a and rel is None else fmt(rel, 2)} | {q.text[:50]}")
        if a is None and q.end_s is not None:
            rep.problem(3, f"question {q.question_id} ({q.text[:40]!r}) was never analysed")
        if lag is not None and lag > 45:
            rep.problem(2, f"the answer to {q.question_id} was analysed {lag:.0f} s after it ended")
    auto = [q for q in qs if q.question_id.startswith("AUTO")]
    if iv.platform != "interlens" and not qs:
        rep.problem(2, "this was a Meet / Zoom / Teams interview but no question was detected from the interviewer's voice")
    if auto:
        rep.line(f"questions detected from the voice: {len(auto)}")
    rep.line(f"follow-up suggestions: {Counter(f.reason for f in fus).most_common()}")

    # ---- 7. logs
    rep.head("7. WORKER AND CONTAINER LOGS DURING THE INTERVIEW")
    log = ROOT / "runs" / "workers.log"
    shown = 0
    if log.exists() and started:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        idx = 0
        for i, ln in enumerate(lines):
            m = re.match(r"===== workers started (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", ln)
            if m and datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() <= started + 5:
                idx = i
        pat = re.compile(r"Traceback|Error|Exception|falling behind|could not be transcribed|skipped|unavailable|disabled for|Fatal|Segmentation", re.I)
        hits = [ln for ln in lines[idx:] if pat.search(ln)]
        rep.line(f"runs/workers.log since the workers started: {len(hits)} suspicious lines")
        for ln in hits[:8]:
            rep.line("   " + ln[:150])
            shown += 1
        if any("falling behind" in h for h in hits):
            rep.problem(3, "the speech worker logged 'falling behind real time' and skipped speech")
        if any("Traceback" in h for h in hits):
            rep.problem(3, "the workers logged a Python error (see section 7)")
    else:
        rep.line("runs/workers.log not found")
    if started:
        since = datetime.utcfromtimestamp(started).strftime("%Y-%m-%dT%H:%M:%SZ")
        for name in ("mbas-livekit-1", "mbas-caddy-1"):
            try:
                out = subprocess.run(["docker", "logs", "--since", since, name], capture_output=True, text=True, timeout=30)
                text = (out.stdout or "") + (out.stderr or "")
                bad_lines = [ln for ln in text.splitlines() if re.search(r"error|fail|timeout|disconnect|reconnect|ice .*(fail|disconnect)|closed|refused|\bwarn", ln, re.I)]
                rep.line(f"{name}: {len(text.splitlines())} lines since the start, {len(bad_lines)} look like problems")
                for ln in bad_lines[:5]:
                    rep.line("   " + re.sub(r"\s+", " ", ln)[:170])
                if name == "mbas-livekit-1" and len(bad_lines) > 5:
                    rep.problem(2, f"LiveKit logged {len(bad_lines)} warning / error lines (connection trouble between the browsers and the server)")
            except Exception as e:  # noqa: BLE001
                rep.line(f"{name}: could not read ({type(e).__name__})")

    # ---- 8. the computer
    rep.head("8. THE COMPUTER (from the monitor recording)")
    mon = monitor or (sorted(glob.glob(str(ROOT / "runs" / "monitor-*.csv"))) or [None])[-1]
    if mon and started:
        rows = []
        with open(mon, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    if started - 5 <= float(row["time"]) <= (ended or float(row["time"])) + 30:
                        rows.append(row)
                except Exception:  # noqa: BLE001
                    pass
        col = lambda k: [float(r_[k]) for r_ in rows if r_.get(k) not in ("", None) and not str(r_[k]).startswith("ERR")]  # noqa: E731
        if rows:
            rep.line(f"recording {Path(mon).name}: {len(rows)} samples during the interview")
            tot, chrome, work, dock, ram = col("cpu_total"), col("cpu_chrome"), col("cpu_workers"), col("cpu_docker"), col("ram_free_gb")
            rep.line(f"CPU total median {pct(tot, 50):.0f}%  p90 {pct(tot, 90):.0f}%  max {max(tot or [0]):.0f}%   |   Chrome {pct(chrome, 50):.0f}%   analysis workers {pct(work, 50):.0f}%   Docker {pct(dock, 50):.0f}%")
            rep.line(f"free RAM min {min(ram or [0]):.1f} GB   Redis round trip p90 {pct(col('redis_ms'), 90):.0f} ms (normally 1-5)")
            for k in ("vision_frame", "vad", "asr_chunk", "e2e_vision", "e2e_transcript"):
                v = col(f"{k}_ms")
                if v:
                    rep.line(f"  {k:15s} median {pct(v, 50):7.0f} ms   max {max(v):7.0f} ms   (idle computer: about {IDLE_MS[k]} ms; {pct(v, 50) / IDLE_MS[k]:.0f}x)")
            sb, vb = col("speech_backlog_s"), col("vision_backlog_s")
            rep.line(f"unprocessed audio (speech worker behind): median {pct(sb, 50):.1f} s, max {max(sb or [0]):.1f} s;   unprocessed video: median {pct(vb, 50):.1f} s, max {max(vb or [0]):.1f} s")
            if pct(tot, 50) > 85:
                rep.problem(3, f"the CPU was at {pct(tot, 50):.0f}% most of the time (Chrome {pct(chrome, 50):.0f}%, workers {pct(work, 50):.0f}%, Docker {pct(dock, 50):.0f}%): this computer is overloaded, which is the likely cause of every lag above")
            elif pct(tot, 90) > 90:
                rep.problem(2, f"the CPU peaked above 90% (p90 {pct(tot, 90):.0f}%)")
            if ram and min(ram) < 1.5:
                rep.problem(2, f"free memory fell to {min(ram):.1f} GB: the computer was swapping")
            if pct(col("redis_ms"), 90) > 50:
                rep.problem(2, f"Redis answered slowly (p90 {pct(col('redis_ms'), 90):.0f} ms): the computer was too busy to serve its own messages")
        else:
            rep.line(f"{Path(mon).name} does not overlap this interview")
    else:
        rep.line("no monitor recording (start `python -m tools.monitor_run` before the next test to see CPU and backlog over time)")

    # ---- 9. optional: listen again offline
    if retranscribe and audio:
        rep.head("9. RE-LISTENING OFFLINE (what a patient pass over the same audio hears)")
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(settings().whisper_model, device="cpu", compute_type="int8", cpu_threads=4)
            for role, a in audio.items():
                x = a.wav().astype(np.float32) / 32768.0
                segs_o, _ = model.transcribe(x, language="en", beam_size=1, condition_on_previous_text=False, vad_filter=True)
                off = [(a.t[0] + s_.start, a.t[0] + s_.end, s_.text.strip()) for s_ in segs_o if s_.text.strip()]
                live = [(y.start_s, y.end_s, y.text) for y in segs if y.speaker == role]
                missed = [o for o in off if len(o[2].split()) >= 3 and not any(o[0] - 1.5 <= l[1] and o[1] + 1.5 >= l[0] for l in live)]
                ow = Counter(w for o in off for w in re.findall(r"[a-z']+", o[2].lower()))
                lw = Counter(w for l in live for w in re.findall(r"[a-z']+", l[2].lower()))
                recall = sum(min(c, lw.get(w, 0)) for w, c in ow.items()) / max(sum(ow.values()), 1)
                rep.line(f"{role:11s}: offline pass heard {sum(ow.values())} words in {len(off)} lines; the live transcript has {sum(lw.values())} words; "
                         f"{recall * 100:.0f}% of the offline words also appear live; {len(missed)} offline lines have no live line nearby")
                for o in missed[:8]:
                    rep.line(f"   missed live {mmss(o[0])}: {o[2][:110]}")
                if role == "candidate" and len(missed) >= 3:
                    rep.problem(2, f"{len(missed)} candidate sentences heard offline were never in the live transcript")
        except Exception as e:  # noqa: BLE001
            rep.line(f"could not re-transcribe: {type(e).__name__}: {e}")

    if export:
        out = Path(export)
        out.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        for role, a in audio.items():
            sf.write(str(out / f"interview{iid}_{role}.wav"), a.wav(), SR)
        rep.line(f"\naudio saved to {out}")

    # ---- verdict
    rep.head("WHAT IS STILL LAGGY OR NOT WORKING (worst first)")
    if not rep.problems:
        rep.line("nothing stood out")
    for sev, text in sorted(rep.problems, key=lambda p: -p[0]):
        rep.line(f"[{'BROKEN' if sev == 3 else 'LAGGY / DEGRADED' if sev == 2 else 'minor'}] {text}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("interview", nargs="?", type=int, help="interview id (default: the newest)")
    ap.add_argument("--retranscribe", action="store_true", help="re-listen to the audio offline and list what the live transcript missed")
    ap.add_argument("--export", metavar="DIR", help="save both audio tracks as WAV files")
    ap.add_argument("--monitor", metavar="CSV", help="a monitor_run recording (default: the newest in runs/)")
    args = ap.parse_args(argv)
    return analyse(args.interview, args.retranscribe, args.export, args.monitor)


if __name__ == "__main__":
    sys.exit(main())
