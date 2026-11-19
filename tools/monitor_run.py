"""Record, every 2 seconds, how busy the computer is and how far behind the analysis is, while an interview runs.

    python -m tools.monitor_run                 # until you press Ctrl+C (or --minutes N), writes runs/monitor-<time>.csv

Why: lag is a moving quantity, and after the interview only the last value of each timing metric remains. This keeps the whole curve:

  cpu_total / ram_free_gb          the computer
  cpu_chrome / cpu_workers / ...   who uses the CPU (the browser, the analysis workers, the API, the web server, Docker)
  speech_backlog_s                 how many seconds of audio the speech worker has NOT processed yet (newest audio minus the
                                   time of its latest health message): 0 = real time, growing = falling behind
  vision_backlog_s                 the same for camera frames (newest frame minus the latest gaze message)
  *_ms                             the workers' own latest timings (vision frame, VAD, ASR chunk, disfluency, end to end ...)
  redis_ms                         round trip to Redis

Read it afterwards with `python -m tools.analyze_interview`, which merges it with the stored interview.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import settings  # noqa: E402

METRICS = ("vision_frame", "vad", "asr_chunk", "disfluency", "fusion", "answer_quality", "e2e_vision", "e2e_transcript", "e2e_answer_analysis")
FIELDS = (["time", "wall", "interview", "cpu_total", "ram_free_gb", "cpu_chrome", "cpu_workers", "cpu_api", "cpu_web", "cpu_docker",
           "speech_backlog_s", "vision_backlog_s", "audio_age_s", "last_audio_t", "redis_ms"] + [f"{m}_ms" for m in METRICS])


def _group(proc) -> str:
    try:
        name = (proc.info["name"] or "").lower()
        cmd = " ".join(proc.info["cmdline"] or []).lower()
    except Exception:  # noqa: BLE001
        return ""
    if name.startswith("chrome") or name.startswith("msedge"):
        return "chrome"
    if "backend.workers.run_all" in cmd:
        return "workers"
    if "uvicorn" in cmd and "backend.main" in cmd:
        return "api"
    if name == "node.exe":
        return "web"
    if name.startswith("com.docker") or name in ("vmmem", "vmmemwsl", "docker desktop.exe", "wslhost.exe"):
        return "docker"
    return ""


def _id_ms(mid: bytes | str) -> int:
    return int((mid.decode() if isinstance(mid, bytes) else mid).split("-")[0])


def active_interview(r) -> int | None:
    """The newest interview that has started and not stopped."""
    state: dict[int, str] = {}
    for _, f in reversed(r.xrevrange("control", count=200)):          # oldest first
        m = json.loads(f[b"json"])
        if m.get("type") in ("start", "stop") and m.get("interview_id") is not None:
            state[int(m["interview_id"])] = m["type"]
    live = [i for i, t in state.items() if t == "start"]
    return max(live) if live else None


def last_t(r, stream: str, kind: str | None = None, key: str = "t_s", count: int = 60) -> float | None:
    for _, f in r.xrevrange(stream, count=count):
        m = json.loads(f[b"json"])
        if kind is None or m.get("kind") == kind:
            v = m.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--every", type=float, default=2.0)
    args = ap.parse_args(argv)

    import redis

    r = redis.Redis.from_url(settings().redis_url)
    out = ROOT / "runs" / f"monitor-{datetime.now():%Y%m%d-%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    print(f"monitoring to {out} (Ctrl+C to stop)", flush=True)
    cores = psutil.cpu_count() or 1
    psutil.cpu_percent(None)
    for p in psutil.process_iter():
        try:
            p.cpu_percent(None)
        except Exception:  # noqa: BLE001
            pass
    deadline = time.time() + args.minutes * 60
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        while time.time() < deadline:
            time.sleep(args.every)
            row = {k: "" for k in FIELDS}
            now = time.time()
            row["time"], row["wall"] = f"{now:.1f}", datetime.now().strftime("%H:%M:%S")
            row["cpu_total"] = psutil.cpu_percent(None)
            row["ram_free_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
            groups = {"chrome": 0.0, "workers": 0.0, "api": 0.0, "web": 0.0, "docker": 0.0}
            for p in psutil.process_iter(["name", "cmdline"]):
                g = _group(p)
                if g:
                    try:
                        groups[g] += p.cpu_percent(None) / cores
                    except Exception:  # noqa: BLE001
                        pass
            for g, v in groups.items():
                row[f"cpu_{g}"] = round(v, 1)
            try:
                t0 = time.perf_counter()
                r.ping()
                row["redis_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                for k, v in r.hgetall("metrics:latency").items():
                    name = k.decode()
                    if name in METRICS:
                        row[f"{name}_ms"] = json.loads(v).get("ms", "")
                iid = active_interview(r)
                row["interview"] = iid if iid is not None else ""
                if iid is not None:
                    last = r.xrevrange(f"audio:{iid}:candidate", count=1)
                    if last:
                        row["audio_age_s"] = round(now - _id_ms(last[0][0]) / 1000, 1)
                        row["last_audio_t"] = round(json.loads(last[0][1][b"json"]).get("t_s", 0), 1)
                        health = last_t(r, f"analytics:{iid}", "health")
                        if health is not None:
                            row["speech_backlog_s"] = round(max(0.0, row["last_audio_t"] - health), 1)
                    fr = r.xrevrange(f"frames:{iid}", count=1)
                    if fr:
                        ft = json.loads(fr[0][1][b"json"]).get("t_s")
                        gz = last_t(r, f"analytics:{iid}", "gaze")
                        if ft is not None and gz is not None:
                            row["vision_backlog_s"] = round(max(0.0, ft - gz), 1)
            except Exception as e:  # noqa: BLE001 - never stop recording because Redis hiccuped
                row["redis_ms"] = f"ERR {type(e).__name__}"
            w.writerow(row)
            fh.flush()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("stopped", flush=True)
