"""Run the LiveKit bot + vision worker + speech worker in one process (demo machine).
In production each runs as its own process/container (see docker-compose.yml, profile "app").

    python -m backend.workers.run_all
"""
from __future__ import annotations

import asyncio
import faulthandler
import sys
import threading
import time
from pathlib import Path

from backend.workers import bot, speech_worker, vision_worker


class _Tee:
    """Copy everything printed to the terminal into a log file too (PowerShell loses output when it is redirected)."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        try:
            self.stream.write(text)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.log.write(text)
        except Exception:  # noqa: BLE001
            pass
        return len(text)

    def flush(self):
        for f in (self.stream, self.log):
            try:
                f.flush()
            except Exception:  # noqa: BLE001
                pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log() -> Path:
    path = Path(__file__).resolve().parents[2] / "runs" / "workers.log"
    path.parent.mkdir(exist_ok=True)
    log = open(path, "a", encoding="utf-8", buffering=1)
    log.write(f"\n===== workers started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    faulthandler.enable(file=log, all_threads=True)          # a native crash leaves its stack trace here
    return path


def main():
    print(f"[workers] log file: {start_log()}", flush=True)
    for target in (vision_worker.run, speech_worker.run):
        threading.Thread(target=target, daemon=True).start()
    asyncio.run(bot.main_async())


if __name__ == "__main__":
    main()
