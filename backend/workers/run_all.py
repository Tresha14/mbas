"""Run the LiveKit bot + vision worker + speech worker in one process (demo machine).
In production each runs as its own process/container (see docker-compose.yml, profile "app").

    python -m backend.workers.run_all
"""
from __future__ import annotations

import asyncio
import threading

from backend.workers import bot, speech_worker, vision_worker


def main():
    for target in (vision_worker.run, speech_worker.run):
        threading.Thread(target=target, daemon=True).start()
    asyncio.run(bot.main_async())


if __name__ == "__main__":
    main()
