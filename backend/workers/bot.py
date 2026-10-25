"""LiveKit analysis bot: joins each live room as a hidden participant and forwards media to the bus.

  candidate camera  -> frames:{id}        JPEG 640 px wide, 10 fps, with capture time (session seconds)
  every audio track -> audio:{id}:{role}  16 kHz mono PCM16, 640 ms chunks
The bot never publishes anything into the room, so the candidate cannot see that analysis is running other than
through the consent banner. Screen shares are relayed to the interviewer by LiveKit itself (not analysed).

    python -m backend.workers.bot
"""
from __future__ import annotations

import asyncio
import json
import time

import cv2
import jwt
import numpy as np

from backend.bus import make_bus
from backend.config import settings

FPS = 10
CHUNK = int(0.64 * 16000)


def bot_token(cfg, room: str) -> str:
    now = int(time.time())
    return jwt.encode({"iss": cfg.livekit_key, "sub": "mbas-analyzer", "name": "MBAS analyzer", "nbf": now - 10,
                       "exp": now + 12 * 3600, "metadata": json.dumps({"role": "bot"}),
                       "video": {"room": room, "roomJoin": True, "canPublish": False, "canSubscribe": True,
                                 "canPublishData": False, "hidden": True}}, cfg.livekit_secret, algorithm="HS256")


def role_of(participant) -> str:
    try:
        return json.loads(participant.metadata or "{}").get("role", "participant")
    except json.JSONDecodeError:
        return "participant"


class RoomBot:
    def __init__(self, cfg, bus, interview_id: int, room_name: str, started_at: float):
        self.cfg, self.bus, self.id, self.room_name, self.t0 = cfg, bus, interview_id, room_name, started_at
        self.tasks: list[asyncio.Task] = []

    async def run(self):
        from livekit import rtc

        self.room = rtc.Room()

        @self.room.on("track_subscribed")
        def _on(track, publication, participant):
            role = role_of(participant)
            if track.kind == rtc.TrackKind.KIND_VIDEO and publication.source == rtc.TrackSource.SOURCE_CAMERA \
                    and role in ("candidate", "participant"):
                self.tasks.append(asyncio.create_task(self.video(track, participant.identity, role)))
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                self.tasks.append(asyncio.create_task(self.audio(track, participant.identity, role)))

        await self.room.connect(self.cfg.livekit_url, bot_token(self.cfg, self.room_name),
                                rtc.RoomOptions(auto_subscribe=True))
        print(f"[bot] joined {self.room_name} for interview {self.id}", flush=True)

    async def video(self, track, identity: str, role: str):
        from livekit import rtc

        last = 0.0
        async for ev in rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24):
            now = time.time()
            if now - last < 1 / FPS:
                continue
            last = now
            f = ev.frame
            img = np.frombuffer(f.data, np.uint8).reshape(f.height, f.width, 3)
            scale = 640 / f.width
            if scale < 1:
                img = cv2.resize(img, (640, int(f.height * scale)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.bus.publish(f"frames:{self.id}", {"t_s": round(now - self.t0, 3), "captured_at": now,
                                                       "identity": identity, "role": role}, jpg.tobytes())

    async def audio(self, track, identity: str, role: str):
        from livekit import rtc

        buf = np.zeros(0, np.int16)
        start = None
        key = "candidate" if role in ("candidate",) else ("interviewer" if role == "interviewer" else identity)
        async for ev in rtc.AudioStream(track, sample_rate=16000, num_channels=1):
            pcm = np.frombuffer(ev.frame.data, np.int16)
            if start is None:
                start = time.time() - len(pcm) / 16000
            buf = np.concatenate([buf, pcm])
            while len(buf) >= CHUNK:
                chunk, buf = buf[:CHUNK], buf[CHUNK:]
                self.bus.publish(f"audio:{self.id}:{key}", {"t_s": round(start - self.t0, 3), "captured_at": time.time(),
                                                            "identity": identity, "role": role, "samples": CHUNK},
                                 chunk.tobytes())
                start += CHUNK / 16000

    async def close(self):
        for t in self.tasks:
            t.cancel()
        await self.room.disconnect()


async def main_async():
    cfg = settings()
    bus = make_bus(cfg.redis_url)
    bots: dict[int, RoomBot] = {}
    last = bus.last_id("control")
    print("[bot] waiting for interviews to start ...", flush=True)
    while True:
        items = await asyncio.to_thread(bus.read, {"control": last}, 1000, 50)
        for _, mid, msg, _ in items:
            last = mid
            if msg["type"] == "start" and msg["interview_id"] not in bots:
                b = RoomBot(cfg, bus, msg["interview_id"], msg["room"], msg["started_at"])
                bots[msg["interview_id"]] = b
                try:
                    await b.run()
                except Exception as e:  # noqa: BLE001
                    print(f"[bot] cannot join {msg['room']}: {e}", flush=True)
                    bots.pop(msg["interview_id"], None)
            elif msg["type"] == "stop" and msg["interview_id"] in bots:
                await bots.pop(msg["interview_id"]).close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
