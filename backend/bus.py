"""Message bus between the API, the LiveKit bot and the inference workers: Redis streams.

Streams (per interview id):
  control              start / stop / ask / answer_end events for all workers
  frames:{id}          candidate camera frames (JPEG, 10 fps) from the bot
  audio:{id}:{role}    16 kHz PCM16 chunks (640 ms) per participant track
  analytics:{id}       results for the interviewer channel (window scores, transcript, answer scores, alerts)
Hash metrics:latency  per-model latency (ms) reported by the workers.
Keys state:{id}:{name} small shared state between workers (e.g. current speech activity per track).

`MemoryBus` implements the same interface in-process (tests, single-process demo).
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from typing import Any

MAXLEN = {"frames": 300, "audio": 2000, "analytics": 20000, "control": 5000}


def _maxlen(stream: str) -> int:
    return MAXLEN.get(stream.split(":")[0], 5000)


class RedisBus:
    def __init__(self, url: str):
        import redis

        self.r = redis.Redis.from_url(url)

    def publish(self, stream: str, msg: dict[str, Any], binary: bytes | None = None) -> str:
        fields = {"json": json.dumps(msg, default=str)}
        if binary is not None:
            fields["bin"] = binary
        return self.r.xadd(stream, fields, maxlen=_maxlen(stream), approximate=True).decode()

    def read(self, streams: dict[str, str], block_ms: int = 1000, count: int = 100) -> list[tuple[str, str, dict, bytes | None]]:
        out = []
        res = self.r.xread(streams, block=block_ms, count=count) or []
        for name, items in res:
            for mid, f in items:
                out.append((name.decode(), mid.decode(), json.loads(f[b"json"]), f.get(b"bin")))
        return out

    def history(self, stream: str, count: int = 2000) -> list[dict]:
        return [json.loads(f[b"json"]) for _, f in self.r.xrange(stream, count=count)]

    def last_id(self, stream: str) -> str:
        last = self.r.xrevrange(stream, count=1)
        return last[0][0].decode() if last else "0-0"

    def set_metric(self, key: str, value: float) -> None:
        self.r.hset("metrics:latency", key, json.dumps({"ms": round(value, 2), "t": time.time()}))

    def metrics(self) -> dict[str, Any]:
        return {k.decode(): json.loads(v) for k, v in self.r.hgetall("metrics:latency").items()}

    def set_state(self, key: str, value: dict) -> None:
        self.r.set(f"state:{key}", json.dumps(value, default=str), ex=3600)

    def get_state(self, key: str) -> dict:
        v = self.r.get(f"state:{key}")
        return json.loads(v) if v else {}


class MemoryBus:
    def __init__(self):
        self.streams: dict[str, list[tuple[str, dict, bytes | None]]] = defaultdict(list)
        self.cv = threading.Condition()
        self.seq = 0
        self._metrics: dict[str, Any] = {}

    def publish(self, stream: str, msg: dict, binary: bytes | None = None) -> str:
        with self.cv:
            self.seq += 1
            mid = f"{int(time.time() * 1000)}-{self.seq}"
            s = self.streams[stream]
            s.append((mid, msg, binary))
            del s[: max(0, len(s) - _maxlen(stream))]
            self.cv.notify_all()
            return mid

    def read(self, streams: dict[str, str], block_ms: int = 1000, count: int = 100):
        def collect():
            out = []
            for name, last in streams.items():
                for mid, msg, b in self.streams.get(name, []):
                    if last == "$":
                        continue
                    if _after(mid, last):
                        out.append((name, mid, msg, b))
            return out[:count]

        with self.cv:
            for name, last in list(streams.items()):     # '$' = only new messages from now on
                if last == "$":
                    s = self.streams.get(name, [])
                    streams[name] = s[-1][0] if s else "0-0"
            got = collect()
            if not got and block_ms:
                self.cv.wait(block_ms / 1000)
                got = collect()
            return got

    def history(self, stream: str, count: int = 2000) -> list[dict]:
        return [m for _, m, _ in self.streams.get(stream, [])[-count:]]

    def last_id(self, stream: str) -> str:
        s = self.streams.get(stream)
        return s[-1][0] if s else "0-0"

    def set_metric(self, key: str, value: float) -> None:
        self._metrics[key] = {"ms": round(value, 2), "t": time.time()}

    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def set_state(self, key: str, value: dict) -> None:
        self._metrics.setdefault("_state", {})[key] = value

    def get_state(self, key: str) -> dict:
        return self._metrics.get("_state", {}).get(key, {})


def _after(mid: str, last: str) -> bool:
    a, b = (tuple(int(x) for x in m.split("-")) for m in (mid, last))
    return a > b


def make_bus(url: str):
    return MemoryBus() if url.startswith("memory") else RedisBus(url)
