"""Put video frames, audio samples and question events of a recorded session on one clock.

Session time t = seconds from the recording start (timeline.json recording.start_ms,
server clock). Each track starts at its own offset; a video frame's session time
is offset + its presentation timestamp, an audio sample's is offset + n / 16000.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from ml.common.audio_features import SR, load_audio
from tools.session_index import SessionIndex, load_session


def video_pts(path: str | Path) -> np.ndarray:
    """Presentation time (s) of every frame, from the container (no decoding)."""
    from tools.check_session import _ffmpeg

    p = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "framemd5", "-"],
                       capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"cannot read {path}")
    tb, pts = 1 / 1000, []
    for line in p.stdout.splitlines():
        if line.startswith("#tb 0:"):
            num, den = line.split(":", 1)[1].strip().split("/")
            tb = int(num) / int(den)
        elif line and not line.startswith("#"):
            pts.append(int(line.split(",")[2]))
    a = np.asarray(pts, dtype=np.float64) * tb
    return a - a[0] if len(a) else a


@dataclass
class SessionClock:
    index: SessionIndex

    @classmethod
    def load(cls, session_dir: str | Path) -> "SessionClock":
        return cls(load_session(session_dir))

    def offset(self, track: str) -> float:
        return self.index.track_offsets_s.get(track, 0.0)

    @cached_property
    def webcam_times(self) -> np.ndarray:
        """Session time of every webcam frame."""
        return self.offset("candidate_webcam") + video_pts(self.index.dir / "candidate_webcam.webm")

    def frame_at(self, t: float) -> int:
        """Index of the webcam frame shown at session time t (last frame with time <= t)."""
        times = self.webcam_times
        return int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1))

    def frames_between(self, t0: float, t1: float) -> np.ndarray:
        times = self.webcam_times
        return np.nonzero((times >= t0) & (times < t1))[0]

    def sample_at(self, t: float, track: str = "candidate_mic") -> int:
        return int(round((t - self.offset(track)) * SR))

    def audio(self, t0: float, t1: float, track: str = "candidate_mic") -> np.ndarray:
        """Audio between session times t0 and t1, zero-padded where the track has no samples."""
        off = self.offset(track)
        n = int(round((t1 - t0) * SR))
        a0 = t0 - off
        start = max(a0, 0.0)
        x = load_audio(self.index.dir / f"{track}.wav", start_s=start, end_s=max(t1 - off, start))
        out = np.zeros(n, np.float32)
        lead = int(round((start - a0) * SR))
        m = max(0, min(len(x), n - lead))
        out[lead:lead + m] = x[:m]
        return out

    def events(self) -> list[dict]:
        """Answers with ask / end on the session clock."""
        return [{"answer_id": a.answer_id, "question_id": a.question_id, "ask_s": a.ask_s, "end_s": a.end_s}
                for a in self.index.answers]

    def calibration(self) -> list[dict]:
        """Calibration dot samples with session time, matched to the nearest webcam frame."""
        import json

        p = self.index.dir / "calibration.json"
        if not p.exists():
            return []
        cal = json.loads(p.read_text(encoding="utf-8"))
        out = []
        for s in cal.get("samples", []):
            t = (s["t_ms"] - self.index.start_ms) / 1000.0
            out.append({**s, "t_s": round(t, 4), "frame": self.frame_at(t)})
        return out
