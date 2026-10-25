"""Conversation analyser on separate audio tracks per participant (no diarisation model needed).

Input: speech segments per participant (from our VAD on each WebRTC track) and the question events.
Output: talk-time ratio, turn count, average turn length, interruptions, overlaps, question-answer pairs.
An interruption is a turn that starts while another participant is speaking and overlaps it > 0.5 s.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Turn:
    speaker: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def turns_from_segments(segs: dict[str, list[tuple[float, float]]], merge_gap_s: float = 1.0) -> list[Turn]:
    """Merge each speaker's segments separated by < merge_gap_s into turns; sort all turns by start."""
    out = []
    for spk, ss in segs.items():
        cur = None
        for a, b in sorted(ss):
            if cur and a - cur.end_s < merge_gap_s:
                cur.end_s = max(cur.end_s, b)
            else:
                if cur:
                    out.append(cur)
                cur = Turn(spk, a, b)
        if cur:
            out.append(cur)
    return sorted(out, key=lambda t: t.start_s)


def analyse(segs: dict[str, list[tuple[float, float]]], duration_s: float, answers: list[dict] | None = None,
            interruption_s: float = 0.5) -> dict:
    turns = turns_from_segments(segs)
    speakers = sorted(segs)
    talk = {s: round(sum(b - a for a, b in segs[s]), 3) for s in speakers}
    total_talk = sum(talk.values()) or 1.0
    overlaps, interruptions = 0.0, {s: 0 for s in speakers}
    for i, t in enumerate(turns):
        for u in turns[:i]:
            if u.speaker == t.speaker or u.end_s <= t.start_s:
                continue
            ov = min(u.end_s, t.end_s) - t.start_s
            if ov > 0:
                overlaps += ov
                if ov > interruption_s and u.start_s < t.start_s:
                    interruptions[t.speaker] += 1
    per = {s: {"talk_time_s": talk[s], "talk_share": round(talk[s] / total_talk, 4),
               "talk_time_ratio": round(talk[s] / max(duration_s, 1e-6), 4),
               "turns": sum(1 for t in turns if t.speaker == s),
               "avg_turn_s": round(sum(t.duration_s for t in turns if t.speaker == s) /
                                   max(1, sum(1 for t in turns if t.speaker == s)), 3),
               "interruptions_made": interruptions[s]} for s in speakers}
    shares = sorted(p["talk_share"] for p in per.values())
    n = len(shares)
    gini = (sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(shares)) / (n * sum(shares))) if n and sum(shares) else 0.0
    out = {"duration_s": duration_s, "turn_count": len(turns), "overlap_s": round(overlaps, 3),
           "talk_balance_gini": round(gini, 4), "participants": per}
    if answers:
        pairs = []
        for a in answers:
            reply = next((t for t in turns if t.speaker == "candidate" and a["ask_s"] < t.start_s < a["end_s"] + 1), None)
            pairs.append({"answer_id": a["answer_id"], "answered": reply is not None,
                          "first_reply_s": round(reply.start_s, 3) if reply else None})
        out["question_answer_pairs"] = pairs
    return out
