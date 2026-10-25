"""Speech + answer worker: audio:{id}:{role} -> VAD, streaming ASR, speaking patterns, conversation metrics;
on 'answer_end' -> behaviour JSON, answer-quality model, feature assembler, fusion + top-3 reasons + follow-ups.

Publishes to analytics:{id}:
  transcript    {speaker, text, final, start_s, end_s, words, fillers}       (fillers = word indices to highlight)
  speaking      {wpm, pause_ratio, pitch_variation_st, speech_ratio}          every 5 s (candidate)
  conversation  {talk_time_ratio, turns, interruptions, overlap_s, participants} every 10 s
  answer        {question_ref, effectiveness, subscores, follow_up, reasons, behaviour summary, ...}
  follow_up     {t_s, reason, text, probability, source}
  status        {models: {...}}  which models are trained / missing

    python -m backend.workers.speech_worker
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from backend.bus import make_bus
from backend.config import settings
from backend.db import AnswerScore, Database, FollowUp, TranscriptSegment, WindowScore, clean_json, row_dict, select
from backend.features import assemble
from backend.inference import Models, Timer
from ml.behaviour.answer_behaviour import answer_behaviour
from ml.behaviour.conversation import analyse
from ml.behaviour.pauses import question_end, segments
from ml.behaviour.text_hesitation import FILLERS, words as split_words

SR = 16000
FRAME = 160


@dataclass
class Track:
    name: str
    audio: list[tuple[float, np.ndarray]] = field(default_factory=list)   # (t_s, float32 chunk)
    speech: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # 10 ms frames from t = 0
    vad_ctx: np.ndarray = field(default_factory=lambda: np.zeros(8000, np.float32))
    asr: object | None = None
    finals: list[dict] = field(default_factory=list)
    conf: list[tuple[float, float]] = field(default_factory=list)          # (t_s, mean max-prob) per final

    def clip(self, t0: float, t1: float) -> np.ndarray:
        n = int(round((t1 - t0) * SR))
        out = np.zeros(max(n, 0), np.float32)
        for t, x in self.audio:
            a = int(round((t - t0) * SR))
            if a >= n or a + len(x) <= 0:
                continue
            s = max(0, -a)
            e = min(len(x), n - a)
            out[a + s:a + e] = x[s:e]
        return out

    def speech_between(self, t0: float, t1: float) -> np.ndarray:
        a, b = int(t0 * 100), int(t1 * 100)
        sp = self.speech[max(a, 0):max(b, 0)]
        return np.pad(sp, (0, max(0, (b - a) - len(sp))))

    def segs(self, t0: float = 0.0, t1: float = 1e9) -> list[tuple[float, float]]:
        a = max(0, int(t0 * 100))
        return [(round(s, 2), round(e, 2)) for s, e in segments(self.speech[a:max(a, int(t1 * 100))], t0=a / 100)]


class InterviewSpeech:
    def __init__(self, interview_id: int, mode: str, bus, db, models: Models):
        self.id, self.mode, self.bus, self.db, self.m = interview_id, mode, bus, db, models
        self.tracks: dict[str, Track] = {}
        self.next_speaking, self.next_conv = 5.0, 10.0
        self.questions: dict[int, dict] = {}
        bus.publish(f"analytics:{self.id}", {"kind": "status", "models": {
            "vad": models.vad is not None, "asr": models.asr is not None, "disfluency": models.disfluency is not None,
            "answer_quality": models.answer is not None, "fusion": models.fusion is not None}})

    def track(self, name: str) -> Track:
        if name not in self.tracks:
            t = Track(name)
            if self.m.asr is not None:
                from ml.speech.asr.streaming import StreamingASR

                model, tok, dev = self.m.asr
                t.asr = StreamingASR(model, tok, None, device=dev)
            self.tracks[name] = t
        return self.tracks[name]

    # ------------------------------------------------------------------ audio
    def audio(self, name: str, msg: dict, pcm: bytes):
        tr = self.track(name)
        x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        t0 = msg["t_s"]
        tr.audio.append((t0, x))
        tr.audio = [a for a in tr.audio if a[0] > t0 - 900]          # keep 15 min for answer analysis
        # VAD with 0.5 s of context
        if self.m.vad is not None:
            with Timer(self.bus, "vad"):
                p = self.m.vad(np.concatenate([tr.vad_ctx, x]))
            n = len(x) // FRAME
            probs = p[-n - 1:-1] if len(p) > n else p
            from ml.speech.vad.model import smooth

            sp = smooth(np.concatenate([np.zeros(0), probs]), hangover=10, min_speech=5)[:n]
        else:   # energy fallback until the VAD is trained
            fr = x[: len(x) // FRAME * FRAME].reshape(-1, FRAME)
            sp = 10 * np.log10((fr ** 2).mean(1) + 1e-10) > -45
        tr.vad_ctx = x[-8000:]
        f0 = int(round(t0 * 100))
        if len(tr.speech) < f0 + len(sp):
            tr.speech = np.pad(tr.speech, (0, f0 + len(sp) - len(tr.speech)))
        tr.speech[f0:f0 + len(sp)] = sp
        st = {k: float(v.speech_between(max(0, t0 - 5), t0 + 0.64).mean()) for k, v in self.tracks.items()}
        self.bus.set_state(f"{self.id}:vad", st)
        # streaming ASR
        if tr.asr is not None:
            with Timer(self.bus, "asr_chunk"):
                results = tr.asr.accept(x)
            for r in results:
                self.transcript(name, r, msg)
        now_t = t0 + len(x) / SR
        if name == "candidate" and now_t >= self.next_speaking:
            self.speaking(tr, now_t)
            self.next_speaking += 5.0
        if now_t >= self.next_conv:
            self.conversation(now_t)
            self.next_conv += 10.0

    def transcript(self, name: str, r, msg: dict):
        if not r.text and not r.final:
            return
        ws = r.words
        fill = [i for i, w in enumerate(ws) if w["word"].lower() in FILLERS]
        start = ws[0]["start_s"] if ws else r.t_s
        end = ws[-1]["end_s"] if ws else r.t_s
        out = {"kind": "transcript", "speaker": name, "text": r.text, "final": r.final, "start_s": start, "end_s": end,
               "words": ws, "fillers": fill}
        self.bus.publish(f"analytics:{self.id}", out)
        if r.final and r.text:
            tr = self.tracks[name]
            tr.finals.append({"start_s": start, "end_s": end, "text": r.text, "words": ws})
            with self.db.session() as s:
                s.add(TranscriptSegment(interview_id=self.id, speaker=name, start_s=start, end_s=end, text=r.text,
                                        words=ws, fillers=fill))
                s.commit()
            self.bus.set_metric("e2e_transcript", (time.time() - msg["captured_at"]) * 1000 + 640)

    def speaking(self, tr: Track, t: float):
        win = 30.0
        sp = tr.speech_between(max(0, t - win), t)
        words = sum(len(f["text"].split()) for f in tr.finals if f["end_s"] > t - win)
        speech_s = sp.sum() / 100
        x = tr.clip(max(0, t - 10), t)
        pitch_var = None
        if len(x) > SR:
            from ml.common.audio_features import pitch_yin

            f0 = pitch_yin(x)
            v = f0[~np.isnan(f0)]
            if len(v) > 10:
                pitch_var = float(np.std(12 * np.log2(v / np.median(v))))
        self.bus.publish(f"analytics:{self.id}", {
            "kind": "speaking", "t_s": t, "wpm": round(60 * words / speech_s, 1) if speech_s > 3 else None,
            "pause_ratio": round(1 - speech_s / min(win, t), 3) if t > 0 else None,
            "pitch_variation_st": round(pitch_var, 2) if pitch_var is not None else None,
            "speech_ratio": round(float(sp.mean()), 3) if len(sp) else 0.0})

    def conversation(self, t: float):
        segs = {k: v.segs() for k, v in self.tracks.items()}
        if not segs:
            return
        r = analyse(segs, t)
        msg = {"kind": "conversation", "t_s": t, **r}
        if self.mode == "meeting":
            msg["meeting"] = self.meeting_scores(r)
        self.bus.publish(f"analytics:{self.id}", msg)

    def meeting_scores(self, conv: dict) -> dict:
        import torch

        from ml.common.config import CHECKPOINTS
        from ml.fusion.model import MeetingHead

        p = CHECKPOINTS / "meeting_head" / "best.pt"
        if not p.exists():
            return {}
        st = torch.load(p, map_location="cpu", weights_only=False)
        head = MeetingHead(len(st["features"]))
        head.load_state_dict(st["model"])
        parts = conv["participants"]
        n = len(parts)
        tot_turns = sum(v["turns"] for v in parts.values()) or 1
        mean_turn = np.mean([v["avg_turn_s"] for v in parts.values()]) or 1
        out = {}
        for k, v in parts.items():
            row = {"talk_share": v["talk_share"], "share_vs_fair": v["talk_share"] * n,
                   "turn_share_vs_fair": v["turns"] / tot_turns * n, "turn_length_vs_meeting": v["avg_turn_s"] / mean_turn,
                   "on_screen_ratio": 0.0, "camera_on": 0.0, "face_motion_energy": 0.0, "head_yaw_std_deg": 0.0,
                   "multitasking_share": 0.0}
            x = np.array([row[c] for c in st["features"]], float)
            x = np.nan_to_num((x - np.array(st["std"]["mu"])) / np.array(st["std"]["sd"])).astype(np.float32)
            with torch.no_grad():
                e, b = head(torch.from_numpy(x[None]))[0].numpy()
            out[k] = {"engagement": round(float(e), 3), "contribution_balance": round(float(b), 3),
                      "talk_share": v["talk_share"]}
        return out

    # ------------------------------------------------------------------ answers
    def answer_end(self, ev: dict):
        t_start = time.time()
        ask, end = ev["ask_s"], ev["end_s"]
        cand = self.track("candidate")
        inter = self.tracks.get("interviewer")
        clip_t0, clip_t1 = ask, end + 0.5
        audio = cand.clip(clip_t0, clip_t1)
        cand_segs = cand.segs(ask, clip_t1)
        q_end = question_end(inter.segs(ask - 1, end) if inter else [], ask, end)
        transcript = " ".join(f["text"] for f in cand.finals if ask - 0.5 <= f["start_s"] <= end + 1)
        events = self.audio_events(cand, q_end, end)
        beh = answer_behaviour(f"{self.id}_A{ev['question_ref']}", audio, transcript, cand_segs, q_end, end, clip_t0, events)
        ans = self.m.answer_scores(ev.get("text", ""), transcript) if transcript.strip() else None
        with self.db.session() as s:
            wins = [row_dict(w) for w in s.scalars(select(WindowScore).where(
                WindowScore.interview_id == self.id, WindowScore.end_s > ask, WindowScore.start_s < end))]
        conv = self.answer_conversation(cand, inter, q_end, end)
        speech = cand.speech_between(clip_t0, clip_t1)
        feats = assemble(beh, ans, wins, conv, ev.get("question_id", ""), transcript, audio, speech, None)
        fus = self.m.fusion_scores(feats)
        flags = self.rule_flags(beh, ans, wins)
        out = {"kind": "answer", "question_ref": ev["question_ref"], "question_id": ev.get("question_id"),
               "question_text": ev.get("text"), "ask_s": ask, "end_s": end, "transcript": transcript,
               "behaviour": {"speaking_pattern": (beh.get("speaking_pattern") or {}).get("label"),
                             "hesitation_index": (beh.get("hesitation_index") or {}).get("value"),
                             "response_latency_s": beh["timing"]["response_latency_s"],
                             "wpm": beh["prosody"].get("speaking_rate_wpm"),
                             "fillers": beh["text_hesitation"]["filler_count"],
                             "long_pauses": beh["timing"]["long_pause_count"]},
               "answer_model": ans, "fusion": fus, "rule_flags": flags}
        with self.db.session() as s:
            s.add(AnswerScore(interview_id=self.id, question_ref=ev["question_ref"],
                              effectiveness=fus["effectiveness"] if fus else None,
                              subscores=clean_json(fus["subscores"]) if fus else {}, answer_model=clean_json(ans or {}),
                              behaviour=clean_json(beh), reasons=clean_json(fus["reasons"]) if fus else [],
                              modality_weights=clean_json(fus["modality_weights"]) if fus else {}, transcript=transcript))
            fu_list = []
            if fus and fus["follow_up_p"] >= 0.5 and fus["reason"] != "none":
                fu_list.append(("model", fus["reason"], fus["follow_up_p"], fus["reasons"][0]["text"] if fus["reasons"] else ""))
            for reason, text in flags:
                if not any(f[1] == reason for f in fu_list):
                    fu_list.append(("rule", reason, 1.0, text))
            for src, reason, p, text in fu_list:
                s.add(FollowUp(interview_id=self.id, question_ref=ev["question_ref"], t_s=end, reason=reason,
                               probability=p, source=src, text=text))
                self.bus.publish(f"analytics:{self.id}", {"kind": "follow_up", "question_ref": ev["question_ref"],
                                                         "t_s": end, "reason": reason, "probability": round(p, 3),
                                                         "source": src, "text": text})
            s.commit()
        self.bus.publish(f"analytics:{self.id}", clean_json(out))
        self.bus.set_metric("e2e_answer_analysis", (time.time() - t_start) * 1000)

    def audio_events(self, cand: Track, t0: float, t1: float) -> dict:
        if self.m.disfluency is None or t1 - t0 < 1.0:
            return {}
        x = cand.clip(t0, t1)
        wins = [x[i:i + 3 * SR] for i in range(0, max(len(x) - SR, 1), int(1.5 * SR))]
        wins = [np.pad(w, (0, 3 * SR - len(w))) for w in wins]
        with Timer(self.bus, "disfluency"):
            p = self.m.disfluency(np.stack(wins))
        th = self.m.disfluency_thresholds
        names = ["filler", "repetition", "prolongation", "block"]
        hits = {n: int((p[:, i] >= th.get(n, 0.5)).sum()) for i, n in enumerate(names)}
        return {"filler_events": hits["filler"], "repetition_events": hits["repetition"],
                "prolongation_count": hits["prolongation"], "block_count": hits["block"]}

    def answer_conversation(self, cand: Track, inter: Track | None, t0: float, t1: float) -> dict:
        if inter is None:
            return {"interrupted": 0, "overlap_s": 0.0}
        both = cand.speech_between(t0, t1) & inter.speech_between(t0, t1)
        r = analyse({"candidate": cand.segs(t0, t1), "interviewer": inter.segs(t0, t1)}, t1 - t0)
        return {"interrupted": int(r["participants"]["interviewer"]["interruptions_made"] > 0),
                "overlap_s": round(float(both.sum()) / 100, 2)}

    @staticmethod
    def rule_flags(beh: dict, ans: dict | None, wins: list[dict]) -> list[tuple[str, str]]:
        out = []
        hi = (beh.get("hesitation_index") or {}).get("value")
        if hi is not None and hi >= 65:
            out.append(("low_confidence", f"high hesitation (index {hi:.0f}/100)"))
        att = [w["attention"] for w in wins if w.get("attention") is not None]
        if att and np.mean(att) < 50:
            out.append(("attention_drop", f"attention dropped to {np.mean(att):.0f}/100 during the answer"))
        if any(w["features"].get("reading_pattern_flag") for w in wins):
            out.append(("attention_drop", "eye movements looked like reading from another screen"))
        if ans and ans["relevance"] < 0.4:
            out.append(("topic_not_covered", f"answer relevance {ans['relevance']:.2f}"))
        return out


def run(bus=None, db=None, stop=lambda: False):
    cfg = settings()
    bus = bus or make_bus(cfg.redis_url)
    db = db or Database(cfg.database_url)
    models = Models(bus)
    live: dict[int, InterviewSpeech] = {}
    cursors = {"control": bus.last_id("control")}
    print("[speech] ready", flush=True)
    while not stop():
        for name, mid, msg, binary in bus.read(dict(cursors), 500, 100):
            cursors[name] = mid
            try:
                handle(name, msg, binary, bus, db, models, live, cursors)
            except Exception:  # noqa: BLE001 - one bad message must not stop the worker
                import traceback

                traceback.print_exc()


def handle(name, msg, binary, bus, db, models, live, cursors):
    """One bus message: control events (start / answer_end / stop) or a 640 ms audio chunk."""
    if name == "control":
        iid = msg.get("interview_id")
        if msg["type"] == "start":
            live[iid] = InterviewSpeech(iid, msg.get("mode", "interview"), bus, db, models)
            for k in [k for k in list(cursors) if k.startswith(f"audio:{iid}:")]:
                cursors.pop(k)
            cursors[f"audio:{iid}:candidate"] = bus.last_id(f"audio:{iid}:candidate")
            cursors[f"audio:{iid}:interviewer"] = bus.last_id(f"audio:{iid}:interviewer")
        elif msg["type"] == "answer_end" and iid in live:
            live[iid].answer_end(msg)
        elif msg["type"] == "stop" and iid in live:
            for t in live[iid].tracks.values():
                if t.asr is not None:
                    for r in t.asr.flush():
                        live[iid].transcript(t.name, r, {"captured_at": time.time()})
            live.pop(iid)
    elif name.startswith("audio:") and binary is not None:
        _, iid, role = name.split(":", 2)
        if int(iid) in live:
            live[int(iid)].audio(role, msg, binary)

if __name__ == "__main__":
    run()
