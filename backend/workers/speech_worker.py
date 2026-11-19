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

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from backend.bus import make_bus
from backend.config import settings
from backend.db import (AnswerScore, Database, FollowUp, SpeechStat, TranscriptSegment, WindowScore, clean_json, row_dict,
                        select)
from backend.flags import raise_flag
from backend.features import assemble
from backend.inference import Models, Timer
from ml.behaviour.answer_behaviour import answer_behaviour
from ml.behaviour.conversation import analyse
from ml.behaviour.pauses import question_end, segments
from ml.behaviour.text_hesitation import FILLERS, HEDGES, count_phrases, filler_count, words as split_words

SR = 16000
FRAME = 160
# Answer analysis takes about 10 s (the disfluency model alone 6 s). It runs on its own thread so the audio loop keeps
# reading: a loop that blocks for 10 s on every question falls behind real time and loses the end of the interview.
_ANALYSIS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="answer-analysis")
TEXT_WIN_S, TEXT_MIN_WORDS = 60.0, 8       # filler / hedge rates: the last minute of words, from 8 words on
SILENCE_DB = -52.0           # absolute floor under the VAD (dBFS per 10 ms frame)
SOUND_DB = -55.0             # a chunk louder than this means the microphone is alive (room noise is about -60 .. -50)


class ChunkVAD:
    """Speech probabilities for consecutive 640 ms chunks, with 0.5 s of context between calls. Falls back to an
    energy gate until the VAD is trained. StreamingASR needs this to know when an utterance has ended."""

    def __init__(self, fn):
        self.fn, self.ctx = fn, np.zeros(8000, np.float32)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        n = len(chunk) // FRAME
        if self.fn is None:
            fr = chunk[: n * FRAME].reshape(-1, FRAME)
            return (10 * np.log10((fr ** 2).mean(1) + 1e-10) > -45).astype(np.float32)
        x = np.concatenate([self.ctx, chunk])
        p = self.fn(x)
        self.ctx = x[-len(self.ctx):]
        return p[-n - 1:-1] if n and len(p) > n else p[:n]


@dataclass
class Track:
    name: str
    audio: list[tuple[float, np.ndarray]] = field(default_factory=list)   # (t_s, float32 chunk)
    speech: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # 10 ms frames from t = 0
    vad_ctx: np.ndarray = field(default_factory=lambda: np.zeros(8000, np.float32))
    asr: object | None = None
    finals: list[dict] = field(default_factory=list)
    asr_anchored: bool = False                    # StreamingASR clock set to the first chunk's session time
    conf: list[tuple[float, float]] = field(default_factory=list)          # (t_s, mean max-prob) per final
    # signal health (what the cockpit shows: is this microphone alive, is it hearing speech, are words coming out)
    first_t: float | None = None
    level_db: float | None = None                 # smoothed loudness of the last chunks, dBFS
    last_sound_t: float | None = None             # last chunk louder than SOUND_DB
    last_speech_t: float | None = None            # last chunk the VAD called speech
    last_final_t: float | None = None             # when the last recognised words ended

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
    def __init__(self, interview_id: int, mode: str, bus, db, models: Models, platform: str | None = None):
        self.id, self.mode, self.bus, self.db, self.m = interview_id, mode, bus, db, models
        self.tracks: dict[str, Track] = {}
        self.next_speaking, self.next_conv = 5.0, 10.0
        self.next_health = 0.0
        self.lock = threading.RLock()            # guards the tracks: the analysis thread reads them while audio arrives
        self.candidate_name = ""
        try:                                     # the name helps the recogniser spell it
            from backend.db import Interview

            with db.session() as s:
                iv = s.get(Interview, interview_id)
                self.candidate_name = (iv.candidate_name or "").strip() if iv else ""
        except Exception:  # noqa: BLE001 - a missing row or database must not stop the worker
            pass
        self.in_pause = False
        self.questions: dict[int, dict] = {}
        self.platform = platform or "interlens"
        a = settings().auto_questions
        # questions found in the interviewer's voice: on for interviews held on Meet / Zoom / Teams (where nobody clicks
        # Ask), off in our own room (the interviewer picks from the list) unless MBAS_AUTO_QUESTIONS=on
        self.auto_questions = mode == "interview" and (a == "on" or (a == "auto" and self.platform != "interlens"))
        self.open_q: dict | None = None          # the question being answered right now (asked by click or by voice)
        self.last_click_ask_t = -1e9             # session time of the latest question asked by a click
        bus.publish(f"analytics:{self.id}", {"kind": "status", "models": {
            "vad": models.vad is not None, "asr": getattr(models, "asr_ready", models.asr is not None), "asr_pretrained": getattr(models, "asr_backend", "own") == "whisper", "disfluency": models.disfluency is not None,
            "answer_quality": models.answer is not None, "fusion": models.fusion is not None}})

    def track(self, name: str) -> Track:
        if name not in self.tracks:
            t = Track(name)
            if getattr(self.m, "asr_backend", "own") == "whisper":
                from ml.speech.asr.whisper_streaming import WhisperStreamingASR

                from ml.speech.asr.whisper_streaming import FILLER_PROMPT

                model = self.m.whisper_for(name) if hasattr(self.m, "whisper_for") else self.m.whisper
                t.asr = WhisperStreamingASR(model, ChunkVAD(self.m.vad), pool=getattr(self.m, "whisper_pool", None),
                                            prompt=None if settings().whisper_prompt == "off" else FILLER_PROMPT)
                t.asr.set_context(self.asr_context(name))
            elif self.m.asr is not None:
                from ml.speech.asr.streaming import StreamingASR

                model, tok, dev = self.m.asr
                t.asr = StreamingASR(model, tok, ChunkVAD(self.m.vad), device=dev, decoder=self.m.lm_decoder)
            self.tracks[name] = t
        return self.tracks[name]

    # ------------------------------------------------------------------ audio
    def audio(self, name: str, msg: dict, pcm: bytes):
        with self.lock:
            self._audio(name, msg, pcm)

    def _audio(self, name: str, msg: dict, pcm: bytes):
        tr = self.track(name)
        x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        t0 = msg["t_s"]
        tr.audio.append((t0, x))
        tr.audio = [a for a in tr.audio if a[0] > t0 - 900]          # keep 15 min for answer analysis
        # VAD with 0.5 s of context
        fr = x[: len(x) // FRAME * FRAME].reshape(-1, FRAME)
        frame_db = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
        if self.m.vad is not None:
            with Timer(self.bus, "vad"):
                p = self.m.vad(np.concatenate([tr.vad_ctx, x]))
            n = len(x) // FRAME
            probs = p[-n - 1:-1] if len(p) > n else p
            # the VAD normalises by the mean level of its 1.1 s input, so a window of pure room noise looks like
            # speech; frames below an absolute floor (quiet room ~ -60 dBFS, speech -35..-20) are never speech
            probs = np.where(frame_db[:len(probs)] < SILENCE_DB, 0.0, probs)
            from ml.speech.vad.model import smooth

            sp = smooth(np.concatenate([np.zeros(0), probs]), hangover=10, min_speech=5)[:n]
        else:   # energy fallback until the VAD is trained
            sp = frame_db > -45
        tr.vad_ctx = x[-8000:]
        db = 10 * np.log10(float((x ** 2).mean()) + 1e-10)
        tr.first_t = t0 if tr.first_t is None else tr.first_t
        tr.level_db = db if tr.level_db is None else 0.6 * tr.level_db + 0.4 * db
        if db > SOUND_DB:
            tr.last_sound_t = t0
        if len(sp) and bool(np.any(sp)):
            tr.last_speech_t = t0
        f0 = int(round(t0 * 100))
        if len(tr.speech) < f0 + len(sp):
            tr.speech = np.pad(tr.speech, (0, f0 + len(sp) - len(tr.speech)))
        tr.speech[f0:f0 + len(sp)] = sp
        st = {k: float(v.speech_between(max(0, t0 - 5), t0 + 0.64).mean()) for k, v in self.tracks.items()}
        self.bus.set_state(f"{self.id}:vad", st)
        # streaming ASR
        if tr.asr is not None:
            if not tr.asr_anchored:     # StreamingASR counts from 0; this track's audio starts at the first chunk's t_s
                tr.asr.t0, tr.asr_anchored = t0, True
            with Timer(self.bus, "asr_chunk"):
                results = tr.asr.accept(x)
            for r in results:
                self.transcript(name, r, msg)
        now_t = t0 + len(x) / SR
        if name == "candidate" and now_t >= self.next_health:
            self.next_health = now_t + 1.0
            self.publish_health(now_t)
        if name == "candidate":
            self.pause_check(now_t)
        if name == "candidate" and now_t >= self.next_speaking:
            self.speaking(tr, now_t)
            self.next_speaking += 5.0
        if now_t >= self.next_conv:
            self.conversation(now_t)
            self.next_conv += 10.0

    def pause_check(self, t: float):
        """Alert once per episode when the candidate has been silent for 4 s (and the interviewer is not talking)."""
        cand, inter = self.tracks["candidate"], self.tracks.get("interviewer")
        n = 400
        silent = len(cand.speech) >= n and cand.speech.any() and not cand.speech[-n:].any() \
            and not (inter is not None and inter.speech[-n:].any())
        if silent and not self.in_pause:
            raise_flag(self.bus, self.db, self.id, "long_pause", t, "Long pause: candidate silent for 4 s")
        self.in_pause = silent

    def transcript(self, name: str, r, msg: dict):
        if not r.text and not r.final:
            return
        ws = r.words
        fill = [i for i, w in enumerate(ws) if w["word"].lower() in FILLERS]
        start = ws[0]["start_s"] if ws else r.t_s
        end = ws[-1]["end_s"] if ws else r.t_s
        if r.final and r.text and self.is_leak(name, r.text, start, end):
            return                                   # the same words are on the other track, louder there: this is a leak
        out = {"kind": "transcript", "speaker": name, "text": r.text, "final": r.final, "start_s": start, "end_s": end,
               "words": ws, "fillers": fill}
        self.bus.publish(f"analytics:{self.id}", out)
        if r.final and r.text:
            tr = self.tracks[name]
            tr.last_final_t = end
            tr.finals.append({"start_s": start, "end_s": end, "text": r.text, "words": ws})
            if name == "interviewer" and self.auto_questions:
                self.auto_question(r.text, start)
            with self.db.session() as s:
                s.add(TranscriptSegment(interview_id=self.id, speaker=name, start_s=start, end_s=end, text=r.text,
                                        words=ws, fillers=fill))
                s.commit()
            self.bus.set_metric("e2e_transcript", (time.time() - msg["captured_at"]) * 1000 + 640)

    # ------------------------------------------------------------------ questions found in the interviewer's voice
    def asr_context(self, role: str) -> str:
        """Words that help the recogniser: the candidate's name (both tracks) and, for the candidate, the open question."""
        parts = []
        if self.candidate_name and self.candidate_name.lower() not in ("candidate", "a"):
            parts.append(f"Candidate: {self.candidate_name}.")
        if role == "candidate" and self.open_q and self.open_q.get("text"):
            parts.append(f"Question: {str(self.open_q['text'])[:160]}")
        return " ".join(parts)

    def refresh_asr_context(self):
        for name, tr in self.tracks.items():
            if tr.asr is not None and hasattr(tr.asr, "set_context"):
                tr.asr.set_context(self.asr_context(name))

    def on_ask(self, ev: dict):
        """Every question that starts (by click or by voice) becomes the open one."""
        self.open_q = {k: ev.get(k) for k in ("question_ref", "question_id", "text", "category", "ask_s", "bank_text")}
        self.refresh_asr_context()
        if not ev.get("auto"):
            self.last_click_ask_t = ev.get("ask_s") or 0.0

    def auto_question(self, text: str, start_s: float):
        from backend.db import Question
        from backend.question_bank import match_bank
        from backend.question_detect import extract_question

        q = extract_question(text)
        if not q:
            return
        if start_s - self.last_click_ask_t < 30.0:                 # the interviewer also clicked Ask: do not double up
            return
        if self.open_q and start_s - (self.open_q.get("ask_s") or 0.0) < 3.0:     # the same question, said in two pieces
            return
        if self.open_q:                                            # a new question ends the previous answer
            self.end_open(start_s - 0.1)
        from sqlalchemy import func, select

        with self.db.session() as s:
            n = s.scalar(select(func.count()).select_from(Question).where(Question.interview_id == self.id)) or 0
            row = Question(interview_id=self.id, order=n + 1, question_id=f"AUTO{n + 1}", text=q[:300], category="auto", ask_s=round(start_s, 2))
            s.add(row)
            s.commit()
            match = match_bank(row.text, str(settings().questions_path))          # a bank question? then relevance can be judged
            ev = {"type": "ask", "interview_id": self.id, "question_ref": row.id, "question_id": row.question_id, "text": row.text,
                  "category": "auto", "ask_s": row.ask_s, "end_s": None, "t_s": row.ask_s, "auto": True,
                  "bank_id": match[0] if match else None, "bank_text": match[1] if match else None}
        self.on_ask(ev)
        self.bus.publish("control", ev)
        self.bus.publish(f"analytics:{self.id}", {"kind": "question", **ev})

    def end_open(self, end_s: float, direct: bool = False):
        """Close the open question: store its end and queue the answer analysis (through the bus, like a click)."""
        from backend.db import Question

        q, self.open_q = self.open_q, None
        self.refresh_asr_context()
        if not q or q.get("question_ref") is None:
            return
        end_s = max(float(end_s), float(q.get("ask_s") or 0.0) + 0.5)
        with self.db.session() as s:
            row = s.get(Question, q["question_ref"])
            if row is not None and row.end_s is None:
                row.end_s = round(end_s, 2)
                s.commit()
        ev = {"type": "answer_end", "interview_id": self.id, **q, "end_s": round(end_s, 2), "t_s": round(end_s, 2), "auto": True}
        self.bus.publish(f"analytics:{self.id}", {"kind": "question", **ev})
        if direct:                                   # the interview is ending: nobody will read a queued message
            submit_answer(self, ev)
        else:
            self.bus.publish("control", ev)

    # ------------------------------------------------------------------ one voice on both tracks
    def level_db(self, name: str, t0: float, t1: float) -> float:
        tr = self.tracks.get(name)
        x = tr.clip(max(0.0, t0), max(t1, t0 + 0.3)) if tr is not None else np.zeros(0, np.float32)
        return 10 * np.log10(float((x ** 2).mean()) + 1e-10) if len(x) else -120.0

    def is_leak(self, name: str, text: str, start: float, end: float) -> bool:
        """When two people share a room (or a microphone hears the speakers) the same words are transcribed on both tracks.
        The quieter track is the leak. True: this new line is the leak, drop it. Otherwise an already shown line of the
        other track that turns out to be the leak is removed (and retracted on the live screen)."""
        other = "interviewer" if name == "candidate" else "candidate"
        tr_o = self.tracks.get(other)
        if tr_o is None:
            return False
        mine = set(_leak_words(text))
        for f in list(tr_o.finals):
            if not (start - 4.0 <= f["start_s"] <= end + 4.0):
                continue
            theirs = set(_leak_words(f["text"]))
            if min(len(mine), len(theirs)) < 3 or len(mine & theirs) / min(len(mine), len(theirs)) < 0.7:
                continue
            my_db, their_db = self.level_db(name, start, end), self.level_db(other, f["start_s"], f["end_s"])
            i_am_leak = my_db < their_db - 1.0 or (abs(my_db - their_db) <= 1.0 and name == "interviewer")
            if i_am_leak:
                return True
            self.retract(other, f)
        return False

    def retract(self, speaker: str, f: dict):
        tr = self.tracks.get(speaker)
        if tr is not None and f in tr.finals:
            tr.finals.remove(f)
        try:
            from sqlalchemy import delete

            with self.db.session() as s:
                s.execute(delete(TranscriptSegment).where(TranscriptSegment.interview_id == self.id, TranscriptSegment.speaker == speaker,
                                                          TranscriptSegment.start_s == f["start_s"], TranscriptSegment.text == f["text"]))
                s.commit()
        except Exception:  # noqa: BLE001 - never let storage break the live analysis
            import traceback

            traceback.print_exc()
        self.bus.publish(f"analytics:{self.id}", {"kind": "transcript_retract", "speaker": speaker, "start_s": f["start_s"]})

    def publish_health(self, now_t: float):
        """Once a second: is each microphone alive, is it hearing speech, and are words coming out of it?"""
        mics = {}
        for name, tr in self.tracks.items():
            first = tr.first_t if tr.first_t is not None else now_t
            heard = tr.last_sound_t if tr.last_sound_t is not None else first
            mics[name] = {
                "db": None if tr.level_db is None else round(tr.level_db),
                "silent_s": round(max(0.0, now_t - heard), 1),               # how long nothing louder than room noise
                "speech_recent": tr.last_speech_t is not None and now_t - tr.last_speech_t < 5.0,
                "since_words_s": None if tr.last_final_t is None else round(max(0.0, now_t - tr.last_final_t), 1),
                "listening_s": round(now_t - first, 1)}
        self.bus.publish(f"analytics:{self.id}", {"kind": "health", "t_s": round(now_t, 1), "mics": mics,
                                                 "asr": "whisper" if getattr(self.m, "asr_backend", "own") == "whisper" else "own"})

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
        text = " ".join(f["text"] for f in tr.finals if f["end_s"] > t - TEXT_WIN_S)     # a minute of words: 30 s rarely has enough
        ws = split_words(text)
        per100 = (lambda n: round(100 * n / len(ws), 1)) if len(ws) >= TEXT_MIN_WORDS else (lambda n: None)
        msg = {
            "kind": "speaking", "t_s": t, "wpm": round(60 * words / speech_s, 1) if speech_s > 3 and words > 0 else None,   # no words = no transcript yet (ASR missing), not "0 wpm"
            "fillers_per_100w": per100(filler_count(ws, text)),
            "hedges_per_100w": per100(count_phrases(text, HEDGES)),
            "pause_ratio": round(1 - speech_s / min(win, t), 3) if t > 0 else None,
            "pitch_variation_st": round(pitch_var, 2) if pitch_var is not None else None,
            "speech_ratio": round(float(sp.mean()), 3) if len(sp) else 0.0}
        self.bus.publish(f"analytics:{self.id}", msg)
        try:
            with self.db.session() as s:
                s.add(SpeechStat(interview_id=self.id, t_s=round(t, 2), wpm=msg["wpm"], pause_ratio=msg["pause_ratio"],
                                 fillers_per_100w=msg["fillers_per_100w"], hedges_per_100w=msg["hedges_per_100w"],
                                 pitch_variation_st=msg["pitch_variation_st"]))
                s.commit()
        except Exception:  # noqa: BLE001 - storage problems must not stop the live analysis
            import traceback

            traceback.print_exc()

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
        clip_t0, clip_t1 = ask, end + 0.5
        with self.lock:             # a consistent snapshot of the tracks; the slow model work below runs without the lock
            cand = self.track("candidate")
            inter = self.tracks.get("interviewer")
            audio = cand.clip(clip_t0, clip_t1)
            cand_segs = cand.segs(ask, clip_t1)
            q_end = question_end(inter.segs(ask - 1, end) if inter else [], ask, end)
            transcript = " ".join(f["text"] for f in cand.finals if ask - 0.5 <= f["start_s"] <= end + 1)
            answer_clip = cand.clip(q_end, end) if end - q_end >= 1.0 else None
            conv = self.answer_conversation(cand, inter, q_end, end)
            speech = cand.speech_between(clip_t0, clip_t1)
        events = self.audio_events(answer_clip, q_end, end)
        beh = answer_behaviour(f"{self.id}_A{ev['question_ref']}", audio, transcript, cand_segs, q_end, end, clip_t0, events)
        ans = None
        # The answer-quality model was trained on the bank's questions: its relevance is meaningful only for a bank question
        # (clicked from the list, or heard and matched to the bank). For a free-form question it abstains instead of
        # inventing "topic not covered" for a correct answer.
        in_bank = bool(ev.get("bank_text")) or str(ev.get("question_id") or "").upper().startswith("Q")
        if transcript.strip() and in_bank:
            try:
                ans = self.m.answer_scores(ev.get("bank_text") or ev.get("text", ""), transcript)
            except Exception as e:  # noqa: BLE001 - a broken answer model must not lose the rest of the answer
                print(f"[speech] answer-quality model failed for interview {self.id}: {type(e).__name__}: {e}", flush=True)
        with self.db.session() as s:
            wins = [row_dict(w) for w in s.scalars(select(WindowScore).where(
                WindowScore.interview_id == self.id, WindowScore.end_s > ask, WindowScore.start_s < end))]
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
                             "long_pauses": beh["timing"]["long_pause_count"],
                             "longest_pause_s": beh["timing"].get("longest_pause_s"),
                             "voice_breaks": beh["prosody"].get("voice_break_count"),
                             "jitter_pct": beh["prosody"].get("pitch_jitter_pct")},
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

    def audio_events(self, x: np.ndarray | None, t0: float, t1: float) -> dict:
        if self.m.disfluency is None or x is None or t1 - t0 < 1.0:
            return {}
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
    # load every model now (ASR, VAD, disfluency, answer, fusion: ~20 s the first time), not when the first
    # interview starts, so the first answer of the day is analysed as fast as the rest
    for m in ("vad", "asr_backend", "whisper", "whisper_candidate", "asr", "disfluency", "answer", "fusion"):
        getattr(models, m)
    if models.asr_backend == "whisper":
        cand = settings().whisper_model_candidate or settings().whisper_model
        print(f"[speech] recogniser: Whisper (pre-trained): candidate {cand}, interviewer {settings().whisper_model}", flush=True)
    else:
        print("[speech] recogniser: our own ASR", flush=True)
    live: dict[int, InterviewSpeech] = {}
    cursors = {"control": bus.last_id("control")}
    print("[speech] ready", flush=True)
    while not stop():
        for name, mid, msg, binary in bus.read(dict(cursors), 500, 100):
            cursors[name] = mid
            try:
                handle(name, msg, binary, bus, db, models, live, cursors, mid)
            except Exception:  # noqa: BLE001 - one bad message must not stop the worker
                import traceback

                traceback.print_exc()


def _leak_words(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z']+", text.lower())


def submit_answer(sp: "InterviewSpeech", ev: dict):
    """Analyse one finished answer on the analysis thread (the audio loop must not wait for it)."""
    def job():
        try:
            sp.answer_end(ev)
        except Exception:  # noqa: BLE001 - one bad answer must not stop the others
            import traceback

            traceback.print_exc()

    _ANALYSIS.submit(job)


def wait_for_analysis(timeout: float = 120.0):
    """Block until every submitted answer analysis has finished (used by tests and at shutdown)."""
    _ANALYSIS.submit(lambda: None).result(timeout=timeout)


def drain_audio(bus, live, cursors, iid: int, limit_s: float = 90.0):
    """Process every audio chunk of this interview that is still queued on the bus. The stop message sits on a different
    stream and is read first, so without this the end of a slowly processed interview would be thrown away."""
    keys = [k for k in cursors if k.startswith(f"audio:{iid}:")]
    deadline = time.time() + limit_s
    while keys and iid in live and time.time() < deadline:
        items = bus.read({k: cursors[k] for k in keys}, 20, 200)
        if not items:
            break
        for name, mid, msg, binary in items:
            cursors[name] = mid
            if binary is not None:
                live[iid].audio(name.split(":", 2)[2], msg, binary)


def handle(name, msg, binary, bus, db, models, live, cursors, mid: str | None = None):
    """One bus message: control events (start / answer_end / stop) or a 640 ms audio chunk."""
    if name == "control":
        iid = msg.get("interview_id")
        if msg["type"] == "start":
            # read this interview's audio from the moment of the start message (stream ids are time-based), not
            # from whenever the setup below finishes: audio published meanwhile would otherwise be skipped
            for k in [k for k in list(cursors) if k.startswith(f"audio:{iid}:")]:
                cursors.pop(k)
            for role in ("candidate", "interviewer"):
                cursors[f"audio:{iid}:{role}"] = mid or bus.last_id(f"audio:{iid}:{role}")
            live[iid] = InterviewSpeech(iid, msg.get("mode", "interview"), bus, db, models, platform=msg.get("platform"))
        elif msg["type"] == "ask" and iid in live:
            if not msg.get("auto"):
                live[iid].on_ask(msg)
        elif msg["type"] == "answer_end" and iid in live:
            if live[iid].open_q and live[iid].open_q.get("question_ref") == msg.get("question_ref"):
                live[iid].open_q = None
            submit_answer(live[iid], msg)
        elif msg["type"] == "stop" and iid in live:
            drain_audio(bus, live, cursors, iid)           # audio that was still queued when the stop arrived
            for t in live[iid].tracks.values():
                if t.asr is not None:
                    for r in t.asr.flush():
                        live[iid].transcript(t.name, r, {"captured_at": time.time()})
            if live[iid].open_q:        # the interview ended mid-answer: analyse it too
                live[iid].end_open(msg.get("t_s") or live[iid].open_q.get("ask_s") or 0.0, direct=True)
            live.pop(iid)
    elif name.startswith("audio:") and binary is not None:
        _, iid, role = name.split(":", 2)
        if int(iid) in live:
            live[int(iid)].audio(role, msg, binary)

if __name__ == "__main__":
    run()
