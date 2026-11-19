"""Optional pre-trained speech recogniser: OpenAI Whisper (via faster-whisper) behind the same interface as StreamingASR.

This is a PRE-TRAINED model, not one we trained. It is off by default (MBAS_ASR=own keeps our own Conformer-CTC ASR);
select it with MBAS_ASR=whisper. Why it exists: our ASR was trained on 100 h of clean audiobook speech and does not
cope with real laptop / phone microphones, rooms and accents (see docs/pretrained_models.md for the measurements).

How it works: the VAD splits the audio into utterances (an utterance ends after `end_silence_s` of silence, or at
`max_utt_s`), and each finished utterance is transcribed once. So there are no partial results: a line appears about
a second after the speaker pauses. Word timestamps come from Whisper and are shifted to session time. Whisper is
known to invent text such as "Thank you." on silence or noise, so segments it itself marks as probably not speech,
low-confidence repetitive output, and stock phrases on very short clips are dropped.

Whisper drops spoken fillers ("um", "uh") by default, which would make every filler count zero. A short prompt full of
fillers makes it write them down; text that is only the prompt echoed back (it can happen on noise) is discarded.

With a `pool` (a one-thread executor) the transcribing runs in the background: accept() keeps splitting the audio and
returns results of earlier utterances as they finish, so a slow recogniser delays the words but never blocks the audio
loop (a blocked loop falls behind real time and loses audio). flush() waits for everything still running.

Same interface as StreamingASR: accept(audio) -> [Result], flush() -> [Result], and `t0` (session time of the next
sample, set by the worker to the first chunk's time).
"""
from __future__ import annotations

import re
from collections import deque
from concurrent.futures import Future

import numpy as np

from ml.speech.asr.streaming import Result

SR = 16000
# phrases Whisper tends to produce from silence / noise when there is no speech
FILLER_PROMPT = "Umm, let me think, like, hmm... Okay, so, uh, I mean, you know, I think maybe it's, like, kind of, um, basically."
_NORM = re.compile(r"[^a-z' ]+")


def _norm(t: str) -> str:
    return " ".join(_NORM.sub(" ", t.lower()).split())


STOCK = {"thank you", "thank you.", "thanks for watching", "thanks for watching.", "thanks for watching!", "you", "you.",
         "bye", "bye.", "bye!", "okay", "okay.", ".", "so", "so."}


class WhisperStreamingASR:
    def __init__(self, model, vad=None, chunk_s: float = 0.64, end_silence_s: float = 0.7, max_utt_s: float = 12.0,
                 min_speech_s: float = 0.4, preroll_chunks: int = 1, beam_size: int = 1, prompt: str | None = FILLER_PROMPT,
                 pool=None, max_pending: int = 16):
        self.model, self.vad = model, vad
        self.chunk = int(chunk_s * SR)
        self.end_silence, self.max_utt, self.min_speech = end_silence_s, max_utt_s, min_speech_s
        self.preroll_chunks, self.beam_size = preroll_chunks, beam_size
        self.prompt, self._prompt_norm = prompt, " " + _norm(prompt or "") + " "
        self.pool, self.max_pending = pool, max_pending
        self.pending: deque[Future] = deque()          # utterances being transcribed in the background, oldest first
        self.t0 = 0.0                                  # session time of the next sample (the worker sets it)
        self.buf = np.zeros(0, np.float32)
        self.utt: list[np.ndarray] = []                # chunks of the utterance being heard (with a little lead-in)
        self.utt_t0 = 0.0
        self.speech_s = 0.0
        self.silence = 0.0
        self.recent: list[tuple[float, np.ndarray]] = []   # last quiet chunks, kept as lead-in for the next utterance

    # ------------------------------------------------------------------ speech / no speech
    def _is_speech(self, chunk: np.ndarray) -> bool:
        if self.vad is not None:
            return bool((self.vad(chunk) > 0.5).mean() > 0.1)
        return bool(10 * np.log10((chunk ** 2).mean() + 1e-10) > -45)

    def accept(self, audio: np.ndarray) -> list[Result]:
        out: list[Result] = []
        self.buf = np.concatenate([self.buf, audio.astype(np.float32)])
        while len(self.buf) >= self.chunk:
            c, self.buf = self.buf[: self.chunk], self.buf[self.chunk:]
            t_c, self.t0 = self.t0, self.t0 + self.chunk / SR
            if self._is_speech(c):
                if not self.utt:
                    lead = self.recent[-self.preroll_chunks:] if self.preroll_chunks else []
                    self.utt = [a for _, a in lead]
                    self.utt_t0 = lead[0][0] if lead else t_c
                self.utt.append(c)
                self.speech_s += self.chunk / SR
                self.silence = 0.0
            elif self.utt:
                self.utt.append(c)
                self.silence += self.chunk / SR
            else:
                self.recent = (self.recent + [(t_c, c)])[-max(self.preroll_chunks, 1):]
            if self.utt and (self.silence >= self.end_silence or len(self.utt) * self.chunk / SR >= self.max_utt):
                out += self._finish()
        return out + self._collect()

    def flush(self) -> list[Result]:
        """End of the stream: finish the utterance in progress and wait for every background job."""
        out: list[Result] = []
        if self.utt:
            out += self._finish()
        out += self._collect(wait=True)
        return out

    def _collect(self, wait: bool = False, timeout: float = 120.0) -> list[Result]:
        out: list[Result] = []
        while self.pending and (wait or self.pending[0].done()):
            try:
                out += self.pending.popleft().result(timeout=timeout if wait else None)
            except Exception as e:  # noqa: BLE001 - one failed utterance must not stop the transcript
                print(f"[whisper] an utterance could not be transcribed ({type(e).__name__}: {e})", flush=True)
        return out

    # ------------------------------------------------------------------ transcribe one utterance
    def _finish(self) -> list[Result]:
        audio, t0, speech_s = np.concatenate(self.utt), self.utt_t0, self.speech_s
        self.utt, self.speech_s, self.silence, self.recent = [], 0.0, 0.0, []
        if speech_s < self.min_speech:
            return []
        if self.pool is None:
            return self._transcribe(audio, t0, speech_s)
        if len(self.pending) >= self.max_pending:                       # hopelessly behind: skip rather than lag forever
            print("[whisper] falling behind real time: an utterance was skipped", flush=True)
            return []
        self.pending.append(self.pool.submit(self._transcribe, audio, t0, speech_s))
        return []

    def _transcribe(self, audio: np.ndarray, t0: float, speech_s: float) -> list[Result]:
        kw = {"initial_prompt": self.prompt} if self.prompt else {}
        segs, _ = self.model.transcribe(audio, language="en", beam_size=self.beam_size, condition_on_previous_text=False,
                                        word_timestamps=True, vad_filter=False, temperature=0.0, **kw)
        words, parts = [], []
        for s in segs:
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:          # Whisper itself doubts there is speech
                continue
            if getattr(s, "compression_ratio", 0.0) > 2.4:                # repetitive loop
                continue
            text = s.text.strip()
            if not text:
                continue
            parts.append(text)
            for w in s.words or []:
                words.append({"word": w.word.strip(), "start_s": round(t0 + w.start, 2), "end_s": round(t0 + w.end, 2)})
        text = " ".join(parts).strip()
        if not text or (speech_s < 3.0 and text.lower() in STOCK):
            return []
        n = _norm(text)
        if self.prompt and len(n.split()) >= 3 and f" {n} " in self._prompt_norm:      # only the prompt echoed back
            return []
        return [Result(text=text, final=True, words=words, t_s=t0 + len(audio) / SR)]
