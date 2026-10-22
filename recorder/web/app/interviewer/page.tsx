"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Track, type Room } from "livekit-client";
import ConsentForm, { CONSENT_VERSION, type ParticipantInfo } from "@/components/ConsentForm";
import MediaSlot from "@/components/MediaSlot";
import { api, type CheckReport, type Question } from "@/lib/api";
import { clock } from "@/lib/clock";
import { PcmRecorder, type RecorderStatus } from "@/lib/recorders";
import { joinRoom, publish, send } from "@/lib/room";
import { fmtBytes, useSession } from "@/lib/useSession";

type Cat = "all" | Question["category"];

function InterviewerInner() {
  const router = useRouter();
  const params = useSearchParams();
  const code = params.get("room")?.toUpperCase() ?? null;

  const [joined, setJoined] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [camEl, setCamEl] = useState<HTMLMediaElement | null>(null);
  const [screenEl, setScreenEl] = useState<HTMLMediaElement | null>(null);
  const [view, setView] = useState<"both" | "camera" | "screen">("both");
  const [selfStream, setSelfStream] = useState<MediaStream | null>(null);
  const [peers, setPeers] = useState<string[]>([]);
  const [micStatus, setMicStatus] = useState<RecorderStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<CheckReport | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [cat, setCat] = useState<Cat>("all");
  const [openQ, setOpenQ] = useState<string | null>(null);
  const [doneQ, setDoneQ] = useState<Set<string>>(new Set());
  const [elapsed, setElapsed] = useState(0);

  const room = useRef<Room | null>(null);
  const micTrack = useRef<MediaStreamTrack | null>(null);
  const micRec = useRef<PcmRecorder | null>(null);
  const audioEls = useRef<HTMLMediaElement[]>([]);
  const { summary, refresh } = useSession(code);

  // Create a room when none is given.
  useEffect(() => {
    if (code) return;
    api
      .createSession()
      .then((s) => router.replace(`/interviewer?room=${s.room_code}`))
      .catch((e) => setError(`Cannot reach recorder API: ${(e as Error).message}`));
  }, [code, router]);

  useEffect(() => {
    api.questions().then((q) => setQuestions(q.questions)).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (summary?.status !== "recording" || !summary.recording.start_ms) return;
    const id = setInterval(() => setElapsed((clock.now() - summary.recording.start_ms!) / 1000), 500);
    return () => clearInterval(id);
  }, [summary?.status, summary?.recording.start_ms]);

  useEffect(
    () => () => {
      room.current?.disconnect();
      audioEls.current.forEach((e) => e.remove());
    },
    [],
  );

  const join = useCallback(
    async (info: ParticipantInfo) => {
      if (!code) return;
      await api.register(code, {
        role: "interviewer",
        participant_id: info.participant_id,
        consent: { agreed: true, version: CONSENT_VERSION },
        user_agent: navigator.userAgent,
      });
      const c = await clock.sync();
      await api.clock(code, "interviewer", c.offset, c.rtt);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: false },
        video: { width: 640, height: 360, frameRate: 15 },
      });
      micTrack.current = stream.getAudioTracks()[0];
      setSelfStream(new MediaStream(stream.getVideoTracks()));
      const r = await joinRoom(code, "interviewer", `interviewer-${info.participant_id}`, {
        onTrack: (kind, el) => {
          if (kind === "audio") {
            if (el) {
              el.style.display = "none";
              document.body.appendChild(el);
              audioEls.current.push(el);
            }
          } else if (kind === "camera") setCamEl(el);
          else setScreenEl(el);
        },
        onMessage: () => undefined,
        onParticipants: setPeers,
      });
      room.current = r;
      await publish(r, stream.getAudioTracks()[0], Track.Source.Microphone);
      await publish(r, stream.getVideoTracks()[0], Track.Source.Camera);
      setJoined(true);
    },
    [code],
  );

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const startRecording = () =>
    run(async () => {
      if (!code || !room.current || !micTrack.current) return;
      await clock.sync(6);
      await api.clock(code, "interviewer", clock.offsetMs, clock.rttMs);
      const t = clock.now();
      await api.recording(code, "start", t);
      await send(room.current, { type: "rec_start", t_ms: t });
      const rec = new PcmRecorder(code, "interviewer_mic", micTrack.current, setMicStatus);
      micRec.current = rec;
      await rec.start();
    });

  const stopRecording = () =>
    run(async () => {
      if (!code || !room.current) return;
      if (openQ) await answerEnded(openQ);
      const t = clock.now();
      await send(room.current, { type: "rec_stop", t_ms: t });
      await api.recording(code, "stop", t);
      await micRec.current?.stop();
    });

  const finalize = () =>
    run(async () => {
      if (!code) return;
      const res = await api.finalize(code);
      setReport(res.check);
    });

  const ask = async (q: Question) => {
    if (!code) return;
    if (openQ && openQ !== q.question_id) await answerEnded(openQ);
    await api.event(code, { type: "ask", question_id: q.question_id, t_ms: clock.now(), question_text: q.text, category: q.category });
    setOpenQ(q.question_id);
  };

  const answerEnded = async (qid: string) => {
    if (!code) return;
    await api.event(code, { type: "answer_end", question_id: qid, t_ms: clock.now() });
    setOpenQ(null);
    setDoneQ((s) => new Set(s).add(qid));
  };

  const shown = useMemo(() => questions.filter((q) => cat === "all" || q.category === cat), [questions, cat]);
  const candidateLink = typeof window !== "undefined" && code ? `${location.origin}/candidate?room=${code}` : "";
  const recording = summary?.status === "recording";
  const candidateTracksOpen = Object.entries(summary?.tracks ?? {}).some(([, t]) => t && t.end_ms === null);
  const canFinalize = summary?.status === "stopped" || summary?.status === "finalized";

  if (!code) return <p className="p-6 text-zinc-400">{error ?? "Creating room…"}</p>;
  if (!joined) return <ConsentForm role="interviewer" onSubmit={join} />;

  return (
    <div className="flex min-h-screen flex-col gap-4 p-4">
      <header className="card flex flex-wrap items-center gap-3">
        <div className="mr-auto">
          <div className="text-xs text-zinc-400">Room code</div>
          <div className="font-mono text-2xl tracking-widest">{code}</div>
          <div className="text-xs text-zinc-500">{summary?.session_id}</div>
        </div>
        <button className="btn btn-ghost" onClick={() => navigator.clipboard.writeText(candidateLink)}>
          Copy candidate link
        </button>
        <span className="text-sm text-zinc-400">
          Candidate: {summary?.participants.candidate ? summary.participants.candidate.participant_id : "not joined"}
          {peers.length === 0 && " · not in room"}
        </span>
        {recording ? (
          <>
            <span className="flex items-center gap-2 text-sm text-rose-400">
              <span className="h-2 w-2 animate-pulse rounded-full bg-rose-500" />
              REC {Math.floor(elapsed / 60)}:{String(Math.floor(elapsed % 60)).padStart(2, "0")}
            </span>
            <button className="btn btn-ghost" disabled={busy} onClick={() => room.current && send(room.current, { type: "calibrate" })}>
              Run calibration
            </button>
            <button className="btn btn-danger" disabled={busy} onClick={stopRecording}>
              Stop recording
            </button>
          </>
        ) : summary?.status === "created" ? (
          <button className="btn btn-primary" disabled={busy || !summary?.participants.candidate} onClick={startRecording}>
            Start recording
          </button>
        ) : (
          <button className="btn btn-primary" disabled={busy || !canFinalize || candidateTracksOpen} onClick={finalize}>
            {candidateTracksOpen ? "Waiting for uploads…" : "Finalize & check"}
          </button>
        )}
      </header>

      {error && <div className="card border-rose-800 text-sm text-rose-300">{error}</div>}

      <div className="grid flex-1 grid-cols-1 gap-4 xl:grid-cols-[1fr_420px]">
        <section className="flex flex-col gap-4">
          <div className="flex gap-2">
            {(["both", "camera", "screen"] as const).map((v) => (
              <button key={v} className={`btn ${view === v ? "btn-primary" : "btn-ghost"}`} onClick={() => setView(v)}>
                {v === "both" ? "Side by side" : v === "camera" ? "Camera" : "Screen"}
              </button>
            ))}
          </div>
          <div className={`grid gap-4 ${view === "both" ? "md:grid-cols-2" : ""}`}>
            {view !== "screen" && <MediaSlot el={camEl} label="Candidate camera" className="aspect-video" />}
            {view !== "camera" && <MediaSlot el={screenEl} label="Candidate screen" className="aspect-video" />}
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-[240px_1fr]">
            <MediaSlot stream={selfStream} muted label="You" className="aspect-video" />
            <div className="card text-sm">
              <h3 className="mb-2 font-medium">Recording status</h3>
              <table className="w-full text-left text-zinc-300">
                <thead className="text-xs text-zinc-500">
                  <tr>
                    <th>Track</th>
                    <th>State</th>
                    <th>Chunks</th>
                    <th>Size</th>
                  </tr>
                </thead>
                <tbody>
                  {(["candidate_webcam", "candidate_mic", "candidate_screen", "interviewer_mic"] as const).map((t) => {
                    const s = summary?.tracks[t];
                    return (
                      <tr key={t}>
                        <td className="font-mono text-xs">{t}</td>
                        <td>{!s ? "—" : s.end_ms === null ? "recording" : "stopped"}</td>
                        <td>{s?.chunks ?? 0}</td>
                        <td>{fmtBytes(s?.bytes ?? 0)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-2 text-xs text-zinc-500">
                Calibration: {summary?.calibration_saved ? "saved" : "not yet"}
                {micStatus?.error && <span className="text-rose-400"> · mic upload error: {micStatus.error}</span>}
              </p>
            </div>
          </div>
          {report && (
            <div className={`card text-sm ${report.passed ? "border-emerald-700" : "border-rose-700"}`}>
              <h3 className="mb-2 font-medium">
                check_session: {report.passed ? <span className="text-emerald-400">PASS</span> : <span className="text-rose-400">FAIL</span>}
              </h3>
              <pre className="max-h-80 overflow-auto whitespace-pre-wrap text-xs text-zinc-300">
                {JSON.stringify({ errors: report.errors, warnings: report.warnings, tracks: report.tracks, info: report.info }, null, 2)}
              </pre>
            </div>
          )}
        </section>

        <aside className="card flex max-h-[calc(100vh-140px)] flex-col">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-medium">Questions</h2>
            <span className="text-xs text-zinc-500">
              {doneQ.size} answered / {questions.length}
            </span>
          </div>
          <div className="mb-3 flex gap-1">
            {(["all", "hr", "behavioural", "technical"] as const).map((c) => (
              <button key={c} className={`btn px-3 py-1 text-xs ${cat === c ? "btn-primary" : "btn-ghost"}`} onClick={() => setCat(c)}>
                {c}
              </button>
            ))}
          </div>
          <ol className="flex-1 space-y-2 overflow-auto pr-1">
            {shown.map((q) => {
              const isOpen = openQ === q.question_id;
              const done = doneQ.has(q.question_id);
              return (
                <li
                  key={q.question_id}
                  className={`rounded-xl border p-3 text-sm ${isOpen ? "border-indigo-400 bg-indigo-500/10" : done ? "border-zinc-800 opacity-60" : "border-zinc-800"}`}
                >
                  <div className="mb-1 flex items-center gap-2 text-xs text-zinc-500">
                    <span className="font-mono">{q.question_id}</span>
                    <span>{q.category}</span>
                    {done && <span className="text-emerald-400">✓</span>}
                  </div>
                  <p className="mb-2">{q.text}</p>
                  <div className="flex gap-2">
                    <button className="btn btn-primary px-3 py-1 text-xs" disabled={!recording || isOpen} onClick={() => ask(q).catch((e) => setError(e.message))}>
                      Ask
                    </button>
                    <button
                      className="btn btn-ghost px-3 py-1 text-xs"
                      disabled={!recording || !isOpen}
                      onClick={() => answerEnded(q.question_id).catch((e) => setError(e.message))}
                    >
                      Answer ended
                    </button>
                  </div>
                </li>
              );
            })}
          </ol>
        </aside>
      </div>
    </div>
  );
}

export default function InterviewerPage() {
  return (
    <Suspense fallback={<p className="p-6 text-zinc-400">Loading…</p>}>
      <InterviewerInner />
    </Suspense>
  );
}
