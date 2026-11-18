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
  const [copied, setCopied] = useState(false);

  const room = useRef<Room | null>(null);
  const micTrack = useRef<MediaStreamTrack | null>(null);
  const micRec = useRef<PcmRecorder | null>(null);
  const audioEls = useRef<HTMLMediaElement[]>([]);
  const { summary, refresh } = useSession(code);

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

  const copyLink = () => {
    if (!candidateLink) return;
    navigator.clipboard.writeText(candidateLink);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const shown = useMemo(() => questions.filter((q) => cat === "all" || q.category === cat), [questions, cat]);
  const candidateLink = typeof window !== "undefined" && code ? `${location.origin}/candidate?room=${code}` : "";
  const recording = summary?.status === "recording";
  const candidateJoined = !!summary?.participants.candidate;
  const candidateInRoom = peers.length > 0;
  const candidateTracksOpen = Object.entries(summary?.tracks ?? {}).some(([, t]) => t && t.end_ms === null);
  const canFinalize = summary?.status === "stopped" || summary?.status === "finalized";

  const elapsedStr = `${Math.floor(elapsed / 60)}:${String(Math.floor(elapsed % 60)).padStart(2, "0")}`;

  if (!code) return <p className="p-6 text-zinc-400">{error ?? "Creating room…"}</p>;
  if (!joined) return <ConsentForm role="interviewer" onSubmit={join} />;

  return (
    <div className="flex min-h-screen flex-col gap-4 p-4">
      {/* Header */}
      <header className="card flex flex-wrap items-center gap-3">
        <div className="mr-auto">
          <div className="text-xs text-zinc-500">Room code</div>
          <div className="font-mono text-2xl font-bold tracking-widest text-white">{code}</div>
        </div>

        {/* Candidate status */}
        <div className="flex items-center gap-2 rounded-xl border border-zinc-800 px-3 py-2">
          <span className={`h-2 w-2 rounded-full ${candidateInRoom ? "bg-green-400" : candidateJoined ? "bg-amber-400" : "bg-zinc-600"}`} />
          <span className="text-sm text-zinc-400">
            {candidateInRoom ? `Candidate connected` : candidateJoined ? "Candidate joined" : "Waiting for candidate"}
          </span>
        </div>

        <button className="btn btn-ghost gap-2" onClick={copyLink}>
          {copied ? (
            <>
              <svg className="h-4 w-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
              </svg>
              Copied!
            </>
          ) : (
            <>
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 0 1 1.242 7.244l-4.5 4.5a4.5 4.5 0 0 1-6.364-6.364l1.757-1.757m13.35-.622 1.757-1.757a4.5 4.5 0 0 0-6.364-6.364l-4.5 4.5a4.5 4.5 0 0 0 1.242 7.244" />
              </svg>
              Copy candidate link
            </>
          )}
        </button>

        {recording ? (
          <>
            <span className="flex items-center gap-2 rounded-xl bg-rose-500/10 px-3 py-1.5 text-sm font-medium text-rose-300 ring-1 ring-rose-500/20">
              <span className="h-2 w-2 animate-pulse rounded-full bg-rose-500" />
              REC {elapsedStr}
            </span>
            <button
              className="btn btn-ghost"
              disabled={busy}
              onClick={() => room.current && send(room.current, { type: "calibrate" })}
            >
              Calibrate gaze
            </button>
            <button className="btn btn-danger" disabled={busy} onClick={stopRecording}>
              Stop recording
            </button>
          </>
        ) : summary?.status === "created" ? (
          <button
            className="btn btn-primary"
            disabled={busy || !candidateJoined}
            title={!candidateJoined ? "Waiting for candidate to join" : undefined}
            onClick={startRecording}
          >
            Start recording
          </button>
        ) : (
          <button
            className="btn btn-primary"
            disabled={busy || !canFinalize || candidateTracksOpen}
            onClick={finalize}
          >
            {candidateTracksOpen ? (
              <><span className="h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white" /> Waiting for uploads…</>
            ) : "Finalize & check"}
          </button>
        )}
      </header>

      {error && (
        <div className="card border-rose-800 bg-rose-950/30 text-sm text-rose-300">{error}</div>
      )}

      <div className="grid flex-1 grid-cols-1 gap-4 xl:grid-cols-[1fr_400px]">
        {/* Left: video + status */}
        <section className="flex flex-col gap-4">
          {/* View toggle */}
          <div className="flex gap-2">
            {(["both", "camera", "screen"] as const).map((v) => (
              <button
                key={v}
                className={`btn text-xs ${view === v ? "btn-primary" : "btn-ghost"}`}
                onClick={() => setView(v)}
              >
                {v === "both" ? "Side by side" : v === "camera" ? "Camera" : "Screen"}
              </button>
            ))}
          </div>

          {/* Candidate video feeds */}
          <div className={`grid gap-4 ${view === "both" ? "md:grid-cols-2" : ""}`}>
            {view !== "screen" && <MediaSlot el={camEl} label="Candidate camera" className="aspect-video" />}
            {view !== "camera" && <MediaSlot el={screenEl} label="Candidate screen" className="aspect-video" />}
          </div>

          {/* Self + track status row */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-[200px_1fr]">
            <MediaSlot stream={selfStream} muted label="You" className="aspect-video" />

            <div className="card">
              <h3 className="mb-3 text-sm font-medium">Track status</h3>
              <div className="grid grid-cols-2 gap-2">
                {(["candidate_webcam", "candidate_mic", "candidate_screen", "interviewer_mic"] as const).map((t) => {
                  const s = summary?.tracks[t];
                  const label = t.replace("candidate_", "").replace("interviewer_", "interviewer ").replace("_", " ");
                  const active = s && s.end_ms === null;
                  const done = s && s.end_ms !== null;
                  return (
                    <div key={t} className="flex items-center gap-2 rounded-lg border border-zinc-800 px-3 py-2">
                      <span className={`h-2 w-2 shrink-0 rounded-full ${active ? "animate-pulse bg-rose-500" : done ? "bg-emerald-500" : "bg-zinc-700"}`} />
                      <span className="text-xs capitalize text-zinc-300">{label}</span>
                      {done && <span className="ml-auto text-xs text-zinc-500">{fmtBytes(s.bytes)}</span>}
                    </div>
                  );
                })}
              </div>
              <div className="mt-3 flex gap-4 text-xs text-zinc-500">
                <span>Calibration: {summary?.calibration_saved ? <span className="text-emerald-400">saved</span> : "not yet"}</span>
                {micStatus?.error && <span className="text-rose-400">Mic upload error: {micStatus.error}</span>}
              </div>
            </div>
          </div>

          {/* Finalize report */}
          {report && (
            <div className={`card ${report.passed ? "border-emerald-700 bg-emerald-950/20" : "border-rose-700 bg-rose-950/20"}`}>
              <div className="flex items-center gap-3">
                {report.passed ? (
                  <svg className="h-5 w-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                  </svg>
                ) : (
                  <svg className="h-5 w-5 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                  </svg>
                )}
                <div>
                  <div className="font-medium">
                    Session check: <span className={report.passed ? "text-emerald-400" : "text-rose-400"}>{report.passed ? "Passed" : "Failed"}</span>
                  </div>
                  {report.errors.length > 0 && (
                    <div className="mt-1 text-sm text-rose-300">{report.errors.join(" · ")}</div>
                  )}
                  {report.warnings.length > 0 && (
                    <div className="mt-1 text-sm text-amber-400">{report.warnings.join(" · ")}</div>
                  )}
                  {report.passed && <div className="mt-1 text-sm text-zinc-400">Session is ready for analysis.</div>}
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Right: questions panel */}
        <aside className="card flex max-h-[calc(100vh-120px)] flex-col">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-medium">Questions</h2>
            <span className="text-xs text-zinc-500">{doneQ.size} / {questions.length} asked</span>
          </div>

          <div className="mb-3 flex gap-1 flex-wrap">
            {(["all", "hr", "behavioural", "technical"] as const).map((c) => (
              <button
                key={c}
                className={`btn px-3 py-1 text-xs capitalize ${cat === c ? "btn-primary" : "btn-ghost"}`}
                onClick={() => setCat(c)}
              >
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
                  className={`rounded-xl border p-3 text-sm transition ${
                    isOpen
                      ? "border-indigo-500/60 bg-indigo-500/10"
                      : done
                      ? "border-zinc-800/50 opacity-50"
                      : "border-zinc-800 hover:border-zinc-700"
                  }`}
                >
                  <div className="mb-1.5 flex items-center gap-2 text-xs text-zinc-500">
                    <span className="capitalize">{q.category}</span>
                    <span>·</span>
                    <span>Difficulty {q.difficulty}</span>
                    {done && (
                      <span className="ml-auto flex items-center gap-1 text-emerald-400">
                        <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                        </svg>
                        Done
                      </span>
                    )}
                    {isOpen && (
                      <span className="ml-auto flex items-center gap-1 text-indigo-400">
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-400" />
                        Active
                      </span>
                    )}
                  </div>
                  <p className="mb-3 leading-snug text-zinc-200">{q.text}</p>
                  <div className="flex gap-2">
                    <button
                      className="btn btn-primary px-3 py-1 text-xs"
                      disabled={!recording || isOpen || done}
                      onClick={() => ask(q).catch((e) => setError(e.message))}
                    >
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
