"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Track, type Room } from "livekit-client";
import Calibration, { type CalibrationResult } from "@/components/Calibration";
import ConsentForm, { CONSENT_VERSION, type ParticipantInfo } from "@/components/ConsentForm";
import MediaSlot from "@/components/MediaSlot";
import { api } from "@/lib/api";
import { clock } from "@/lib/clock";
import { PcmRecorder, VideoTrackRecorder, type RecorderStatus } from "@/lib/recorders";
import { joinRoom, publish, type ControlMessage } from "@/lib/room";
import { fmtBytes, useSession } from "@/lib/useSession";

function CandidateInner() {
  const params = useSearchParams();
  const [code, setCode] = useState<string | null>(params.get("room")?.toUpperCase() ?? null);
  const [codeInput, setCodeInput] = useState("");
  const [joined, setJoined] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selfStream, setSelfStream] = useState<MediaStream | null>(null);
  const [interviewerEl, setInterviewerEl] = useState<HTMLMediaElement | null>(null);
  const [sharing, setSharing] = useState(false);
  const [recording, setRecording] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [status, setStatus] = useState<Record<string, RecorderStatus>>({});
  const [camInfo, setCamInfo] = useState("");

  const room = useRef<Room | null>(null);
  const cam = useRef<MediaStreamTrack | null>(null);
  const mic = useRef<MediaStreamTrack | null>(null);
  const screen = useRef<MediaStreamTrack | null>(null);
  const recs = useRef<Record<string, VideoTrackRecorder | PcmRecorder>>({});
  const recState = useRef<"idle" | "recording" | "stopped">("idle");
  const audioEls = useRef<HTMLMediaElement[]>([]);
  const { summary } = useSession(joined ? code : null);

  const onStatus = useCallback((s: RecorderStatus) => setStatus((p) => ({ ...p, [s.track]: s })), []);
  const fail = (e: unknown) => setError((e as Error).message);

  const startScreenRecorder = useCallback(async () => {
    if (!code || !screen.current || recs.current.candidate_screen) return;
    const r = new VideoTrackRecorder(code, "candidate_screen", screen.current, onStatus, 1_500_000);
    recs.current.candidate_screen = r;
    await r.start();
  }, [code, onStatus]);

  const startRecording = useCallback(async () => {
    if (!code || recState.current !== "idle" || !cam.current || !mic.current) return;
    recState.current = "recording";
    setRecording(true);
    try {
      await clock.sync(6);
      await api.clock(code, "candidate", clock.offsetMs, clock.rttMs);
      const webcam = new VideoTrackRecorder(code, "candidate_webcam", cam.current, onStatus);
      const micRec = new PcmRecorder(code, "candidate_mic", mic.current, onStatus);
      recs.current.candidate_webcam = webcam;
      recs.current.candidate_mic = micRec;
      await Promise.all([webcam.start(), micRec.start()]);
      if (screen.current) await startScreenRecorder();
    } catch (e) {
      fail(e);
    }
  }, [code, onStatus, startScreenRecorder]);

  const stopRecording = useCallback(async () => {
    if (recState.current !== "recording") return;
    recState.current = "stopped";
    setRecording(false);
    setCalibrating(false);
    try {
      await Promise.all(Object.values(recs.current).map((r) => r.stop()));
    } catch (e) {
      fail(e);
    }
  }, []);

  // Fallback if a data message is missed: follow the server's session status.
  useEffect(() => {
    if (summary?.status === "recording") startRecording();
    if (summary?.status === "stopped" || summary?.status === "finalized") stopRecording();
  }, [summary?.status, startRecording, stopRecording]);

  useEffect(
    () => () => {
      room.current?.disconnect();
      audioEls.current.forEach((e) => e.remove());
    },
    [],
  );

  const onMessage = useCallback(
    (msg: ControlMessage) => {
      if (msg.type === "rec_start") startRecording();
      else if (msg.type === "rec_stop") stopRecording();
      else if (msg.type === "calibrate" && recState.current === "recording") setCalibrating(true);
    },
    [startRecording, stopRecording],
  );
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const join = async (info: ParticipantInfo) => {
    if (!code) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 15, max: 15 } },
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });
    cam.current = stream.getVideoTracks()[0];
    mic.current = stream.getAudioTracks()[0];
    const s = cam.current.getSettings();
    setCamInfo(`${s.width}x${s.height} @ ${s.frameRate?.toFixed?.(0) ?? s.frameRate} fps`);
    setSelfStream(new MediaStream([cam.current]));

    await api.register(code, {
      role: "candidate",
      ...info,
      consent: { agreed: true, version: CONSENT_VERSION },
      user_agent: navigator.userAgent,
    });
    const c = await clock.sync();
    await api.clock(code, "candidate", c.offset, c.rtt);
    const r = await joinRoom(code, "candidate", `candidate-${info.participant_id}`, {
      onTrack: (kind, el) => {
        if (kind === "audio") {
          if (el) {
            el.style.display = "none";
            document.body.appendChild(el);
            audioEls.current.push(el);
          }
        } else if (kind === "camera") setInterviewerEl(el);
      },
      onMessage: (m) => onMessageRef.current(m),
      onParticipants: () => undefined,
    });
    room.current = r;
    await publish(r, cam.current, Track.Source.Camera);
    await publish(r, mic.current, Track.Source.Microphone);
    setJoined(true);
  };

  const shareScreen = async () => {
    try {
      const ds = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 15 }, audio: false });
      const track = ds.getVideoTracks()[0];
      screen.current = track;
      setSharing(true);
      track.addEventListener("ended", () => {
        setSharing(false);
        screen.current = null;
        room.current?.localParticipant.unpublishTrack(track).catch(() => undefined);
      });
      if (room.current) await publish(room.current, track, Track.Source.ScreenShare);
      if (recState.current === "recording") await startScreenRecorder();
    } catch (e) {
      if ((e as Error).name !== "NotAllowedError") fail(e);
    }
  };

  const onCalibrationDone = useCallback(
    async (r: CalibrationResult) => {
      setCalibrating(false);
      if (!code) return;
      try {
        await api.calibration(code, r as unknown as Record<string, unknown>);
      } catch (e) {
        fail(e);
      }
    },
    [code],
  );

  if (!code)
    return (
      <div className="mx-auto max-w-sm p-6">
        <div className="card space-y-3">
          <h1 className="text-lg font-semibold">Join interview</h1>
          <input className="input font-mono uppercase" placeholder="Room code" value={codeInput} onChange={(e) => setCodeInput(e.target.value)} />
          <button className="btn btn-primary w-full" disabled={codeInput.trim().length < 4} onClick={() => setCode(codeInput.trim().toUpperCase())}>
            Continue
          </button>
        </div>
      </div>
    );
  if (!joined) return <ConsentForm role="candidate" onSubmit={join} />;

  const screenRec = status.candidate_screen;
  const screenRecorded = summary?.tracks.candidate_screen;

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-4 p-4">
      <header className="card flex flex-wrap items-center gap-3">
        <div className="mr-auto">
          <div className="text-xs text-zinc-400">Interview room</div>
          <div className="font-mono text-xl tracking-widest">{code}</div>
        </div>
        {recording ? (
          <span className="flex items-center gap-2 rounded-xl bg-rose-500/10 px-3 py-1 text-sm text-rose-300">
            <span className="h-2 w-2 animate-pulse rounded-full bg-rose-500" /> Recording
          </span>
        ) : (
          <span className="text-sm text-zinc-400">{recState.current === "stopped" ? "Recording finished — thank you!" : "Waiting for the interviewer to start"}</span>
        )}
        <button className="btn btn-ghost" onClick={shareScreen} disabled={sharing || (!!screenRecorded && !screenRec)}>
          {sharing ? "Sharing screen" : "Share screen"}
        </button>
        {recording && (
          <button className="btn btn-ghost" onClick={() => setCalibrating(true)}>
            Calibrate gaze
          </button>
        )}
      </header>

      {error && <div className="card border-rose-800 text-sm text-rose-300">{error}</div>}
      {sharing && recording && (
        <div className="card border-amber-800 text-sm text-amber-200">
          Please keep screen sharing on until the interview ends; it can only be recorded once per session.
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <MediaSlot el={interviewerEl} label="Interviewer" className="aspect-video" />
        <MediaSlot stream={selfStream} muted label={`You · ${camInfo}`} className="aspect-video" />
      </div>

      <div className="card text-xs text-zinc-400">
        Uploads:{" "}
        {Object.values(status).length === 0
          ? "none yet"
          : Object.values(status)
              .map((s) => `${s.track} ${s.state} ${s.chunks} chunks ${fmtBytes(s.bytes)}${s.pending > 1 ? ` (${s.pending} queued)` : ""}${s.error ? ` ERROR ${s.error}` : ""}`)
              .join(" · ")}
      </div>

      {calibrating && <Calibration onDone={onCalibrationDone} onCancel={() => setCalibrating(false)} />}
    </div>
  );
}

export default function CandidatePage() {
  return (
    <Suspense fallback={<p className="p-6 text-zinc-400">Loading…</p>}>
      <CandidateInner />
    </Suspense>
  );
}
