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
import { useSession } from "@/lib/useSession";

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

  const hasUploadError = Object.values(status).some((s) => s.error);

  if (!code)
    return (
      <div className="flex min-h-screen items-center justify-center p-6">
        <div className="w-full max-w-sm">
          <div className="card space-y-5">
            <div>
              <h1 className="text-xl font-semibold">Join interview</h1>
              <p className="mt-1 text-sm text-zinc-500">Enter the room code from your interviewer</p>
            </div>
            <input
              className="input font-mono text-center text-lg tracking-widest uppercase"
              placeholder="e.g. ABC123"
              value={codeInput}
              onChange={(e) => setCodeInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && codeInput.trim().length >= 4 && setCode(codeInput.trim().toUpperCase())}
              autoFocus
            />
            <button
              className="btn btn-primary w-full py-2.5"
              disabled={codeInput.trim().length < 4}
              onClick={() => setCode(codeInput.trim().toUpperCase())}
            >
              Continue
            </button>
          </div>
        </div>
      </div>
    );

  if (!joined) return <ConsentForm role="candidate" onSubmit={join} />;

  const screenRecorded = summary?.tracks.candidate_screen;

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4 p-4">
      <header className="card flex flex-wrap items-center gap-3">
        <div className="mr-auto flex items-center gap-3">
          <div>
            <div className="text-xs text-zinc-500">Room</div>
            <div className="font-mono text-lg font-semibold tracking-widest">{code}</div>
          </div>
        </div>

        {recording ? (
          <span className="flex items-center gap-2 rounded-xl bg-rose-500/10 px-3 py-1.5 text-sm font-medium text-rose-300 ring-1 ring-rose-500/20">
            <span className="h-2 w-2 animate-pulse rounded-full bg-rose-500" />
            Recording in progress
          </span>
        ) : recState.current === "stopped" ? (
          <span className="flex items-center gap-2 rounded-xl bg-emerald-500/10 px-3 py-1.5 text-sm text-emerald-300 ring-1 ring-emerald-500/20">
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
            </svg>
            Interview complete — thank you!
          </span>
        ) : (
          <span className="text-sm text-zinc-500">Waiting for interviewer to start…</span>
        )}

        <button
          className="btn btn-ghost"
          onClick={shareScreen}
          disabled={sharing || (!!screenRecorded && !status.candidate_screen)}
        >
          {sharing ? (
            <>
              <span className="h-2 w-2 rounded-full bg-green-400" />
              Sharing screen
            </>
          ) : (
            <>
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 17.25v1.007a3 3 0 0 1-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0 1 15 18.257V17.25m6-12V15a2.25 2.25 0 0 1-2.25 2.25H5.25A2.25 2.25 0 0 1 3 15V5.25m18 0A2.25 2.25 0 0 0 18.75 3H5.25A2.25 2.25 0 0 0 3 5.25m18 0H3" />
              </svg>
              Share screen
            </>
          )}
        </button>

        {recording && (
          <button className="btn btn-ghost" onClick={() => setCalibrating(true)}>
            Calibrate gaze
          </button>
        )}
      </header>

      {error && (
        <div className="card border-rose-800 bg-rose-950/30 text-sm text-rose-300">{error}</div>
      )}
      {sharing && recording && (
        <div className="card border-amber-800 bg-amber-950/30 text-sm text-amber-300">
          Keep screen sharing on until the interview ends — it can only be recorded once per session.
        </div>
      )}
      {hasUploadError && (
        <div className="card border-rose-800 bg-rose-950/30 text-sm text-rose-300">
          Upload error — check your connection. Data is being retried automatically.
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <MediaSlot el={interviewerEl} label="Interviewer" className="aspect-video" />
        <MediaSlot stream={selfStream} muted label="You" className="aspect-video" />
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
