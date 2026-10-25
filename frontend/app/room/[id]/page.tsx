"use client";

/* Candidate room: a plain video call. This page never opens the analytics channel and imports nothing from
   lib/analytics - candidates see no scores (the backend also rejects candidate tokens on /ws/analytics). */
import { Mic, MicOff, MonitorUp, PhoneOff, Video, VideoOff } from "lucide-react";
import { Track, type Room } from "livekit-client";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Logo, VideoSlot } from "@/components/shell";
import { Button, cn, Dot, mmss } from "@/components/ui";
import { api } from "@/lib/api";
import { joinRoom, roleOf } from "@/lib/room";

function Control({ on, label, onClick, children, danger = false }: { on: boolean; label: string; onClick: () => void; children: React.ReactNode; danger?: boolean }) {
  return (
    <button onClick={onClick} aria-label={label} title={label}
      className={cn("grid h-12 w-12 place-items-center rounded-full transition",
        danger ? "bg-bad text-white hover:opacity-90" : on ? "bg-white/10 text-white hover:bg-white/20" : "bg-white text-[#141413]")}>
      {children}
    </button>
  );
}

export default function CandidateRoom() {
  const id = Number(useParams<{ id: string }>().id);
  const router = useRouter();
  const room = useRef<Room | null>(null);
  const preview = useRef<HTMLVideoElement | null>(null);
  const [remote, setRemote] = useState<HTMLMediaElement | null>(null);
  const [self, setSelf] = useState<HTMLMediaElement | null>(null);
  const [mic, setMic] = useState(true);
  const [cam, setCam] = useState(true);
  const [sharing, setSharing] = useState(false);
  const [ok, setOk] = useState<{ cam: boolean | null; mic: boolean | null }>({ cam: null, mic: null });
  const [level, setLevel] = useState(0);
  const [joined, setJoined] = useState(false);
  const [joining, setJoining] = useState(false);
  const [t0, setT0] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const [error, setError] = useState<string | null>(null);
  const [title, setTitle] = useState("");

  useEffect(() => {
    let stream: MediaStream | null = null;
    let raf = 0;
    navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then((s) => {
      stream = s;
      setOk({ cam: s.getVideoTracks().length > 0, mic: s.getAudioTracks().length > 0 });
      if (preview.current) { preview.current.srcObject = s; preview.current.play().catch(() => undefined); }
      const ctx = new AudioContext();
      const an = ctx.createAnalyser();
      ctx.createMediaStreamSource(s).connect(an);
      const buf = new Uint8Array(an.fftSize);
      const tick = () => {
        an.getByteTimeDomainData(buf);
        setLevel(Math.min(1, Math.max(...buf.map((v) => Math.abs(v - 128))) / 48));
        raf = requestAnimationFrame(tick);
      };
      tick();
    }).catch(() => setOk({ cam: false, mic: false }));
    api.interview(id).then((iv) => setTitle(iv.title)).catch(() => router.replace("/"));
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
      cancelAnimationFrame(raf);
      clearInterval(timer);
      room.current?.disconnect();
    };
  }, [id, router]);

  async function join() {
    setJoining(true);
    setError(null);
    try {
      const r = await joinRoom(id, {
        onTrack: (kind, el, p) => {
          if (kind === "camera" && roleOf(p) === "interviewer") setRemote(el);
          if (kind === "audio" && el) document.body.appendChild(el);
        },
      });
      room.current = r;
      setSelf(r.localParticipant.getTrackPublication(Track.Source.Camera)?.videoTrack?.attach() ?? null);
      setJoined(true);
      setT0(Date.now());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setJoining(false);
    }
  }

  if (!joined) {
    return (
      <main className="flex min-h-screen flex-col px-6 py-8">
        <Logo />
        <div className="mx-auto grid w-full max-w-5xl flex-1 items-center gap-12 py-10 lg:grid-cols-[1.3fr_1fr]">
          <div className="relative aspect-video overflow-hidden rounded-3xl bg-[#121211]">
            <video ref={preview} muted playsInline className="h-full w-full -scale-x-100 object-cover" />
            <div className="absolute bottom-4 left-4 flex h-1.5 w-28 overflow-hidden rounded-full bg-white/15">
              <div className="bg-white/90 transition-all" style={{ width: `${level * 100}%` }} />
            </div>
          </div>
          <div>
            <p className="text-[13px] text-muted">{title}</p>
            <h1 className="mt-2 font-display text-3xl tracking-tight">Ready to join?</h1>
            <p className="mt-3 text-[15px] leading-relaxed text-muted">Check that you can see yourself and the bar moves when you speak. Sit about an arm&apos;s length away, with light on your face.</p>
            <div className="mt-6 space-y-2 text-sm">
              <p className="flex items-center gap-2.5"><Dot tone={ok.cam ? "good" : ok.cam === false ? "bad" : "neutral"} /> Camera {ok.cam === null ? "checking…" : ok.cam ? "ready" : "not found"}</p>
              <p className="flex items-center gap-2.5"><Dot tone={ok.mic ? "good" : ok.mic === false ? "bad" : "neutral"} /> Microphone {ok.mic === null ? "checking…" : ok.mic ? "ready" : "not found"}</p>
            </div>
            {error && <p className="mt-5 rounded-2xl bg-bad/10 px-4 py-3 text-sm text-bad">{error}. Please try again.</p>}
            <Button size="lg" className="mt-8 w-full" onClick={join} disabled={!ok.cam || !ok.mic || joining}>{joining ? "Connecting…" : "Join interview"}</Button>
            <p className="mt-4 text-[13px] text-faint">The call is recorded with your consent. You won&apos;t see scores during the call.</p>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="relative flex h-screen flex-col bg-[#0d0d0c] text-white">
      <div className="flex items-center justify-between px-6 py-4">
        <span className="text-sm text-white/70">{title}</span>
        <span className="flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-[13px] tabular-nums"><Dot tone="bad" pulse /> {mmss(t0 ? (now - t0) / 1000 : 0)}</span>
      </div>
      <div className="relative flex-1 px-6 pb-28">
        <VideoSlot el={remote} label="Interviewer" className="h-full w-full" />
        <div className="absolute right-10 bottom-32 w-56 overflow-hidden rounded-2xl shadow-2xl ring-1 ring-white/10">
          <VideoSlot el={self} label="You" muted className="aspect-video rounded-2xl" />
        </div>
      </div>
      <div className="absolute inset-x-0 bottom-6 flex justify-center">
        <div className="flex items-center gap-3 rounded-full bg-white/5 px-4 py-3 ring-1 ring-white/10 backdrop-blur-xl">
          <Control on={mic} label={mic ? "Mute" : "Unmute"} onClick={async () => { await room.current?.localParticipant.setMicrophoneEnabled(!mic); setMic(!mic); }}>
            {mic ? <Mic size={19} /> : <MicOff size={19} />}
          </Control>
          <Control on={cam} label={cam ? "Turn camera off" : "Turn camera on"} onClick={async () => { await room.current?.localParticipant.setCameraEnabled(!cam); setCam(!cam); }}>
            {cam ? <Video size={19} /> : <VideoOff size={19} />}
          </Control>
          <Control on={!sharing} label={sharing ? "Stop sharing" : "Share screen"} onClick={async () => { await room.current?.localParticipant.setScreenShareEnabled(!sharing, { audio: false }); setSharing(!sharing); }}>
            <MonitorUp size={19} />
          </Control>
          <span className="mx-1 h-6 w-px bg-white/10" />
          <Control on danger label="Leave" onClick={() => { room.current?.disconnect(); router.push("/"); }}><PhoneOff size={19} /></Control>
        </div>
      </div>
    </main>
  );
}
