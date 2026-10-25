"use client";

/* Candidate room: plain video call. This page never opens the analytics channel and imports nothing from
   lib/analytics - candidates see no scores (the backend also rejects candidate tokens on /ws/analytics). */
import { Mic, MicOff, MonitorUp, PhoneOff, ShieldCheck, Video, VideoOff } from "lucide-react";
import { Track, type Room } from "livekit-client";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Header, VideoSlot } from "@/components/shell";
import { Badge, Button, Card, mmss } from "@/components/ui";
import { api } from "@/lib/api";
import { joinRoom, roleOf } from "@/lib/room";

export default function CandidateRoom() {
  const id = Number(useParams<{ id: string }>().id);
  const router = useRouter();
  const room = useRef<Room | null>(null);
  const [remote, setRemote] = useState<HTMLMediaElement | null>(null);
  const [self, setSelf] = useState<HTMLMediaElement | null>(null);
  const [mic, setMic] = useState(true);
  const [cam, setCam] = useState(true);
  const [sharing, setSharing] = useState(false);
  const [checks, setChecks] = useState<{ cam: boolean | null; mic: boolean | null }>({ cam: null, mic: null });
  const [level, setLevel] = useState(0);
  const [joined, setJoined] = useState(false);
  const [t0, setT0] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const [error, setError] = useState<string | null>(null);
  const [title, setTitle] = useState("");

  // device check before joining
  useEffect(() => {
    let stream: MediaStream | null = null;
    let raf = 0;
    navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then((s) => {
      stream = s;
      setChecks({ cam: s.getVideoTracks().length > 0, mic: s.getAudioTracks().length > 0 });
      const ctx = new AudioContext();
      const an = ctx.createAnalyser();
      ctx.createMediaStreamSource(s).connect(an);
      const buf = new Uint8Array(an.fftSize);
      const tick = () => {
        an.getByteTimeDomainData(buf);
        setLevel(Math.min(1, Math.max(...buf.map((v) => Math.abs(v - 128))) / 64));
        raf = requestAnimationFrame(tick);
      };
      tick();
    }).catch(() => setChecks({ cam: false, mic: false }));
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
    try {
      const r = await joinRoom(id, {
        onTrack: (kind, el, p) => {
          if (kind === "camera" && roleOf(p) === "interviewer") setRemote(el);
          if (kind === "audio" && el) document.body.appendChild(el);
        },
      });
      room.current = r;
      const pub = r.localParticipant.getTrackPublication(Track.Source.Camera);
      setSelf(pub?.videoTrack?.attach() ?? null);
      setJoined(true);
      setT0(Date.now());
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function toggleShare() {
    if (!room.current) return;
    await room.current.localParticipant.setScreenShareEnabled(!sharing, { audio: false });
    setSharing(!sharing);
  }

  return (
    <>
      <Header right={joined && <Badge tone="good">{mmss(t0 ? (now - t0) / 1000 : 0)}</Badge>} />
      <main className="mx-auto max-w-6xl space-y-4 px-4 py-6">
        <div className="flex items-center gap-2 rounded-2xl border border-line bg-accent-soft px-4 py-2 text-sm">
          <ShieldCheck size={16} className="text-accent" />
          This call is recorded and analysed for your feedback report (you consented on the previous page). You will not see live scores.
        </div>
        <h1 className="text-xl font-semibold">{title}</h1>
        {!joined ? (
          <Card className="space-y-4">
            <h2 className="font-medium">Camera and microphone check</h2>
            <div className="flex gap-4 text-sm">
              <Badge tone={checks.cam ? "good" : checks.cam === false ? "bad" : "neutral"}>camera {checks.cam === null ? "..." : checks.cam ? "ok" : "not found"}</Badge>
              <Badge tone={checks.mic ? "good" : checks.mic === false ? "bad" : "neutral"}>microphone {checks.mic === null ? "..." : checks.mic ? "ok" : "not found"}</Badge>
            </div>
            <div className="h-2 w-64 overflow-hidden rounded-full bg-surface-2"><div className="h-full bg-good transition-all" style={{ width: `${level * 100}%` }} /></div>
            <p className="text-xs text-muted">Speak to see the microphone level move. Sit about an arm&apos;s length from the screen with your face well lit.</p>
            {error && <p className="text-sm text-bad">{error}</p>}
            <Button onClick={join} disabled={!checks.cam || !checks.mic}>Join interview</Button>
          </Card>
        ) : (
          <>
            <div className="grid gap-4 md:grid-cols-[1fr_280px]">
              <VideoSlot el={remote} label="Interviewer" className="aspect-video" />
              <VideoSlot el={self} label="You" className="aspect-video md:aspect-auto" />
            </div>
            <div className="flex justify-center gap-3">
              <Button variant="outline" onClick={async () => { await room.current?.localParticipant.setMicrophoneEnabled(!mic); setMic(!mic); }}>
                {mic ? <Mic size={16} /> : <MicOff size={16} />} {mic ? "Mute" : "Unmute"}
              </Button>
              <Button variant="outline" onClick={async () => { await room.current?.localParticipant.setCameraEnabled(!cam); setCam(!cam); }}>
                {cam ? <Video size={16} /> : <VideoOff size={16} />} Camera
              </Button>
              <Button variant={sharing ? "good" : "outline"} onClick={toggleShare}><MonitorUp size={16} /> {sharing ? "Stop sharing" : "Share screen"}</Button>
              <Button variant="danger" onClick={() => { room.current?.disconnect(); router.push("/"); }}><PhoneOff size={16} /> Leave</Button>
            </div>
          </>
        )}
      </main>
    </>
  );
}
