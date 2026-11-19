"use client";

import { Mic, MicOff, Video, VideoOff } from "lucide-react";
import { RoomEvent, type Room } from "livekit-client";
import { useCallback, useEffect, useState } from "react";
import { cn } from "@/components/ui";

type Kind = "mic" | "cam";

/** Mute / camera buttons for the call. They read the real state of your devices from the call (not a guess), so they
 *  stay right when a device failed to start or was switched off from elsewhere. Renders two buttons; the page decides
 *  the wrapper. `dark` is for buttons over video, `light` for the normal page. */
export function CallControls({ room, tone = "light", mic }: { room: Room | null; tone?: "dark" | "light"; mic?: { echoCancellation?: boolean; noiseSuppression?: boolean; autoGainControl?: boolean } }) {
  const [on, setOn] = useState<Record<Kind, boolean>>({ mic: true, cam: true });
  const [busy, setBusy] = useState<Kind | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sync = useCallback(() => {
    if (room) setOn({ mic: room.localParticipant.isMicrophoneEnabled, cam: room.localParticipant.isCameraEnabled });
  }, [room]);

  useEffect(() => {
    if (!room) return;
    sync();
    room.on(RoomEvent.TrackMuted, sync).on(RoomEvent.TrackUnmuted, sync)
      .on(RoomEvent.LocalTrackPublished, sync).on(RoomEvent.LocalTrackUnpublished, sync);
    return () => {
      room.off(RoomEvent.TrackMuted, sync).off(RoomEvent.TrackUnmuted, sync)
        .off(RoomEvent.LocalTrackPublished, sync).off(RoomEvent.LocalTrackUnpublished, sync);
    };
  }, [room, sync]);

  async function toggle(kind: Kind) {
    if (!room || busy) return;
    setBusy(kind);
    setError(null);
    try {
      if (kind === "mic") await room.localParticipant.setMicrophoneEnabled(!on.mic, mic ?? { echoCancellation: true, noiseSuppression: false });
      else await room.localParticipant.setCameraEnabled(!on.cam, { resolution: { width: 1280, height: 720, frameRate: 15 } });
    } catch {
      setError(kind === "mic" ? "Could not start the microphone. Another app or tab may be using it." : "Could not start the camera. Another app or tab may be using it.");
    } finally {
      setBusy(null);
      sync();
    }
  }

  const btn = (kind: Kind, label: string, icon: React.ReactNode) => (
    <button key={kind} type="button" onClick={() => toggle(kind)} disabled={!room || busy === kind} aria-label={label} aria-pressed={!on[kind]} title={label}
      className={cn("grid place-items-center rounded-full transition disabled:opacity-60",
        tone === "dark"
          ? cn("h-12 w-12", on[kind] ? "bg-white/10 text-white hover:bg-white/20" : "bg-white text-[#141413]")
          : cn("h-10 w-10 border", on[kind] ? "border-border bg-surface text-fg hover:bg-surface-2" : "border-transparent bg-bad/10 text-bad"))}>
      {icon}
    </button>
  );

  return (
    <>
      {btn("mic", on.mic ? "Mute microphone" : "Unmute microphone", on.mic ? <Mic size={19} /> : <MicOff size={19} />)}
      {btn("cam", on.cam ? "Turn camera off" : "Turn camera on", on.cam ? <Video size={19} /> : <VideoOff size={19} />)}
      {error && <span role="alert" className="absolute bottom-full left-1/2 mb-2 w-max max-w-64 -translate-x-1/2 rounded-xl bg-bad px-3 py-1.5 text-xs text-white">{error}</span>}
    </>
  );
}
