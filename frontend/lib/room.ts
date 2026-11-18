"use client";

import { Room, RoomEvent, Track, type RemoteParticipant, type RemoteTrack } from "livekit-client";
import { api, livekitUrl } from "./api";

export type Kind = "camera" | "screen" | "audio";

export interface RoomHandlers {
  onTrack: (kind: Kind, el: HTMLMediaElement | null, participant: RemoteParticipant) => void;
  onParticipants?: (ps: RemoteParticipant[]) => void;
}

function kindOf(track: RemoteTrack): Kind {
  if (track.kind === Track.Kind.Audio) return "audio";
  return track.source === Track.Source.ScreenShare ? "screen" : "camera";
}

export async function joinRoom(interviewId: number, h: RoomHandlers, publish = { camera: true, mic: true }): Promise<Room> {
  const { token, url } = await api.livekit(interviewId);
  const room = new Room({ adaptiveStream: true, dynacast: true });
  const refresh = () => h.onParticipants?.(Array.from(room.remoteParticipants.values()));
  room
    .on(RoomEvent.TrackSubscribed, (track, _pub, p) => h.onTrack(kindOf(track), track.attach(), p))
    .on(RoomEvent.TrackUnsubscribed, (track, _pub, p) => {
      track.detach().forEach((e) => e.remove());
      h.onTrack(kindOf(track), null, p);
    })
    .on(RoomEvent.ParticipantConnected, refresh)
    .on(RoomEvent.ParticipantDisconnected, refresh);
  await room.connect(livekitUrl(url), token);
  if (publish.camera) await room.localParticipant.setCameraEnabled(true, { resolution: { width: 1280, height: 720, frameRate: 15 } });
  if (publish.mic) await room.localParticipant.setMicrophoneEnabled(true, { echoCancellation: true, noiseSuppression: false });
  refresh();
  return room;
}

export const CAPTURE_PREFIX = "call-capture-";

export type CallSource = "meet" | "zoom_web" | "zoom_app" | "teams";

/** How each external call is captured, and what the interviewer has to pick in Chrome's share dialog. */
export const CALL_SOURCES: Record<CallSource, { label: string; platform: "meet" | "zoom" | "teams"; surface: "browser" | "monitor"; steps: string }> = {
  meet: { label: "Google Meet", platform: "meet", surface: "browser",
    steps: "Pick the Meet tab and keep “Also share tab audio” ticked." },
  zoom_web: { label: "Zoom (in the browser)", platform: "zoom", surface: "browser",
    steps: "Join with “Join from your browser”, then pick that Zoom tab and keep “Also share tab audio” ticked." },
  zoom_app: { label: "Zoom (desktop app)", platform: "zoom", surface: "monitor",
    steps: "Pick “Entire screen” and tick “Also share system audio”. Put the candidate in speaker view, maximise Zoom and mute other apps: every sound on this computer is analysed." },
  teams: { label: "Microsoft Teams (in the browser)", platform: "teams", surface: "browser",
    steps: "Open Teams in the browser, then pick that tab and keep “Also share tab audio” ticked." },
};

export interface CallCapture {
  source: CallSource;
  hasAudio: boolean;
  stop: () => void;
}

/** Capture an external call (a Meet / Zoom / Teams tab, or the screen with system audio for the Zoom desktop app) and
 *  publish it into our room as the candidate's camera + mic. The call's audio carries the other side only (call apps
 *  never play your own mic back), so the workers hear the candidate; the hidden bot analyses it like any candidate
 *  joining our own room. The face detector keeps the largest face, so the candidate should be in speaker view. */
export async function captureCall(interviewId: number, source: CallSource, onEnded: () => void): Promise<CallCapture> {
  const cfg = CALL_SOURCES[source];
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: { frameRate: 15, width: { ideal: 1280 }, displaySurface: cfg.surface },
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    // Chrome hints: offer the right surface first, keep our own tab out of the list, pre-tick tab / system audio
    preferCurrentTab: false, selfBrowserSurface: "exclude", systemAudio: "include",
    surfaceSwitching: "include", monitorTypeSurfaces: cfg.surface === "monitor" ? "include" : "exclude",
  } as DisplayMediaStreamOptions);
  const { token, url } = await api.captureToken(interviewId, cfg.platform);
  const room = new Room();
  await room.connect(livekitUrl(url), token);
  const [video] = stream.getVideoTracks();
  const [audio] = stream.getAudioTracks();
  await room.localParticipant.publishTrack(video, { source: Track.Source.Camera, name: `${source}-video` });
  if (audio) await room.localParticipant.publishTrack(audio, { source: Track.Source.Microphone, name: `${source}-audio` });
  let stopped = false;
  const stop = () => {
    if (stopped) return;
    stopped = true;
    stream.getTracks().forEach((t) => t.stop());
    room.disconnect();
    onEnded();
  };
  video.addEventListener("ended", stop); // Chrome's own "Stop sharing" bar
  return { source, hasAudio: !!audio, stop };
}

export function roleOf(p: RemoteParticipant): string {
  try {
    return JSON.parse(p.metadata ?? "{}").role ?? "participant";
  } catch {
    return "participant";
  }
}
