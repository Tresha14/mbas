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

export const CAPTURE_PREFIX = "meet-capture-";

export interface MeetCapture {
  hasAudio: boolean;
  stop: () => void;
}

/** Capture a Google Meet (or any call) tab and publish it into our room as the candidate's camera + mic.
 *  The tab's audio carries the other side of the call only (Meet never plays your own mic back), so the
 *  workers hear the candidate; the hidden bot analyses it like any candidate joining our own room. */
export async function captureMeetTab(interviewId: number, onEnded: () => void): Promise<MeetCapture> {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: { frameRate: 15, width: { ideal: 1280 } },
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    // Chrome hints: offer tabs first, keep our own tab out of the list, and pre-tick "share tab audio"
    preferCurrentTab: false, selfBrowserSurface: "exclude", systemAudio: "include",
  } as DisplayMediaStreamOptions);
  const { token, url } = await api.captureToken(interviewId);
  const room = new Room();
  await room.connect(livekitUrl(url), token);
  const [video] = stream.getVideoTracks();
  const [audio] = stream.getAudioTracks();
  await room.localParticipant.publishTrack(video, { source: Track.Source.Camera, name: "meet-video" });
  if (audio) await room.localParticipant.publishTrack(audio, { source: Track.Source.Microphone, name: "meet-audio" });
  let stopped = false;
  const stop = () => {
    if (stopped) return;
    stopped = true;
    stream.getTracks().forEach((t) => t.stop());
    room.disconnect();
    onEnded();
  };
  video.addEventListener("ended", stop); // Chrome's own "Stop sharing" bar
  return { hasAudio: !!audio, stop };
}

export function roleOf(p: RemoteParticipant): string {
  try {
    return JSON.parse(p.metadata ?? "{}").role ?? "participant";
  } catch {
    return "participant";
  }
}
