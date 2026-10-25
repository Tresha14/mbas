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

export function roleOf(p: RemoteParticipant): string {
  try {
    return JSON.parse(p.metadata ?? "{}").role ?? "participant";
  } catch {
    return "participant";
  }
}
