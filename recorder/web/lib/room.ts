import { Room, RoomEvent, Track, type RemoteTrack, type RemoteParticipant } from "livekit-client";
import { api, livekitUrl, type Role } from "./api";

export type ControlMessage =
  | { type: "rec_start"; t_ms: number }
  | { type: "rec_stop"; t_ms: number }
  | { type: "calibrate" }
  | { type: "hello"; role: Role };

export interface RoomHandlers {
  onTrack: (kind: "camera" | "screen" | "audio", el: HTMLMediaElement | null, participant: RemoteParticipant) => void;
  onMessage: (msg: ControlMessage, from?: RemoteParticipant) => void;
  onParticipants: (identities: string[]) => void;
}

function kindOf(track: RemoteTrack): "camera" | "screen" | "audio" {
  if (track.kind === Track.Kind.Audio) return "audio";
  return track.source === Track.Source.ScreenShare ? "screen" : "camera";
}

export async function joinRoom(code: string, role: Role, identity: string, handlers: RoomHandlers): Promise<Room> {
  const { token } = await api.token(code, role, identity);
  const room = new Room({ adaptiveStream: false, dynacast: true });

  const refresh = () => handlers.onParticipants(Array.from(room.remoteParticipants.values()).map((p) => p.identity));

  room
    .on(RoomEvent.TrackSubscribed, (track, _pub, participant) => {
      const el = track.attach();
      handlers.onTrack(kindOf(track), el, participant);
    })
    .on(RoomEvent.TrackUnsubscribed, (track, _pub, participant) => {
      track.detach().forEach((e) => e.remove());
      handlers.onTrack(kindOf(track), null, participant);
    })
    .on(RoomEvent.DataReceived, (payload, participant) => {
      try {
        handlers.onMessage(JSON.parse(new TextDecoder().decode(payload)) as ControlMessage, participant);
      } catch {
        /* ignore malformed */
      }
    })
    .on(RoomEvent.ParticipantConnected, refresh)
    .on(RoomEvent.ParticipantDisconnected, refresh);

  await room.connect(livekitUrl(), token);
  refresh();
  return room;
}

export async function send(room: Room, msg: ControlMessage): Promise<void> {
  await room.localParticipant.publishData(new TextEncoder().encode(JSON.stringify(msg)), { reliable: true });
}

export async function publish(room: Room, track: MediaStreamTrack, source: Track.Source): Promise<void> {
  await room.localParticipant.publishTrack(track, { source, simulcast: source === Track.Source.Camera });
}
