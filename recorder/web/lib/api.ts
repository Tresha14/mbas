export type Role = "candidate" | "interviewer";

export type TrackName = "candidate_webcam" | "candidate_screen" | "candidate_mic" | "interviewer_mic";

export interface SessionSummary {
  session_id: string;
  room_code: string;
  status: "created" | "recording" | "stopped" | "finalized";
  recording: { start_ms: number | null; end_ms: number | null };
  participants: Partial<Record<Role, { participant_id: string }>>;
  tracks: Partial<Record<TrackName, { start_ms: number; end_ms: number | null; chunks: number; bytes: number }>>;
  calibration_saved: boolean;
}

export interface Question {
  question_id: string;
  category: "hr" | "behavioural" | "technical";
  difficulty: number;
  text: string;
  key_points: string[];
}

export interface CheckReport {
  session: string;
  passed: boolean;
  errors: string[];
  warnings: string[];
  tracks: Record<string, Record<string, unknown>>;
  info: Record<string, unknown>;
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(`${method} ${path}: ${res.status} ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  createSession: () => req<SessionSummary>("POST", "/api/sessions"),
  session: (code: string) => req<SessionSummary>("GET", `/api/sessions/${code}`),
  questions: () => req<{ questions: Question[] }>("GET", "/api/questions"),
  register: (code: string, body: Record<string, unknown>) =>
    req<SessionSummary>("POST", `/api/sessions/${code}/participants`, body),
  clock: (code: string, role: Role, offset_ms: number, rtt_ms: number) =>
    req("POST", `/api/sessions/${code}/clock`, { role, offset_ms, rtt_ms }),
  token: (code: string, role: Role, identity: string) =>
    req<{ token: string; room: string }>("POST", `/api/sessions/${code}/token`, { role, identity }),
  recording: (code: string, action: "start" | "stop", t_ms: number) =>
    req<SessionSummary>("POST", `/api/sessions/${code}/recording`, { action, t_ms }),
  trackStart: (code: string, track: TrackName, body: Record<string, unknown>) =>
    req("POST", `/api/sessions/${code}/tracks/${track}/start`, body),
  trackStop: (code: string, track: TrackName, end_ms: number, total_chunks: number) =>
    req("POST", `/api/sessions/${code}/tracks/${track}/stop`, { end_ms, total_chunks }),
  event: (code: string, body: Record<string, unknown>) => req("POST", `/api/sessions/${code}/events`, body),
  calibration: (code: string, body: Record<string, unknown>) =>
    req("POST", `/api/sessions/${code}/calibration`, body),
  finalize: (code: string) =>
    req<{ session_id: string; dir: string; check: CheckReport }>("POST", `/api/sessions/${code}/finalize`),
};

/** Upload one chunk, retrying on network errors. */
export async function putChunk(code: string, track: TrackName, seq: number, data: Blob | ArrayBuffer): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(`/api/sessions/${code}/tracks/${track}/chunks/${seq}`, {
        method: "PUT",
        headers: { "Content-Type": "application/octet-stream" },
        body: data,
      });
      if (res.ok) return;
      if (res.status < 500 && res.status !== 409) throw new Error(`chunk ${seq}: ${res.status} ${await res.text()}`);
    } catch (e) {
      if (attempt >= 8) throw e;
    }
    await new Promise((r) => setTimeout(r, Math.min(4000, 250 * 2 ** attempt)));
  }
}

export function livekitUrl(): string {
  const env = process.env.NEXT_PUBLIC_LIVEKIT_URL;
  if (env) return env;
  // Plain http (localhost dev): talk to LiveKit directly. Behind Caddy (https): same origin, /rtc is proxied.
  if (location.protocol === "http:") return `ws://${location.hostname}:7880`;
  return `wss://${location.host}`;
}
