/* REST client for the MBAS API (FastAPI, proxied at /api by next.config.ts or Caddy). */

export type Role = "interviewer" | "meeting_host" | "candidate" | "participant";

export interface Interview {
  id: number;
  title: string;
  mode: "interview" | "meeting";
  room: string;
  status: "scheduled" | "live" | "ended";
  candidate_name: string | null;
  scheduled_at: string | null;
  started_at: string | null;
  ended_at: string | null;
  candidate_link: string;
  invite_token: string;
  notes: string;
}

export interface Question {
  question_id: string;
  category: string;
  difficulty: number;
  text: string;
  key_points: string[];
}

const KEY = "mbas_token";

export function getToken(): string | null {
  return typeof window === "undefined" ? null : localStorage.getItem(KEY) ?? sessionStorage.getItem(KEY);
}

export function setToken(token: string, session = false) {
  (session ? sessionStorage : localStorage).setItem(KEY, token);
}

export function clearToken() {
  localStorage.removeItem(KEY);
  sessionStorage.removeItem(KEY);
}

export function claims(): { sub: string; role: Role; name?: string; interview_id?: number } | null {
  const t = getToken();
  if (!t) return null;
  try {
    return JSON.parse(atob(t.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
  } catch {
    return null;
  }
}

export async function req<T>(method: string, path: string, body?: unknown, token = getToken()): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  register: (email: string, name: string, password: string, role: Role) =>
    req<{ token: string }>("POST", "/api/auth/register", { email, name, password, role }),
  login: (email: string, password: string) => req<{ token: string }>("POST", "/api/auth/login", { email, password }),
  interviews: () => req<Interview[]>("GET", "/api/interviews"),
  interview: (id: number) => req<Interview>("GET", `/api/interviews/${id}`),
  create: (body: { title: string; mode: string; candidate_name?: string; scheduled_at?: string | null }) =>
    req<Interview>("POST", "/api/interviews", body),
  questions: () => req<Question[]>("GET", "/api/questions"),
  join: (invite: string, name: string, consent: boolean) =>
    req<{ token: string; interview: { id: number; title: string; mode: string }; identity: string }>(
      "POST", `/api/join/${invite}`, { name, consent }, null),
  livekit: (id: number) => req<{ token: string; url: string; identity: string }>("POST", `/api/interviews/${id}/livekit-token`),
  start: (id: number) => req<Interview>("POST", `/api/interviews/${id}/start`),
  stop: (id: number) => req<Interview>("POST", `/api/interviews/${id}/stop`),
  event: (id: number, type: "ask" | "answer_end", q: { question_id: string; text?: string; category?: string }) =>
    req<{ question_ref: number; ask_s: number; end_s: number | null }>("POST", `/api/interviews/${id}/events`,
      { type, question_id: q.question_id, text: q.text ?? "", category: q.category ?? "" }),
  notes: (id: number, notes: string) => req("PUT", `/api/interviews/${id}/notes`, { notes }),
  data: (id: number) => req<InterviewData>("GET", `/api/interviews/${id}/data`),
  buildReport: (id: number) => req<ReportRow>("POST", `/api/interviews/${id}/report`),
  report: (id: number) => req<ReportRow>("GET", `/api/interviews/${id}/report`),
  editReport: (id: number, candidate_view: CandidateView) => req<ReportRow>("PUT", `/api/interviews/${id}/report`, { candidate_view }),
  release: (id: number) => req<{ release_token: string; url: string }>("POST", `/api/interviews/${id}/report/release`),
  feedback: (token: string) => req<CandidateView & { title: string; released_at: string }>("GET", `/api/feedback/${token}`, undefined, null),
};

export interface InterviewData {
  interview: Interview;
  questions: { id: number; order: number; question_id: string; text: string; category: string; ask_s: number; end_s: number | null }[];
  transcript: { speaker: string; start_s: number; end_s: number; text: string; fillers: number[]; words: { word: string }[] }[];
  windows: { start_s: number; end_s: number; attention: number | null; engagement: string | null }[];
  answers: unknown[];
  follow_ups: { t_s: number; reason: string; text: string; source: string; question_ref: number | null }[];
}

export interface CandidateView {
  strengths: string[];
  improvements: { area: string; observation: string; tip: string; timestamp: string | null }[];
  closing: string;
}

export interface Card {
  question_ref: number;
  order: number;
  question: string;
  category: string;
  ask_s: number;
  end_s: number | null;
  transcript: string;
  effectiveness: number | null;
  subscores: Record<string, number>;
  answer_model: { relevance?: number; quality?: number; star?: Record<string, number>; completeness?: number };
  reasons: { feature: string; modality: string; contribution: number; text: string }[];
  modality_weights: Record<string, number>;
  speaking_pattern: string | null;
  hesitation_index: number | null;
  response_latency_s: number | null;
  wpm: number | null;
  fillers: number | null;
  follow_ups: { reason: string; text: string; source: string }[];
}

export interface ReportRow {
  interview_id: number;
  interviewer_view: {
    summary: { questions: number; mean_effectiveness: number | null; mean_attention: number | null; engaged_share: number | null; follow_ups: number };
    radar: Record<string, number | null>;
    cards: Card[];
    timeline: { t_s: number; attention: number | null; engagement: string | null }[];
    follow_ups: { t_s: number; time: string; reason: string; text: string; source: string }[];
    note: string;
  };
  candidate_view: CandidateView;
  released: boolean;
  release_token: string | null;
}

export function wsUrl(path: string): string {
  const env = process.env.NEXT_PUBLIC_WS_URL;
  if (env) return `${env}${path}`;
  if (location.port === "3020") return `${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:8020${path}`;
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;
}

export function livekitUrl(fromApi: string): string {
  const env = process.env.NEXT_PUBLIC_LIVEKIT_URL;
  if (env) return env;
  if (location.protocol === "http:") return fromApi.replace("localhost", location.hostname);
  return `wss://${location.host}`;
}
