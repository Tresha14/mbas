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
  platform?: "interlens" | "meet" | "zoom_web" | "zoom_app" | "teams";
}

export interface Question {
  question_id: string;
  category: string;
  difficulty: number;
  text: string;
  key_points: string[];
}

const KEY = "mbas_token";

/** Candidate pages: the per-tab candidate token wins. Everywhere else (interviewer pages): the stored interviewer login
    wins, so signing in as an interviewer in a tab that was used as a candidate works. */
function onCandidatePage(): boolean {
  return /^\/(room|join)(\/|$)/.test(window.location.pathname);
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return onCandidatePage() ? sessionStorage.getItem(KEY) ?? localStorage.getItem(KEY) : localStorage.getItem(KEY);
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
    const c = JSON.parse(atob(t.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    if (c.exp && c.exp * 1000 < Date.now()) {
      clearToken(); // expired login behaves as signed out, so pages send the user to /login
      return null;
    }
    return c;
  } catch {
    return null;
  }
}

export async function req<T>(method: string, path: string, body?: unknown, token = getToken()): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (res.status === 401 && !path.startsWith("/api/auth/") && !onCandidatePage()) {
    clearToken();                                   // login expired: go back to sign-in instead of showing empty pages
    window.location.assign("/login");
    return new Promise<T>(() => undefined);
  }
  if (!res.ok) {
    // expired or revoked interviewer login: sign out and go to /login (candidate pages keep their own flow)
    if (res.status === 401 && token && typeof window !== "undefined" && !/^\/(login|join|room|feedback)/.test(location.pathname)) {
      clearToken();
      location.href = "/login";
    }
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
  create: (body: { title: string; mode: string; candidate_name?: string; scheduled_at?: string | null; platform?: string }) =>
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
  captureToken: (id: number, platform: "meet" | "zoom" | "teams" = "meet") =>
    req<{ token: string; url: string; identity: string }>("POST", `/api/interviews/${id}/capture-token?platform=${platform}`),
  notes: (id: number, notes: string) => req("PUT", `/api/interviews/${id}/notes`, { notes }),
  data: (id: number) => req<InterviewData>("GET", `/api/interviews/${id}/data`),
  buildReport: (id: number) => req<ReportRow>("POST", `/api/interviews/${id}/report`),
  report: (id: number) => req<ReportRow>("GET", `/api/interviews/${id}/report`),
  /** The interviewer report as a single self-contained HTML file (text). */
  reportHtml: async (id: number): Promise<string> => {
    const res = await fetch(`/api/interviews/${id}/report.html`, { headers: { Authorization: `Bearer ${getToken()}` } });
    if (!res.ok) throw new Error(res.status === 404 ? "Build the report first (press Refresh)." : res.statusText);
    return res.text();
  },
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
  long_pauses?: number | null;
  longest_pause_s?: number | null;
  longest_pause_at_s?: number | null;
  very_long_pauses?: number | null;
  voice_breaks?: number | null;
  voice_break_times_s?: number[];
  pitch_jitter_pct?: number | null;
  filler_rate?: number | null;
  words?: WordBreakdown;
  eyes?: { blink_rate: number | null; on_screen_pct: number | null; reading: boolean };
  quadrant?: string;
}

export interface WordBreakdown {
  fillers: Record<string, number>;
  hedges: Record<string, number>;
  crutch_words: Record<string, number>;
}

export type Status = "good" | "watch" | "concern" | "neutral";

export interface MetricRow {
  key: string;
  label: string;
  value: number | null;
  unit: string;
  typical: string;
  status: Status;
  note: string;
  at_s?: number | null;
  at?: string | null;
  extra?: { breakdown?: Record<string, number>; words?: Record<string, number>; times?: string[]; min?: number | null; max?: number | null };
}

export interface BehaviourSection {
  key: string;
  title: string;
  rows: MetricRow[];
  series?: { t_s: number; blink: number | null; on_screen: number | null }[];
}

export interface Scorecard {
  dimensions: Record<string, { score: number | null; weight: number; basis: string }>;
  overall: number | null;
  grade: string | null;
  grade_word: string | null;
  strongest: string | null;
  weakest: string | null;
  note?: string | null;
  method: string;
}

export interface Matrix {
  points: { question_ref: number; order: number; question: string; content: number; confidence: number; quadrant: string; label: string }[];
  counts: Record<string, number>;
  note: string | null;
  cut: number;
}

export interface Pressure {
  rows: { key: string; label: string; unit: string; easier: number; harder: number; change: number; worse: boolean }[];
  summary: string;
  easier_n: number;
  harder_n: number;
}

export interface Moment {
  t_s: number;
  time: string;
  kind: string;
  label: string;
  detail: string;
}

export interface ReportRow {
  interview_id: number;
  interviewer_view: {
    summary: { questions: number; mean_effectiveness: number | null; mean_attention: number | null; engaged_share: number | null; follow_ups: number };
    radar: Record<string, number | null>;
    cards: Card[];
    timeline: { t_s: number; attention: number | null; engagement: string | null; facing?: number | null; blink?: number | null }[];
    delivery?: {
      wpm: number | null; pause_pct: number | null; fillers_per_100w: number | null; hedges_per_100w: number | null;
      camera_facing_pct: number | null; blinks_per_min: number | null; flag_counts: Record<string, number>;
    };
    flags?: { t_s: number; time: string; kind: string; text: string }[];
    follow_ups: { t_s: number; time: string; reason: string; text: string; source: string }[];
    note: string;
    scorecard?: Scorecard;
    behaviour?: BehaviourSection[];
    matrix?: Matrix;
    pressure?: Pressure | null;
    moments?: Moment[];
    combined_flags?: { t_start: number; t_end: number; time: string; signals: string[]; phrases: string[]; text: string; question_order: number | null; question_ref: number | null }[];
  };
  candidate_view: CandidateView;
  released: boolean;
  release_token: string | null;
  updated_at: string;
}

export function wsUrl(path: string): string {
  const env = process.env.NEXT_PUBLIC_WS_URL;
  if (env) return `${env}${path}`;
  // Next dev (3000) and next start (3020) do not proxy WebSockets: go straight to the API
  if (location.port === "3020" || location.port === "3000") return `${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:8020${path}`;
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;
}

export function livekitUrl(fromApi: string): string {
  const env = process.env.NEXT_PUBLIC_LIVEKIT_URL;
  if (env) return env;
  if (location.protocol === "http:") return fromApi.replace("localhost", location.hostname);
  return `wss://${location.host}`;
}
