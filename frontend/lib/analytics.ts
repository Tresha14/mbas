"use client";

/* Interviewer-only analytics channel -> live state for the cockpit. The candidate page never imports this. */
import { useEffect, useReducer, useRef, useState } from "react";
import { getToken, wsUrl } from "./api";

export interface Word {
  word: string;
  start_s: number;
  end_s: number;
}
export interface Line {
  speaker: string;
  text: string;
  start_s: number;
  end_s: number;
  words: Word[];
  fillers: number[];
  final: boolean;
}
export interface WindowMsg {
  start_s: number;
  end_s: number;
  attention: number | null;
  engagement: string | null;
  engagement_probs?: Record<string, number> | null;
  on_screen_ratio: number | null;
  glances: number | null;
  blink_rate: number | null;
  reading: number | null;
}
export interface GazeMsg {
  t_s: number;
  face: boolean;
  on_screen: boolean;
  yaw: number | null;
  pitch: number | null;
  iris: number[] | null;
  head: number[] | null;
}
export interface AnswerMsg {
  question_ref: number;
  question_id: string;
  question_text: string;
  ask_s: number;
  end_s: number;
  transcript: string;
  behaviour: {
    speaking_pattern: string | null; hesitation_index: number | null; response_latency_s: number; wpm: number | null; fillers: number;
    long_pauses: number; longest_pause_s?: number | null; voice_breaks?: number | null; jitter_pct?: number | null;
  };
  answer_model: { relevance: number; quality: number; completeness: number; star_count: number } | null;
  fusion: {
    effectiveness: number;
    subscores: Record<string, number>;
    follow_up_p: number;
    reason: string;
    modality_weights: Record<string, number>;
    reasons: { text: string; feature: string; contribution: number }[];
  } | null;
}
export interface FollowUpMsg {
  question_ref: number;
  t_s: number;
  reason: string;
  text: string;
  probability: number;
  source: string;
}
export interface Speaking {
  t_s: number;
  wpm: number | null;
  pause_ratio: number | null;
  pitch_variation_st: number | null;
  speech_ratio: number;
  fillers_per_100w?: number | null;
  hedges_per_100w?: number | null;
}
export interface State {
  connected: boolean;
  lines: Line[];
  partial: Record<string, Line | undefined>;
  windows: WindowMsg[];
  gaze: GazeMsg[];
  speaking: Speaking | null;
  speakingHist: Speaking[];
  conversation: {
    turn_count: number;
    overlap_s: number;
    talk_balance_gini: number;
    participants: Record<string, { talk_share: number; talk_time_ratio: number; turns: number; avg_turn_s: number; interruptions_made: number }>;
    meeting?: Record<string, { engagement: number; contribution_balance: number; talk_share: number }>;
  } | null;
  answers: AnswerMsg[];
  followUps: FollowUpMsg[];
  alerts: { t_s: number; text: string; alert: string }[];
  questions: { question_ref: number; question_id: string; ask_s: number; end_s: number | null; type: string; text?: string; auto?: boolean }[];
  models: Record<string, boolean> | null;
  health: HealthMsg | null;
  latencyMs: number | null;
}

export interface MicHealth { db: number | null; silent_s: number; speech_recent: boolean; since_words_s: number | null; listening_s: number }
export interface HealthMsg { t_s: number; mics: Record<string, MicHealth>; asr: "own" | "whisper" }

const initial: State = {
  connected: false, lines: [], partial: {}, windows: [], gaze: [], speaking: null, speakingHist: [], conversation: null, answers: [],
  followUps: [], alerts: [], questions: [], models: null, health: null, latencyMs: null,
};

type Msg = Record<string, unknown> & { kind?: string };

function reduce(s: State, a: { type: "msg"; m: Msg } | { type: "conn"; v: boolean } | { type: "reset" }): State {
  if (a.type === "reset") return { ...initial, connected: true };
  if (a.type === "conn") return { ...s, connected: a.v };
  const m = a.m;
  const latency = typeof m.server_sent_at === "number" ? Math.max(0, Date.now() - (m.server_sent_at as number) * 1000) : s.latencyMs;
  switch (m.kind) {
    case "transcript": {
      const line = m as unknown as Line;
      if (line.final) return { ...s, lines: [...s.lines, line].slice(-400), partial: { ...s.partial, [line.speaker]: undefined }, latencyMs: latency };
      return { ...s, partial: { ...s.partial, [line.speaker]: line } };
    }
    case "transcript_retract":      // a line that turned out to be one voice leaking onto the other microphone
      return { ...s, lines: s.lines.filter((l) => !(l.speaker === m.speaker && l.start_s === m.start_s)) };
    case "window":
      return { ...s, windows: [...s.windows, m as unknown as WindowMsg].slice(-720) };
    case "gaze":
      return { ...s, gaze: [...s.gaze, m as unknown as GazeMsg].slice(-60), latencyMs: latency };
    case "speaking":
      return { ...s, speaking: m as unknown as Speaking, speakingHist: [...s.speakingHist, m as unknown as Speaking].slice(-360) };
    case "conversation":
      return { ...s, conversation: m as unknown as State["conversation"] };
    case "answer":
      return { ...s, answers: [...s.answers.filter((x) => x.question_ref !== (m.question_ref as number)), m as unknown as AnswerMsg] };
    case "follow_up":
      return { ...s, followUps: [...s.followUps, m as unknown as FollowUpMsg] };
    case "alert":
      return { ...s, alerts: [...s.alerts, m as unknown as State["alerts"][0]].slice(-200) };
    case "question":
      return { ...s, questions: [...s.questions.filter((q) => !(q.question_ref === m.question_ref && m.type === "answer_end" && q.type === "ask")), m as unknown as State["questions"][0]] };
    case "health":
      return { ...s, health: m as unknown as HealthMsg };
    case "status":
      return { ...s, models: { ...s.models, ...(m.models as Record<string, boolean>) } };
    default:
      return s;
  }
}

export function useAnalytics(interviewId: number | null) {
  const [state, dispatch] = useReducer(reduce, initial);
  const [error, setError] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);
  useEffect(() => {
    if (!interviewId) return;
    let stop = false;
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      const sock = new WebSocket(wsUrl(`/ws/analytics/${interviewId}?token=${getToken() ?? ""}`));
      ws.current = sock;
      sock.onopen = () => dispatch({ type: "reset" });   // the server replays history on connect: start clean so nothing shows twice
      sock.onmessage = (e) => dispatch({ type: "msg", m: JSON.parse(e.data) });
      sock.onclose = (e) => {
        dispatch({ type: "conn", v: false });
        if (e.code === 4403 || e.code === 4401) {
          setError("Analytics are only available to the interviewer.");
          return;
        }
        if (!stop) retry = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      stop = true;
      clearTimeout(retry);
      ws.current?.close();
    };
  }, [interviewId]);
  return { state, error };
}
