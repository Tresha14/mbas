"use client";

import { AlertTriangle, Check, Columns2, FileText, Gauge as GaugeIcon, Play, Square, Timer } from "lucide-react";
import type { Room } from "livekit-client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { GazePlot, Gauge, TrendChart } from "@/components/charts";
import { Header, VideoSlot } from "@/components/shell";
import { Badge, Button, Card, CardTitle, cn, mmss, scoreTone, Tabs } from "@/components/ui";
import { useAnalytics, type Line } from "@/lib/analytics";
import { api, type Interview, type Question } from "@/lib/api";
import { joinRoom, roleOf } from "@/lib/room";

function TranscriptLine({ l, highlight }: { l: Line; highlight: boolean }) {
  const words = l.words?.length ? l.words.map((w) => w.word) : l.text.split(" ");
  return (
    <div id={`t-${Math.floor(l.start_s)}`} className={cn("rounded-xl px-2 py-1 transition", highlight && "bg-accent-soft")}>
      <span className={cn("mr-2 text-xs font-semibold", l.speaker === "candidate" ? "text-accent" : "text-muted")}>
        {l.speaker === "candidate" ? "Candidate" : l.speaker === "interviewer" ? "You" : l.speaker} {mmss(l.start_s)}
      </span>
      {words.map((w, i) => <span key={i} className={l.fillers?.includes(i) ? "filler" : ""}>{w} </span>)}
      {!l.final && <span className="animate-pulse text-muted">...</span>}
    </div>
  );
}

export default function Cockpit() {
  const id = Number(useParams<{ id: string }>().id);
  const [iv, setIv] = useState<Interview | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [asked, setAsked] = useState<Record<string, "asking" | "done">>({});
  const [notes, setNotes] = useState("");
  const [cam, setCam] = useState<HTMLMediaElement | null>(null);
  const [screen, setScreen] = useState<HTMLMediaElement | null>(null);
  const [layout, setLayout] = useState("side by side");
  const [jump, setJump] = useState<number | null>(null);
  const [filter, setFilter] = useState("all");
  const room = useRef<Room | null>(null);
  const { state, error } = useAnalytics(iv?.id ?? null);

  useEffect(() => {
    api.interview(id).then((x) => { setIv(x); setNotes(x.notes); });
    api.questions().then(setQuestions);
    joinRoom(id, {
      onTrack: (kind, el, p) => {
        const r = roleOf(p);
        if (r !== "candidate") return;
        if (kind === "camera") setCam(el);
        if (kind === "screen") setScreen(el);
        if (kind === "audio" && el) document.body.appendChild(el);
      },
    }).then((r) => (room.current = r)).catch(() => undefined);
    return () => { room.current?.disconnect(); };
  }, [id]);

  const last = state.windows[state.windows.length - 1];
  const lastAnswer = state.answers[state.answers.length - 1];
  const hesitation = lastAnswer?.behaviour.hesitation_index ?? null;
  const confidence = lastAnswer?.fusion?.subscores.confidence ?? null;
  const engagedPct = last?.engagement_probs ? Math.round(100 * (last.engagement_probs.engaged + 0.5 * last.engagement_probs.neutral)) : null;
  const trend = useMemo(() => state.windows.map((w) => ({ t: w.end_s, attention: w.attention, engaged: w.engagement === "engaged" ? 90 : w.engagement === "neutral" ? 50 : 10 })), [state.windows]);
  const markers = state.followUps.map((f) => ({ t: f.t_s, label: f.reason }));
  const lines = [...state.lines, ...Object.values(state.partial).filter((x): x is Line => !!x)];

  useEffect(() => {
    if (jump == null) return;
    const target = [...state.lines].reverse().find((l) => l.start_s <= jump);
    if (target) document.getElementById(`t-${Math.floor(target.start_s)}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [jump, state.lines]);

  async function toggleQuestion(q: Question) {
    if (!iv) return;
    if (asked[q.question_id] !== "asking") {
      await api.event(iv.id, "ask", q);
      setAsked({ ...asked, [q.question_id]: "asking" });
    } else {
      await api.event(iv.id, "answer_end", q);
      setAsked({ ...asked, [q.question_id]: "done" });
    }
  }

  async function startStop() {
    if (!iv) return;
    setIv(iv.status === "live" ? await api.stop(iv.id) : await api.start(iv.id));
  }

  const shownQs = questions.filter((q) => filter === "all" || q.category === filter);
  return (
    <>
      <Header right={
        <div className="flex items-center gap-2">
          <Badge tone={state.connected ? "good" : "bad"}>{state.connected ? "analytics live" : "connecting"}</Badge>
          {state.latencyMs != null && <Badge tone={state.latencyMs < 2000 ? "good" : "warn"}><Timer size={12} /> {Math.round(state.latencyMs)} ms</Badge>}
          {iv?.status === "ended" && <Link href={`/report/${id}`}><Button size="sm" variant="outline"><FileText size={14} /> Report</Button></Link>}
          <Button size="sm" variant={iv?.status === "live" ? "danger" : "good"} onClick={startStop} disabled={iv?.status === "ended"}>
            {iv?.status === "live" ? <><Square size={14} /> End</> : <><Play size={14} /> Start analysis</>}
          </Button>
        </div>} />
      {error && <div className="mx-auto max-w-7xl px-4 pt-4 text-sm text-bad">{error}</div>}
      <main className="mx-auto grid max-w-[1600px] gap-4 px-4 py-4 xl:grid-cols-[1fr_380px]">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-semibold">{iv?.title} {iv?.candidate_name && <span className="text-muted">- {iv.candidate_name}</span>}</h1>
            <Tabs tabs={["side by side", "camera", "screen"]} value={layout} onChange={setLayout} />
          </div>
          <div className={cn("grid gap-4", layout === "side by side" ? "md:grid-cols-2" : "")}>
            {layout !== "screen" && <VideoSlot el={cam} label="Candidate camera" className="aspect-video" />}
            {layout !== "camera" && <VideoSlot el={screen} label="Candidate screen" className="aspect-video" />}
          </div>
          <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
            <Card>
              <CardTitle className="flex items-center gap-2"><GaugeIcon size={14} /> Live signals</CardTitle>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <Gauge label="Attention" value={last?.attention} />
                <Gauge label="Engagement" value={engagedPct} suffix="%" />
                <Gauge label="Hesitation" value={hesitation} invert />
                <Gauge label="Confidence" value={confidence} />
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <Badge tone="accent">WPM {state.speaking?.wpm ?? "-"}</Badge>
                <Badge tone={(state.speaking?.pause_ratio ?? 0) > 0.45 ? "warn" : "neutral"}>pauses {state.speaking?.pause_ratio != null ? `${Math.round(state.speaking.pause_ratio * 100)}%` : "-"}</Badge>
                <Badge tone={(state.speaking?.pitch_variation_st ?? 3) < 1.5 ? "warn" : "neutral"}>pitch variation {state.speaking?.pitch_variation_st ?? "-"} st</Badge>
                <Badge>glances {last?.glances ?? "-"}</Badge>
                <Badge>blinks/min {last?.blink_rate != null ? Math.round(last.blink_rate) : "-"}</Badge>
                {last?.reading ? <Badge tone="bad">reading pattern</Badge> : null}
                {state.conversation && <Badge>talk share {Math.round(100 * (state.conversation.participants.candidate?.talk_share ?? 0))}%</Badge>}
              </div>
            </Card>
            <Card>
              <CardTitle>Iris / gaze (degrees, box = screen)</CardTitle>
              <GazePlot points={state.gaze} />
              <div className="mt-2 max-h-20 space-y-1 overflow-auto">
                {state.alerts.slice(-4).reverse().map((a, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs text-warn"><AlertTriangle size={12} />{mmss(a.t_s)} {a.text}</div>
                ))}
              </div>
            </Card>
          </div>
          <Card>
            <CardTitle>Timeline (attention line, engagement steps, ! = follow-up; click to jump in the transcript)</CardTitle>
            <TrendChart data={trend} markers={markers} onPick={setJump} />
          </Card>
          <Card>
            <CardTitle>Live transcript</CardTitle>
            <div className="max-h-72 space-y-1 overflow-auto text-sm leading-relaxed">
              {lines.length === 0 && <p className="text-muted">{state.models && !state.models.asr ? "Speech model not trained yet: transcript unavailable." : "Waiting for speech..."}</p>}
              {lines.map((l, i) => <TranscriptLine key={i} l={l} highlight={jump != null && Math.abs(l.start_s - jump) < 3} />)}
            </div>
          </Card>
        </div>
        <aside className="space-y-4">
          <Card>
            <div className="mb-2 flex items-center justify-between">
              <CardTitle className="mb-0">Questions</CardTitle>
              <Tabs tabs={["all", "hr", "behavioural", "technical"]} value={filter} onChange={setFilter} />
            </div>
            <div className="max-h-80 space-y-1 overflow-auto">
              {shownQs.map((q) => (
                <div key={q.question_id} className="flex items-start gap-2 rounded-xl p-2 hover:bg-surface-2">
                  <Button size="sm" variant={asked[q.question_id] === "asking" ? "danger" : asked[q.question_id] === "done" ? "ghost" : "outline"}
                    onClick={() => toggleQuestion(q)} disabled={iv?.status !== "live" || asked[q.question_id] === "done"}>
                    {asked[q.question_id] === "asking" ? "Answer ended" : asked[q.question_id] === "done" ? <Check size={14} /> : "Ask"}
                  </Button>
                  <div className="text-sm"><span className="text-xs text-muted">{q.question_id} · {q.category}</span><br />{q.text}</div>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <CardTitle>Answers</CardTitle>
            {state.answers.length === 0 && <p className="text-sm text-muted">Scores appear when you press &quot;Answer ended&quot;.</p>}
            <div className="space-y-3">
              {[...state.answers].reverse().map((a) => (
                <div key={a.question_ref} className="rounded-xl border border-line p-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium">{a.question_text}</span>
                    <Badge tone={scoreTone(a.fusion?.effectiveness)}>{a.fusion ? Math.round(a.fusion.effectiveness) : "n/a"}</Badge>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1 text-xs">
                    {a.behaviour.speaking_pattern && <Badge>{a.behaviour.speaking_pattern}</Badge>}
                    <Badge>delay {a.behaviour.response_latency_s?.toFixed(1)} s</Badge>
                    <Badge>fillers {a.behaviour.fillers}</Badge>
                    {a.answer_model && <Badge>relevance {a.answer_model.relevance.toFixed(2)}</Badge>}
                  </div>
                  {a.fusion && <ul className="mt-2 list-disc pl-4 text-xs text-muted">{a.fusion.reasons.map((r, i) => <li key={i}>{r.text}</li>)}</ul>}
                  {state.followUps.filter((f) => f.question_ref === a.question_ref).map((f, i) => (
                    <div key={i} className="mt-2 rounded-lg bg-amber-500/10 px-2 py-1 text-xs text-amber-700 dark:text-amber-300">
                      Follow up ({f.reason.replace("_", " ")}): {f.text}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <CardTitle className="flex items-center gap-2"><Columns2 size={14} /> Private notes</CardTitle>
            <textarea className="h-28 w-full rounded-xl border border-line bg-surface p-2 text-sm outline-none focus:border-accent"
              value={notes} onChange={(e) => setNotes(e.target.value)} onBlur={() => iv && api.notes(iv.id, notes)} />
          </Card>
          <p className="text-xs text-muted">Decision support only: scores never decide on their own. The candidate cannot see this page.</p>
        </aside>
      </main>
    </>
  );
}
