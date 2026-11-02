"use client";

import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle, ArrowLeftRight, Check, ChevronRight, FileText, Play, Square, Video } from "lucide-react";
import type { Room } from "livekit-client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { GazePlot, Signal, TrendChart } from "@/components/charts";
import { Header, VideoSlot } from "@/components/shell";
import { Badge, Button, Card, cn, Disclosure, Dot, Empty, mmss, scoreTone, SectionLabel, Segmented, Tabs, Textarea } from "@/components/ui";
import { useAnalytics, type Line } from "@/lib/analytics";
import { api, type Interview, type Question } from "@/lib/api";
import { CAPTURE_PREFIX, captureMeetTab, joinRoom, roleOf, type MeetCapture } from "@/lib/room";

function TranscriptLine({ l, active }: { l: Line; active: boolean }) {
  const words = l.words?.length ? l.words.map((w) => w.word) : l.text.split(" ");
  const you = l.speaker === "interviewer";
  return (
    <div id={`t-${Math.floor(l.start_s)}`} className={cn("grid grid-cols-[64px_1fr] gap-3 rounded-2xl px-3 py-2.5 transition", active && "bg-accent-soft")}>
      <span className="pt-0.5 text-xs tabular-nums text-faint">{mmss(l.start_s)}</span>
      <p className={cn("text-[15px] leading-relaxed", you ? "text-muted" : "text-fg")}>
        <span className={cn("mr-2 text-[11px] font-medium tracking-wider uppercase", you ? "text-faint" : "text-accent")}>{you ? "You" : "Candidate"}</span>
        {words.map((w, i) => <span key={i} className={l.fillers?.includes(i) ? "filler" : ""}>{w} </span>)}
        {!l.final && <span className="text-faint">…</span>}
      </p>
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
  const [swap, setSwap] = useState(false);
  const [detail, setDetail] = useState("transcript");
  const [flow, setFlow] = useState("questions");
  const [cat, setCat] = useState("all");
  const [jump, setJump] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const room = useRef<Room | null>(null);
  const [capture, setCapture] = useState<MeetCapture | null>(null);
  const [captureMsg, setCaptureMsg] = useState("");
  const { state, error } = useAnalytics(iv?.id ?? null);

  useEffect(() => {
    api.interview(id).then((x) => { setIv(x); setNotes(x.notes); });
    api.questions().then(setQuestions);
    joinRoom(id, {
      onTrack: (kind, el, p) => {
        if (roleOf(p) !== "candidate") return;
        if (kind === "camera") setCam(el);
        if (kind === "screen") setScreen(el);
        // a captured Meet tab is already playing in its own tab: playing it here too would echo
        if (kind === "audio" && el && !p.identity.startsWith(CAPTURE_PREFIX)) document.body.appendChild(el);
      },
    }).then((r) => (room.current = r)).catch(() => undefined);
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => { clearInterval(t); room.current?.disconnect(); };
  }, [id]);

  const last = state.windows[state.windows.length - 1];
  const engagedPct = (w?: typeof last) => w?.engagement_probs ? 100 * (w.engagement_probs.engaged + 0.5 * w.engagement_probs.neutral) : null;
  const hesitationHist = state.answers.map((a) => a.behaviour.hesitation_index);
  const trend = useMemo(() => state.windows.map((w) => ({ t: w.end_s, attention: w.attention, engaged: w.engagement === "engaged" ? 90 : w.engagement === "neutral" ? 50 : 10 })), [state.windows]);
  const lines = [...state.lines, ...Object.values(state.partial).filter((x): x is Line => !!x)];
  const current = questions.find((q) => asked[q.question_id] === "asking");
  const elapsed = iv?.started_at && iv.status === "live" ? (now - new Date(iv.started_at).getTime()) / 1000 : null;
  const recentAlert = state.alerts.length ? state.alerts[state.alerts.length - 1] : null;
  const showAlert = recentAlert && elapsed != null && elapsed - recentAlert.t_s < 6;

  useEffect(() => {
    if (jump == null) return;
    setDetail("transcript");
    const target = [...state.lines].reverse().find((l) => l.start_s <= jump);
    if (target) setTimeout(() => document.getElementById(`t-${Math.floor(target.start_s)}`)?.scrollIntoView({ behavior: "smooth", block: "center" }), 50);
  }, [jump, state.lines]);

  async function ask(q: Question) {
    if (!iv) return;
    if (current && current.question_id !== q.question_id) await api.event(iv.id, "answer_end", current);
    await api.event(iv.id, "ask", q);
    setAsked((a) => ({ ...a, ...(current ? { [current.question_id]: "done" } : {}), [q.question_id]: "asking" }));
  }
  async function endAnswer() {
    if (!iv || !current) return;
    await api.event(iv.id, "answer_end", current);
    setAsked((a) => ({ ...a, [current.question_id]: "done" }));
    setFlow("answers");
  }
  async function startStop() {
    if (!iv) return;
    if (iv.status === "live") capture?.stop();
    setIv(iv.status === "live" ? await api.stop(iv.id) : await api.start(iv.id));
  }
  async function toggleMeet() {
    if (!iv) return;
    if (capture) return capture.stop();
    try {
      const c = await captureMeetTab(iv.id, () => { setCapture(null); setCaptureMsg(""); });
      setCapture(c);
      setCaptureMsg(c.hasAudio ? "" : "No audio: share again and tick “Also share tab audio”.");
    } catch (e) {
      setCaptureMsg(e instanceof Error && e.name === "NotAllowedError" ? "" : `Couldn't capture the tab: ${e}`);
    }
  }
  useEffect(() => () => capture?.stop(), [capture]);

  const main = swap ? screen : cam;
  const inset = swap ? cam : screen;
  const shownQs = questions.filter((q) => (cat === "all" || q.category === cat) && asked[q.question_id] !== "asking");

  return (
    <>
      <Header wide right={
        <div className="flex items-center gap-3">
          {iv?.status === "live" && <span className="flex items-center gap-2 text-[13px] tabular-nums text-muted"><Dot tone="bad" pulse /> {mmss(elapsed)}</span>}
          {state.models && !state.connected && <Badge tone="warn">reconnecting</Badge>}
          {iv && iv.status !== "ended" && (
            <Button size="sm" variant={capture ? "danger" : "outline"} onClick={toggleMeet}
              title="Run the interview on Google Meet: pick the Meet tab and tick “Also share tab audio”">
              <Video size={14} /> {capture ? "Stop Meet capture" : "Use Google Meet"}
            </Button>)}
          {iv?.status === "ended"
            ? <Link href={`/report/${id}`}><Button size="sm"><FileText size={14} /> View report</Button></Link>
            : <Button size="sm" variant={iv?.status === "live" ? "danger" : "accent"} onClick={startStop}>
                {iv?.status === "live" ? <><Square size={13} /> End interview</> : <><Play size={13} /> Start analysis</>}
              </Button>}
        </div>} />
      <main className="mx-auto grid max-w-[1440px] gap-6 px-6 py-6 xl:grid-cols-[1fr_400px]">
        <section className="min-w-0 space-y-6">
          <div className="flex items-baseline justify-between">
            <div>
              <h1 className="font-display text-2xl tracking-tight">{iv?.candidate_name ?? "Candidate"}</h1>
              <p className="text-[13px] text-muted">{iv?.title}</p>
            </div>
            {(captureMsg || error) && <span className="text-[13px] text-bad">{captureMsg || error}</span>}
          </div>

          <div className="relative">
            <VideoSlot el={main} label={swap ? "Shared screen" : "Camera"} className="aspect-video w-full" />
            {inset && (
              <button onClick={() => setSwap(!swap)} className="group absolute top-4 right-4 w-56 overflow-hidden rounded-2xl ring-1 ring-white/15 transition hover:ring-white/40">
                <VideoSlot el={inset} label={swap ? "Camera" : "Screen"} className="pointer-events-none aspect-video rounded-2xl" />
                <span className="absolute inset-0 grid place-items-center bg-black/40 opacity-0 transition group-hover:opacity-100"><ArrowLeftRight size={18} className="text-white" /></span>
              </button>
            )}
            <AnimatePresence>
              {showAlert && (
                <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  className="absolute top-4 left-4 flex items-center gap-2 rounded-full bg-black/55 px-4 py-2 text-[13px] text-white backdrop-blur-md">
                  <AlertCircle size={15} className="text-warn" /> {recentAlert?.text}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <Card className="p-0">
            <div className="px-6 pt-5"><Tabs items={[{ value: "transcript", label: "Transcript" }, { value: "timeline", label: "Timeline" }, { value: "gaze", label: "Gaze" }]} value={detail} onChange={setDetail} /></div>
            <div className="px-3 py-4">
              {detail === "transcript" && (
                <div className="max-h-80 space-y-0.5 overflow-y-auto">
                  {lines.length === 0
                    ? <Empty title={state.models && !state.models.asr ? "Transcript unavailable" : "Listening…"} text={state.models && !state.models.asr ? "The speech model hasn't been trained yet." : "Words appear here as people speak. Filler words are underlined."} />
                    : lines.map((l, i) => <TranscriptLine key={i} l={l} active={jump != null && Math.abs(l.start_s - jump) < 3} />)}
                </div>
              )}
              {detail === "timeline" && (
                <div className="px-3">
                  {trend.length < 2 ? <Empty title="Not enough data yet" text="Attention and engagement appear every few seconds once analysis starts." /> : (
                    <>
                      <TrendChart data={trend} markers={state.followUps.map((f) => ({ t: f.t_s, label: f.reason }))} onPick={setJump} height={220} />
                      <p className="mt-2 text-xs text-faint">Solid line: attention · dashed: engagement · amber marks: follow-up suggestions. Click to jump in the transcript.</p>
                    </>
                  )}
                </div>
              )}
              {detail === "gaze" && (
                <div className="px-3">
                  <GazePlot points={state.gaze} />
                  <p className="mt-2 text-xs text-faint">Where the candidate is looking, in degrees. The dashed box is the screen.</p>
                </div>
              )}
            </div>
          </Card>
        </section>

        <aside className="space-y-6">
          <Card className="pb-3">
            <SectionLabel>Signals</SectionLabel>
            <div className="divide-y divide-line">
              <Signal label="Attention" value={last?.attention} history={state.windows.map((w) => w.attention)} />
              <Signal label="Engagement" value={engagedPct(last)} unit="%" history={state.windows.map((w) => engagedPct(w))} />
              <Signal label="Hesitation" value={state.answers.length ? hesitationHist[hesitationHist.length - 1] : null} history={hesitationHist} invert hint="Per answer" />
            </div>
            {state.speaking && (
              <p className="mt-1 text-[13px] text-muted">
                {state.speaking.wpm ? `${Math.round(state.speaking.wpm)} words/min · ` : ""}
                {state.speaking.pause_ratio != null ? `${Math.round(state.speaking.pause_ratio * 100)}% pauses` : ""}
                {state.speaking.pitch_variation_st != null && state.speaking.pitch_variation_st < 1.5 ? " · flat voice" : ""}
              </p>
            )}
          </Card>

          <Card className="p-0">
            <div className="px-6 pt-5"><Tabs items={[{ value: "questions", label: "Questions" }, { value: "answers", label: `Answers${state.answers.length ? ` · ${state.answers.length}` : ""}` }, { value: "notes", label: "Notes" }]} value={flow} onChange={setFlow} /></div>

            {flow === "questions" && (
              <div className="p-5">
                {current ? (
                  <div className="mb-5 rounded-2xl bg-accent-soft p-4">
                    <p className="text-[11px] tracking-wider text-accent uppercase">Now asking · {current.category}</p>
                    <p className="mt-1.5 font-display text-lg leading-snug">{current.text}</p>
                    <Button className="mt-4 w-full" onClick={endAnswer}><Check size={15} /> Answer ended</Button>
                  </div>
                ) : iv?.status === "ended" ? (
                  <p className="mb-5 rounded-2xl bg-surface-2 px-4 py-3 text-[13px] text-muted">This interview has ended. The report has the full picture.</p>
                ) : iv?.status !== "live" ? (
                  <p className="mb-5 rounded-2xl bg-surface-2 px-4 py-3 text-[13px] text-muted">Start the analysis, then pick a question to ask.</p>
                ) : null}
                <Segmented className="mb-3" items={[{ value: "all", label: "All" }, { value: "hr", label: "HR" }, { value: "behavioural", label: "Behavioural" }, { value: "technical", label: "Technical" }]} value={cat} onChange={setCat} />
                <ul className="max-h-[420px] space-y-1 overflow-y-auto pr-1">
                  {shownQs.map((q) => (
                    <li key={q.question_id}>
                      <button disabled={iv?.status !== "live" || asked[q.question_id] === "done"} onClick={() => ask(q)}
                        className="group flex w-full items-start gap-3 rounded-2xl px-3 py-2.5 text-left transition hover:bg-surface-2 disabled:cursor-default disabled:hover:bg-transparent">
                        <span className={cn("mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full border text-[10px]", asked[q.question_id] === "done" ? "border-good bg-good text-white" : "border-faint/60 text-transparent")}>
                          <Check size={11} />
                        </span>
                        <span className={cn("flex-1 text-sm leading-snug", asked[q.question_id] === "done" && "text-faint line-through decoration-line")}>{q.text}</span>
                        {iv?.status === "live" && asked[q.question_id] !== "done" && <ChevronRight size={16} className="mt-0.5 text-faint opacity-0 transition group-hover:opacity-100" />}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {flow === "answers" && (
              <div className="px-6 pb-2">
                {state.answers.length === 0 ? <Empty title="No answers yet" text="Each answer is scored when you press “Answer ended”." /> :
                  [...state.answers].reverse().map((a) => {
                    const fu = state.followUps.filter((f) => f.question_ref === a.question_ref);
                    return (
                      <Disclosure key={a.question_ref} defaultOpen={a === state.answers[state.answers.length - 1]} summary={
                        <div className="flex items-center gap-3">
                          <span className={cn("font-display text-2xl tabular-nums", { good: "text-good", warn: "text-warn", bad: "text-bad", neutral: "text-faint" }[scoreTone(a.fusion?.effectiveness)])}>
                            {a.fusion ? Math.round(a.fusion.effectiveness) : "–"}
                          </span>
                          <span className="line-clamp-2 text-sm">{a.question_text}</span>
                        </div>}>
                        <ul className="space-y-2 text-sm text-muted">
                          {a.fusion?.reasons.map((r, i) => (
                            <li key={i} className="flex gap-2.5"><span className={cn("mt-2 h-1.5 w-1.5 shrink-0 rounded-full", r.contribution >= 0 ? "bg-good" : "bg-bad")} />{r.text}</li>
                          ))}
                        </ul>
                        <p className="mt-3 text-xs text-faint">
                          {a.behaviour.speaking_pattern ?? ""} · {a.behaviour.response_latency_s?.toFixed(1)} s to start · {a.behaviour.fillers} fillers
                          {a.answer_model ? ` · relevance ${a.answer_model.relevance.toFixed(2)}` : ""}
                        </p>
                        {fu.map((f, i) => (
                          <p key={i} className="mt-3 rounded-2xl bg-warn/10 px-4 py-2.5 text-[13px] text-warn">Consider a follow-up: {f.text}</p>
                        ))}
                      </Disclosure>
                    );
                  })}
              </div>
            )}

            {flow === "notes" && (
              <div className="p-5">
                <Textarea className="h-56" placeholder="Private notes. Only you can see these." value={notes} onChange={(e) => setNotes(e.target.value)} onBlur={() => iv && api.notes(iv.id, notes)} />
              </div>
            )}
          </Card>
          <p className="px-2 text-xs leading-relaxed text-faint">Decision support only: scores never decide on their own. The candidate can&apos;t see this page.</p>
        </aside>
      </main>
    </>
  );
}
