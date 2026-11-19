"use client";

import { Check, Copy, Download, Pause, Play, RefreshCw, Send } from "lucide-react";
import { useParams } from "next/navigation";
import { flushSync } from "react-dom";
import { useEffect, useMemo, useRef, useState } from "react";
import { ConfidenceMatrix, EyeTrend, quadrantTone, ScoreRing, SubscoreRadar, TrendChart } from "@/components/charts";
import { FlaggedMoments } from "@/components/live";
import { Header } from "@/components/shell";
import { Badge, Button, Card, cn, Disclosure, Empty, Input, mmss, scoreTone, SectionLabel, Tabs, Textarea } from "@/components/ui";
import { copyText } from "@/lib/clipboard";
import { api, type BehaviourSection, type CandidateView, type InterviewData, type MetricRow, type Moment, type ReportRow,
  type Scorecard, type Status } from "@/lib/api";

function verdict(v: number | null | undefined) {
  if (v == null) return "Not enough data to score this interview yet.";
  if (v >= 70) return "Clear, confident communication overall.";
  if (v >= 55) return "Solid overall, with a few answers worth revisiting.";
  return "Several answers would benefit from follow-up questions.";
}

const statusText: Record<Status, string> = { good: "text-good", watch: "text-warn", concern: "text-bad", neutral: "text-faint" };
const statusWord: Record<Status, string> = { good: "good", watch: "watch", concern: "concern", neutral: "" };
const dimLabel: Record<string, string> = {
  content: "Content", fluency: "Fluency", composure: "Composure", delivery: "Delivery", eye_contact: "Eye contact",
  responsiveness: "Responsiveness",
};
const momentTone: Record<string, "good" | "warn" | "bad" | "neutral"> = {
  best: "good", weakest: "bad", delay: "warn", silence: "warn", voice: "warn", attention: "bad", reading: "bad",
};
const quadrants: [string, string][] = [["confident_on_point", "Confident and on point"], ["unsure_but_right", "Knows it, sounds unsure"],
  ["sure_but_off", "Sounds sure, but off target"], ["struggled", "Struggled"]];

function Chips({ data, tone = "neutral" }: { data?: Record<string, number>; tone?: "neutral" | "warn" }) {
  const e = Object.entries(data ?? {});
  if (!e.length) return null;
  return <div className="mt-2 flex flex-wrap gap-1.5">{e.map(([w, n]) => <Badge key={w} tone={tone}>“{w}” ×{n}</Badge>)}</div>;
}

function MetricLine({ r, onJump }: { r: MetricRow; onJump: (t: number) => void }) {
  const chips = r.extra?.breakdown ?? r.extra?.words;
  return (
    <div className="grid grid-cols-[1fr_auto] gap-x-6 gap-y-1 border-b border-border py-4 last:border-0 sm:grid-cols-[13rem_7rem_1fr]">
      <div>
        <p className="text-[15px]">{r.label}</p>
        {r.typical && <p className="text-xs text-faint">typical: {r.typical}</p>}
      </div>
      <div className="text-right sm:text-left">
        <span className="font-display text-2xl tabular-nums">{r.value ?? "–"}</span>
        {r.value != null && r.unit && <span className="ml-1 text-xs text-faint">{r.unit}</span>}
        {statusWord[r.status] && <p className={cn("text-xs", statusText[r.status])}>{statusWord[r.status]}</p>}
      </div>
      <div className="col-span-2 text-[13px] text-muted sm:col-span-1">
        {r.note}
        {r.extra?.min != null && r.extra?.max != null && <span className="text-faint"> Range {r.extra.min}–{r.extra.max}.</span>}
        {r.at_s != null && (
          <button className="no-print ml-2 text-accent underline-offset-2 hover:underline" onClick={() => onJump(r.at_s!)}>at {r.at}</button>)}
        {r.extra?.times && r.extra.times.length > 1 && <p className="mt-1 text-xs text-faint">at {r.extra.times.join(", ")}</p>}
        <Chips data={chips} tone={r.status === "concern" || r.status === "watch" ? "warn" : "neutral"} />
      </div>
    </div>
  );
}

function ScorecardView({ sc }: { sc: Scorecard }) {
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-8">
        <div className="text-center">
          <div className={cn("font-display text-6xl", { good: "text-good", warn: "text-warn", bad: "text-bad", neutral: "text-faint" }[scoreTone(sc.overall)])}>{sc.grade ?? "–"}</div>
          <p className="text-[13px] text-muted">{sc.grade_word}</p>
        </div>
        <div className="min-w-60 flex-1">
          <SectionLabel>Final rating</SectionLabel>
          <p className="font-display text-3xl tabular-nums">{sc.overall ?? "–"}<span className="text-base text-faint"> / 100</span></p>
          {sc.overall == null && <p className="mt-1 text-[13px] text-muted">{sc.note ?? "Not enough data to rate this interview yet."}</p>}
          <p className="mt-1 text-[13px] text-muted">
            {sc.strongest && <>Strongest: <b className="font-medium text-fg">{dimLabel[sc.strongest]}</b>. </>}
            {sc.weakest && <>Most room to grow: <b className="font-medium text-fg">{dimLabel[sc.weakest]}</b>.</>}
          </p>
        </div>
      </div>
      <div className="mt-8 space-y-5">
        {Object.entries(sc.dimensions).map(([k, d]) => (
          <div key={k}>
            <div className="flex items-baseline justify-between text-[14px]">
              <span>{dimLabel[k] ?? k} <span className="text-xs text-faint">· weight {Math.round(d.weight * 100)}%</span></span>
              <span className="font-display text-lg tabular-nums">{d.score ?? "–"}</span>
            </div>
            <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-surface-2">
              <div className={cn("h-full rounded-full", { good: "bg-good", warn: "bg-warn", bad: "bg-bad", neutral: "bg-faint" }[scoreTone(d.score)])}
                style={{ width: `${d.score ?? 0}%` }} />
            </div>
            <p className="mt-1 text-xs text-faint">{d.basis}</p>
          </div>
        ))}
      </div>
      <p className="mt-6 text-xs text-faint">{sc.method}</p>
    </Card>
  );
}

function Moments({ items, onJump }: { items: Moment[]; onJump: (t: number) => void }) {
  return (
    <div className="divide-y divide-border">
      {items.map((m, i) => (
        <button key={i} onClick={() => onJump(m.t_s)} className="flex w-full items-center gap-4 py-3 text-left hover:bg-surface-2/40">
          <span className="w-12 text-xs tabular-nums text-faint">{m.time}</span>
          <Badge tone={momentTone[m.kind] ?? "neutral"}>{m.label}</Badge>
          <span className="min-w-0 truncate text-[13px] text-muted">{m.detail}</span>
        </button>
      ))}
    </div>
  );
}

function CombinedFlags({ items, onJump }: { items: NonNullable<ReportRow["interviewer_view"]["combined_flags"]>; onJump: (t: number) => void }) {
  return (
    <div className="divide-y divide-border">
      {items.map((f, i) => (
        <button key={i} onClick={() => onJump(f.t_start)} className="flex w-full flex-wrap items-center gap-x-4 gap-y-1 py-3 text-left hover:bg-surface-2/40">
          <span className="w-24 text-xs tabular-nums text-faint">{f.time}</span>
          <span className="min-w-0 flex-1 text-[13px] text-fg">{f.text}</span>
          {f.question_order != null && <Badge tone="neutral">Question {f.question_order}</Badge>}
        </button>
      ))}
    </div>
  );
}

function BehaviourView({ sections, onJump, cursor }: { sections: BehaviourSection[]; onJump: (t: number) => void; cursor: number }) {
  return (
    <div className="space-y-6">
      {sections.map((s) => (
        <Card key={s.key} className="px-6">
          <SectionLabel>{s.title}</SectionLabel>
          {s.series && s.series.length > 1 && (
            <div className="mt-3">
              <EyeTrend data={s.series} onPick={onJump} cursor={cursor} />
              <p className="text-xs text-faint"><span className="text-warn">—</span> blinks per minute (shaded: typical 8–30) · <span className="text-accent">—</span> eye contact %</p>
            </div>
          )}
          <div className="mt-2">{s.rows.map((r) => <MetricLine key={r.key} r={r} onJump={onJump} />)}</div>
        </Card>
      ))}
    </div>
  );
}

export default function Report() {
  const id = Number(useParams<{ id: string }>().id);
  const [rep, setRep] = useState<ReportRow | null>(null);
  const [data, setData] = useState<InterviewData | null>(null);
  const [cv, setCv] = useState<CandidateView | null>(null);
  const [tab, setTab] = useState("overview");
  const [link, setLink] = useState<string | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [copied, setCopied] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const [printAll, setPrintAll] = useState(false);
  useEffect(() => {   // the PDF button prints every tab as one report
    const before = () => setPrintAll(true);
    const after = () => setPrintAll(false);
    window.addEventListener("beforeprint", before);
    window.addEventListener("afterprint", after);
    return () => { window.removeEventListener("beforeprint", before); window.removeEventListener("afterprint", after); };
  }, []);

  async function load(rebuild = false) {
    try {
      const [existing, d] = await Promise.all([api.report(id).catch(() => null), api.data(id)]);
      // a report built while the interview was still running is out of date once it has ended
      const endedAt = d.interview.ended_at ? new Date(d.interview.ended_at).getTime() : null;
      const stale = !!existing && endedAt != null && new Date(existing.updated_at).getTime() < endedAt;
      const r = rebuild || !existing || stale ? await api.buildReport(id) : existing;
      setRep(r);
      setCv(r.candidate_view);
      if (r.release_token) setLink(`${location.origin}/feedback/${r.release_token}`);
      setData(d);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  useEffect(() => { load(); }, [id]);   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    // printing (button or Ctrl+P): show every section and open every question, not just the visible tab
    const before = () => flushSync(() => setPrintAll(true));
    const after = () => setPrintAll(false);
    window.addEventListener("beforeprint", before);
    window.addEventListener("afterprint", after);
    return () => { window.removeEventListener("beforeprint", before); window.removeEventListener("afterprint", after); };
  }, []);

  const duration = useMemo(() => Math.max(1, ...(data?.transcript.map((s) => s.end_s) ?? [0]), ...(rep?.interviewer_view.timeline.map((x) => x.t_s) ?? [0])), [data, rep]);
  useEffect(() => {
    if (!playing) return;
    timer.current = setInterval(() => setT((v) => (v >= duration ? (setPlaying(false), v) : v + 0.5)), 500);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [playing, duration]);

  if (error) return <><Header /><main className="mx-auto max-w-4xl p-10"><Card><Empty title="Couldn't load the report" text={error} /></Card></main></>;
  if (!rep || !cv) return <><Header /><main className="mx-auto max-w-4xl space-y-4 p-10">{[0, 1, 2].map((i) => <div key={i} className="h-32 animate-pulse rounded-3xl bg-surface-2" />)}</main></>;
  const v = rep.interviewer_view;
  const show = (k: string) => printAll || tab === k;
  const jump = (s: number) => { setT(Math.max(0, s)); setTab("overview"); };
  const trend = v.timeline.map((x) => ({ t: x.t_s, attention: x.attention, engaged: x.engagement === "engaged" ? 90 : x.engagement === "neutral" ? 50 : 10 }));
  const nowLines = data?.transcript.filter((s) => s.start_s <= t + 0.5 && s.end_s >= t - 5) ?? [];
  const date = data?.interview.started_at ? new Date(data.interview.started_at).toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" }) : "";

  async function saveAndRelease() {
    if (!cv) return;
    setSaving(true);
    setActionError(null);
    try {
      await api.editReport(id, cv);
      const r = await api.release(id);
      setLink(`${location.origin}/feedback/${r.release_token}`);
      setRep({ ...rep!, released: true });
    } catch (e) {
      setActionError(`Could not save and release: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <Header right={
        <div className="no-print flex gap-1">
          <Button variant="ghost" size="sm" onClick={() => load(true)}><RefreshCw size={14} /> Refresh</Button>
          <Button variant="ghost" size="sm" onClick={() => window.print()}><Download size={14} /> PDF</Button>
          <Button variant="ghost" size="sm" onClick={async () => {
            try {
              const html = await api.reportHtml(id);
              const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
              const link = Object.assign(document.createElement("a"), { href: url, download: `interlens-report-${id}.html` });
              document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url);
            } catch (e) { setActionError((e as Error).message); }
          }}><Download size={14} /> HTML</Button>
        </div>} />
      <main className="mx-auto max-w-4xl px-6 py-12">
        <p className="text-[13px] text-muted">{data?.interview.title}{date ? ` · ${date}` : ""}</p>
        <div className="mt-4 flex flex-wrap items-center gap-10">
          <ScoreRing value={v.scorecard?.overall ?? v.summary.mean_effectiveness} label={v.scorecard?.grade ? `grade ${v.scorecard.grade}` : "overall"} />
          <div className="min-w-60 flex-1">
            <h1 className="font-display text-[34px] leading-tight tracking-tight">{data?.interview.candidate_name ?? "Candidate"}</h1>
            <p className="mt-2 text-[17px] text-muted">{verdict(v.scorecard?.overall ?? v.summary.mean_effectiveness)}</p>
            <div className="mt-5 flex flex-wrap gap-x-8 gap-y-2 text-sm">
              <span><span className="font-display text-xl">{v.summary.questions}</span> <span className="text-muted">questions</span></span>
              <span><span className="font-display text-xl">{v.summary.mean_effectiveness != null ? Math.round(v.summary.mean_effectiveness) : "–"}</span> <span className="text-muted">avg. answer score</span></span>
              <span><span className="font-display text-xl">{v.summary.mean_attention != null ? Math.round(v.summary.mean_attention) : "–"}</span> <span className="text-muted">avg. attention</span></span>
              <span><span className="font-display text-xl">{v.summary.follow_ups}</span> <span className="text-muted">follow-ups suggested</span></span>
            </div>
          </div>
        </div>

        <div className="no-print mt-12"><Tabs items={[{ value: "overview", label: "Overview" }, { value: "scorecard", label: "Scorecard" }, { value: "behaviour", label: "Behaviour" },
          { value: "questions", label: "Questions" }, { value: "feedback", label: "Candidate feedback" }]} value={tab} onChange={setTab} /></div>

        {show("overview") && (
          <div className="mt-8 grid gap-6 md:grid-cols-[1fr_1.3fr]">
            <Card>
              <SectionLabel>Communication profile</SectionLabel>
              <SubscoreRadar data={v.radar} />
            </Card>
            <Card>
              <SectionLabel>Replay</SectionLabel>
              <TrendChart data={trend} markers={v.follow_ups.map((f) => ({ t: f.t_s, label: f.reason }))} onPick={setT} cursor={t} height={170} />
              <div className="no-print mt-3 flex items-center gap-3">
                <Button variant="quiet" size="icon" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause size={15} /> : <Play size={15} />}</Button>
                <input type="range" min={0} max={duration} step={0.5} value={t} onChange={(e) => setT(Number(e.target.value))} className="flex-1 accent-[var(--accent)]" />
                <span className="w-12 text-right text-xs tabular-nums text-muted">{mmss(t)}</span>
              </div>
              <div className="mt-3 min-h-14 space-y-1.5 text-sm">
                {nowLines.length === 0 ? <p className="text-faint">Move the slider to replay the transcript.</p> :
                  nowLines.map((s, i) => <p key={i}><span className="mr-2 text-[11px] tracking-wider text-accent uppercase">{s.speaker === "interviewer" ? "You" : "Candidate"}</span>{s.text}</p>)}
              </div>
            </Card>
          </div>
        )}

        {show("overview") && !!v.moments?.length && (
          <Card className="mt-6 px-6">
            <SectionLabel>Moments to rewatch</SectionLabel>
            <Moments items={v.moments} onJump={jump} />
          </Card>
        )}

        {show("overview") && !!v.combined_flags?.length && (
          <Card className="mt-6 px-6">
            <SectionLabel>Where several signals dropped together</SectionLabel>
            <p className="mt-1 text-[13px] text-muted">One pause or glance away is normal. These are stretches where more than one signal got worse at the same time, compared with this candidate&apos;s own average.</p>
            <CombinedFlags items={v.combined_flags} onJump={jump} />
          </Card>
        )}

        {show("scorecard") && (
          <div className="mt-8 space-y-6">
            {v.scorecard ? <ScorecardView sc={v.scorecard} /> : <Card><Empty title="No scorecard yet" text="Press Refresh to rebuild this report." /></Card>}
            {v.matrix && v.matrix.points.length > 0 && (
              <Card>
                <SectionLabel>Confidence vs content</SectionLabel>
                <p className="mt-1 text-[13px] text-muted">Each number is a question. Right = what was said was relevant; up = how sure it sounded.</p>
                <ConfidenceMatrix points={v.matrix.points} cut={v.matrix.cut}
                  onPick={(ref) => { const c = v.cards.find((x) => x.question_ref === ref); if (c) jump(c.ask_s); }} />
                <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
                  {quadrants.map(([k, l]) => (
                    <span key={k} className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: quadrantTone[k] }} />{l} ({v.matrix!.counts[k] ?? 0})</span>))}
                </div>
                {v.matrix.note && <p className="mt-3 rounded-2xl bg-accent-soft px-4 py-2.5 text-[13px] text-accent">{v.matrix.note}</p>}
              </Card>
            )}
            {v.pressure && (
              <Card>
                <SectionLabel>Under pressure</SectionLabel>
                <p className="mt-1 text-[15px]">{v.pressure.summary}</p>
                <table className="mt-4 w-full text-[13px]">
                  <thead><tr className="text-left text-xs text-faint"><th className="py-1.5 font-normal">Signal</th>
                    <th className="font-normal">Easier ({v.pressure.easier_n})</th><th className="font-normal">Harder ({v.pressure.harder_n})</th><th className="font-normal">Change</th></tr></thead>
                  <tbody>{v.pressure.rows.map((r) => (
                    <tr key={r.key} className="border-t border-border"><td className="py-2">{r.label}</td>
                      <td className="tabular-nums">{r.easier}{r.unit}</td><td className="tabular-nums">{r.harder}{r.unit}</td>
                      <td className={cn("tabular-nums", r.worse ? "text-warn" : "text-good")}>{r.change > 0 ? "+" : ""}{r.change}{r.unit}</td></tr>))}
                  </tbody>
                </table>
              </Card>
            )}
          </div>
        )}

        {show("behaviour") && (
          <div className="mt-8">
            {v.behaviour ? <BehaviourView sections={v.behaviour} onJump={jump} cursor={t} /> : <Card><Empty title="No behaviour details yet" text="Press Refresh to rebuild this report." /></Card>}
          </div>
        )}

        {show("overview") && v.delivery && (
          <Card className="mt-6">
            <SectionLabel>Delivery</SectionLabel>
            <div className="mt-1 grid grid-cols-2 gap-x-8 gap-y-5 sm:grid-cols-3">
              {([
                ["Pace", v.delivery.wpm, "wpm"], ["Pauses", v.delivery.pause_pct, "% of speech time"], ["Fillers", v.delivery.fillers_per_100w, "per 100 words"],
                ["Hedges", v.delivery.hedges_per_100w, "per 100 words"], ["Camera-facing", v.delivery.camera_facing_pct, "% of the time"], ["Blinks", v.delivery.blinks_per_min, "per minute"],
              ] as [string, number | null, string][]).map(([label, val, unit]) => (
                <div key={label}>
                  <p className="text-[13px] text-muted">{label}</p>
                  <p className="font-display text-2xl tabular-nums">{val == null ? "–" : Math.round(val * 10) / 10}</p>
                  <p className="text-xs text-faint">{unit}</p>
                </div>
              ))}
            </div>
            <p className="mt-4 text-xs text-faint">Averages over the whole interview. Camera-facing is head pose and gaze toward the screen, not exact eye tracking.</p>
            <SectionLabel className="mt-6">Flagged moments</SectionLabel>
            <FlaggedMoments alerts={(v.flags ?? []).map((f) => ({ t_s: f.t_s, text: f.text, alert: f.kind }))} onPick={setT} />
          </Card>
        )}

        {show("questions") && (
          <Card className="mt-8 px-6 py-1">
            {v.cards.length === 0 ? <Empty title="No answers were scored" /> : v.cards.map((c) => (
              <Disclosure key={c.question_ref} forceOpen={printAll} summary={
                <div className="flex items-center gap-5">
                  <span className={cn("w-10 font-display text-2xl tabular-nums", { good: "text-good", warn: "text-warn", bad: "text-bad", neutral: "text-faint" }[scoreTone(c.effectiveness)])}>
                    {c.effectiveness != null ? Math.round(c.effectiveness) : "–"}
                  </span>
                  <div className="min-w-0">
                    <p className="truncate font-medium">{c.question}</p>
                    <p className="text-[13px] text-muted">{c.category} · {mmss(c.ask_s)}{c.follow_ups.length ? " · follow-up suggested" : ""}</p>
                  </div>
                </div>}>
                <div className="space-y-5 pl-[60px]">
                  <p className="text-[15px] leading-relaxed text-muted">{c.transcript || "No transcript for this answer."}</p>
                  {c.reasons.length > 0 && (
                    <div>
                      <SectionLabel className="mb-2">Why this score</SectionLabel>
                      <ul className="space-y-1.5 text-sm">
                        {c.reasons.map((r, i) => <li key={i} className="flex gap-2.5"><span className={cn("mt-2 h-1.5 w-1.5 shrink-0 rounded-full", r.contribution >= 0 ? "bg-good" : "bg-bad")} />{r.text}</li>)}
                      </ul>
                    </div>
                  )}
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(c.subscores).map(([k, s]) => <Badge key={k} tone={scoreTone(s)}>{k} {Math.round(s)}</Badge>)}
                  </div>
                  <p className="text-xs text-faint">
                    {[c.speaking_pattern, c.response_latency_s != null && `${c.response_latency_s.toFixed(1)} s to start`,
                      c.hesitation_index != null && `hesitation ${Math.round(c.hesitation_index)}`,
                      c.answer_model.relevance != null && `relevance ${c.answer_model.relevance.toFixed(2)}`].filter(Boolean).join(" · ")}
                  </p>
                  <div className="grid gap-x-8 gap-y-1 text-[13px] text-muted sm:grid-cols-2">
                    {c.longest_pause_s != null && <p>Longest pause <b className="font-medium text-fg">{c.longest_pause_s.toFixed(1)} s</b>{c.long_pauses ? ` · ${c.long_pauses} over 1 s` : ""}</p>}
                    {c.voice_breaks != null && <p>Voice breaks <b className="font-medium text-fg">{c.voice_breaks}</b>{c.pitch_jitter_pct != null ? ` · jitter ${c.pitch_jitter_pct.toFixed(1)} %` : ""}</p>}
                    {c.eyes?.blink_rate != null && <p>Blink rate <b className="font-medium text-fg">{Math.round(c.eyes.blink_rate)}/min</b>{c.eyes.on_screen_pct != null ? ` · eye contact ${Math.round(c.eyes.on_screen_pct)} %` : ""}{c.eyes.reading ? " · reading pattern" : ""}</p>}
                    {c.wpm != null && <p>Pace <b className="font-medium text-fg">{Math.round(c.wpm)} wpm</b>{c.filler_rate != null ? ` · ${c.filler_rate} fillers / 100 words` : ""}</p>}
                  </div>
                  <Chips data={c.words?.fillers} tone="warn" />
                  {c.quadrant && <Badge tone="accent">{c.quadrant}</Badge>}
                  {c.follow_ups.map((f, i) => <p key={i} className="rounded-2xl bg-warn/10 px-4 py-2.5 text-[13px] text-warn">Follow-up ({f.reason.replace(/_/g, " ")}): {f.text}</p>)}
                </div>
              </Disclosure>
            ))}
          </Card>
        )}

        {show("feedback") && (
          <div className="mt-8 space-y-6">
            <p className="text-[15px] text-muted">This is what the candidate will receive. Edit anything you like, then release it. The candidate never sees scores or analytics.</p>
            <Card className="space-y-6">
              <div>
                <SectionLabel className="mb-3">What went well</SectionLabel>
                <div className="space-y-2">
                  {cv.strengths.map((s, i) => <Input key={i} value={s} onChange={(e) => setCv({ ...cv, strengths: cv.strengths.map((x, j) => (j === i ? e.target.value : x)) })} />)}
                </div>
              </div>
              <div>
                <SectionLabel className="mb-3">Things to work on</SectionLabel>
                <div className="space-y-4">
                  {cv.improvements.map((imp, i) => (
                    <div key={i} className="space-y-2 rounded-2xl bg-surface-2/60 p-4">
                      <Input value={imp.observation} onChange={(e) => setCv({ ...cv, improvements: cv.improvements.map((x, j) => (j === i ? { ...x, observation: e.target.value } : x)) })} />
                      <Textarea rows={2} value={imp.tip} onChange={(e) => setCv({ ...cv, improvements: cv.improvements.map((x, j) => (j === i ? { ...x, tip: e.target.value } : x)) })} />
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <SectionLabel className="mb-3">Closing note</SectionLabel>
                <Textarea rows={3} value={cv.closing} onChange={(e) => setCv({ ...cv, closing: e.target.value })} />
              </div>
            </Card>
            <div className="flex flex-wrap items-center gap-3">
              <Button size="lg" variant="accent" onClick={saveAndRelease} disabled={saving}><Send size={15} /> {rep.released ? "Save and update" : "Release to candidate"}</Button>
              {link && (
                <Button size="lg" variant="outline" onClick={async () => { if (await copyText(link)) { setCopied(true); setTimeout(() => setCopied(false), 1600); } }}>
                  {copied ? <Check size={15} /> : <Copy size={15} />} Copy feedback link
                </Button>
              )}
              {rep.released && <Badge tone="good">Released</Badge>}
            </div>
            {actionError && <p className="rounded-2xl bg-bad/10 px-4 py-3 text-sm text-bad">{actionError}</p>}
          </div>
        )}
        <p className="mt-14 text-xs text-faint">{v.note}</p>
      </main>
    </>
  );
}
