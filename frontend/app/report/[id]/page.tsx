"use client";

import { Check, Copy, Download, Pause, Play, RefreshCw, Send } from "lucide-react";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { SubscoreRadar, TrendChart } from "@/components/charts";
import { Header } from "@/components/shell";
import { Badge, Button, Card, CardTitle, cn, Input, mmss, scoreTone } from "@/components/ui";
import { api, type CandidateView, type InterviewData, type ReportRow } from "@/lib/api";

export default function Report() {
  const id = Number(useParams<{ id: string }>().id);
  const [rep, setRep] = useState<ReportRow | null>(null);
  const [data, setData] = useState<InterviewData | null>(null);
  const [cv, setCv] = useState<CandidateView | null>(null);
  const [link, setLink] = useState<string | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  async function load(rebuild = false) {
    try {
      const r = rebuild ? await api.buildReport(id) : await api.report(id).catch(() => api.buildReport(id));
      setRep(r);
      setCv(r.candidate_view);
      if (r.release_token) setLink(`${location.origin}/feedback/${r.release_token}`);
      setData(await api.data(id));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  useEffect(() => { load(); }, [id]);   // eslint-disable-line react-hooks/exhaustive-deps

  const duration = useMemo(() => Math.max(0, ...(data?.transcript.map((s) => s.end_s) ?? [0]), ...(rep?.interviewer_view.timeline.map((x) => x.t_s) ?? [0])), [data, rep]);
  useEffect(() => {
    if (!playing) return;
    timer.current = setInterval(() => setT((v) => (v >= duration ? (setPlaying(false), v) : v + 0.5)), 500);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [playing, duration]);

  if (error) return <><Header /><main className="mx-auto max-w-5xl p-6 text-bad">{error}</main></>;
  if (!rep || !cv) return <><Header /><main className="mx-auto max-w-5xl p-6 text-muted">Building report...</main></>;
  const v = rep.interviewer_view;
  const trend = v.timeline.map((x) => ({ t: x.t_s, attention: x.attention, engaged: x.engagement === "engaged" ? 90 : x.engagement === "neutral" ? 50 : 10 }));
  const current = data?.transcript.filter((s) => s.start_s <= t + 0.5 && s.end_s >= t - 4) ?? [];

  async function release() {
    if (!cv) return;
    await api.editReport(id, cv);
    const r = await api.release(id);
    setLink(`${location.origin}/feedback/${r.release_token}`);
  }

  return (
    <>
      <Header right={
        <div className="no-print flex gap-2">
          <Button variant="outline" size="sm" onClick={() => load(true)}><RefreshCw size={14} /> Rebuild</Button>
          <Button variant="outline" size="sm" onClick={() => window.print()}><Download size={14} /> Export PDF</Button>
        </div>} />
      <main className="mx-auto max-w-6xl space-y-6 px-4 py-6">
        <div>
          <h1 className="text-2xl font-semibold">{data?.interview.title}</h1>
          <p className="text-sm text-muted">{data?.interview.candidate_name} · {v.summary.questions} questions · {v.summary.follow_ups} follow-up flags</p>
        </div>
        <div className="grid gap-4 md:grid-cols-4">
          {[["Communication effectiveness", v.summary.mean_effectiveness], ["Mean attention", v.summary.mean_attention],
            ["Engaged time", v.summary.engaged_share != null ? v.summary.engaged_share * 100 : null], ["Follow-ups", v.summary.follow_ups]].map(([k, val]) => (
            <Card key={k as string}><CardTitle>{k}</CardTitle><div className="text-3xl font-semibold tabular-nums">{val == null ? "-" : Math.round(val as number)}</div></Card>
          ))}
        </div>
        <div className="grid gap-4 md:grid-cols-[1fr_1.4fr]">
          <Card><CardTitle>Sub-scores</CardTitle><SubscoreRadar data={v.radar} /></Card>
          <Card>
            <CardTitle>Replay (timeline synced with the transcript)</CardTitle>
            <TrendChart data={trend} markers={v.follow_ups.map((f) => ({ t: f.t_s, label: f.reason }))} onPick={setT} />
            <div className="no-print mt-2 flex items-center gap-3">
              <Button size="sm" variant="outline" onClick={() => setPlaying(!playing)}>{playing ? <Pause size={14} /> : <Play size={14} />}</Button>
              <input type="range" min={0} max={duration} step={0.5} value={t} onChange={(e) => setT(Number(e.target.value))} className="flex-1 accent-[var(--accent)]" />
              <span className="w-12 text-xs tabular-nums">{mmss(t)}</span>
            </div>
            <div className="mt-2 min-h-16 space-y-1 text-sm">
              {current.map((s, i) => <div key={i}><span className="mr-2 text-xs font-semibold text-accent">{s.speaker}</span>{s.text}</div>)}
            </div>
          </Card>
        </div>
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Questions</h2>
          {v.cards.map((c) => (
            <Card key={c.question_ref} className="print-break-avoid">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-xs text-muted">Q{c.order} · {c.category} · {mmss(c.ask_s)}</div>
                  <div className="font-medium">{c.question}</div>
                </div>
                <Badge tone={scoreTone(c.effectiveness)} className="text-sm">{c.effectiveness ?? "n/a"}</Badge>
              </div>
              <p className="mt-2 text-sm text-muted">{c.transcript || "(no transcript)"}</p>
              <div className="mt-3 flex flex-wrap gap-2 text-xs">
                {Object.entries(c.subscores).map(([k, s]) => <Badge key={k} tone={scoreTone(s)}>{k} {Math.round(s)}</Badge>)}
                {c.speaking_pattern && <Badge>{c.speaking_pattern}</Badge>}
                {c.hesitation_index != null && <Badge>hesitation {Math.round(c.hesitation_index)}</Badge>}
                {c.response_latency_s != null && <Badge>delay {c.response_latency_s.toFixed(1)} s</Badge>}
                {c.answer_model.relevance != null && <Badge>relevance {c.answer_model.relevance.toFixed(2)}</Badge>}
              </div>
              {c.reasons.length > 0 && (
                <ul className="mt-3 space-y-1 text-sm">
                  {c.reasons.map((r, i) => <li key={i} className="flex gap-2"><span className={cn("mt-1.5 h-2 w-2 rounded-full", r.contribution >= 0 ? "bg-good" : "bg-bad")} />{r.text}</li>)}
                </ul>
              )}
              {c.follow_ups.map((f, i) => <div key={i} className="mt-2 rounded-lg bg-amber-500/10 px-2 py-1 text-xs text-amber-700 dark:text-amber-300">Follow-up: {f.reason.replace(/_/g, " ")} - {f.text}</div>)}
            </Card>
          ))}
        </section>
        <Card className="no-print space-y-4">
          <div className="flex items-center justify-between">
            <CardTitle className="mb-0">Candidate feedback (edit, then release)</CardTitle>
            {rep.released ? <Badge tone="good">released</Badge> : <Badge tone="warn">not released</Badge>}
          </div>
          <div>
            <div className="mb-1 text-sm font-medium">Strengths</div>
            {cv.strengths.map((s, i) => (
              <Input key={i} className="mb-2" value={s} onChange={(e) => setCv({ ...cv, strengths: cv.strengths.map((x, j) => (j === i ? e.target.value : x)) })} />
            ))}
          </div>
          <div>
            <div className="mb-1 text-sm font-medium">Areas to improve</div>
            {cv.improvements.map((imp, i) => (
              <div key={i} className="mb-3 grid gap-2 rounded-xl border border-line p-3">
                <Input value={imp.observation} onChange={(e) => setCv({ ...cv, improvements: cv.improvements.map((x, j) => (j === i ? { ...x, observation: e.target.value } : x)) })} />
                <textarea className="rounded-xl border border-line bg-surface p-2 text-sm" value={imp.tip}
                  onChange={(e) => setCv({ ...cv, improvements: cv.improvements.map((x, j) => (j === i ? { ...x, tip: e.target.value } : x)) })} />
              </div>
            ))}
          </div>
          <textarea className="w-full rounded-xl border border-line bg-surface p-2 text-sm" value={cv.closing} onChange={(e) => setCv({ ...cv, closing: e.target.value })} />
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={release}><Send size={14} /> {rep.released ? "Update released feedback" : "Release to candidate"}</Button>
            {link && (
              <Button variant="outline" onClick={async () => { await navigator.clipboard.writeText(link); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>
                {copied ? <Check size={14} /> : <Copy size={14} />} Copy candidate link
              </Button>
            )}
          </div>
        </Card>
        <p className="text-xs text-muted">{v.note}</p>
      </main>
    </>
  );
}
