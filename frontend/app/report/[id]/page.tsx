"use client";

import { Check, Copy, Download, Pause, Play, RefreshCw, Send } from "lucide-react";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { ScoreRing, SubscoreRadar, TrendChart } from "@/components/charts";
import { Header } from "@/components/shell";
import { Badge, Button, Card, cn, Disclosure, Empty, Input, mmss, scoreTone, SectionLabel, Tabs, Textarea } from "@/components/ui";
import { api, type CandidateView, type InterviewData, type ReportRow } from "@/lib/api";

function verdict(v: number | null | undefined) {
  if (v == null) return "Not enough data to score this interview yet.";
  if (v >= 70) return "Clear, confident communication overall.";
  if (v >= 55) return "Solid overall, with a few answers worth revisiting.";
  return "Several answers would benefit from follow-up questions.";
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

  const duration = useMemo(() => Math.max(1, ...(data?.transcript.map((s) => s.end_s) ?? [0]), ...(rep?.interviewer_view.timeline.map((x) => x.t_s) ?? [0])), [data, rep]);
  useEffect(() => {
    if (!playing) return;
    timer.current = setInterval(() => setT((v) => (v >= duration ? (setPlaying(false), v) : v + 0.5)), 500);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [playing, duration]);

  if (error) return <><Header /><main className="mx-auto max-w-4xl p-10"><Card><Empty title="Couldn't load the report" text={error} /></Card></main></>;
  if (!rep || !cv) return <><Header /><main className="mx-auto max-w-4xl space-y-4 p-10">{[0, 1, 2].map((i) => <div key={i} className="h-32 animate-pulse rounded-3xl bg-surface-2" />)}</main></>;
  const v = rep.interviewer_view;
  const trend = v.timeline.map((x) => ({ t: x.t_s, attention: x.attention, engaged: x.engagement === "engaged" ? 90 : x.engagement === "neutral" ? 50 : 10 }));
  const nowLines = data?.transcript.filter((s) => s.start_s <= t + 0.5 && s.end_s >= t - 5) ?? [];
  const date = data?.interview.started_at ? new Date(data.interview.started_at).toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" }) : "";

  async function saveAndRelease() {
    if (!cv) return;
    setSaving(true);
    await api.editReport(id, cv);
    const r = await api.release(id);
    setLink(`${location.origin}/feedback/${r.release_token}`);
    setRep({ ...rep!, released: true });
    setSaving(false);
  }

  return (
    <>
      <Header right={
        <div className="no-print flex gap-1">
          <Button variant="ghost" size="sm" onClick={() => load(true)}><RefreshCw size={14} /> Refresh</Button>
          <Button variant="ghost" size="sm" onClick={() => window.print()}><Download size={14} /> PDF</Button>
        </div>} />
      <main className="mx-auto max-w-4xl px-6 py-12">
        <p className="text-[13px] text-muted">{data?.interview.title}{date ? ` · ${date}` : ""}</p>
        <div className="mt-4 flex flex-wrap items-center gap-10">
          <ScoreRing value={v.summary.mean_effectiveness} label="overall" />
          <div className="min-w-60 flex-1">
            <h1 className="font-display text-[34px] leading-tight tracking-tight">{data?.interview.candidate_name ?? "Candidate"}</h1>
            <p className="mt-2 text-[17px] text-muted">{verdict(v.summary.mean_effectiveness)}</p>
            <div className="mt-5 flex flex-wrap gap-x-8 gap-y-2 text-sm">
              <span><span className="font-display text-xl">{v.summary.questions}</span> <span className="text-muted">questions</span></span>
              <span><span className="font-display text-xl">{v.summary.mean_attention != null ? Math.round(v.summary.mean_attention) : "–"}</span> <span className="text-muted">avg. attention</span></span>
              <span><span className="font-display text-xl">{v.summary.follow_ups}</span> <span className="text-muted">follow-ups suggested</span></span>
            </div>
          </div>
        </div>

        <div className="no-print mt-12"><Tabs items={[{ value: "overview", label: "Overview" }, { value: "questions", label: "Questions" }, { value: "feedback", label: "Candidate feedback" }]} value={tab} onChange={setTab} /></div>

        {(tab === "overview") && (
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

        {(tab === "questions") && (
          <Card className="mt-8 px-6 py-1">
            {v.cards.length === 0 ? <Empty title="No answers were scored" /> : v.cards.map((c) => (
              <Disclosure key={c.question_ref} summary={
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
                  {c.follow_ups.map((f, i) => <p key={i} className="rounded-2xl bg-warn/10 px-4 py-2.5 text-[13px] text-warn">Follow-up ({f.reason.replace(/_/g, " ")}): {f.text}</p>)}
                </div>
              </Disclosure>
            ))}
          </Card>
        )}

        {(tab === "feedback") && (
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
                <Button size="lg" variant="outline" onClick={async () => { await navigator.clipboard.writeText(link); setCopied(true); setTimeout(() => setCopied(false), 1600); }}>
                  {copied ? <Check size={15} /> : <Copy size={15} />} Copy feedback link
                </Button>
              )}
              {rep.released && <Badge tone="good">Released</Badge>}
            </div>
          </div>
        )}
        <p className="mt-14 text-xs text-faint">{v.note}</p>
      </main>
    </>
  );
}
