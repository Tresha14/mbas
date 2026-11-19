"use client";

import { useMemo } from "react";
import { AlertCircle, ArrowDown, ArrowRight, ArrowUp, Eye, EyeOff, Lightbulb, Pause, ScanFace } from "lucide-react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { State } from "@/lib/analytics";
import { toneVsBaseline, NEEDED, progress, useBaseline, type MetricKey, type Sample } from "@/lib/baseline";
import { Button, Card, cn, Dot, Empty, mmss, SectionLabel } from "./ui";

type Tone = "good" | "warn" | "bad" | "neutral";
const toneText: Record<Tone, string> = { good: "text-good", warn: "text-warn", bad: "text-bad", neutral: "text-faint" };

const pct = (r: number | null | undefined) => (r == null ? null : r <= 1.5 ? r * 100 : r);
const num = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

/** Change versus the average of the samples before it; null until there is enough history. */
function delta(series: (number | null | undefined)[], lookback: number): number | null {
  const v = series.filter(num);
  if (v.length < 3) return null;
  const last = v[v.length - 1];
  const prev = v.slice(-1 - lookback, -1);
  return last - prev.reduce((a, b) => a + b, 0) / prev.length;
}

function Tile({ label, value, unit, tone, change, higherIsBetter, hint, digits = 0, base, learning }: {
  label: string; value: number | null | undefined; unit: string; tone: Tone; change: number | null; higherIsBetter: boolean; hint: string; digits?: number;
  base?: number | null; learning?: string;
}) {
  const small = Math.abs(change ?? 0) < (unit === "%" ? 2 : 0.6);
  const up = (change ?? 0) > 0;
  const Arrow = change == null || small ? ArrowRight : up ? ArrowUp : ArrowDown;
  const arrowTone: Tone = change == null || small ? "neutral" : up === higherIsBetter ? "good" : "bad";
  return (
    <div className="rounded-2xl border border-line bg-surface px-4 py-3.5" title={hint}>
      <p className="text-[12px] text-muted">{label}</p>
      <div className="mt-1 flex items-baseline gap-1">
        <span className={cn("font-display text-[28px] leading-none tabular-nums", toneText[value == null ? "neutral" : tone])}>
          {value == null ? "–" : value.toFixed(digits)}
        </span>
        {value != null && <span className="text-xs text-faint">{unit}</span>}
      </div>
      <p className={cn("mt-2 flex items-center gap-1 text-[12px] tabular-nums", toneText[arrowTone])}>
        <Arrow size={12} />
        {change == null ? "gathering data" : small ? "steady" : `${up ? "+" : "−"}${Math.abs(change).toFixed(digits)} ${unit}`}
      </p>
      <p className="mt-1 text-[11px] text-faint">{base != null ? `Their usual: ${base.toFixed(digits)} ${unit}` : learning}</p>
    </div>
  );
}

function MiniArea({ title, data, unit, domain }: { title: string; data: { t: number; y: number | null }[]; unit: string; domain?: [number, number] }) {
  const id = `fill-${title.replace(/\W/g, "")}`;
  return (
    <div>
      <p className="mb-1 text-[12px] text-muted">{title}</p>
      <div className="h-36 w-full">
        {data.filter((d) => d.y != null).length < 2 ? <Empty title="Waiting for data" /> : (
          <ResponsiveContainer>
            <AreaChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: -26 }}>
              <defs>
                <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.22} />
                  <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} stroke="var(--border)" />
              <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => mmss(v)} tick={{ fontSize: 10, fill: "var(--faint)" }} axisLine={false} tickLine={false} />
              <YAxis domain={domain ?? ["auto", "auto"]} tick={{ fontSize: 10, fill: "var(--faint)" }} axisLine={false} tickLine={false} />
              <Tooltip labelFormatter={(v) => mmss(Number(v))} formatter={(v) => [`${Math.round(Number(v))}${unit}`, title]}
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 14, fontSize: 12 }} />
              <Area type="monotone" dataKey="y" stroke="var(--accent)" strokeWidth={1.75} fill={`url(#${id})`} isAnimationActive={false} connectNulls />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

export function tips(state: State): string[] {
  const w = state.windows[state.windows.length - 1];
  const sp = state.speaking;
  const facing = pct(w?.on_screen_ratio);
  const out: string[] = [];
  if (facing != null && facing < 45) out.push("Candidate is mostly looking away from the screen. Ask a direct question to bring them back.");
  if (w?.reading) out.push("Eye movements look like reading. Ask a follow-up that cannot be answered from notes.");
  if (sp?.wpm != null && sp.wpm > 175) out.push("Speaking very fast. They may be nervous, so slow the pace or give them a moment.");
  if (sp?.wpm != null && sp.wpm < 90) out.push("Speaking slowly. Check whether they understood the question.");
  if (sp?.pause_ratio != null && sp.pause_ratio > 0.5) out.push("Many long pauses. Offer a hint or rephrase the question.");
  if (sp?.fillers_per_100w != null && sp.fillers_per_100w > 8) out.push("Lots of filler words, which can signal uncertainty. Probe the details of this answer.");
  if (sp?.hedges_per_100w != null && sp.hedges_per_100w > 5) out.push("Frequent hedging (“I think”, “maybe”). Ask how confident they are and what they have actually done.");
  if (sp?.pitch_variation_st != null && sp.pitch_variation_st < 1.5) out.push("Flat voice. Ask something they care about to draw out more energy.");
  return out.slice(0, 3);
}

/** Six live tiles, two live charts and coaching tips. Updates every few seconds while the interview is running. */
export function LiveMetrics({ state, id }: { state: State; id: number }) {
  const w = state.windows;
  const lastW = w[w.length - 1];
  const sp = state.speaking;
  const facingSeries = w.map((x) => pct(x.on_screen_ratio));
  const facing = pct(lastW?.on_screen_ratio);
  const blink = lastW?.blink_rate ?? null;
  const pauseSeries = state.speakingHist.map((s) => (s.pause_ratio == null ? null : s.pause_ratio * 100));
  const pause = pauseSeries[pauseSeries.length - 1] ?? null;
  const wpm = sp?.wpm ?? null;
  const fillers = sp?.fillers_per_100w ?? null;
  const hedges = sp?.hedges_per_100w ?? null;
  const tipList = tips(state);

  const series = useMemo<Record<MetricKey, Sample[]>>(() => ({
    facing: w.map((x, i) => ({ t: x.end_s, v: facingSeries[i] })),
    blink: w.map((x) => ({ t: x.end_s, v: x.blink_rate })),
    wpm: state.speakingHist.map((x) => ({ t: x.t_s, v: x.wpm })),
    pause: state.speakingHist.map((x, i) => ({ t: x.t_s, v: pauseSeries[i] })),
    fillers: state.speakingHist.map((x) => ({ t: x.t_s, v: x.fillers_per_100w })),
    hedges: state.speakingHist.map((x) => ({ t: x.t_s, v: x.hedges_per_100w })),
  }), [state.windows, state.speakingHist]);
  const now = Math.max(lastW?.end_s ?? 0, sp?.t_s ?? 0);
  const cal = useBaseline(id, series, now);

  const tone = (key: MetricKey, v: number | null | undefined, good: (x: number) => boolean, warn: (x: number) => boolean): Tone => {
    if (v == null) return "neutral";
    const b = cal.baseline[key];
    if (b != null) return toneVsBaseline(key, v, b);
    return good(v) ? "good" : warn(v) ? "warn" : "bad";
  };
  const learning = (key: MetricKey) => `Learning their normal ${progress(key, series[key], cal.since)}/${NEEDED[key]}`;
  const ready = cal.learned === cal.total;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3 rounded-2xl bg-surface-2 px-4 py-2.5 text-[13px] text-muted">
        <span>
          {ready
            ? "Calibrated: tiles compare this candidate with their own normal."
            : `Calibrating: learning this candidate's normal (${cal.learned}/${cal.total} metrics). Needs about 20 s of face and 20 s of speech.`}
        </span>
        <Button size="sm" variant="ghost" onClick={cal.reset}>Recalibrate</Button>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        <Tile label="Camera-facing" value={facing} unit="%" digits={0} higherIsBetter hint="Share of the last 5 s the candidate's head and gaze point at the screen (head pose + gaze models, not true eye tracking)"
          tone={tone("facing", facing, (x) => x >= 70, (x) => x >= 45)} change={delta(facingSeries, 12)} base={cal.baseline.facing} learning={learning("facing")} />
        <Tile label="Words per minute" value={wpm} unit="wpm" higherIsBetter hint="Speaking pace over the last 30 s. Comfortable range is about 110 to 170"
          tone={tone("wpm", wpm, (x) => x >= 110 && x <= 170, (x) => x >= 90 && x <= 190)} change={delta(state.speakingHist.map((s) => s.wpm), 6)} base={cal.baseline.wpm} learning={learning("wpm")} />
        <Tile label="Pause rate" value={pause} unit="%" higherIsBetter={false} hint="Share of the last 30 s with no speech"
          tone={tone("pause", pause, (x) => x <= 35, (x) => x <= 55)} change={delta(pauseSeries, 6)} base={cal.baseline.pause} learning={learning("pause")} />
        <Tile label="Fillers per 100 words" value={fillers} unit="/100w" digits={1} higherIsBetter={false} hint="“um”, “uh”, “you know”, “matlab” and similar, last 30 s"
          tone={tone("fillers", fillers, (x) => x <= 4, (x) => x <= 8)} change={delta(state.speakingHist.map((s) => s.fillers_per_100w), 6)} base={cal.baseline.fillers} learning={learning("fillers")} />
        <Tile label="Blinks per minute" value={blink} unit="/min" higherIsBetter={false} hint="Blink rate over the last 5 s window. Typical is 10 to 25"
          tone={tone("blink", blink, (x) => x >= 8 && x <= 28, (x) => x <= 40)} change={delta(w.map((x) => x.blink_rate), 12)} base={cal.baseline.blink} learning={learning("blink")} />
        <Tile label="Hedges per 100 words" value={hedges} unit="/100w" digits={1} higherIsBetter={false} hint="“I think”, “maybe”, “not sure” and similar, last 30 s"
          tone={tone("hedges", hedges, (x) => x <= 2, (x) => x <= 5)} change={delta(state.speakingHist.map((s) => s.hedges_per_100w), 6)} base={cal.baseline.hedges} learning={learning("hedges")} />
      </div>
      <p className="px-1 text-xs text-faint">Speech tiles need about 15 s of talking and use the last 30 s. “Camera-facing” is head pose and gaze toward the screen, not exact eye tracking.</p>

      <Card>
        <div className="grid gap-6 md:grid-cols-2">
          <MiniArea title="Camera-facing %" unit="%" domain={[0, 100]} data={w.map((x) => ({ t: x.end_s, y: pct(x.on_screen_ratio) }))} />
          <MiniArea title="Words per minute" unit=" wpm" data={state.speakingHist.map((s) => ({ t: s.t_s, y: s.wpm }))} />
        </div>
      </Card>

      <Card>
        <SectionLabel>Coaching</SectionLabel>
        {tipList.length === 0 ? (
          <p className="flex items-center gap-2 text-[13px] text-muted"><Lightbulb size={14} /> Nothing to flag right now.</p>
        ) : (
          <ul className="space-y-2">
            {tipList.map((t) => <li key={t} className="flex gap-2.5 text-[13px] leading-snug"><Lightbulb size={14} className="mt-0.5 shrink-0 text-warn" />{t}</li>)}
          </ul>
        )}
      </Card>
    </div>
  );
}

const flagMeta: Record<string, { label: string; icon: React.ReactNode }> = {
  long_pause: { label: "Long pause", icon: <Pause size={14} /> },
  off_screen: { label: "Looking away", icon: <EyeOff size={14} /> },
  no_face: { label: "Face not in frame", icon: <ScanFace size={14} /> },
  reading: { label: "Possible reading", icon: <Eye size={14} /> },
};

/** Persistent list of flagged moments (not just the 6 s pop-up). Click one to jump to that point in the transcript. */
export function FlaggedMoments({ alerts, onPick }: { alerts: State["alerts"]; onPick: (t: number) => void }) {
  if (alerts.length === 0) return <Empty title="No flagged moments yet" text="Long pauses, looking away and face-not-in-frame events are listed here as they happen." />;
  return (
    <ul className="max-h-80 divide-y divide-line overflow-y-auto px-3">
      {[...alerts].reverse().map((a, i) => {
        const m = flagMeta[a.alert] ?? { label: a.alert, icon: <AlertCircle size={14} /> };
        return (
          <li key={`${a.t_s}-${i}`}>
            <button onClick={() => onPick(a.t_s)} className="flex w-full items-center gap-3 py-2.5 text-left text-[13px] transition hover:text-accent">
              <span className="w-12 tabular-nums text-faint">{mmss(a.t_s)}</span>
              <span className="text-warn">{m.icon}</span>
              <span className="font-medium">{m.label}</span>
              <span className="truncate text-muted">{a.text}</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}


/** Is everything the analysis needs actually working? Camera (a face in frame), the candidate's microphone (any sound),
 *  and the transcript (words coming out when someone speaks). Each chip says what is wrong in plain words. */
export function SignalHealth({ state }: { state: State }) {
  const gaze = state.gaze.slice(-15);                          // about 3 s
  const h = state.health;
  const cand = h?.mics.candidate;
  const inter = h?.mics.interviewer;

  const face: [Tone, string] = gaze.length === 0 ? ["neutral", "Waiting for the camera"]
    : gaze.some((g) => g.face) ? ["good", state.models?.face_pretrained ? "Face detected (MediaPipe)" : "Face detected"] : ["bad", "No face in frame: ask the candidate to face the camera"];
  const mic = (m: typeof cand, who: string): [Tone, string] => !m ? ["neutral", `Waiting for ${who} audio`]
    : m.listening_s >= 6 && m.silent_s >= 6 ? ["bad", `No sound from ${who} microphone for ${Math.round(m.silent_s)} s: muted or not working`]
    : ["good", `${who} microphone is live`];
  const words: [Tone, string] = !cand ? ["neutral", "Waiting for speech"]
    : cand.speech_recent && (cand.since_words_s == null ? cand.listening_s > 15 : cand.since_words_s > 15)
      ? ["warn", `Hearing speech but no words for ${Math.round(cand.since_words_s ?? cand.listening_s)} s`]
      : cand.since_words_s == null ? ["neutral", "No words yet"] : ["good", `Transcript running (${h?.asr === "whisper" ? "Whisper" : "our model"})`];
  const items: [string, [Tone, string]][] = [["Camera", face], ["Candidate mic", mic(cand, "the candidate's")], ["Transcript", words]];
  if (inter) items.push(["Your mic", mic(inter, "your")]);
  const level = cand?.db == null ? 0 : Math.max(0, Math.min(1, (cand.db + 60) / 50));

  return (
    <div className="flex flex-wrap items-stretch gap-x-6 gap-y-2 rounded-2xl border border-border bg-surface px-4 py-2.5" aria-label="Signal health">
      {items.map(([label, [tone, text]]) => (
        <div key={label} className="flex min-w-0 items-center gap-2 text-[13px]" title={text}>
          <Dot tone={tone} pulse={tone === "bad"} />
          <span className="text-faint">{label}</span>
          <span className={cn("truncate", tone === "good" || tone === "neutral" ? "text-muted" : toneText[tone])}>{text}</span>
          {label === "Candidate mic" && cand?.db != null && (
            <span className="ml-1 h-1.5 w-14 overflow-hidden rounded-full bg-surface-2" aria-hidden><span className="block h-full bg-accent transition-all" style={{ width: `${level * 100}%` }} /></span>
          )}
        </div>
      ))}
    </div>
  );
}
