"use client";

import { motion } from "framer-motion";
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart,
  ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import type { GazeMsg } from "@/lib/analytics";
import { cn, mmss } from "./ui";

type Tone = "good" | "warn" | "bad" | "neutral";
const toneVar: Record<Tone, string> = { good: "var(--good)", warn: "var(--warn)", bad: "var(--bad)", neutral: "var(--faint)" };

/** One live signal: name, big number, plain-language status and a 2-minute sparkline. */
export function Signal({ label, value, unit = "", history, invert = false, hint }: {
  label: string; value: number | null | undefined; unit?: string; history: (number | null)[]; invert?: boolean; hint?: string;
}) {
  const v = value ?? null;
  const tone: Tone = v == null ? "neutral" : invert ? (v < 40 ? "good" : v < 65 ? "warn" : "bad") : v >= 65 ? "good" : v >= 40 ? "warn" : "bad";
  const word = v == null ? "waiting" : invert ? (v < 40 ? "low" : v < 65 ? "moderate" : "high") : v >= 65 ? "strong" : v >= 40 ? "fair" : "low";
  const data = history.slice(-48).map((y, i) => ({ i, y }));
  return (
    <div className="flex items-center gap-4 py-4">
      <div className="w-28 shrink-0">
        <p className="text-[13px] text-muted">{label}</p>
        <div className="mt-1 flex items-baseline gap-1">
          <motion.span key={v == null ? "-" : Math.round(v)} initial={{ opacity: 0.4, y: 2 }} animate={{ opacity: 1, y: 0 }}
            className="font-display text-3xl tabular-nums">{v == null ? "–" : Math.round(v)}</motion.span>
          {v != null && unit && <span className="text-sm text-faint">{unit}</span>}
        </div>
      </div>
      <div className="h-10 min-w-0 flex-1">
        {data.length > 1 && (
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 4, right: 0, bottom: 4, left: 0 }}>
              <YAxis hide domain={[0, 100]} />
              <Line type="monotone" dataKey="y" stroke={toneVar[tone]} strokeWidth={1.75} dot={false} isAnimationActive={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
      <span className={cn("w-20 text-right text-[13px]", tone === "good" ? "text-good" : tone === "warn" ? "text-warn" : tone === "bad" ? "text-bad" : "text-faint")} title={hint}>
        {word}
      </span>
    </div>
  );
}

export function GazePlot({ points }: { points: GazeMsg[] }) {
  const data = points.filter((p) => p.face && p.yaw != null).map((p, i, a) => ({ x: p.yaw, y: p.pitch, age: i - a.length }));
  const last = data[data.length - 1];
  const lastOn = points.filter((p) => p.face).slice(-1)[0]?.on_screen;
  return (
    <div className="h-52 w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 8, bottom: 4, left: -24 }}>
          <XAxis type="number" dataKey="x" domain={[-45, 45]} tick={{ fontSize: 10, fill: "var(--faint)" }} axisLine={false} tickLine={false} unit="°" />
          <YAxis type="number" dataKey="y" domain={[-40, 25]} tick={{ fontSize: 10, fill: "var(--faint)" }} axisLine={false} tickLine={false} unit="°" />
          <ZAxis dataKey="age" range={[10, 40]} />
          <ReferenceArea x1={-24} x2={24} y1={-26} y2={11} fill="var(--accent)" fillOpacity={0.06} stroke="var(--accent)" strokeOpacity={0.35} strokeDasharray="3 4" radius={12} />
          <Scatter data={data} fill="var(--accent)" fillOpacity={0.22} isAnimationActive={false} />
          {last && <Scatter data={[last]} fill={lastOn ? "var(--good)" : "var(--bad)"} isAnimationActive={false} />}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TrendChart({ data, onPick, markers = [], height = 180, cursor }: {
  data: { t: number; attention: number | null; engaged: number }[];
  onPick?: (t: number) => void;
  markers?: { t: number; label: string }[];
  height?: number;
  cursor?: number;
}) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 10, right: 6, bottom: 0, left: -28 }}
          onClick={(e) => { const t = (e as { activeLabel?: number })?.activeLabel; if (onPick && typeof t === "number") onPick(t); }}>
          <defs>
            <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.18} />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => mmss(v)} tick={{ fontSize: 10, fill: "var(--faint)" }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} ticks={[0, 50, 100]} tick={{ fontSize: 10, fill: "var(--faint)" }} axisLine={false} tickLine={false} />
          <Tooltip labelFormatter={(v) => mmss(Number(v))} cursor={{ stroke: "var(--border)" }}
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 14, fontSize: 12, boxShadow: "var(--shadow)" }} />
          <Area type="monotone" dataKey="attention" name="attention" stroke="var(--accent)" fill="url(#trendFill)" strokeWidth={1.75} isAnimationActive={false} />
          <Line type="stepAfter" dataKey="engaged" name="engagement" stroke="var(--muted)" strokeDasharray="2 4" dot={false} isAnimationActive={false} />
          {markers.map((m) => <ReferenceLine key={`${m.t}-${m.label}`} x={m.t} stroke="var(--warn)" strokeOpacity={0.7} />)}
          {cursor != null && <ReferenceLine x={cursor} stroke="var(--fg)" strokeOpacity={0.5} />}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function SubscoreRadar({ data }: { data: Record<string, number | null> }) {
  const rows = Object.entries(data).map(([k, v]) => ({ k: k[0].toUpperCase() + k.slice(1), v: v ?? 0 }));
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <RadarChart data={rows} outerRadius="72%">
          <PolarGrid stroke="var(--border)" />
          <PolarAngleAxis dataKey="k" tick={{ fontSize: 12, fill: "var(--muted)" }} />
          <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
          <Radar dataKey="v" stroke="var(--accent)" strokeWidth={1.5} fill="var(--accent)" fillOpacity={0.14} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Large ring for a headline score. */
export function ScoreRing({ value, size = 132, label }: { value: number | null | undefined; size?: number; label?: string }) {
  const v = value ?? 0;
  const r = size / 2 - 8;
  const c = 2 * Math.PI * r;
  const tone = value == null ? "var(--faint)" : v >= 70 ? "var(--good)" : v >= 50 ? "var(--warn)" : "var(--bad)";
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--surface-2)" strokeWidth={8} fill="none" />
        <motion.circle cx={size / 2} cy={size / 2} r={r} stroke={tone} strokeWidth={8} fill="none" strokeLinecap="round"
          strokeDasharray={c} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: c * (1 - v / 100) }} transition={{ duration: 1, ease: "easeOut" }} />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="font-display text-4xl tabular-nums">{value == null ? "–" : Math.round(v)}</div>
          {label && <div className="text-[11px] tracking-wide text-faint uppercase">{label}</div>}
        </div>
      </div>
    </div>
  );
}

export function Meter({ label, value, max = 1 }: { label: string; value: number; max?: number }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-[13px]"><span className="truncate">{label}</span><span className="tabular-nums text-muted">{Math.round(pct)}%</span></div>
      <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
        <motion.div className="h-full rounded-full bg-accent" animate={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
