"use client";

import { motion } from "framer-motion";
import {
  Area, AreaChart, CartesianGrid, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, RadialBar, RadialBarChart,
  ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import type { GazeMsg } from "@/lib/analytics";
import { cn, mmss } from "./ui";

export function Gauge({ label, value, suffix = "", invert = false }: { label: string; value: number | null | undefined; suffix?: string; invert?: boolean }) {
  const v = value ?? 0;
  const good = invert ? v < 40 : v >= 60;
  const mid = invert ? v < 65 : v >= 40;
  const color = value == null ? "var(--border)" : good ? "var(--good)" : mid ? "var(--warn)" : "var(--bad)";
  return (
    <div className="relative flex flex-col items-center">
      <div className="h-28 w-28">
        <ResponsiveContainer>
          <RadialBarChart innerRadius="72%" outerRadius="100%" data={[{ v }]} startAngle={220} endAngle={-40}>
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar dataKey="v" cornerRadius={8} fill={color} background={{ fill: "var(--surface-2)" }} isAnimationActive />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <motion.div key={Math.round(v)} initial={{ scale: 0.9, opacity: 0.6 }} animate={{ scale: 1, opacity: 1 }}
        className="absolute top-9 text-2xl font-semibold tabular-nums">
        {value == null ? "-" : Math.round(v)}<span className="text-sm text-muted">{suffix}</span>
      </motion.div>
      <div className="-mt-3 text-xs font-medium text-muted">{label}</div>
    </div>
  );
}

export function GazePlot({ points }: { points: GazeMsg[] }) {
  const data = points.filter((p) => p.face && p.yaw != null).map((p, i, a) => ({ x: p.yaw, y: p.pitch, age: a.length - i, on: p.on_screen }));
  const last = data[data.length - 1];
  return (
    <div className="h-44 w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: -16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis type="number" dataKey="x" domain={[-45, 45]} tick={{ fontSize: 10, fill: "var(--muted)" }} unit="°" />
          <YAxis type="number" dataKey="y" domain={[-40, 25]} tick={{ fontSize: 10, fill: "var(--muted)" }} unit="°" />
          <ZAxis dataKey="age" range={[8, 60]} />
          <ReferenceArea x1={-24} x2={24} y1={-26} y2={11} fill="var(--accent)" fillOpacity={0.08} stroke="var(--accent)" strokeDasharray="4 3" />
          <Scatter data={data} fill="var(--accent)" fillOpacity={0.35} isAnimationActive={false} />
          {last && <Scatter data={[last]} fill={last.on ? "var(--good)" : "var(--bad)"} isAnimationActive={false} />}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TrendChart({ data, onPick, markers = [] }: {
  data: { t: number; attention: number | null; engaged: number }[];
  onPick?: (t: number) => void;
  markers?: { t: number; label: string }[];
}) {
  return (
    <div className="h-40 w-full">
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}
          onClick={(e) => { const t = (e as { activeLabel?: number })?.activeLabel; if (onPick && typeof t === "number") onPick(t); }}>
          <defs>
            <linearGradient id="att" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.35} />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => mmss(v)} tick={{ fontSize: 10, fill: "var(--muted)" }} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "var(--muted)" }} />
          <Tooltip labelFormatter={(v) => mmss(Number(v))} contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12 }} />
          <Area type="monotone" dataKey="attention" stroke="var(--accent)" fill="url(#att)" strokeWidth={2} isAnimationActive={false} />
          <Area type="stepAfter" dataKey="engaged" stroke="var(--good)" fill="none" strokeDasharray="4 3" isAnimationActive={false} />
          {markers.map((m) => <ReferenceLine key={`${m.t}-${m.label}`} x={m.t} stroke="var(--warn)" strokeWidth={2} label={{ value: "!", fill: "var(--warn)", fontSize: 12 }} />)}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function SubscoreRadar({ data }: { data: Record<string, number | null> }) {
  const rows = Object.entries(data).map(([k, v]) => ({ k, v: v ?? 0 }));
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <RadarChart data={rows} outerRadius="75%">
          <PolarGrid stroke="var(--border)" />
          <PolarAngleAxis dataKey="k" tick={{ fontSize: 12, fill: "var(--muted)" }} />
          <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
          <Radar dataKey="v" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.3} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Bar({ label, value, max = 1, className }: { label: string; value: number; max?: number; className?: string }) {
  return (
    <div className={cn("space-y-1", className)}>
      <div className="flex justify-between text-xs"><span className="truncate">{label}</span><span className="tabular-nums text-muted">{Math.round((value / max) * 100)}%</span></div>
      <div className="h-2 overflow-hidden rounded-full bg-surface-2">
        <motion.div className="h-full rounded-full bg-accent" animate={{ width: `${Math.min(100, (value / max) * 100)}%` }} />
      </div>
    </div>
  );
}
