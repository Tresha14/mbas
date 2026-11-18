"use client";

/* Per-candidate calibration: learn what is normal for THIS candidate from their first few samples of each metric,
   then judge later values against that instead of one fixed range. Kept in the interviewer's browser only. */
import { useEffect, useMemo, useState } from "react";

export type MetricKey = "facing" | "wpm" | "pause" | "fillers" | "blink" | "hedges";
export type Sample = { t: number; v: number | null | undefined };
export type Baseline = Partial<Record<MetricKey, number>>;

/** Valid samples needed before a metric's baseline is fixed (windows arrive every 2.5 s, speech stats every 5 s). */
export const NEEDED: Record<MetricKey, number> = { facing: 8, blink: 8, wpm: 4, pause: 4, fillers: 4, hedges: 4 };

/** Which direction is worse than their normal, and how far counts as a clear change. */
export const WATCH: Record<MetricKey, { bad: "low" | "high" | "both"; tol: number }> = {
  facing: { bad: "low", tol: 15 },
  wpm: { bad: "both", tol: 25 },
  pause: { bad: "high", tol: 15 },
  fillers: { bad: "high", tol: 3 },
  blink: { bad: "both", tol: 8 },
  hedges: { bad: "high", tol: 2 },
};

const num = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

export function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Median of the first NEEDED valid samples after `since` (session seconds); null until there are enough. */
export function learn(key: MetricKey, samples: Sample[], since: number): number | null {
  const v = samples.filter((s) => s.t > since && num(s.v)).map((s) => s.v as number);
  return v.length >= NEEDED[key] ? median(v.slice(0, NEEDED[key])) : null;
}

export function progress(key: MetricKey, samples: Sample[], since: number): number {
  return Math.min(NEEDED[key], samples.filter((s) => s.t > since && num(s.v)).length);
}

/** good / warn / bad of a value against the candidate's own baseline. */
export function toneVsBaseline(key: MetricKey, value: number, base: number): "good" | "warn" | "bad" {
  const { bad, tol } = WATCH[key];
  const d = value - base;
  const worse = bad === "both" ? Math.abs(d) : bad === "low" ? -d : d;
  return worse <= tol ? "good" : worse <= 2 * tol ? "warn" : "bad";
}

interface Saved { baseline: Baseline; since: number }
const storeKey = (id: number) => `mbas-baseline-${id}`;

export function useBaseline(id: number, series: Record<MetricKey, Sample[]>, now: number) {
  const [saved, setSaved] = useState<Saved>({ baseline: {}, since: -1 });
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(storeKey(id));
      if (raw) setSaved(JSON.parse(raw));
    } catch { /* private mode or bad data: start fresh */ }
    setLoaded(true);
  }, [id]);

  const baseline = useMemo(() => {
    const out: Baseline = { ...saved.baseline };
    (Object.keys(NEEDED) as MetricKey[]).forEach((k) => {
      if (out[k] == null) {
        const b = learn(k, series[k], saved.since);
        if (b != null) out[k] = b;
      }
    });
    return out;
  }, [saved, series]);

  const learned = Object.keys(baseline).length;
  useEffect(() => {
    if (!loaded || learned === Object.keys(saved.baseline).length) return;
    const next = { baseline, since: saved.since };
    setSaved(next);
    try { localStorage.setItem(storeKey(id), JSON.stringify(next)); } catch { /* ignore */ }
  }, [loaded, learned, baseline, saved, id]);

  const reset = () => {
    const next = { baseline: {}, since: now };
    setSaved(next);
    try { localStorage.setItem(storeKey(id), JSON.stringify(next)); } catch { /* ignore */ }
  };

  const done = (Object.keys(NEEDED) as MetricKey[]).filter((k) => baseline[k] != null).length;
  return { baseline, reset, learned: done, total: Object.keys(NEEDED).length, since: saved.since };
}
