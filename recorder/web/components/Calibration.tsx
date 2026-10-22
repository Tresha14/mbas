"use client";

import { useEffect, useRef, useState } from "react";
import { clock } from "@/lib/clock";

export const CALIBRATION_MS = 60_000;

export interface CalibrationSample {
  t_ms: number;
  x: number;
  y: number;
  x_norm: number;
  y_norm: number;
  phase: "fixation" | "pursuit";
  target: number;
}

export interface CalibrationResult {
  start_ms: number;
  end_ms: number;
  pattern: string;
  screen: Record<string, number>;
  samples: CalibrationSample[];
}

// 0-18 s: 9 fixation points, 2 s each.  18-60 s: smooth pursuit on a Lissajous curve.
const GRID = [0.1, 0.5, 0.9];
const FIX_POINTS = [
  [0.5, 0.5], ...GRID.flatMap((y) => GRID.map((x) => [x, y])).filter(([x, y]) => !(x === 0.5 && y === 0.5)),
] as [number, number][];
const FIX_MS = 2000;
const PATTERN = "9-point fixation (2 s each, centre first, then row-major 10/50/90%) then Lissajous pursuit x=0.5+0.4sin(2πt/7s), y=0.5+0.4sin(2πt/5s+π/4)";

function dotAt(elapsed: number): { xn: number; yn: number; phase: CalibrationSample["phase"]; target: number } {
  const fixTotal = FIX_POINTS.length * FIX_MS;
  if (elapsed < fixTotal) {
    const i = Math.floor(elapsed / FIX_MS);
    return { xn: FIX_POINTS[i][0], yn: FIX_POINTS[i][1], phase: "fixation", target: i };
  }
  const t = (elapsed - fixTotal) / 1000;
  return {
    xn: 0.5 + 0.4 * Math.sin((2 * Math.PI * t) / 7),
    yn: 0.5 + 0.4 * Math.sin((2 * Math.PI * t) / 5 + Math.PI / 4),
    phase: "pursuit",
    target: -1,
  };
}

/**
 * Full-screen moving-dot calibration. The candidate must click to start
 * (fullscreen needs a user gesture). Logs the dot position on every animation frame.
 */
export default function Calibration({ onDone, onCancel }: { onDone: (r: CalibrationResult) => void; onCancel: () => void }) {
  const root = useRef<HTMLDivElement>(null);
  const dot = useRef<HTMLDivElement>(null);
  const [running, setRunning] = useState(false);
  const [left, setLeft] = useState(CALIBRATION_MS / 1000);

  useEffect(() => {
    if (!running) return;
    const samples: CalibrationSample[] = [];
    const t0perf = performance.now();
    const start_ms = clock.fromPerf(t0perf);
    let raf = 0;
    let lastSec = -1;
    const tick = (now: number) => {
      const elapsed = now - t0perf;
      const w = window.innerWidth;
      const h = window.innerHeight;
      if (elapsed >= CALIBRATION_MS) {
        const end_ms = clock.fromPerf(now);
        document.exitFullscreen?.().catch(() => undefined);
        onDone({
          start_ms,
          end_ms,
          pattern: PATTERN,
          screen: {
            viewport_w: w,
            viewport_h: h,
            screen_w: window.screen.width,
            screen_h: window.screen.height,
            device_pixel_ratio: window.devicePixelRatio,
          },
          samples,
        });
        return;
      }
      const d = dotAt(elapsed);
      const x = d.xn * w;
      const y = d.yn * h;
      if (dot.current) dot.current.style.transform = `translate(${x - 12}px, ${y - 12}px)`;
      samples.push({ t_ms: clock.fromPerf(now), x, y, x_norm: d.xn, y_norm: d.yn, phase: d.phase, target: d.target });
      const sec = Math.ceil((CALIBRATION_MS - elapsed) / 1000);
      if (sec !== lastSec) {
        lastSec = sec;
        setLeft(sec);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running, onDone]);

  return (
    <div ref={root} className="fixed inset-0 z-50 bg-zinc-950">
      {!running ? (
        <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
          <h2 className="text-2xl font-semibold">Gaze calibration (60 s)</h2>
          <p className="max-w-md text-zinc-400">
            Keep your head still and follow the dot with your eyes only. First it jumps between 9 points, then it moves
            smoothly.
          </p>
          <div className="flex gap-3">
            <button
              className="btn btn-primary"
              onClick={async () => {
                await root.current?.requestFullscreen?.().catch(() => undefined);
                setRunning(true);
              }}
            >
              Start calibration
            </button>
            <button className="btn btn-ghost" onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <>
          <div ref={dot} className="absolute left-0 top-0 h-6 w-6 rounded-full bg-rose-500 ring-4 ring-rose-500/30" />
          <span className="absolute bottom-4 right-4 text-sm text-zinc-500">{left}s</span>
        </>
      )}
    </div>
  );
}
