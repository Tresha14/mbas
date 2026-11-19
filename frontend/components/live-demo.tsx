"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { cn } from "./ui";

/** Animated mock of a live interview (login + home page): candidate tile with face tracking and gaze, live signals,
 *  a transcript that types itself with fillers highlighted, and a follow-up nudge. Pure decoration, no data. */

const LINES: { who: "You" | "Candidate"; words: string[]; fillers?: number[] }[] = [
  { who: "You", words: "Tell me about a project you are proud of.".split(" ") },
  { who: "Candidate", words: "um so I built a campus event portal that, like, handled seat booking for 2,000 students".split(" "), fillers: [0, 9] },
  { who: "You", words: "How did you stop two people booking the same seat?".split(" ") },
  { who: "Candidate", words: "we used a database lock and hmm a short queue so each booking is checked once".split(" "), fillers: [6] },
];

const NUDGES = [
  { tone: "warn", title: "Sounds sure, but off target", text: "Ask for the exact steps they took." },
  { tone: "good", title: "Confident and on point", text: "Strong, specific answer. Move on." },
];

function useTick(ms: number) {
  const [t, setT] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setT((v) => v + 1), ms);
    return () => clearInterval(id);
  }, [ms]);
  return t;
}

function Gauge({ label, value, invert = false }: { label: string; value: number; invert?: boolean }) {
  const good = invert ? value < 40 : value >= 65;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-[11px] tracking-wide text-white/50 uppercase">{label}</span>
        <motion.span key={value} initial={{ opacity: 0.4 }} animate={{ opacity: 1 }} className="font-display text-lg tabular-nums text-white">
          {value}
        </motion.span>
      </div>
      <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-white/10">
        <motion.div className={cn("h-full rounded-full", good ? "bg-[#5eb8ad]" : "bg-[#d9a54a]")}
          animate={{ width: `${value}%` }} transition={{ duration: 0.8, ease: "easeOut" }} />
      </div>
    </div>
  );
}

function CandidateTile({ t }: { t: number }) {
  // gaze wanders gently around the screen centre, glancing away now and then
  const away = t % 14 === 9 || t % 14 === 10;
  const gx = away ? 34 : Math.sin(t * 0.9) * 7;
  const gy = away ? -6 : Math.cos(t * 0.7) * 4;
  const blink = t % 6 === 0;
  return (
    <div className="relative aspect-[4/3] overflow-hidden rounded-2xl bg-gradient-to-b from-[#2b3532] to-[#161a19]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_35%,rgba(94,184,173,0.18),transparent_60%)]" />
      <svg viewBox="0 0 200 150" className="absolute inset-0 h-full w-full">
        {/* shoulders + head */}
        <path d="M30 150 C35 112 70 102 100 102 C130 102 165 112 170 150 Z" fill="#3a4441" />
        <ellipse cx="100" cy="66" rx="25" ry="31" fill="#4a5652" />
        <path d="M75 58 C76 34 124 34 125 58 C121 46 79 46 75 58 Z" fill="#2c3431" />
        {/* eyes (blink) */}
        <motion.ellipse cx="90" cy="66" rx="3.2" initial={{ ry: 2.2 }} animate={{ ry: blink ? 0.4 : 2.2 }} transition={{ duration: 0.12 }} fill="#1c2220" />
        <motion.ellipse cx="110" cy="66" rx="3.2" initial={{ ry: 2.2 }} animate={{ ry: blink ? 0.4 : 2.2 }} transition={{ duration: 0.12 }} fill="#1c2220" />
        {/* face box with corner brackets */}
        <g stroke="#5eb8ad" strokeWidth="1.4" fill="none" strokeLinecap="round">
          <path d="M70 38 v-6 h7 M130 38 v-6 h-7 M70 96 v6 h7 M130 96 v6 h-7" />
        </g>
        {/* landmarks */}
        {[[90, 66], [110, 66], [100, 78], [91, 88], [109, 88]].map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="1.3" fill="#5eb8ad" />
        ))}
        {/* gaze ray */}
        <motion.line x1="100" y1="66" initial={{ x2: 100, y2: 40 }} animate={{ x2: 100 + gx * 1.6, y2: 40 + gy }} transition={{ duration: 0.9, ease: "easeInOut" }}
          stroke={away ? "#d9a54a" : "#5eb8ad"} strokeWidth="1" strokeDasharray="2 2" opacity="0.8" />
        <motion.circle r="3" initial={{ cx: 100, cy: 40 }} animate={{ cx: 100 + gx * 1.6, cy: 40 + gy }} transition={{ duration: 0.9, ease: "easeInOut" }}
          fill={away ? "#d9a54a" : "#5eb8ad"} />
      </svg>
      <div className="absolute top-2.5 left-2.5 flex items-center gap-1.5 rounded-full bg-black/40 px-2 py-0.5 text-[10px] text-white/80 backdrop-blur">
        <span className={cn("h-1.5 w-1.5 rounded-full", away ? "bg-[#d9a54a]" : "bg-[#5eb8ad]")} />
        {away ? "glancing away" : "eyes on screen"}
      </div>
      <div className="absolute bottom-2.5 left-2.5 rounded-full bg-black/40 px-2 py-0.5 text-[10px] text-white/70 backdrop-blur">
        Candidate · face 0.98
      </div>
      <div className="absolute right-2.5 bottom-2.5 h-[22%] w-[24%] rounded-lg border border-white/10 bg-gradient-to-b from-[#3b3a36] to-[#22211f]">
        <span className="absolute bottom-1 left-1.5 text-[9px] text-white/60">You</span>
      </div>
    </div>
  );
}

function Transcript({ t }: { t: number }) {
  // reveal one word per tick across the conversation, then start over
  const total = LINES.reduce((n, l) => n + l.words.length, 0) + 6;
  let left = t % total;
  return (
    <div className="space-y-2 text-[13px] leading-relaxed">
      {LINES.map((l, li) => {
        const shown = Math.max(0, Math.min(l.words.length, left));
        left -= l.words.length;
        if (!shown) return null;
        return (
          <p key={li} className={l.who === "You" ? "text-white/45" : "text-white/85"}>
            <span className={cn("mr-2 text-[10px] font-medium tracking-wider uppercase", l.who === "You" ? "text-white/35" : "text-[#5eb8ad]")}>{l.who}</span>
            {l.words.slice(0, shown).map((w, i) => (
              <span key={i} className={l.fillers?.includes(i) ? "rounded bg-[#d9a54a]/25 px-0.5 text-[#f0c77a]" : ""}>{w} </span>
            ))}
          </p>
        );
      })}
    </div>
  );
}

export function LiveDemo({ className }: { className?: string }) {
  const t = useTick(450);
  const slow = Math.floor(t / 4);
  const wobble = (base: number, amp: number, k: number) => Math.round(base + Math.sin(slow * k) * amp);
  const nudge = NUDGES[Math.floor(t / 40) % NUDGES.length];
  const showNudge = t % 40 > 18;
  const secs = 252 + Math.floor(t * 0.45);
  return (
    <div className={cn("w-full rounded-[26px] border border-white/10 bg-white/[0.04] p-4 shadow-2xl shadow-black/40 backdrop-blur-sm", className)}>
      <div className="mb-3 flex items-center justify-between px-1 text-[12px] text-white/60">
        <span className="flex items-center gap-2">
          <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#e5675a] opacity-60" /><span className="relative h-2 w-2 rounded-full bg-[#e5675a]" /></span>
          Live · {String(Math.floor(secs / 60)).padStart(2, "0")}:{String(secs % 60).padStart(2, "0")} · SDE intern, round 1
        </span>
        <span className="rounded-full bg-[#5eb8ad]/15 px-2 py-0.5 text-[11px] text-[#7fd1c6]">analysing</span>
      </div>
      <div className="grid gap-4 sm:grid-cols-[1.25fr_1fr]">
        <CandidateTile t={t} />
        <div className="flex flex-col justify-between gap-3 rounded-2xl bg-white/[0.03] p-4">
          <Gauge label="Attention" value={wobble(84, 5, 0.9)} />
          <Gauge label="Engagement" value={wobble(76, 6, 0.6)} />
          <Gauge label="Hesitation" value={wobble(31, 8, 0.8)} invert />
          <Gauge label="Composure" value={wobble(79, 4, 0.5)} />
        </div>
      </div>
      <div className="relative mt-4 min-h-[118px] rounded-2xl bg-white/[0.03] p-4">
        <Transcript t={t} />
        <AnimatePresence>
          {showNudge && (
            <motion.div key={nudge.title} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}
              className={cn("absolute right-3 -bottom-3 max-w-[230px] rounded-xl border px-3 py-2 text-[12px] shadow-lg backdrop-blur",
                nudge.tone === "warn" ? "border-[#d9a54a]/30 bg-[#2a2418]/90 text-[#f0c77a]" : "border-[#5eb8ad]/30 bg-[#16201e]/90 text-[#7fd1c6]")}>
              <p className="font-medium">{nudge.title}</p>
              <p className="text-white/60">{nudge.text}</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
