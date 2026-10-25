"use client";

import { motion } from "framer-motion";
import { ArrowRight, Eye, MessageSquareText, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Header } from "@/components/shell";
import { claims } from "@/lib/api";
import { Button, Dot } from "@/components/ui";

const pillars = [
  { icon: Eye, title: "Sees attention", text: "Gaze, glances and engagement from the webcam, measured by models we trained ourselves." },
  { icon: MessageSquareText, title: "Hears the answer", text: "Live transcript, hesitation and speaking pace, then relevance and structure of every answer." },
  { icon: ShieldCheck, title: "Fair by design", text: "Only the interviewer sees analytics. Candidates receive reviewed, constructive feedback." },
];

function Preview() {
  const rows = [["Attention", 82, "strong", "text-good"], ["Engagement", 71, "strong", "text-good"], ["Hesitation", 38, "low", "text-good"]] as const;
  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, duration: 0.6 }}
      className="relative rounded-[28px] border border-line bg-surface p-6 shadow-soft">
      <div className="mb-5 flex items-center justify-between">
        <span className="text-[13px] text-muted">Live · Software engineer, round 1</span>
        <span className="flex items-center gap-2 text-[13px] text-muted"><Dot tone="good" pulse /> analysing</span>
      </div>
      <div className="mb-5 aspect-video rounded-2xl bg-gradient-to-br from-[#2a2a27] to-[#141413]" />
      <div className="divide-y divide-line">
        {rows.map(([l, v, w, c]) => (
          <div key={l} className="flex items-center gap-4 py-3">
            <span className="w-24 text-[13px] text-muted">{l}</span>
            <span className="font-display text-2xl tabular-nums">{v}</span>
            <svg className="h-6 flex-1" viewBox="0 0 120 24" preserveAspectRatio="none"><path d={`M0 ${24 - v / 6} C30 ${20 - v / 7}, 60 ${26 - v / 5}, 120 ${24 - v / 4.5}`} fill="none" stroke="var(--good)" strokeWidth="1.5" /></svg>
            <span className={`w-16 text-right text-[13px] ${c}`}>{w}</span>
          </div>
        ))}
      </div>
      <div className="mt-4 rounded-2xl bg-accent-soft px-4 py-3 text-[13px] text-accent">
        Suggested follow-up: ask for a concrete example of the result.
      </div>
    </motion.div>
  );
}

export default function Landing() {
  const [signedIn, setSignedIn] = useState(false);
  useEffect(() => { const c = claims(); setSignedIn(!!c && (c.role === "interviewer" || c.role === "meeting_host")); }, []);
  return (
    <>
      <Header nav={false} right={signedIn
        ? <Link href="/dashboard"><Button size="sm" variant="outline">Open dashboard</Button></Link>
        : <Link href="/login"><Button size="sm" variant="outline">Sign in</Button></Link>} />
      <main className="mx-auto max-w-6xl px-6">
        <section className="grid items-center gap-14 py-20 lg:grid-cols-[1.05fr_1fr] lg:py-28">
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <p className="mb-5 text-[13px] tracking-wide text-accent">Multimodal Behavioural Analysis System</p>
            <h1 className="font-display text-5xl leading-[1.05] tracking-tight sm:text-6xl">
              Interviews that are <em className="text-accent">clearer</em> for both sides of the table.
            </h1>
            <p className="mt-6 max-w-lg text-[17px] leading-relaxed text-muted">
              A quiet copilot for interviewers: it reads attention, engagement, hesitation and the substance of each answer,
              and suggests what to ask next. Every model is trained from scratch on our own data.
            </p>
            <div className="mt-10 flex flex-wrap gap-3">
              <Link href={signedIn ? "/dashboard" : "/login?mode=register"}><Button size="lg">{signedIn ? "Go to your interviews" : "Get started"} <ArrowRight size={16} /></Button></Link>
              {!signedIn && <Link href="/login"><Button size="lg" variant="ghost">I have an account</Button></Link>}
            </div>
          </motion.div>
          <Preview />
        </section>
        <section className="grid gap-10 border-t border-line py-16 sm:grid-cols-3">
          {pillars.map(({ icon: Icon, title, text }) => (
            <div key={title}>
              <Icon size={20} className="mb-4 text-accent" strokeWidth={1.75} />
              <h3 className="font-display text-xl">{title}</h3>
              <p className="mt-2 text-[15px] leading-relaxed text-muted">{text}</p>
            </div>
          ))}
        </section>
        <footer className="border-t border-line py-10 text-[13px] text-faint">
          Scores support the interviewer&apos;s judgement and never decide on their own. Recording requires the candidate&apos;s consent.
        </footer>
      </main>
    </>
  );
}
