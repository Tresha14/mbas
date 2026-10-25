"use client";

import { motion } from "framer-motion";
import { Brain, Eye, LineChart, Lock, Mic, Users } from "lucide-react";
import Link from "next/link";
import { Header } from "@/components/shell";
import { Button, Card } from "@/components/ui";

const features = [
  { icon: Eye, title: "Attention & gaze", text: "Our own face, iris and gaze models track on-screen attention, glances away and reading patterns." },
  { icon: Mic, title: "Live transcript", text: "A Conformer speech recogniser we trained from scratch transcribes each speaker with fillers highlighted." },
  { icon: Brain, title: "Hesitation & speaking patterns", text: "Response delay, pauses, fillers, pace and pitch variation per answer." },
  { icon: LineChart, title: "Answer quality", text: "Relevance, STAR structure and a 0-100 communication score with the top 3 reasons." },
  { icon: Users, title: "Meeting mode", text: "Talk-time balance and engagement for every participant." },
  { icon: Lock, title: "Interviewer-only analytics", text: "Candidates never see live scores; they get a reviewed feedback report afterwards." },
];

export default function Landing() {
  return (
    <>
      <Header right={<Link href="/login"><Button size="sm">Sign in</Button></Link>} />
      <main className="mx-auto max-w-6xl px-4 py-16">
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="max-w-3xl">
          <p className="mb-3 text-sm font-medium text-accent">Multimodal Behavioural Analysis System</p>
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
            An interview and meeting copilot that analyses video, audio and conversation.
          </h1>
          <p className="mt-4 text-lg text-muted">
            Attention, engagement, hesitation, speaking patterns and responses, combined into a communication-effectiveness
            score with clear reasons and follow-up suggestions. Every model is trained by us from scratch.
          </p>
          <div className="mt-8 flex gap-3">
            <Link href="/login"><Button size="lg">Start an interview</Button></Link>
            <Link href="/login?mode=register"><Button size="lg" variant="outline">Create account</Button></Link>
          </div>
        </motion.section>
        <section className="mt-16 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {features.map(({ icon: Icon, title, text }) => (
            <Card key={title}>
              <Icon className="mb-3 text-accent" size={22} />
              <h3 className="font-semibold">{title}</h3>
              <p className="mt-1 text-sm text-muted">{text}</p>
            </Card>
          ))}
        </section>
        <p className="mt-12 text-xs text-muted">
          Scores are decision support for the interviewer, never an automatic rejection. Recording requires the candidate&apos;s consent.
        </p>
      </main>
    </>
  );
}
