"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Logo } from "@/components/shell";
import { Empty, ThemeToggle } from "@/components/ui";
import { api, type CandidateView } from "@/lib/api";

export default function Feedback() {
  const { token } = useParams<{ token: string }>();
  const [fb, setFb] = useState<(CandidateView & { title: string; released_at: string }) | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.feedback(token).then(setFb).catch((e) => setError(e.message)); }, [token]);

  return (
    <main className="min-h-screen px-6 py-8">
      <div className="mx-auto flex max-w-2xl items-center justify-between"><Logo /><ThemeToggle /></div>
      <article className="mx-auto max-w-2xl py-16">
        {error && <Empty title="This feedback isn't available" text="The link may be wrong, or your interviewer hasn't released it yet." />}
        {fb && (
          <>
            <p className="text-[13px] text-accent">Your interview feedback</p>
            <h1 className="mt-3 font-display text-4xl leading-tight tracking-tight">{fb.title}</h1>
            <p className="mt-3 text-sm text-faint">{new Date(fb.released_at).toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" })}</p>

            <section className="mt-14">
              <h2 className="font-display text-2xl">What went well</h2>
              <ul className="mt-5 space-y-3">
                {fb.strengths.map((s, i) => <li key={i} className="flex gap-3 text-[16px] leading-relaxed"><span className="mt-2.5 h-1.5 w-1.5 shrink-0 rounded-full bg-good" />{s}</li>)}
              </ul>
            </section>

            <section className="mt-14">
              <h2 className="font-display text-2xl">Things to work on</h2>
              <ol className="mt-6 space-y-8">
                {fb.improvements.map((imp, i) => (
                  <li key={i} className="grid grid-cols-[36px_1fr] gap-2">
                    <span className="font-display text-2xl text-faint">{i + 1}</span>
                    <div>
                      <p className="text-[16px] leading-relaxed">{imp.observation}{imp.timestamp && <span className="ml-2 text-sm text-faint">at {imp.timestamp}</span>}</p>
                      <p className="mt-2 rounded-2xl bg-accent-soft px-5 py-4 text-[15px] leading-relaxed text-fg/80">{imp.tip}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>

            <p className="mt-16 border-t border-line pt-8 text-[15px] leading-relaxed text-muted">{fb.closing}</p>
          </>
        )}
      </article>
    </main>
  );
}
