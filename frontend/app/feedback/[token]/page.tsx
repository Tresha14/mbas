"use client";

import { Lightbulb, Sparkles } from "lucide-react";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Header } from "@/components/shell";
import { Badge, Card, CardTitle } from "@/components/ui";
import { api, type CandidateView } from "@/lib/api";

export default function Feedback() {
  const { token } = useParams<{ token: string }>();
  const [fb, setFb] = useState<(CandidateView & { title: string; released_at: string }) | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.feedback(token).then(setFb).catch((e) => setError(e.message)); }, [token]);

  return (
    <>
      <Header />
      <main className="mx-auto max-w-3xl space-y-6 px-4 py-10">
        {error && <Card className="text-bad">{error}</Card>}
        {fb && (
          <>
            <div>
              <p className="text-sm text-accent">Your interview feedback</p>
              <h1 className="text-2xl font-semibold">{fb.title}</h1>
              <p className="text-xs text-muted">Released {new Date(fb.released_at).toLocaleString()}</p>
            </div>
            <Card>
              <CardTitle className="flex items-center gap-2"><Sparkles size={14} /> What went well</CardTitle>
              <ul className="list-disc space-y-1 pl-5">{fb.strengths.map((s, i) => <li key={i}>{s}</li>)}</ul>
            </Card>
            <Card>
              <CardTitle className="flex items-center gap-2"><Lightbulb size={14} /> Three things to work on</CardTitle>
              <div className="space-y-4">
                {fb.improvements.map((imp, i) => (
                  <div key={i}>
                    <div className="flex items-center gap-2 font-medium">{i + 1}. {imp.observation} {imp.timestamp && <Badge>{imp.timestamp}</Badge>}</div>
                    <p className="mt-1 text-sm text-muted">{imp.tip}</p>
                  </div>
                ))}
              </div>
            </Card>
            <p className="text-sm text-muted">{fb.closing}</p>
          </>
        )}
      </main>
    </>
  );
}
