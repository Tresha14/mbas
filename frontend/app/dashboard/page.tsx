"use client";

import { CalendarClock, Check, Copy, FileText, Plus, Radio, Users } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Header } from "@/components/shell";
import { Badge, Button, Card, CardTitle, Input, Label, Tabs } from "@/components/ui";
import { api, claims, type Interview } from "@/lib/api";

export default function Dashboard() {
  const router = useRouter();
  const [items, setItems] = useState<Interview[]>([]);
  const [title, setTitle] = useState("");
  const [candidate, setCandidate] = useState("");
  const [when, setWhen] = useState("");
  const [mode, setMode] = useState("interview");
  const [copied, setCopied] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const c = claims();
    if (!c || (c.role !== "interviewer" && c.role !== "meeting_host")) {
      router.replace("/login");
      return;
    }
    if (c.role === "meeting_host") setMode("meeting");
    api.interviews().then(setItems).catch((e) => setError(e.message));
  }, [router]);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    try {
      const iv = await api.create({ title, mode, candidate_name: candidate || undefined, scheduled_at: when ? new Date(when).toISOString() : null });
      setItems([iv, ...items]);
      setTitle("");
      setCandidate("");
      setWhen("");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function copy(iv: Interview) {
    await navigator.clipboard.writeText(iv.candidate_link.replace(/^https?:\/\/[^/]+/, location.origin));
    setCopied(iv.id);
    setTimeout(() => setCopied(null), 1500);
  }

  return (
    <>
      <Header />
      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-8 lg:grid-cols-[360px_1fr]">
        <Card className="h-fit">
          <CardTitle>New {mode}</CardTitle>
          <form onSubmit={create} className="space-y-4">
            <Tabs tabs={["interview", "meeting"]} value={mode} onChange={setMode} />
            <div><Label htmlFor="t">Title</Label><Input id="t" placeholder="SDE intern - round 1" value={title} onChange={(e) => setTitle(e.target.value)} required /></div>
            {mode === "interview" && (
              <div><Label htmlFor="c">Candidate name</Label><Input id="c" value={candidate} onChange={(e) => setCandidate(e.target.value)} /></div>
            )}
            <div><Label htmlFor="w">Schedule (optional)</Label><Input id="w" type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} /></div>
            <Button className="w-full"><Plus size={16} /> Create</Button>
          </form>
          {error && <p className="mt-3 text-sm text-bad">{error}</p>}
        </Card>
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Your interviews and meetings</h2>
          {items.length === 0 && <Card className="text-sm text-muted">Nothing yet. Create an interview, then send the candidate link.</Card>}
          {items.map((iv) => (
            <Card key={iv.id} className="flex flex-wrap items-center gap-4">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-accent-soft text-accent">
                {iv.mode === "meeting" ? <Users size={18} /> : <Radio size={18} />}
              </div>
              <div className="min-w-48 flex-1">
                <div className="font-medium">{iv.title}</div>
                <div className="flex items-center gap-2 text-xs text-muted">
                  {iv.candidate_name && <span>{iv.candidate_name}</span>}
                  {iv.scheduled_at && <span className="flex items-center gap-1"><CalendarClock size={12} />{new Date(iv.scheduled_at).toLocaleString()}</span>}
                </div>
              </div>
              <Badge tone={iv.status === "live" ? "good" : iv.status === "ended" ? "neutral" : "accent"}>{iv.status}</Badge>
              <Button variant="outline" size="sm" onClick={() => copy(iv)}>
                {copied === iv.id ? <Check size={14} /> : <Copy size={14} />} {iv.mode === "meeting" ? "Participant" : "Candidate"} link
              </Button>
              <Link href={iv.mode === "meeting" ? `/meeting/${iv.id}` : `/cockpit/${iv.id}`}><Button size="sm">Open</Button></Link>
              {iv.status === "ended" && <Link href={`/report/${iv.id}`}><Button variant="outline" size="sm"><FileText size={14} /> Report</Button></Link>}
            </Card>
          ))}
        </section>
      </main>
    </>
  );
}
