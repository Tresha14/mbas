"use client";

import { ArrowUpRight, Check, Link2, Plus, Users, Video } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { Header, PageTitle } from "@/components/shell";
import { Badge, Button, Card, Dot, Empty, Input, Label, Segmented, Sheet } from "@/components/ui";
import { api, claims, type Interview } from "@/lib/api";

function when(iv: Interview): string {
  const d = iv.scheduled_at ?? iv.started_at ?? null;
  if (!d) return "Not scheduled";
  return new Date(d).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

export default function Dashboard() {
  const router = useRouter();
  const [items, setItems] = useState<Interview[] | null>(null);
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("all");
  const [title, setTitle] = useState("");
  const [candidate, setCandidate] = useState("");
  const [at, setAt] = useState("");
  const [mode, setMode] = useState("interview");
  const [copied, setCopied] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");

  useEffect(() => {
    const c = claims();
    if (!c || (c.role !== "interviewer" && c.role !== "meeting_host")) {
      router.replace("/login");
      return;
    }
    setName((c.name ?? "").split(" ")[0]);
    if (c.role === "meeting_host") setMode("meeting");
    api.interviews().then(setItems).catch((e) => setError(e.message));
  }, [router]);

  const shown = useMemo(() => (items ?? []).filter((i) => filter === "all" || i.status === filter), [items, filter]);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    try {
      const iv = await api.create({ title, mode, candidate_name: candidate || undefined, scheduled_at: at ? new Date(at).toISOString() : null });
      setItems([iv, ...(items ?? [])]);
      setTitle(""); setCandidate(""); setAt(""); setOpen(false);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function copy(iv: Interview) {
    await navigator.clipboard.writeText(iv.candidate_link.replace(/^https?:\/\/[^/]+/, location.origin));
    setCopied(iv.id);
    setTimeout(() => setCopied(null), 1600);
  }

  return (
    <>
      <Header />
      <main className="mx-auto max-w-6xl px-6 py-12">
        <PageTitle eyebrow={name ? `Good to see you, ${name}` : undefined} title="Interviews"
          subtitle="Schedule a session, send the link, and open the cockpit when you're ready."
          right={<Button onClick={() => setOpen(true)}><Plus size={16} /> New session</Button>} />

        <div className="mt-10 mb-4 flex items-center justify-between">
          <Segmented items={[{ value: "all", label: "All" }, { value: "scheduled", label: "Upcoming" }, { value: "live", label: "Live" }, { value: "ended", label: "Completed" }]} value={filter} onChange={setFilter} />
          {error && <span className="text-sm text-bad">{error}</span>}
        </div>

        <Card className="p-2">
          {items === null ? (
            <div className="space-y-2 p-4">{[0, 1, 2].map((i) => <div key={i} className="h-14 animate-pulse rounded-2xl bg-surface-2" />)}</div>
          ) : shown.length === 0 ? (
            <Empty icon={<Video size={20} />} title={filter === "all" ? "No sessions yet" : "Nothing here"}
              text="Create a session, copy the candidate link and send it. You'll see live insights once both of you join."
              action={filter === "all" && <Button variant="outline" onClick={() => setOpen(true)}><Plus size={16} /> New session</Button>} />
          ) : (
            <ul className="divide-y divide-line">
              {shown.map((iv) => (
                <li key={iv.id} className="group flex flex-wrap items-center gap-4 rounded-2xl px-4 py-4 transition hover:bg-surface-2/60">
                  <span className="grid h-10 w-10 place-items-center rounded-2xl bg-surface-2 text-muted">
                    {iv.mode === "meeting" ? <Users size={17} /> : <Video size={17} />}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium">{iv.title}</p>
                    <p className="text-[13px] text-muted">{iv.candidate_name ? `${iv.candidate_name} · ` : ""}{when(iv)}</p>
                  </div>
                  <span className="flex w-28 items-center gap-2 text-[13px] text-muted">
                    <Dot tone={iv.status === "live" ? "good" : iv.status === "ended" ? "neutral" : "accent"} pulse={iv.status === "live"} />
                    {iv.status === "live" ? "Live now" : iv.status === "ended" ? "Completed" : "Upcoming"}
                  </span>
                  <div className="flex items-center gap-1.5">
                    {iv.status !== "ended" && (
                      <Button variant="ghost" size="sm" onClick={() => copy(iv)}>
                        {copied === iv.id ? <><Check size={14} /> Copied</> : <><Link2 size={14} /> Invite link</>}
                      </Button>
                    )}
                    {iv.status === "ended" ? (
                      <Link href={`/report/${iv.id}`}><Button size="sm" variant="outline">View report <ArrowUpRight size={14} /></Button></Link>
                    ) : (
                      <Link href={iv.mode === "meeting" ? `/meeting/${iv.id}` : `/cockpit/${iv.id}`}><Button size="sm">Open <ArrowUpRight size={14} /></Button></Link>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </main>

      <Sheet open={open} onClose={() => setOpen(false)} title="New session">
        <form onSubmit={create} className="space-y-6 pt-2">
          <div><Label>Type</Label><Segmented items={[{ value: "interview", label: "Interview" }, { value: "meeting", label: "Meeting" }]} value={mode} onChange={setMode} /></div>
          <div><Label htmlFor="t">Title</Label><Input id="t" placeholder="Software engineer · round 1" value={title} onChange={(e) => setTitle(e.target.value)} required /></div>
          {mode === "interview" && <div><Label htmlFor="c">Candidate name</Label><Input id="c" placeholder="Optional" value={candidate} onChange={(e) => setCandidate(e.target.value)} /></div>}
          <div><Label htmlFor="w">Date and time</Label><Input id="w" type="datetime-local" value={at} onChange={(e) => setAt(e.target.value)} /></div>
          <Badge tone="neutral" className="leading-relaxed">The candidate sees only a normal video call. Analytics stay with you.</Badge>
          <Button className="w-full" size="lg">Create session</Button>
        </form>
      </Sheet>
    </>
  );
}
