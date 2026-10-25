"use client";

import type { RemoteParticipant, Room } from "livekit-client";
import { Play, Square } from "lucide-react";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Meter } from "@/components/charts";
import { Header, VideoSlot } from "@/components/shell";
import { Badge, Button, Card, Dot, Empty, SectionLabel } from "@/components/ui";
import { useAnalytics } from "@/lib/analytics";
import { api, type Interview } from "@/lib/api";
import { joinRoom } from "@/lib/room";

export default function Meeting() {
  const id = Number(useParams<{ id: string }>().id);
  const [iv, setIv] = useState<Interview | null>(null);
  const [videos, setVideos] = useState<Record<string, { el: HTMLMediaElement; name: string }>>({});
  const [people, setPeople] = useState<RemoteParticipant[]>([]);
  const room = useRef<Room | null>(null);
  const { state } = useAnalytics(iv?.id ?? null);

  useEffect(() => {
    api.interview(id).then(setIv);
    joinRoom(id, {
      onTrack: (kind, el, p) => {
        if (kind === "camera") setVideos((v) => {
          const n = { ...v };
          if (el) n[p.identity] = { el, name: p.name || p.identity };
          else delete n[p.identity];
          return n;
        });
        if (kind === "audio" && el) document.body.appendChild(el);
      },
      onParticipants: setPeople,
    }).then((r) => (room.current = r)).catch(() => undefined);
    return () => { room.current?.disconnect(); };
  }, [id]);

  const conv = state.conversation;
  const meet = conv?.meeting ?? {};
  const names = Object.fromEntries(people.map((p) => [p.identity, p.name || p.identity]));
  return (
    <>
      <Header wide right={
        <Button size="sm" variant={iv?.status === "live" ? "danger" : "accent"} disabled={iv?.status === "ended"}
          onClick={async () => iv && setIv(iv.status === "live" ? await api.stop(iv.id) : await api.start(iv.id))}>
          {iv?.status === "live" ? <><Square size={13} /> End meeting</> : <><Play size={13} /> Start analysis</>}
        </Button>} />
      <main className="mx-auto grid max-w-[1440px] gap-6 px-6 py-6 xl:grid-cols-[1fr_360px]">
        <section>
          <div className="mb-5 flex items-center gap-3">
            <h1 className="font-display text-2xl tracking-tight">{iv?.title}</h1>
            <Badge><Dot tone={iv?.status === "live" ? "good" : "neutral"} pulse={iv?.status === "live"} /> {people.length + 1} in the call</Badge>
          </div>
          {Object.keys(videos).length === 0 ? (
            <Card><Empty title="Waiting for participants" text="Share the participant link from your dashboard. Their video appears here." /></Card>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
              {Object.entries(videos).map(([k, v]) => <VideoSlot key={k} el={v.el} label={v.name} className="aspect-video" />)}
            </div>
          )}
        </section>
        <aside className="space-y-6">
          <Card>
            <SectionLabel className="mb-4">Talk time</SectionLabel>
            {!conv ? <p className="text-sm text-muted">Appears about ten seconds after analysis starts.</p> : (
              <div className="space-y-4">
                {Object.entries(conv.participants).sort((a, b) => b[1].talk_share - a[1].talk_share)
                  .map(([k, p]) => <Meter key={k} label={names[k] ?? k} value={p.talk_share} />)}
                <p className="pt-1 text-[13px] text-muted">
                  {conv.talk_balance_gini < 0.2 ? "Conversation is well balanced." : conv.talk_balance_gini < 0.4 ? "Slightly uneven conversation." : "One or two voices dominate."}
                </p>
              </div>
            )}
          </Card>
          <Card>
            <SectionLabel className="mb-4">Engagement</SectionLabel>
            {Object.keys(meet).length === 0 ? <p className="text-sm text-muted">Waiting for audio…</p> : (
              <div className="space-y-4">{Object.entries(meet).map(([k, m]) => <Meter key={k} label={names[k] ?? k} value={m.engagement} />)}</div>
            )}
          </Card>
          {state.followUps.length > 0 && (
            <Card>
              <SectionLabel className="mb-3">Follow-up items</SectionLabel>
              <ul className="space-y-2 text-sm">{state.followUps.map((f, i) => <li key={i} className="text-muted">{f.text}</li>)}</ul>
            </Card>
          )}
        </aside>
      </main>
    </>
  );
}
