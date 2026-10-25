"use client";

import type { RemoteParticipant, Room } from "livekit-client";
import { Play, Square } from "lucide-react";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Bar } from "@/components/charts";
import { Header, VideoSlot } from "@/components/shell";
import { Badge, Button, Card, CardTitle } from "@/components/ui";
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
  return (
    <>
      <Header right={
        <Button size="sm" variant={iv?.status === "live" ? "danger" : "good"} disabled={iv?.status === "ended"}
          onClick={async () => iv && setIv(iv.status === "live" ? await api.stop(iv.id) : await api.start(iv.id))}>
          {iv?.status === "live" ? <><Square size={14} /> End meeting</> : <><Play size={14} /> Start analysis</>}
        </Button>} />
      <main className="mx-auto grid max-w-7xl gap-4 px-4 py-4 lg:grid-cols-[1fr_340px]">
        <div>
          <h1 className="mb-3 text-lg font-semibold">{iv?.title} <Badge>{people.length + 1} in call</Badge></h1>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {Object.entries(videos).map(([k, v]) => <VideoSlot key={k} el={v.el} label={v.name} className="aspect-video" />)}
            {Object.keys(videos).length === 0 && <Card className="text-sm text-muted">Share the participant link from the dashboard. Videos appear here.</Card>}
          </div>
        </div>
        <aside className="space-y-4">
          <Card>
            <CardTitle>Talk time</CardTitle>
            {!conv && <p className="text-sm text-muted">Starts after the first 10 s of analysis.</p>}
            <div className="space-y-3">
              {conv && Object.entries(conv.participants).map(([k, p]) => <Bar key={k} label={k} value={p.talk_share} />)}
            </div>
            {conv && <p className="mt-3 text-xs text-muted">Balance (Gini, 0 = perfectly even): {conv.talk_balance_gini.toFixed(2)} · turns {conv.turn_count} · overlap {conv.overlap_s.toFixed(1)} s</p>}
          </Card>
          <Card>
            <CardTitle>Engagement per participant (meeting head)</CardTitle>
            <div className="space-y-3">
              {Object.entries(meet).map(([k, m]) => (
                <div key={k} className="space-y-1">
                  <Bar label={`${k} engagement`} value={m.engagement} />
                  <div className="text-xs text-muted">contribution balance {Math.round(m.contribution_balance * 100)}%</div>
                </div>
              ))}
              {Object.keys(meet).length === 0 && <p className="text-sm text-muted">Waiting for audio...</p>}
            </div>
          </Card>
          <Card>
            <CardTitle>Follow-up items</CardTitle>
            {state.followUps.length === 0 ? <p className="text-sm text-muted">None yet.</p> :
              state.followUps.map((f, i) => <div key={i} className="text-sm">{f.reason}: {f.text}</div>)}
          </Card>
        </aside>
      </main>
    </>
  );
}
