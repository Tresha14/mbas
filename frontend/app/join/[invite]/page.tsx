"use client";

import { ShieldCheck } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import { Logo } from "@/components/shell";
import { Button, Disclosure, Input, Label } from "@/components/ui";
import { api, setToken } from "@/lib/api";

export default function Join() {
  const { invite } = useParams<{ invite: string }>();
  const router = useRouter();
  const [name, setName] = useState("");
  const [agree, setAgree] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function join(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      const r = await api.join(invite, name, agree);
      setToken(r.token, true);   // per-tab session token; candidates have no account
      router.push(`/room/${r.interview.id}`);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen flex-col px-6 py-8">
      <Logo />
      <div className="flex flex-1 items-center justify-center py-10">
        <div className="w-full max-w-md">
          <div className="mb-6 grid h-12 w-12 place-items-center rounded-2xl bg-accent-soft text-accent"><ShieldCheck size={22} strokeWidth={1.75} /></div>
          <h1 className="font-display text-3xl tracking-tight">You&apos;re invited to a mock interview</h1>
          <p className="mt-3 text-[15px] leading-relaxed text-muted">
            The call is recorded and analysed so your interviewer can give you useful feedback afterwards.
            You won&apos;t see any scores during the call.
          </p>
          <div className="mt-6 rounded-3xl border border-line bg-surface px-5">
            <Disclosure summary={<span className="text-sm font-medium">What is recorded, and who sees it</span>}>
              <ul className="space-y-2 text-sm leading-relaxed text-muted">
                <li>Your camera, microphone and shared screen are analysed during the call.</li>
                <li>Only your interviewer sees the analysis. You may receive a reviewed feedback report.</li>
                <li>Data is handled under India&apos;s DPDP Act 2023. Ask your interviewer to delete it at any time.</li>
                <li>Scores support the interviewer&apos;s judgement and never decide anything on their own.</li>
              </ul>
            </Disclosure>
          </div>
          <form onSubmit={join} className="mt-8 space-y-5">
            <div><Label htmlFor="n">Your name</Label><Input id="n" value={name} onChange={(e) => setName(e.target.value)} required /></div>
            <label className="flex cursor-pointer items-start gap-3 text-sm leading-relaxed text-muted">
              <input type="checkbox" className="mt-1 h-4 w-4 accent-[var(--accent)]" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
              I agree to this mock interview being recorded and analysed.
            </label>
            {error && <p className="rounded-2xl bg-bad/10 px-4 py-3 text-sm text-bad">{error}</p>}
            <Button className="w-full" size="lg" disabled={!agree || !name || busy}>{busy ? "Joining…" : "Continue"}</Button>
          </form>
        </div>
      </div>
    </main>
  );
}
