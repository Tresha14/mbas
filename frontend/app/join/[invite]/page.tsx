"use client";

import { ShieldCheck } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import { Header } from "@/components/shell";
import { Button, Card, Input, Label } from "@/components/ui";
import { api, setToken } from "@/lib/api";

export default function Join() {
  const { invite } = useParams<{ invite: string }>();
  const router = useRouter();
  const [name, setName] = useState("");
  const [agree, setAgree] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function join(e: React.FormEvent) {
    e.preventDefault();
    try {
      const r = await api.join(invite, name, agree);
      setToken(r.token, true);   // per-tab session token; candidates have no account
      router.push(`/room/${r.interview.id}`);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <>
      <Header />
      <main className="mx-auto max-w-xl px-4 py-12">
        <Card>
          <div className="mb-4 flex items-center gap-2"><ShieldCheck className="text-accent" /><h1 className="text-lg font-semibold">Before you join</h1></div>
          <div className="space-y-2 text-sm text-muted">
            <p>This mock interview is recorded and analysed by the MBAS project so your interviewer can give you useful feedback.</p>
            <ul className="list-disc space-y-1 pl-5">
              <li>Your camera, microphone and shared screen are analysed during the call. You will <b>not</b> see live scores.</li>
              <li>After the interview your interviewer reviews a feedback report and may release it to you.</li>
              <li>Data is handled under India&apos;s DPDP Act 2023; you can ask the interviewer to delete it at any time.</li>
              <li>Scores support the interviewer&apos;s judgement and never decide on their own.</li>
            </ul>
          </div>
          <form onSubmit={join} className="mt-6 space-y-4">
            <div><Label htmlFor="n">Your name</Label><Input id="n" value={name} onChange={(e) => setName(e.target.value)} required /></div>
            <label className="flex items-start gap-2 text-sm">
              <input type="checkbox" className="mt-1" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
              I have read this and agree to be recorded and analysed for this mock interview.
            </label>
            {error && <p className="text-sm text-bad">{error}</p>}
            <Button className="w-full" disabled={!agree || !name}>Join the call</Button>
          </form>
        </Card>
      </main>
    </>
  );
}
