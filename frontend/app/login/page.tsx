"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { Logo } from "@/components/shell";
import { Button, Input, Label, Segmented } from "@/components/ui";
import { api, setToken, type Role } from "@/lib/api";

function Form() {
  const router = useRouter();
  const params = useSearchParams();
  const [mode, setMode] = useState(params.get("mode") === "register" ? "register" : "signin");
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("interviewer");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = mode === "signin" ? await api.login(email, password) : await api.register(email, name, password, role);
      setToken(r.token);
      router.push("/dashboard");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="w-full max-w-sm">
      <h1 className="font-display text-3xl tracking-tight">{mode === "signin" ? "Welcome back" : "Create your account"}</h1>
      <p className="mt-2 text-[15px] text-muted">{mode === "signin" ? "Sign in to run interviews and read reports." : "For interviewers and meeting hosts."}</p>
      <form onSubmit={submit} className="mt-8 space-y-5">
        {mode === "register" && (
          <>
            <div><Label htmlFor="name">Full name</Label><Input id="name" value={name} onChange={(e) => setName(e.target.value)} required /></div>
            <div>
              <Label>I will mostly run</Label>
              <Segmented items={[{ value: "interviewer", label: "Interviews" }, { value: "meeting_host", label: "Meetings" }]} value={role} onChange={(v) => setRole(v as Role)} />
            </div>
          </>
        )}
        <div><Label htmlFor="email">Email</Label><Input id="email" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></div>
        <div><Label htmlFor="pw">Password</Label><Input id="pw" type="password" minLength={8} autoComplete={mode === "signin" ? "current-password" : "new-password"} value={password} onChange={(e) => setPassword(e.target.value)} required /></div>
        {error && <p className="rounded-2xl bg-bad/10 px-4 py-3 text-sm text-bad">{error}</p>}
        <Button className="w-full" size="lg" disabled={busy}>{busy ? "One moment…" : mode === "signin" ? "Sign in" : "Create account"}</Button>
      </form>
      <p className="mt-6 text-center text-sm text-muted">
        {mode === "signin" ? "New here? " : "Already have an account? "}
        <button className="font-medium text-fg underline-offset-4 hover:underline" onClick={() => setMode(mode === "signin" ? "register" : "signin")}>
          {mode === "signin" ? "Create an account" : "Sign in"}
        </button>
      </p>
    </div>
  );
}

export default function Login() {
  return (
    <main className="grid min-h-screen lg:grid-cols-2">
      <div className="flex flex-col px-8 py-8">
        <Logo />
        <div className="flex flex-1 items-center justify-center py-12"><Suspense><Form /></Suspense></div>
        <p className="text-[13px] text-faint">Candidates don&apos;t need an account. They join with the link you send them.</p>
      </div>
      <div className="relative hidden overflow-hidden bg-[#141413] lg:block">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_20%,rgba(94,184,173,0.25),transparent_60%)]" />
        <div className="relative flex h-full flex-col justify-end p-14 text-white">
          <p className="font-display text-3xl leading-snug text-white/90">
            &ldquo;Instead of a vague rejection, every candidate leaves with three concrete things to work on.&rdquo;
          </p>
          <p className="mt-6 text-sm text-white/50">Designed for placement cells, recruiters and student mock interviews</p>
        </div>
      </div>
    </main>
  );
}
