"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { LiveDemo } from "@/components/live-demo";
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
    <main className="grid min-h-screen lg:h-screen lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      {/* left: logo, form and footer share one centred column, so their left edges line up */}
      <div className="flex justify-center px-8 py-10">
        <div className="flex w-full max-w-sm flex-col">
          <Logo />
          <div className="flex flex-1 items-center py-12"><Suspense><Form /></Suspense></div>
          <p className="text-[13px] leading-relaxed text-faint">Candidates don&apos;t need an account. They join with the link you send them.</p>
        </div>
      </div>
      {/* right: heading, live preview and quote share one centred column */}
      <div className="relative hidden overflow-hidden bg-[#121413] lg:flex lg:items-center lg:justify-center">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_70%_60%_at_50%_30%,rgba(94,184,173,0.22),transparent_70%),radial-gradient(ellipse_60%_50%_at_50%_100%,rgba(217,165,74,0.10),transparent_70%)]" />
        <div className="absolute inset-0 opacity-[0.06] [mask-image:radial-gradient(ellipse_at_center,black,transparent_75%)] [background-image:linear-gradient(rgba(255,255,255,.6)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.6)_1px,transparent_1px)] [background-size:44px_44px]" />
        <div className="relative flex w-full max-w-[640px] flex-col gap-10 px-10 py-12 text-white [@media(max-height:860px)]:gap-6 [@media(max-height:860px)]:py-8">
          <div>
            <p className="text-[12px] font-medium tracking-[0.2em] text-[#7fd1c6] uppercase">InterLens · AI interview copilot</p>
            <h2 className="mt-3 font-display text-[40px] leading-[1.1] tracking-tight xl:text-[46px] [@media(max-height:860px)]:text-[34px]">
              See what an interview <em className="text-[#7fd1c6]">really</em> said.
            </h2>
            <p className="mt-4 max-w-lg text-[15px] leading-relaxed text-white/55 [@media(max-height:700px)]:hidden">
              Eyes, voice and words, read live by models we trained ourselves: attention, hesitation, fillers, pauses and
              answer quality, with a fair report at the end.
            </p>
          </div>
          <LiveDemo />
          <figure className="border-l-2 border-[#5eb8ad]/50 pl-5 [@media(max-height:860px)]:hidden">
            <blockquote className="font-display text-xl leading-snug text-white/80">
              &ldquo;Instead of a vague rejection, every candidate leaves with three concrete things to work on.&rdquo;
            </blockquote>
            <figcaption className="mt-2 text-[13px] text-white/40">Designed for placement cells, recruiters and student mock interviews</figcaption>
          </figure>
        </div>
      </div>
    </main>
  );
}
