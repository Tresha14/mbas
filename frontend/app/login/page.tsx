"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { Header } from "@/components/shell";
import { Button, Card, Input, Label, Tabs } from "@/components/ui";
import { api, setToken, type Role } from "@/lib/api";

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [mode, setMode] = useState(params.get("mode") === "register" ? "Create account" : "Sign in");
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
      const r = mode === "Sign in" ? await api.login(email, password) : await api.register(email, name, password, role);
      setToken(r.token);
      router.push("/dashboard");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto grid max-w-md gap-6 px-4 py-16">
      <Tabs tabs={["Sign in", "Create account"]} value={mode} onChange={setMode} />
      <Card>
        <form onSubmit={submit} className="space-y-4">
          {mode !== "Sign in" && (
            <>
              <div><Label htmlFor="name">Name</Label><Input id="name" value={name} onChange={(e) => setName(e.target.value)} required /></div>
              <div>
                <Label>Account type</Label>
                <Tabs tabs={["interviewer", "meeting_host"]} value={role} onChange={(v) => setRole(v as Role)} />
              </div>
            </>
          )}
          <div><Label htmlFor="email">Email</Label><Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></div>
          <div><Label htmlFor="pw">Password</Label><Input id="pw" type="password" minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} required /></div>
          {error && <p className="text-sm text-bad">{error}</p>}
          <Button className="w-full" disabled={busy}>{busy ? "Please wait..." : mode}</Button>
        </form>
      </Card>
      <p className="text-center text-xs text-muted">Candidates do not need an account: they join with the link from the interviewer.</p>
    </main>
  );
}

export default function Login() {
  return (
    <>
      <Header />
      <Suspense><LoginForm /></Suspense>
    </>
  );
}
