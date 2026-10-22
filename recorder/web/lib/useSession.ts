"use client";

import { useEffect, useState } from "react";
import { api, type SessionSummary } from "./api";

/** Polls the session summary every `ms` milliseconds. */
export function useSession(code: string | null, ms = 2000) {
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!code) return;
    let alive = true;
    const load = () =>
      api
        .session(code)
        .then((s) => {
          if (alive) {
            setSummary(s);
            setError(null);
          }
        })
        .catch((e) => alive && setError((e as Error).message));
    load();
    const id = setInterval(load, ms);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [code, ms, tick]);

  return { summary, error, refresh: () => setTick((t) => t + 1) };
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}
