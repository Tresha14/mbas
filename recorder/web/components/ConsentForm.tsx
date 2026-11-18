"use client";

import { useState } from "react";
import type { Role } from "@/lib/api";

export const CONSENT_VERSION = "v1";

const CONSENT_TEXT = [
  "This session is recorded to build a research dataset for a final-year project (Multimodal Behavioural Analysis System).",
  "We record: your webcam video, your microphone audio, your shared screen (candidate only) and the timing of the questions.",
  "Recordings are stored only on the project machines, are used only to train and evaluate our own models, and are never published. Clips may be viewed by project annotators.",
  "Your name is not stored; you are identified by a participant ID. You may stop at any time and ask for your recordings to be deleted (Digital Personal Data Protection Act, 2023).",
  "Nothing about you is scored or decided during this mock interview.",
];

export interface ParticipantInfo {
  participant_id: string;
  age_band: string;
  gender: string;
  glasses: boolean;
  accent: string;
}

export default function ConsentForm({ role, onSubmit }: { role: Role; onSubmit: (info: ParticipantInfo) => Promise<void> }) {
  const [agree, setAgree] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [f, setF] = useState<ParticipantInfo>({
    participant_id: "",
    age_band: "18-24",
    gender: "",
    glasses: false,
    accent: "",
  });
  const isCandidate = role === "candidate";
  const valid = agree && /^P\d{3,}$/.test(f.participant_id) && (!isCandidate || (f.gender && f.accent));

  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <div className="w-full max-w-lg">
        <div className="card space-y-6">
          <div>
            <div className="mb-1 text-xs font-medium uppercase tracking-widest text-indigo-400">
              Joining as {role}
            </div>
            <h1 className="text-xl font-semibold">Consent to record</h1>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-4">
            <ul className="space-y-2.5 text-sm text-zinc-400">
              {CONSENT_TEXT.map((t, i) => (
                <li key={i} className="flex gap-3">
                  <span className="mt-0.5 h-4 w-4 shrink-0 text-zinc-600">·</span>
                  <span>{t}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="sm:col-span-2">
              <label className="label">Participant ID</label>
              <input
                className="input font-mono"
                value={f.participant_id}
                onChange={(e) => setF({ ...f, participant_id: e.target.value.toUpperCase().trim() })}
                placeholder="e.g. P001"
              />
              {f.participant_id && !/^P\d{3,}$/.test(f.participant_id) && (
                <p className="mt-1 text-xs text-amber-500">Format: P followed by 3+ digits (e.g. P001)</p>
              )}
            </div>

            {isCandidate && (
              <>
                <div>
                  <label className="label">Age band</label>
                  <select className="input" value={f.age_band} onChange={(e) => setF({ ...f, age_band: e.target.value })}>
                    {["18-24", "25-34", "35-44", "45+"].map((a) => (
                      <option key={a}>{a}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="label">Gender</label>
                  <select className="input" value={f.gender} onChange={(e) => setF({ ...f, gender: e.target.value })}>
                    <option value="">Select…</option>
                    <option value="female">Female</option>
                    <option value="male">Male</option>
                    <option value="non_binary">Non-binary</option>
                    <option value="prefer_not_to_say">Prefer not to say</option>
                  </select>
                </div>
                <div>
                  <label className="label">First language region</label>
                  <select className="input" value={f.accent} onChange={(e) => setF({ ...f, accent: e.target.value })}>
                    <option value="">Select…</option>
                    <option value="north_indian">North Indian</option>
                    <option value="south_indian">South Indian</option>
                    <option value="east_indian">East Indian</option>
                    <option value="west_indian">West Indian</option>
                    <option value="other">Other</option>
                  </select>
                </div>
                <div className="flex items-center gap-3">
                  <label className="flex cursor-pointer items-center gap-2.5 text-sm text-zinc-300">
                    <input
                      type="checkbox"
                      className="h-4 w-4 rounded border-zinc-600 accent-indigo-500"
                      checked={f.glasses}
                      onChange={(e) => setF({ ...f, glasses: e.target.checked })}
                    />
                    Wearing glasses
                  </label>
                </div>
              </>
            )}
          </div>

          <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-zinc-800 p-3 text-sm text-zinc-300 transition hover:border-zinc-700">
            <input
              type="checkbox"
              className="mt-0.5 h-4 w-4 rounded border-zinc-600 accent-indigo-500"
              checked={agree}
              onChange={(e) => setAgree(e.target.checked)}
            />
            <span>I have read the above and agree to be recorded.</span>
          </label>

          {error && (
            <div className="rounded-xl border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
              {error}
            </div>
          )}

          <button
            className="btn btn-primary w-full py-2.5"
            disabled={!valid || busy}
            onClick={async () => {
              setBusy(true);
              setError(null);
              try {
                await onSubmit(f);
              } catch (e) {
                setError((e as Error).message);
                setBusy(false);
              }
            }}
          >
            {busy ? "Joining…" : "Agree and join"}
          </button>
        </div>
      </div>
    </div>
  );
}
