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
    <div className="mx-auto max-w-2xl p-6">
      <div className="card space-y-5">
        <div>
          <h1 className="text-xl font-semibold">Consent to record</h1>
          <p className="text-sm text-zinc-400">Joining as {role}</p>
        </div>
        <ul className="list-disc space-y-2 pl-5 text-sm text-zinc-300">
          {CONSENT_TEXT.map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label className="label">Participant ID (e.g. P001)</label>
            <input
              className="input"
              value={f.participant_id}
              onChange={(e) => setF({ ...f, participant_id: e.target.value.toUpperCase().trim() })}
              placeholder="P001"
            />
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
                <label className="label">Accent / first language region</label>
                <select className="input" value={f.accent} onChange={(e) => setF({ ...f, accent: e.target.value })}>
                  <option value="">Select…</option>
                  <option value="north_indian">North Indian</option>
                  <option value="south_indian">South Indian</option>
                  <option value="east_indian">East Indian</option>
                  <option value="west_indian">West Indian</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={f.glasses} onChange={(e) => setF({ ...f, glasses: e.target.checked })} />
                I am wearing glasses
              </label>
            </>
          )}
        </div>

        <label className="flex items-start gap-2 text-sm">
          <input type="checkbox" className="mt-1" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
          I have read the above and agree to be recorded.
        </label>

        {error && <p className="text-sm text-rose-400">{error}</p>}
        <button
          className="btn btn-primary w-full"
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
  );
}
