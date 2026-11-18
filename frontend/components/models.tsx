"use client";

import { Badge, Card, Dot } from "./ui";

interface ModelInfo {
  name: string;
  group: "Vision" | "Speech" | "Behaviour and content";
  does: string;
  feeds: string;
  result: string;
  liveKey?: string;
}

/** Headline numbers are from docs/model_report.md (generated 2026-10-25) and the latest face-detector commit; re-run `python -m tools.model_report` to refresh. */
export const MODELS: ModelInfo[] = [
  { name: "Face detector + 5 landmarks", group: "Vision", does: "Finds the face and eye/nose/mouth points in every frame", feeds: "Everything on the camera side", result: "Landmark NME 0.28 (median)" , liveKey: "vision" },
  { name: "Head-pose CNN", group: "Vision", does: "Yaw, pitch and roll of the head", feeds: "Camera-facing %, attention", result: "MAE 10.8° yaw, 10.7° pitch, 5.5° roll" , liveKey: "vision" },
  { name: "Gaze + iris CNN", group: "Vision", does: "Where the eyes point and iris movement", feeds: "Gaze plot, looking away, reading detection, blinks", result: "Trained on MPIIGaze (see model report)" , liveKey: "vision" },
  { name: "Engagement CNN-GRU", group: "Vision", does: "Engagement from short video clips", feeds: "Engagement signal (when trained)", result: "See model report" },
  { name: "Window engagement MLP + attention score", group: "Vision", does: "Engagement class and 0-100 attention per 5 s window", feeds: "Attention and Engagement signals", result: "Engagement macro-F1 0.75, attention MAE 3.2" },
  { name: "Voice activity detector (VAD)", group: "Speech", does: "Speech vs silence in 10 ms steps", feeds: "Pause rate, long-pause flags, talk time", result: "See model report", liveKey: "vad" },
  { name: "Conformer-CTC ASR (streaming)", group: "Speech", does: "Live speech-to-text with word timings", feeds: "Transcript, words per minute, fillers, hedges", result: "See model report", liveKey: "asr" },
  { name: "4-gram Kneser-Ney language model", group: "Speech", does: "Corrects likely word sequences while decoding", feeds: "Transcript accuracy", result: "Built from LibriSpeech + interview text" },
  { name: "Disfluency CRNN", group: "Speech", does: "Hears fillers, repetitions, prolongations and blocks", feeds: "Hesitation index", result: "Filler F1 0.68, repetition F1 0.40", liveKey: "disfluency" },
  { name: "Speaking pattern + hesitation index", group: "Behaviour and content", does: "Confident, hesitant, rushed or monotone, plus a 0-100 hesitation score", feeds: "Hesitation signal, answer cards", result: "Pattern macro-F1 0.81, hesitation MAE 4.3" },
  { name: "Answer-quality transformer", group: "Behaviour and content", does: "Relevance, quality and completeness of an answer", feeds: "Answer scores, follow-up suggestions", result: "Quality QWK 0.86 (TF-IDF baseline 0.77)", liveKey: "answer_quality" },
  { name: "Fusion network + Shapley reasons", group: "Behaviour and content", does: "Combines everything into one effectiveness score with the top reasons", feeds: "Answer effectiveness, follow-up probability", result: "Effectiveness MAE 5.9, follow-up F1 0.96", liveKey: "fusion" },
  { name: "Meeting head", group: "Behaviour and content", does: "Engagement and contribution balance for meetings", feeds: "Meeting mode scores", result: "AMI balance MAE 0.03" },
];

export function ModelList({ live, compact = false }: { live?: Record<string, boolean> | null; compact?: boolean }) {
  const groups = ["Vision", "Speech", "Behaviour and content"] as const;
  return (
    <div className={compact ? "space-y-5 px-3" : "space-y-6"}>
      <p className={compact ? "text-xs text-faint" : "text-[13px] text-muted"}>
        {MODELS.length} models, all defined in PyTorch and trained from random weights. No pre-trained models or AI APIs. Trained on public datasets plus synthetic data, so treat scores as decision support.
      </p>
      {groups.map((g) => (
        <section key={g}>
          <h3 className="mb-2 text-[11px] font-medium tracking-wider text-faint uppercase">{g}</h3>
          <div className={compact ? "space-y-2" : "grid gap-3 md:grid-cols-2"}>
            {MODELS.filter((m) => m.group === g).map((m) => {
              const on = m.liveKey && live ? live[m.liveKey] : undefined;
              return (
                <Card key={m.name} className="p-4">
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-[14px] font-medium leading-snug">{m.name}</p>
                    {on !== undefined && (
                      <Badge tone={on ? "good" : "warn"} className="shrink-0"><Dot tone={on ? "good" : "warn"} /> {on ? "running" : "not loaded"}</Badge>
                    )}
                  </div>
                  <p className="mt-1.5 text-[13px] text-muted">{m.does}</p>
                  <p className="mt-2 text-xs text-faint">Feeds: {m.feeds}</p>
                  <p className="mt-0.5 text-xs text-faint">Result: {m.result}</p>
                </Card>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
