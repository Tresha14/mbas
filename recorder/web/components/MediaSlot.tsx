"use client";

import { useEffect, useRef } from "react";

/** Mounts a LiveKit-attached media element (or a local MediaStream) into a box. */
export default function MediaSlot({
  el,
  stream,
  label,
  muted = false,
  className = "",
}: {
  el?: HTMLMediaElement | null;
  stream?: MediaStream | null;
  label: string;
  muted?: boolean;
  className?: string;
}) {
  const box = useRef<HTMLDivElement>(null);
  const local = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const b = box.current;
    if (!b || !el) return;
    el.className = "h-full w-full object-contain";
    b.appendChild(el);
    return () => {
      if (el.parentElement === b) b.removeChild(el);
    };
  }, [el]);

  useEffect(() => {
    if (local.current) local.current.srcObject = stream ?? null;
  }, [stream]);

  const empty = !el && !stream;
  return (
    <div className={`relative overflow-hidden rounded-2xl border border-zinc-800 bg-black ${className}`}>
      <div ref={box} className="absolute inset-0" />
      {stream && <video ref={local} autoPlay playsInline muted={muted} className="absolute inset-0 h-full w-full object-contain" />}
      {empty && <div className="absolute inset-0 flex items-center justify-center text-sm text-zinc-500">No {label}</div>}
      <span className="absolute left-3 top-3 rounded-lg bg-black/60 px-2 py-1 text-xs text-zinc-200">{label}</span>
    </div>
  );
}
