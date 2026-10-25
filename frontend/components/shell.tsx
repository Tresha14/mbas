"use client";

import { Activity, LogOut } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { claims, clearToken } from "@/lib/api";
import { Button, ThemeToggle } from "./ui";

export function Header({ right }: { right?: React.ReactNode }) {
  const router = useRouter();
  const [name, setName] = useState<string | null>(null);
  useEffect(() => setName(claims()?.name ?? null), []);
  return (
    <header className="no-print sticky top-0 z-20 border-b border-line bg-bg/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <span className="grid h-8 w-8 place-items-center rounded-xl bg-accent text-white"><Activity size={16} /></span>
          MBAS
        </Link>
        <div className="flex-1" />
        {right}
        <ThemeToggle />
        {name && (
          <Button variant="ghost" size="sm" onClick={() => { clearToken(); router.push("/login"); }}>
            <LogOut size={14} /> {name}
          </Button>
        )}
      </div>
    </header>
  );
}

export function VideoSlot({ el, label, className = "" }: { el: HTMLMediaElement | null; label: string; className?: string }) {
  return (
    <div className={`relative overflow-hidden rounded-2xl border border-line bg-black ${className}`}>
      <div
        className="h-full w-full [&>video]:h-full [&>video]:w-full [&>video]:object-contain"
        ref={(node) => {
          if (node && el && el.parentElement !== node) {
            node.innerHTML = "";
            node.appendChild(el);
          }
        }}
      />
      {!el && <div className="absolute inset-0 grid place-items-center text-sm text-zinc-400">{label}: waiting...</div>}
      <span className="absolute top-2 left-2 rounded-lg bg-black/60 px-2 py-0.5 text-xs text-white">{label}</span>
    </div>
  );
}
