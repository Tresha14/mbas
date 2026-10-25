"use client";

import { LogOut } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { claims, clearToken } from "@/lib/api";
import { Button, cn, ThemeToggle } from "./ui";

export function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2.5">
      <span className="grid h-7 w-7 place-items-center rounded-lg bg-fg text-bg">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
          <path d="M3 12h3l2-6 4 12 3-9 2 3h4" />
        </svg>
      </span>
      <span className="font-display text-[19px] tracking-tight">mbas</span>
    </Link>
  );
}

export function Header({ right, nav = true, wide = false }: { right?: React.ReactNode; nav?: boolean; wide?: boolean }) {
  const router = useRouter();
  const [me, setMe] = useState<{ name?: string; role?: string } | null>(null);
  useEffect(() => setMe(claims()), []);
  const host = me?.role === "interviewer" || me?.role === "meeting_host";
  return (
    <header className="no-print sticky top-0 z-30 border-b border-line/70 bg-bg/80 backdrop-blur-xl">
      <div className={cn("mx-auto flex h-16 items-center gap-6 px-6", wide ? "max-w-[1440px]" : "max-w-6xl")}>
        <Logo />
        {nav && host && (
          <nav className="hidden items-center gap-1 text-[13px] sm:flex">
            <Link href="/dashboard" className="rounded-full px-3 py-1.5 text-muted transition hover:bg-surface-2 hover:text-fg">Interviews</Link>
          </nav>
        )}
        <div className="flex-1" />
        {right}
        <ThemeToggle />
        {me?.name && (
          <div className="flex items-center gap-2">
            <span className="grid h-8 w-8 place-items-center rounded-full bg-accent-soft text-[13px] font-medium text-accent">
              {me.name.split(" ").map((p) => p[0]).slice(0, 2).join("")}
            </span>
            {host && (
              <Button variant="ghost" size="icon" aria-label="Sign out" onClick={() => { clearToken(); router.push("/login"); }}>
                <LogOut size={16} />
              </Button>
            )}
          </div>
        )}
      </div>
    </header>
  );
}

export function VideoSlot({ el, label, className = "", muted = false }: { el: HTMLMediaElement | null; label: string; className?: string; muted?: boolean }) {
  return (
    <div className={cn("group relative overflow-hidden rounded-3xl bg-[#121211]", className)}>
      <div
        className="h-full w-full [&>video]:h-full [&>video]:w-full [&>video]:object-cover"
        ref={(node) => {
          if (node && el && el.parentElement !== node) {
            node.innerHTML = "";
            if (muted && el instanceof HTMLVideoElement) el.muted = true;
            node.appendChild(el);
          }
        }}
      />
      {!el && (
        <div className="absolute inset-0 grid place-items-center">
          <div className="text-center">
            <div className="mx-auto mb-3 h-10 w-10 animate-pulse rounded-full bg-white/5" />
            <p className="text-[13px] text-white/40">Waiting for {label.toLowerCase()}</p>
          </div>
        </div>
      )}
      <span className="absolute bottom-3 left-3 rounded-full bg-black/45 px-3 py-1 text-xs text-white/90 backdrop-blur-md">{label}</span>
    </div>
  );
}

export function PageTitle({ eyebrow, title, subtitle, right }: { eyebrow?: string; title: string; subtitle?: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        {eyebrow && <p className="mb-2 text-[13px] text-accent">{eyebrow}</p>}
        <h1 className="font-display text-[34px] leading-tight tracking-tight">{title}</h1>
        {subtitle && <p className="mt-2 text-[15px] text-muted">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}
