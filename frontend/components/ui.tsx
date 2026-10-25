"use client";

/* shadcn/ui-style primitives (same API shape: variant + size via cva, className merging). */
import { cva, type VariantProps } from "class-variance-authority";
import { clsx, type ClassValue } from "clsx";
import { motion } from "framer-motion";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { forwardRef, useEffect, useState } from "react";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const buttonStyles = cva(
  "inline-flex items-center justify-center gap-2 rounded-xl text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:pointer-events-none disabled:opacity-40",
  {
    variants: {
      variant: {
        default: "bg-accent text-white hover:opacity-90",
        outline: "border border-line bg-surface hover:bg-surface-2",
        ghost: "hover:bg-surface-2",
        danger: "bg-bad text-white hover:opacity-90",
        good: "bg-good text-white hover:opacity-90",
      },
      size: { sm: "h-8 px-3", md: "h-10 px-4", lg: "h-12 px-6 text-base", icon: "h-10 w-10" },
    },
    defaultVariants: { variant: "default", size: "md" },
  },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonStyles> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, ...props }, ref) => (
  <button ref={ref} className={cn(buttonStyles({ variant, size }), className)} {...props} />
));
Button.displayName = "Button";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={cn("rounded-2xl border border-line bg-surface p-4 shadow-sm", className)}
      {...(props as object)}
    />
  );
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn("mb-2 text-xs font-semibold tracking-wide text-muted uppercase", className)} {...props} />;
}

const badgeStyles = cva("inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium", {
  variants: {
    tone: {
      neutral: "bg-surface-2 text-muted",
      accent: "bg-accent-soft text-accent",
      good: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300",
      warn: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
      bad: "bg-rose-500/15 text-rose-600 dark:text-rose-300",
    },
  },
  defaultVariants: { tone: "neutral" },
});

export function Badge({ tone, className, ...props }: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeStyles>) {
  return <span className={cn(badgeStyles({ tone }), className)} {...props} />;
}

export const Input = forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => (
  <input
    ref={ref}
    className={cn("h-10 w-full rounded-xl border border-line bg-surface px-3 text-sm outline-none focus:border-accent", className)}
    {...p}
  />
));
Input.displayName = "Input";

export function Label(props: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return <label {...props} className={cn("mb-1 block text-xs font-medium text-muted", props.className)} />;
}

export function Progress({ value, tone = "accent" }: { value: number; tone?: "accent" | "good" | "warn" | "bad" }) {
  const color = { accent: "bg-accent", good: "bg-good", warn: "bg-warn", bad: "bg-bad" }[tone];
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
      <motion.div className={cn("h-full rounded-full", color)} animate={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

export function Tabs({ tabs, value, onChange }: { tabs: string[]; value: string; onChange: (t: string) => void }) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-surface-2 p-1">
      {tabs.map((t) => (
        <button
          key={t}
          onClick={() => onChange(t)}
          className={cn("rounded-lg px-3 py-1.5 text-sm transition", value === t ? "bg-surface shadow-sm" : "text-muted hover:text-fg")}
        >
          {t}
        </button>
      ))}
    </div>
  );
}

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="h-10 w-10" />;
  const dark = resolvedTheme === "dark";
  return (
    <Button variant="ghost" size="icon" aria-label="Toggle dark mode" onClick={() => setTheme(dark ? "light" : "dark")}>
      {dark ? <Sun size={18} /> : <Moon size={18} />}
    </Button>
  );
}

export function scoreTone(v: number | null | undefined): "good" | "warn" | "bad" | "neutral" {
  if (v === null || v === undefined || Number.isNaN(v)) return "neutral";
  return v >= 70 ? "good" : v >= 50 ? "warn" : "bad";
}

export function mmss(t: number | null | undefined): string {
  if (t === null || t === undefined) return "--:--";
  const s = Math.max(0, Math.floor(t));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
