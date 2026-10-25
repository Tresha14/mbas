"use client";

/* Design-system primitives (shadcn/ui-style API: variant + size via cva, className merging). */
import { cva, type VariantProps } from "class-variance-authority";
import { clsx, type ClassValue } from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, Moon, Sun, X } from "lucide-react";
import { useTheme } from "next-themes";
import { forwardRef, useEffect, useState } from "react";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const buttonStyles = cva(
  "inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-full text-sm font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-2 focus-visible:ring-offset-bg disabled:pointer-events-none disabled:opacity-40 active:scale-[0.98]",
  {
    variants: {
      variant: {
        default: "bg-fg text-bg hover:opacity-90",
        accent: "bg-accent text-accent-ink hover:opacity-90",
        outline: "border border-line bg-surface text-fg hover:bg-surface-2",
        ghost: "text-muted hover:bg-surface-2 hover:text-fg",
        danger: "bg-bad/10 text-bad hover:bg-bad/15",
        quiet: "bg-surface-2 text-fg hover:bg-line",
      },
      size: { sm: "h-8 px-3.5 text-[13px]", md: "h-10 px-5", lg: "h-12 px-7 text-[15px]", icon: "h-9 w-9" },
    },
    defaultVariants: { variant: "default", size: "md" },
  },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonStyles> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, ...props }, ref) => (
  <button ref={ref} className={cn(buttonStyles({ variant, size }), className)} {...props} />
));
Button.displayName = "Button";

export function Card({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("rounded-3xl border border-line bg-surface p-6 shadow-soft", className)} {...props}>
      {children}
    </div>
  );
}

export function SectionLabel({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("text-[11px] font-medium tracking-[0.12em] text-faint uppercase", className)} {...props} />;
}

const badgeStyles = cva("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium", {
  variants: {
    tone: {
      neutral: "bg-surface-2 text-muted",
      accent: "bg-accent-soft text-accent",
      good: "bg-good/10 text-good",
      warn: "bg-warn/10 text-warn",
      bad: "bg-bad/10 text-bad",
    },
  },
  defaultVariants: { tone: "neutral" },
});

export function Badge({ tone, className, ...props }: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeStyles>) {
  return <span className={cn(badgeStyles({ tone }), className)} {...props} />;
}

export function Dot({ tone = "neutral", pulse = false }: { tone?: "good" | "warn" | "bad" | "accent" | "neutral"; pulse?: boolean }) {
  const c = { good: "bg-good", warn: "bg-warn", bad: "bg-bad", accent: "bg-accent", neutral: "bg-faint" }[tone];
  return (
    <span className="relative inline-flex h-2 w-2">
      {pulse && <span className={cn("absolute inline-flex h-full w-full animate-ping rounded-full opacity-50", c)} />}
      <span className={cn("relative inline-flex h-2 w-2 rounded-full", c)} />
    </span>
  );
}

export const Input = forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => (
  <input
    ref={ref}
    className={cn("h-11 w-full rounded-2xl border border-line bg-surface px-4 text-sm outline-none transition placeholder:text-faint focus:border-accent focus:ring-4 focus:ring-accent/10", className)}
    {...p}
  />
));
Input.displayName = "Input";

export const Textarea = forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...p }, ref) => (
  <textarea
    ref={ref}
    className={cn("w-full rounded-2xl border border-line bg-surface px-4 py-3 text-sm leading-relaxed outline-none transition placeholder:text-faint focus:border-accent focus:ring-4 focus:ring-accent/10", className)}
    {...p}
  />
));
Textarea.displayName = "Textarea";

export function Label(props: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return <label {...props} className={cn("mb-2 block text-[13px] font-medium text-fg", props.className)} />;
}

/** Segmented control (pill). */
export function Segmented({ items, value, onChange, className }: { items: { value: string; label: string }[]; value: string; onChange: (v: string) => void; className?: string }) {
  return (
    <div className={cn("inline-flex rounded-full bg-surface-2 p-1", className)}>
      {items.map((it) => (
        <button key={it.value} onClick={() => onChange(it.value)}
          className={cn("relative rounded-full px-3.5 py-1.5 text-[13px] transition-colors", value === it.value ? "text-fg" : "text-muted hover:text-fg")}>
          {value === it.value && <motion.span layoutId={`seg-${items.map((i) => i.value).join("")}`} className="absolute inset-0 rounded-full bg-surface shadow-soft" transition={{ type: "spring", duration: 0.35, bounce: 0.15 }} />}
          <span className="relative">{it.label}</span>
        </button>
      ))}
    </div>
  );
}

/** Underline tabs for sections inside a card. */
export function Tabs({ items, value, onChange }: { items: { value: string; label: string }[]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex gap-6 border-b border-line">
      {items.map((it) => (
        <button key={it.value} onClick={() => onChange(it.value)}
          className={cn("relative -mb-px pb-3 text-[13px] transition-colors", value === it.value ? "text-fg" : "text-muted hover:text-fg")}>
          {it.label}
          {value === it.value && <motion.span layoutId={`tab-${items.map((i) => i.value).join("")}`} className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-fg" />}
        </button>
      ))}
    </div>
  );
}

/** Slide-over panel from the right. */
export function Sheet({ open, onClose, title, children }: { open: boolean; onClose: () => void; title: string; children: React.ReactNode }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 z-40 bg-black/20 backdrop-blur-[2px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.aside className="fixed top-0 right-0 z-50 flex h-full w-full max-w-md flex-col border-l border-line bg-surface shadow-soft"
            initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }} transition={{ type: "spring", duration: 0.45, bounce: 0.05 }}>
            <div className="flex items-center justify-between px-6 py-5">
              <h2 className="font-display text-xl">{title}</h2>
              <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close"><X size={18} /></Button>
            </div>
            <div className="flex-1 overflow-y-auto px-6 pb-8">{children}</div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

/** Expandable row (accordion item). */
export function Disclosure({ summary, children, defaultOpen = false }: { summary: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-line last:border-0">
      <button className="flex w-full items-center gap-4 py-5 text-left" onClick={() => setOpen(!open)}>
        <div className="min-w-0 flex-1">{summary}</div>
        <ChevronDown size={18} className={cn("shrink-0 text-faint transition-transform", open && "rotate-180")} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="pb-6">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="h-9 w-9" />;
  const dark = resolvedTheme === "dark";
  return (
    <Button variant="ghost" size="icon" aria-label="Toggle dark mode" onClick={() => setTheme(dark ? "light" : "dark")}>
      {dark ? <Sun size={17} /> : <Moon size={17} />}
    </Button>
  );
}

export function Empty({ icon, title, text, action }: { icon?: React.ReactNode; title: string; text?: string; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
      {icon && <div className="mb-4 grid h-12 w-12 place-items-center rounded-2xl bg-surface-2 text-muted">{icon}</div>}
      <p className="font-medium">{title}</p>
      {text && <p className="mt-1 max-w-sm text-sm text-muted">{text}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
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
