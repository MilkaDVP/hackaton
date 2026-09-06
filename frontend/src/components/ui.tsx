import * as React from "react";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ Button */
type BtnProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost" | "outline" | "subtle";
  size?: "sm" | "md";
};

export const Button = React.forwardRef<HTMLButtonElement, BtnProps>(
  ({ className, variant = "outline", size = "md", ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium",
        "transition-colors duration-150 disabled:pointer-events-none disabled:opacity-50",
        size === "sm" ? "h-8 px-3 text-xs" : "h-9 px-4 text-sm",
        variant === "primary" && "bg-accent text-white hover:bg-accent/90",
        variant === "outline" && "border border-line bg-surface hover:bg-raised",
        variant === "ghost" && "hover:bg-raised",
        variant === "subtle" && "bg-raised hover:bg-line",
        className,
      )}
      {...props}
    />
  ),
);
Button.displayName = "Button";

/* -------------------------------------------------------------------- Card */
export function Card({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("card", className)} {...p} />;
}

/* ------------------------------------------------------------------- Badge */
export function Badge({
  className, ...p
}: React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5",
        "text-2xs font-medium leading-none",
        className,
      )}
      {...p}
    />
  );
}

/* ------------------------------------------------------------------ Slider */
export function Slider({
  value, onChange, min, max, step = 1, label, id, ...rest
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  label: string;
  id: string;
} & Omit<React.InputHTMLAttributes<HTMLInputElement>, "onChange" | "value">) {
  return (
    <input
      id={id}
      type="range"
      aria-label={label}
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className={cn(
        "h-1.5 w-full cursor-pointer appearance-none rounded-full bg-line",
        "[&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4",
        "[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full",
        "[&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:shadow",
        "[&::-webkit-slider-thumb]:transition-transform",
        "[&::-webkit-slider-thumb]:hover:scale-110",
        "[&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full",
        "[&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-accent",
      )}
      {...rest}
    />
  );
}

/* ---------------------------------------------------------------- Skeleton */
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton", className)} aria-hidden />;
}

/* ------------------------------------------------------------------ Drawer */
export function Drawer({
  open, onClose, title, children,
}: {
  open: boolean;
  onClose: () => void;
  title: React.ReactNode;
  children: React.ReactNode;
}) {
  React.useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div
        className="absolute inset-0 animate-fade-in bg-black/35"
        onClick={onClose}
        aria-hidden
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === "string" ? title : "Карточка студента"}
        className={cn(
          "relative flex h-full w-full max-w-xl animate-slide-in flex-col",
          "border-l border-line bg-surface shadow-2xl",
        )}
      >
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div className="min-w-0">{title}</div>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Закрыть">
            ✕
          </Button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ Fields */
export function Field({
  label, hint, htmlFor, children,
}: {
  label: string;
  hint?: string | null;
  htmlFor?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="block text-sm font-medium text-ink">
        {label}
      </label>
      {children}
      {hint ? <p className="text-xs text-faint">{hint}</p> : null}
    </div>
  );
}

export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, ...p }, ref) => (
  <select
    ref={ref}
    className={cn(
      "h-9 w-full rounded-lg border border-line bg-surface px-3 text-sm",
      "transition-colors hover:border-faint",
      className,
    )}
    {...p}
  />
));
Select.displayName = "Select";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...p }, ref) => (
  <input
    ref={ref}
    className={cn(
      "h-9 w-full rounded-lg border border-line bg-surface px-3 text-sm nums",
      "transition-colors hover:border-faint",
      className,
    )}
    {...p}
  />
));
Input.displayName = "Input";

/* ------------------------------------------------------------------ States */
export function EmptyState({
  icon, title, description, action,
}: {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
      {icon ? <div className="text-faint">{icon}</div> : null}
      <h3 className="text-base font-semibold">{title}</h3>
      {description ? (
        <p className="max-w-md text-sm text-muted">{description}</p>
      ) : null}
      {action}
    </div>
  );
}

export function ErrorState({
  title, hint, onRetry, details,
}: {
  title: string;
  hint?: string;
  onRetry?: () => void;
  details?: React.ReactNode;
}) {
  return (
    <div
      role="alert"
      className="rounded-xl border border-risk-high/30 bg-risk-high/5 px-5 py-4"
    >
      <p className="text-sm font-semibold text-risk-high">{title}</p>
      {hint ? <p className="mt-1 text-sm text-muted">{hint}</p> : null}
      {details}
      {onRetry ? (
        <Button size="sm" className="mt-3" onClick={onRetry}>
          Попробовать снова
        </Button>
      ) : null}
    </div>
  );
}
