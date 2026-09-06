import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { Band, Row, Schema } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const pct = (v: number, digits = 0) =>
  `${(v * 100).toFixed(digits)}%`;

export const BAND_LABEL: Record<Band, string> = {
  high: "высокий",
  medium: "средний",
  low: "низкий",
};

/** Риск никогда не кодируется одним цветом — рядом всегда текстовая метка. */
export const BAND_STYLE: Record<Band, string> = {
  high: "bg-risk-high/12 text-risk-high border-risk-high/30",
  medium: "bg-risk-mid/12 text-risk-mid border-risk-mid/30",
  low: "bg-risk-low/10 text-risk-low border-risk-low/25",
};

export const BAND_BAR: Record<Band, string> = {
  high: "bg-risk-high",
  medium: "bg-risk-mid",
  low: "bg-risk-low",
};

export function bandOf(p: number, bands: { low_max: number; high_min: number }): Band {
  if (p >= bands.high_min) return "high";
  if (p <= bands.low_max) return "low";
  return "medium";
}

/** precision@K и recall@K по уже полученным вероятностям — без запроса к серверу. */
export function metricsAtK(rows: Row[], k: number) {
  const sorted = [...rows].sort((a, b) => b.probability - a.probability);
  const top = sorted.slice(0, k);
  const known = rows.filter((r) => r.actual !== undefined);
  if (known.length === 0) {
    return { precision: null, recall: null, caught: null, total: null };
  }
  const caught = top.filter((r) => r.actual?.no_pass).length;
  const total = known.filter((r) => r.actual?.no_pass).length;
  return {
    precision: top.length ? caught / top.length : 0,
    recall: total ? caught / total : 0,
    caught,
    total,
  };
}

/** Порог, соответствующий списку ровно из K человек. */
export function thresholdForK(rows: Row[], k: number): number {
  if (!rows.length) return 0;
  const sorted = [...rows].sort((a, b) => b.probability - a.probability);
  const i = Math.min(Math.max(k, 1), sorted.length) - 1;
  return sorted[i].probability;
}

/** Человеческая подпись значения признака: studytime=2 -> «2–5 часов». */
export function decode(
  schema: Schema | null,
  name: string,
  value: string | number | null,
): string {
  if (value === null || value === undefined || value === "") return "—";
  const f = schema?.features.find((x) => x.name === name);
  if (!f) return String(value);
  const v = f.values?.[String(value)];
  return v ?? String(value);
}

export function labelOf(schema: Schema | null, name: string): string {
  return schema?.features.find((x) => x.name === name)?.label ?? name;
}

export function downloadDisabled(rows: unknown[]) {
  return !rows || rows.length === 0;
}
