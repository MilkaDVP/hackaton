import * as React from "react";
import type { Row } from "@/lib/types";
import { metricsAtK, pct } from "@/lib/utils";
import { Button, Select, Slider } from "./ui";

export interface Filters {
  school: string;
  sex: string;
  address: string;
  query: string;
}

export const EMPTY_FILTERS: Filters = { school: "all", sex: "all", address: "all", query: "" };

/**
 * Всё, что раньше занимало полэкрана: ёмкость, фильтры, поиск.
 * Значение по умолчанию разумное, поэтому большинству это открывать не нужно.
 */
export function ListSettings({
  rows, capacity, onCapacity, threshold, filters, onFilters, total,
}: {
  rows: Row[];
  capacity: number;
  onCapacity: (k: number) => void;
  threshold: number;
  filters: Filters;
  onFilters: (f: Filters) => void;
  total: number;
}) {
  const [open, setOpen] = React.useState(false);
  const m = React.useMemo(() => metricsAtK(rows, capacity), [rows, capacity]);
  const max = Math.min(rows.length, 300);
  const dirty =
    filters.school !== "all" || filters.sex !== "all" ||
    filters.address !== "all" || filters.query !== "";

  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="text-xs text-muted underline-offset-4 transition-colors hover:text-ink hover:underline"
      >
        Настройки списка{dirty ? " · фильтры включены" : ""}
      </button>

      {open ? (
        <div className="card mt-2 space-y-4 p-4">
          <div>
            <div className="flex items-baseline justify-between gap-3">
              <label htmlFor="cap" className="text-sm font-medium">
                Сколько студентов вы успеете принять
              </label>
              <span className="nums text-base font-semibold">{capacity}</span>
            </div>
            <div className="mt-2">
              <Slider
                id="cap"
                label="Ёмкость списка"
                min={5}
                max={max}
                value={Math.min(capacity, max)}
                onChange={onCapacity}
              />
              <div className="mt-1 flex justify-between text-2xs text-faint">
                <span>5</span><span>{max}</span>
              </div>
            </div>
            <p className="mt-2 text-xs text-muted">
              Столько человек попадёт в список. Порог вероятности при этом —{" "}
              <span className="nums">{threshold.toFixed(2)}</span>.
              {m.precision !== null ? (
                <>
                  {" "}Из них действительно в зоне риска — {pct(m.precision)}
                  {" "}({m.caught} из {capacity}).
                </>
              ) : null}
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label htmlFor="q" className="mb-1 block text-xs text-muted">Поиск</label>
              <input
                id="q"
                value={filters.query}
                placeholder="номер студента"
                onChange={(e) => onFilters({ ...filters, query: e.target.value })}
                className="h-9 w-full rounded-lg border border-line bg-surface px-3 text-sm"
              />
            </div>
            {([
              ["Школа", "school", [["all", "любая"], ["GP", "GP"], ["MS", "MS"]]],
              ["Пол", "sex", [["all", "любой"], ["F", "женский"], ["M", "мужской"]]],
              ["Город/село", "address", [["all", "любой"], ["U", "город"], ["R", "село"]]],
            ] as const).map(([label, key, opts]) => (
              <div key={key}>
                <label htmlFor={key} className="mb-1 block text-xs text-muted">{label}</label>
                <Select
                  id={key}
                  value={filters[key]}
                  onChange={(e) => onFilters({ ...filters, [key]: e.target.value })}
                >
                  {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </Select>
              </div>
            ))}
          </div>

          {dirty ? (
            <div className="flex items-center gap-3">
              <Button size="sm" onClick={() => onFilters(EMPTY_FILTERS)}>
                Сбросить фильтры
              </Button>
              <span className="text-xs text-muted">
                показано {total} студентов
              </span>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
