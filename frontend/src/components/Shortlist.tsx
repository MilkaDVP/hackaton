import * as React from "react";
import type { Row, Schema } from "@/lib/types";
import { BAND_LABEL, cn, pct } from "@/lib/utils";
import { Button } from "./ui";

/**
 * Главный ответ куратору: кто требует внимания.
 *
 * Намеренно НЕ таблица и намеренно без пояснения под номером. В строке ровно
 * три вещи: кто, насколько и словом. Причина риска — это разговор на минуту,
 * её место в карточке студента, а не в списке: в строке она превращается
 * в ярлык («планирует учиться дальше: Нет»), который ничего не решает,
 * а прочитать сорок таких подряд всё равно невозможно.
 */

function RiskDot({ band }: { band: Row["risk_band"] }) {
  // Риск дублируется словом рядом — цвет никогда не единственный носитель смысла
  return (
    <span
      aria-hidden
      className={cn(
        "inline-block h-2 w-2 shrink-0 rounded-full",
        band === "high" ? "bg-risk-high"
          : band === "medium" ? "bg-risk-mid" : "bg-risk-low",
      )}
    />
  );
}

export function ShortlistItem({
  row, onOpen,
}: {
  row: Row;
  onOpen: (r: Row) => void;
}) {
  return (
    <li>
      <button
        onClick={() => onOpen(row)}
        className={cn(
          "group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left",
          "transition-colors hover:bg-raised focus-visible:bg-raised sm:gap-4 sm:px-4",
        )}
        aria-label={`Студент ${row.id}, риск ${BAND_LABEL[row.risk_band]} ${pct(row.probability)}, подробнее`}
      >
        <RiskDot band={row.risk_band} />

        <span className="min-w-0 flex-1 text-sm font-medium">{row.id}</span>

        <span className="shrink-0 text-xs text-faint">
          {BAND_LABEL[row.risk_band]}
        </span>

        <span className="nums w-12 shrink-0 text-right text-sm font-semibold">
          {pct(row.probability)}
        </span>

        <span
          aria-hidden
          className="shrink-0 text-faint opacity-0 transition-opacity group-hover:opacity-100"
        >
          →
        </span>
      </button>
    </li>
  );
}

export function Shortlist({
  rows, onOpen,
}: {
  rows: Row[];
  schema?: Schema | null;
  onOpen: (r: Row) => void;
}) {
  const [limit, setLimit] = React.useState(40);
  const shown = rows.slice(0, limit);

  return (
    <div>
      <ul className="divide-y divide-line">
        {shown.map((r) => (
          <ShortlistItem key={r.id} row={r} onOpen={onOpen} />
        ))}
      </ul>
      {rows.length > limit ? (
        <div className="pt-3">
          <Button size="sm" onClick={() => setLimit((n) => n + 40)}>
            Показать ещё {Math.min(40, rows.length - limit)}
          </Button>
        </div>
      ) : null}
    </div>
  );
}

/** «Всё в порядке» — свёрнуто. Куратору важно, что их посчитали, а не кто они. */
export function CalmBlock({
  rows, onOpen,
}: {
  rows: Row[];
  schema?: Schema | null;
  onOpen: (r: Row) => void;
}) {
  const [open, setOpen] = React.useState(false);
  const [limit, setLimit] = React.useState(50);

  if (!rows.length) return null;

  return (
    <section className="card overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-raised"
      >
        <span className="text-sm font-medium">Всё в порядке</span>
        <span className="nums text-sm text-muted">{rows.length}</span>
        <span className="ml-auto text-xs text-faint">
          {open ? "свернуть" : "показать"}
        </span>
      </button>

      {open ? (
        <div className="border-t border-line px-1 pb-3">
          <ul className="divide-y divide-line">
            {rows.slice(0, limit).map((r) => (
              <ShortlistItem key={r.id} row={r} onOpen={onOpen} />
            ))}
          </ul>
          {rows.length > limit ? (
            <div className="px-3 pt-3">
              <Button size="sm" onClick={() => setLimit((n) => n + 100)}>
                Показать ещё
              </Button>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
