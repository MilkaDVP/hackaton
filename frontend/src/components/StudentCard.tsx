import type { Row, Schema } from "@/lib/types";
import { BAND_LABEL, BAND_STYLE, cn, decode, pct } from "@/lib/utils";
import { Badge, Card, Drawer } from "./ui";
import { FactorList } from "./FactorList";

/** Все признаки студента, сгруппированные по смыслу и расшифрованные. */
export function StudentCard({
  row, schema, open, onClose, level,
}: {
  row: Row | null;
  schema: Schema | null;
  open: boolean;
  onClose: () => void;
  level: string;
}) {
  if (!row) return null;
  const band = row.risk_band;

  const groups = (schema?.groups ?? []).filter((g) =>
    g.id !== "grades" || level === "L1",
  );

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-wide text-faint">
            Студент {row.id} · место {row.rank} в списке
          </p>
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="nums text-2xl font-semibold">
              {pct(row.probability, 1)}
            </span>
            <Badge className={BAND_STYLE[band]}>
              {BAND_LABEL[band]} риск
            </Badge>
            {row.in_shortlist ? (
              <Badge className="border-accent/30 bg-accent-soft text-accent">
                в списке на разговор
              </Badge>
            ) : null}
          </div>
        </div>
      }
    >
      <div className="space-y-6">
        <p className="rounded-lg bg-raised px-3 py-2 text-xs text-muted">
          Это оценка вероятности, а не приговор и не диагноз. Повод поговорить
          и разобраться, а не основание для решений о студенте.
        </p>

        {row.actual ? (
          <Card className="px-4 py-3">
            <p className="text-xs font-medium uppercase tracking-wide text-faint">
              Фактический итог (был в файле)
            </p>
            <p className="mt-1 text-sm">
              Итоговый балл <span className="nums font-semibold">{row.actual.G3}</span>{" "}
              —{" "}
              <span
                className={cn(
                  "font-medium",
                  row.actual.no_pass ? "text-risk-high" : "text-risk-low",
                )}
              >
                {row.actual.no_pass ? "незачёт" : "зачёт"}
              </span>
            </p>
          </Card>
        ) : null}

        <section aria-labelledby="factors-h">
          <h3 id="factors-h" className="mb-2 text-sm font-semibold">
            Что сильнее всего повлияло
          </h3>
          <FactorList factors={row.top_factors} />
        </section>

        {groups.map((g) => {
          const items = (schema?.features ?? []).filter(
            (f) => f.group === g.id && f.name in row.features,
          );
          if (!items.length) return null;
          return (
            <section key={g.id} aria-labelledby={`g-${g.id}`}>
              <h3 id={`g-${g.id}`} className="mb-2 text-sm font-semibold">
                {g.label}
              </h3>
              <dl className="overflow-hidden rounded-lg border border-line">
                {items.map((f, i) => (
                  <div
                    key={f.name}
                    className={cn(
                      "flex items-baseline justify-between gap-4 px-3 py-2 text-sm",
                      i % 2 ? "bg-surface" : "bg-raised/50",
                    )}
                  >
                    <dt className="text-muted">{f.label}</dt>
                    <dd className="nums text-right font-medium">
                      {decode(schema, f.name, row.features[f.name] ?? null)}
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          );
        })}
      </div>
    </Drawer>
  );
}
