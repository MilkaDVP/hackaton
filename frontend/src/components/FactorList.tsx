import type { Factor } from "@/lib/types";
import { cn } from "@/lib/utils";
import { EmptyState } from "./ui";

/**
 * Персональные факторы. Формулировки человеческие, без «shap +0.31».
 * У признака `school` показывается отдельная оговорка: он предсказывает,
 * но не является рычагом для куратора.
 */
export function FactorList({ factors }: { factors: Factor[] }) {
  if (!factors?.length) {
    return (
      <p className="rounded-lg bg-raised px-3 py-2 text-xs text-muted">
        Для этой строки объяснение не рассчитано.
      </p>
    );
  }

  // Все факторы слабые — значит, ответы близки к типичным по потоку.
  // Честнее сказать это прямо, чем рисовать полоски около нуля.
  const allWeak = factors.every((f) => f.weak);
  const max = Math.max(...factors.map((f) => Math.abs(f.effect)), 0.0001);

  return (
    <ul className="space-y-2">
      {allWeak ? (
        <li className="rounded-lg border border-line bg-raised/60 px-3 py-2 text-xs text-muted">
          Сильно выделяющихся факторов нет: ответы близки к типичным по потоку.
          Ниже — то, что всё же сдвигает оценку, но влияние каждого мало.
        </li>
      ) : null}
      {factors.map((f) => {
        const up = f.direction === "up";
        const width = `${Math.max(6, (Math.abs(f.effect) / max) * 100)}%`;
        return (
          <li
            key={f.feature}
            className="rounded-lg border border-line bg-surface px-3 py-2"
          >
            <div className="flex items-baseline justify-between gap-3">
              <p className="text-sm">
                <span className="font-medium">{f.label}</span>
                <span className="text-muted"> — {f.value_label}</span>
              </p>
              <span
                className={cn(
                  "shrink-0 text-2xs font-semibold uppercase tracking-wide",
                  up ? "text-risk-high" : "text-risk-low",
                )}
              >
                {up ? "повышает риск" : "снижает риск"}
              </span>
            </div>

            <div
              className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-raised"
              role="img"
              aria-label={`Сила влияния ${(Math.abs(f.effect) * 100).toFixed(1)} процентных пункта`}
            >
              <div
                className={cn("h-full rounded-full", up ? "bg-risk-high" : "bg-risk-low")}
                style={{ width }}
              />
            </div>

            {f.unusual ? (
              <p className="mt-1.5 text-2xs text-risk-mid">
                У этого студента признак влияет не в ту сторону, что обычно
                ({f.note || "по выборке в целом"}). Модель нелинейна, и на
                отдельном сочетании ответов такое бывает — повод отнестись
                к этому фактору осторожно.
              </p>
            ) : f.not_a_lever ? (
              <p className="mt-1.5 text-2xs text-faint">
                Это не рычаг: внутри одной школы признак одинаков у всех и не
                помогает выбрать, с кем говорить.
              </p>
            ) : null}
          </li>
        );
      })}
      <li className="pt-1 text-2xs text-faint">
        Это статистическая связь, а не причина. Признак — повод спросить,
        а не инструкция, что чинить.
      </li>
    </ul>
  );
}

export function FactorEmpty() {
  return <EmptyState title="Нет данных" />;
}
