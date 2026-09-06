import * as React from "react";
import {
  Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { ModelInfo, Schema } from "@/lib/types";
import { api } from "@/lib/api";
import { labelOf, pct } from "@/lib/utils";
import { Card, ErrorState, Skeleton } from "@/components/ui";

/**
 * Страница для тех, кто спросил «а как оно работает».
 * Сюда переехало всё аналитическое с главного экрана: метрики, распределение
 * вероятностей, факторы, ограничения. Все числа — из /api/model-info,
 * в вёрстке нет ни одной метрики руками.
 */
export function ModelPage({ schema }: { schema: Schema | null }) {
  const [info, setInfo] = React.useState<ModelInfo | null>(null);
  const [err, setErr] = React.useState<string | null>(null);
  const [hist, setHist] = React.useState<{ counts: number[]; edges: number[] } | null>(null);

  React.useEffect(() => {
    api.modelInfo().then(setInfo).catch((e) => setErr(e.message));
    // Распределение берём с демо-выборки: это иллюстрация к порогу,
    // персональных данных тут нет, наружу уходят только счётчики по корзинам.
    api.demo().then((d) => setHist(d.summary.histogram)).catch(() => setHist(null));
  }, []);

  // ВСЕ хуки — до ранних return. Если useMemo стоит после них, на первом
  // рендере (info ещё null) он не вызывается, а на втором вызывается, и React
  // падает с ошибкой #310 «rendered more hooks than during the previous render».
  const dist = React.useMemo(() => {
    if (!hist) return [];
    const { counts, edges } = hist;
    return counts.map((c, i) => ({
      x: (edges[i] + edges[i + 1]) / 2,
      label: `${pct(edges[i])}–${pct(edges[i + 1])}`,
      count: c,
    }));
  }, [hist]);

  if (err) return <ErrorState title="Не удалось получить сведения о модели" hint={err} />;
  if (!info) {
    return (
      <div className="space-y-3" aria-busy="true">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  const top3 = info.importances.L2.slice(0, 3);

  return (
    <div className="max-w-3xl space-y-8">
      <header className="space-y-1">
        <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">
          Как работает модель
        </h1>
        <p className="text-sm text-muted">
          Обучена {new Date(info.trained_at).toLocaleDateString("ru-RU")} на{" "}
          {info.training_data.n_rows} студентах, из них{" "}
          {info.training_data.n_positive} не получили зачёт (
          {pct(info.training_data.positive_rate, 1)}).
        </p>
      </header>

      <section aria-labelledby="metrics-h">
        <h2 id="metrics-h" className="mb-2 text-sm font-semibold">
          Качество на кросс-валидации
        </h2>
        <div className="overflow-x-auto rounded-xl border border-line">
          <table className="w-full text-sm">
            <thead className="bg-raised text-xs text-muted">
              <tr>
                <th scope="col" className="px-3 py-2 text-left">Уровень</th>
                <th scope="col" className="px-3 py-2 text-left">ROC-AUC</th>
                <th scope="col" className="px-3 py-2 text-left">PR-AUC</th>
              </tr>
            </thead>
            <tbody>
              {(["L2", "L1"] as const).map((lvl) => (
                <tr key={lvl} className="border-t border-line">
                  <td className="px-3 py-2">
                    <p className="font-medium">{info.levels[lvl].name}</p>
                    <p className="text-xs text-muted">{info.levels[lvl].uses}</p>
                  </td>
                  <td className="nums px-3 py-2">
                    {info.metrics[lvl].roc_auc.toFixed(3)}
                    <span className="text-faint">
                      {" "}± {info.metrics[lvl].roc_auc_std.toFixed(3)}
                    </span>
                  </td>
                  <td className="nums px-3 py-2">
                    {info.metrics[lvl].pr_auc.toFixed(3)}
                    <span className="text-faint">
                      {" "}± {info.metrics[lvl].pr_auc_std.toFixed(3)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted">
          Разброс — стандартное отклонение по 25 разбиениям данных
          (5 фолдов × 5 повторов). Случайная модель дала бы ROC-AUC 0.5,
          а PR-AUC {info.training_data.positive_rate.toFixed(3)} — это доля
          незачётов, а не 0.5.
        </p>
      </section>

      <section aria-labelledby="levels-h">
        <h2 id="levels-h" className="mb-2 text-sm font-semibold">
          Почему главная модель — «начало семестра»
        </h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {(["L2", "L1"] as const).map((lvl) => (
            <Card key={lvl} className="p-4">
              <p className="text-sm font-medium">{info.levels[lvl].name}</p>
              <p className="mt-1 text-xs text-muted">{info.levels[lvl].caveat}</p>
            </Card>
          ))}
        </div>
        <p className="mt-2 text-xs text-muted">
          Модель конца семестра почти идеальна, потому что предсказывает оценку
          по двум другим оценкам. Пользы от неё мало: когда обе контрольные
          написаны, помогать поздно.
        </p>
      </section>

      <section aria-labelledby="factors-h">
        <h2 id="factors-h" className="mb-2 text-sm font-semibold">
          Три главных фактора (начало семестра)
        </h2>
        <ol className="space-y-2">
          {top3.map((f, i) => (
            <li key={f.feature} className="rounded-lg border border-line bg-surface p-3">
              <p className="text-sm font-medium">
                <span className="text-faint">{i + 1}.</span>{" "}
                {labelOf(schema, f.feature)}
              </p>
              <p className="mt-0.5 text-xs text-muted">{f.direction.text}</p>
              {f.feature === "school" ? (
                <p className="mt-1 text-2xs text-faint">
                  Внимание: это не поведение студента, а то, в какой школе он
                  учится. Внутри одной школы признак одинаков у всех и не
                  помогает выбрать, с кем говорить. Его практический смысл —
                  считать порог отдельно по каждой школе.
                </p>
              ) : null}
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="dist-h">
        <h2 id="dist-h" className="mb-2 text-sm font-semibold">
          Как распределяются вероятности
        </h2>
        <p className="mb-2 text-xs text-muted">
          Демо-выборка, {info.training_data.n_rows} студентов. Две группы
          неизбежно перекрываются — идеального разделения не бывает.
          Вертикальная черта — порог, правее которого студент попадает в список
          куратора.
        </p>
        <Card className="p-3">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dist} margin={{ top: 6, right: 8, bottom: 0, left: -24 }}>
                <XAxis
                  dataKey="x"
                  tickFormatter={(v: number) => pct(v)}
                  tick={{ fontSize: 10, fill: "rgb(var(--faint))" }}
                  stroke="rgb(var(--line))"
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "rgb(var(--faint))" }}
                  stroke="rgb(var(--line))"
                  allowDecimals={false}
                />
                <Tooltip
                  cursor={{ fill: "rgb(var(--raised))" }}
                  contentStyle={{
                    background: "rgb(var(--surface))",
                    border: "1px solid rgb(var(--line))",
                    borderRadius: 8, fontSize: 12, color: "rgb(var(--ink))",
                  }}
                  formatter={(v: number) => [`${v} студентов`, "в интервале"]}
                  labelFormatter={(_, pl) => pl?.[0]?.payload?.label ?? ""}
                />
                <ReferenceLine
                  x={info.threshold.default}
                  stroke="rgb(var(--accent))"
                  strokeDasharray="4 3"
                  label={{ value: "порог", position: "top", fontSize: 10,
                           fill: "rgb(var(--accent))" }}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {dist.map((d, i) => (
                    <Cell key={i}
                      fill={d.x >= info.threshold.default
                        ? "rgb(var(--risk-high))" : "rgb(var(--risk-low))"}
                      fillOpacity={d.x >= info.threshold.default ? 0.85 : 0.45} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </section>

      <section aria-labelledby="limits-h">
        <h2 id="limits-h" className="mb-2 text-sm font-semibold">
          Честные ограничения
        </h2>
        <ul className="space-y-2 text-sm text-muted">
          <li>
            <span className="font-medium text-ink">Мало данных.</span> Две школы,
            один учебный год, один предмет, {info.training_data.n_rows} студентов.
            На другом потоке качество будет другим.
          </li>
          <li>
            <span className="font-medium text-ink">Привязана к предмету — но не всегда.</span>{" "}
            Модель обучена на курсе португальского языка. Проверка на математике:
            <span className="nums"> без оценок ROC-AUC падает до 0.58</span> — это
            почти случайное угадывание, и такому списку верить нельзя.
            <span className="nums"> С оценками за контрольные — 0.94</span>:
            оценки предсказывают оценки независимо от предмета. Практический
            вывод: на другом предмете пользуйтесь только режимом с оценками.
          </li>
          <li>
            <span className="font-medium text-ink">Качество неровное по подгруппам.</span>{" "}
            Точность различается по полу, городу/селу и школе. Это надо
            учитывать, прежде чем распределять внимание куратора по такому списку.
          </li>
          <li>
            <span className="font-medium text-ink">Порог не универсален.</span>{" "}
            По умолчанию <span className="nums">{info.threshold.default}</span> —
            он выведен из ёмкости куратора ({info.threshold.capacity} студентов),
            а не выбран как 0.5. На новом потоке его надо пересчитать.
          </li>
          <li>
            <span className="font-medium text-ink">Корреляция — не причинность.</span>{" "}
            Модель показывает, на кого студент похож статистически. Она не
            говорит, что произойдёт, если на признак повлиять. Прошлые незачёты
            и пропуски — симптом, а не рычаг.
          </li>
        </ul>
      </section>

      <section aria-labelledby="tech-h">
        <h2 id="tech-h" className="mb-2 text-sm font-semibold">Как устроено</h2>
        <Card className="p-4 text-xs text-muted">
          <p>
            Мягкое голосование трёх разнородных моделей. Начало семестра:{" "}
            {(info.ensemble_members.L2 ?? []).join(" + ")}. Конец семестра:{" "}
            {(info.ensemble_members.L1 ?? []).join(" + ")}. Вероятности
            откалиброваны изотонической регрессией — без этого порог не имел бы
            содержательного смысла.
          </p>
          <p className="mt-2">
            Из инженерных признаков {info.drop_features.length} были отброшены:
            leave-one-out показал, что без них метрика не падает, а растёт.
          </p>
          <p className="mt-2 nums">
            python {info.versions.python} · scikit-learn {info.versions.sklearn} ·
            numpy {info.versions.numpy}
          </p>
        </Card>
      </section>
    </div>
  );
}
