import * as React from "react";
import type { BatchResult, Row, Schema } from "@/lib/types";
import { RequestError, api } from "@/lib/api";
import { thresholdForK } from "@/lib/utils";
import { Button, Card, ErrorState, Skeleton } from "@/components/ui";
import { Dropzone } from "@/components/Dropzone";
import { CalmBlock, Shortlist } from "@/components/Shortlist";
import { EMPTY_FILTERS, type Filters, ListSettings } from "@/components/ListSettings";
import { StudentCard } from "@/components/StudentCard";
import { VerificationStrip } from "@/components/VerificationStrip";

/**
 * Главный экран. Отвечает ровно на один вопрос — «с кем поговорить».
 *
 * Всё аналитическое сознательно вынесено: метрики и распределение — на
 * страницу «Как работает модель», ёмкость и фильтры — в «Настройки списка»,
 * подробности по студенту — в карточку по клику.
 */
export function CohortPage({ schema }: { schema: Schema | null }) {
  const [result, setResult] = React.useState<BatchResult | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState<RequestError | null>(null);
  const [capacity, setCapacity] = React.useState(40);
  const [filters, setFilters] = React.useState<Filters>(EMPTY_FILTERS);
  const [open, setOpen] = React.useState<Row | null>(null);

  const run = async (fn: () => Promise<BatchResult>) => {
    setBusy(true);
    setErr(null);
    try {
      const r = await fn();
      setResult(r);
      setCapacity(Math.min(r.capacity, r.rows.length));
      setFilters(EMPTY_FILTERS);
    } catch (e) {
      setErr(e as RequestError);
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const sorted = React.useMemo(
    () => (result ? [...result.rows].sort((a, b) => b.probability - a.probability) : []),
    [result],
  );

  const filtered = React.useMemo(() => {
    const q = filters.query.trim().toLowerCase();
    return sorted.filter((r) => {
      if (filters.school !== "all" && String(r.features.school) !== filters.school) return false;
      if (filters.sex !== "all" && String(r.features.sex) !== filters.sex) return false;
      if (filters.address !== "all" && String(r.features.address) !== filters.address) return false;
      return !q || r.id.toLowerCase().includes(q);
    });
  }, [sorted, filters]);

  const threshold = React.useMemo(
    () => (sorted.length ? thresholdForK(sorted, capacity) : 0.5),
    [sorted, capacity],
  );

  const attention = filtered.slice(0, capacity);
  const calm = filtered.slice(capacity);

  const forExport = React.useMemo(
    () => sorted.map((r, i) => ({ ...r, in_shortlist: i < capacity })),
    [sorted, capacity],
  );

  /* ---------------- состояние 1: до загрузки ---------------- */
  if (!result && !busy) {
    return (
      <div className="mx-auto flex min-h-[calc(100dvh-9rem)] max-w-xl flex-col justify-center py-6">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Кто не получит зачёт
        </h1>
        <p className="mt-2 text-sm text-muted">
          Загрузите выгрузку по группе — покажем, с кем стоит поговорить
          в первую очередь, пока до конца семестра ещё есть время.
        </p>
        <p className="mt-2 text-xs text-faint">
          Нужен CSV или Excel со стандартной выгрузкой по студентам. Файл
          обрабатывается в памяти и никуда не сохраняется. Если файла нет —
          нажмите «Посмотреть на демо-данных».
        </p>

        <div className="mt-6">
          <Dropzone
            busy={busy}
            onFile={(f) => run(() => api.batch(f))}
            onDemo={() => run(() => api.demo())}
          />
        </div>

        {err ? (
          <div className="mt-4">
            <ErrorState title={err.message} hint={err.hint} onRetry={() => setErr(null)} />
          </div>
        ) : null}
      </div>
    );
  }

  /* ---------------- загрузка ---------------- */
  if (busy) {
    return (
      <div className="mx-auto max-w-3xl space-y-4 py-6" aria-busy="true" aria-live="polite">
        <span className="sr-only">Считаем риск…</span>
        <Skeleton className="h-12 w-2/3" />
        <Skeleton className="h-4 w-1/3" />
        <div className="space-y-2 pt-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      </div>
    );
  }

  /* ---------------- состояние 2: результат ---------------- */
  return (
    <div className="mx-auto max-w-3xl space-y-5 py-2">
      <header>
        <h1 className="text-2xl font-semibold leading-snug tracking-tight sm:text-3xl">
          Из {result!.summary.n_rows} студентов внимания требуют{" "}
          <span className="text-risk-high">{attention.length}</span>
        </h1>
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
          <button
            onClick={() => {
              setResult(null);
              setErr(null);
            }}
            className="text-xs text-muted underline-offset-4 transition-colors hover:text-ink hover:underline"
          >
            Загрузить другой файл
          </button>
          <ListSettings
            rows={sorted}
            capacity={capacity}
            onCapacity={setCapacity}
            threshold={threshold}
            filters={filters}
            onFilters={setFilters}
            total={filtered.length}
          />
        </div>
      </header>

      <section aria-labelledby="attn">
        <div className="mb-1 flex items-baseline justify-between gap-3">
          <h2 id="attn" className="text-sm font-semibold">Стоит поговорить</h2>
          <Button size="sm" onClick={() => api.export(forExport, "csv")}>
            Скачать список
          </Button>
        </div>
        <Card className="px-1 py-1">
          <Shortlist rows={attention} schema={schema} onOpen={setOpen} />
        </Card>
      </section>

      <CalmBlock rows={calm} schema={schema} onOpen={setOpen} />

      {result!.verification ? <VerificationStrip v={result!.verification} /> : null}

      <StudentCard
        row={open}
        schema={schema}
        open={open !== null}
        onClose={() => setOpen(null)}
        level={result!.level}
      />
    </div>
  );
}
