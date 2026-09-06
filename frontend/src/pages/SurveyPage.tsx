import * as React from "react";
import type { FeatureDef, Schema, SurveyResult } from "@/lib/types";
import { RequestError, api } from "@/lib/api";
import { BAND_BAR, BAND_LABEL, BAND_STYLE, cn, pct } from "@/lib/utils";
import { Badge, Button, Card, ErrorState, Field, Input, Select, Slider } from "@/components/ui";
import { FactorList } from "@/components/FactorList";

export function SurveyPage({ schema }: { schema: Schema | null }) {
  const [step, setStep] = React.useState(0);
  const [answers, setAnswers] = React.useState<Record<string, string | number>>({});
  const [result, setResult] = React.useState<SurveyResult | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState<RequestError | null>(null);

  // Заполняем значениями по умолчанию только то, что реально спрашиваем.
  // Остальное подставит бэкенд типичным значением потока.
  React.useEffect(() => {
    if (!schema) return;
    const d: Record<string, string | number> = {};
    for (const f of schema.features) {
      if (schema.survey_core.includes(f.name)) d[f.name] = f.default;
    }
    setAnswers(d);
  }, [schema]);

  if (!schema) return <SurveySkeleton />;

  const steps = schema.survey_steps_short;
  const byName = new Map(schema.features.map((f) => [f.name, f]));

  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      const payload = Object.fromEntries(
        Object.entries(answers).filter(([, v]) => v !== "" && v !== undefined),
      );
      setResult(await api.single(payload));
    } catch (e) {
      setErr(e as RequestError);
    } finally {
      setBusy(false);
    }
  };

  if (result) {
    return (
      <ResultView
        result={result}
        onAgain={() => {
          setResult(null);
          setStep(0);
        }}
      />
    );
  }

  const current = steps[step];
  const last = step === steps.length - 1;

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <header className="space-y-2">
        <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">
          Проверить себя
        </h1>
        <p className="text-sm text-muted">
          Тот же расчёт, что видит куратор в своём списке, только для вас одного
          и без чьего-либо участия. Ответьте на вопросы об учёбе — покажем,
          насколько велик риск не получить зачёт и что на это влияет.
        </p>
        <p className="text-xs text-faint">
          Модель обучена на курсе «{schema.subject}» — отвечайте про него.
          Без баллов за контрольные оценка привязана именно к этому предмету:
          на другом она почти не работает. Если баллы указать, расчёт
          переносится и на остальные предметы.
        </p>
        <p className="text-xs text-faint">
          {steps.length} коротких шага, около минуты. Ответы никуда не
          отправляются и не сохраняются — результат видите только вы.
        </p>
      </header>

      <div>
        <div className="flex items-center gap-2" role="list">
          {steps.map((s, i) => (
            <div key={s.id} role="listitem" className="flex flex-1 items-center gap-2">
              <div
                className={cn(
                  "h-1 flex-1 rounded-full transition-colors",
                  i <= step ? "bg-accent" : "bg-line",
                )}
              />
              <span
                className={cn(
                  "shrink-0 text-2xs",
                  i === step ? "font-semibold text-ink" : "text-faint",
                )}
              >
                {s.title}
              </span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-xs text-muted" aria-live="polite">
          Шаг {step + 1} из {steps.length}
        </p>
      </div>

      {err ? <ErrorState title={err.message} hint={err.hint} /> : null}

      <Card className="space-y-5 p-5">
        {current.optional ? (
          <p className="rounded-lg bg-raised px-3 py-2 text-xs text-muted">
            За год по предмету пишут две промежуточные контрольные работы. Если
            они уже были — укажите баллы, и оценка станет заметно точнее. Если
            ещё не было или не помните — пропустите и нажмите «Узнать результат».
          </p>
        ) : null}
        {current.features.map((name) => {
          const f = byName.get(name);
          if (!f) return null;
          return (
            <QuestionField
              key={name}
              f={f}
              optional={current.optional}
              value={answers[name]}
              onChange={(v) => setAnswers((a) => ({ ...a, [name]: v }))}
            />
          );
        })}
      </Card>

      <div className="flex items-center justify-between gap-3">
        <Button
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0 || busy}
        >
          Назад
        </Button>
        {last ? (
          <Button variant="primary" onClick={submit} disabled={busy}>
            {busy ? "Считаем…" : "Узнать результат"}
          </Button>
        ) : (
          <Button variant="primary" onClick={() => setStep((s) => s + 1)}>
            Дальше
          </Button>
        )}
      </div>

      <p className="text-2xs text-faint">
        Ответы обрабатываются в памяти, никуда не сохраняются и никому не отправляются.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------------ */
function QuestionField({
  f, value, onChange, optional,
}: {
  f: FeatureDef;
  value: string | number | undefined;
  onChange: (v: string | number) => void;
  optional?: boolean;
}) {
  const id = `q-${f.name}`;
  const opts = Object.entries(f.values ?? {});
  // Анкету заполняет сам студент — обращаемся на «вы»
  const label = f.self_label || f.label;
  const hint = f.self_hint ?? f.hint;

  // Да/нет — радиогруппа
  if (opts.length === 2 && "yes" in (f.values ?? {})) {
    return (
      <fieldset>
        <legend className="mb-1.5 text-sm font-medium">{label}</legend>
        <div className="flex gap-2">
          {opts.map(([v, l]) => (
            <label
              key={v}
              className={cn(
                "flex-1 cursor-pointer rounded-lg border px-3 py-2 text-center text-sm transition-colors",
                String(value) === v
                  ? "border-accent bg-accent-soft text-accent"
                  : "border-line bg-surface hover:bg-raised",
              )}
            >
              <input
                type="radio"
                name={f.name}
                value={v}
                checked={String(value) === v}
                onChange={() => onChange(v)}
                className="sr-only"
              />
              {l}
            </label>
          ))}
        </div>
        {hint ? <p className="mt-1 text-xs text-faint">{hint}</p> : null}
      </fieldset>
    );
  }

  // Шкала 1–5 — слайдер с текстовой подписью текущего значения
  if (f.kind === "scale") {
    const cur = Number(value ?? f.default);
    return (
      <Field label={label} hint={hint} htmlFor={id}>
        <Slider
          id={id}
          label={label}
          min={f.min ?? 1}
          max={f.max ?? 5}
          value={cur}
          onChange={onChange}
        />
        <div className="mt-1 flex justify-between text-2xs text-faint">
          <span>{f.values?.["1"] ?? f.min}</span>
          <span className="font-medium text-muted">{f.values?.[String(cur)] ?? cur}</span>
          <span>{f.values?.[String(f.max)] ?? f.max}</span>
        </div>
      </Field>
    );
  }

  // Перечисление — селект с человеческими подписями
  if (opts.length) {
    return (
      <Field label={label} hint={hint} htmlFor={id}>
        <Select
          id={id}
          value={String(value ?? f.default)}
          onChange={(e) => {
            const raw = e.target.value;
            const num = Number(raw);
            onChange(Number.isNaN(num) || raw === "" ? raw : num);
          }}
        >
          {opts.map(([v, l]) => (
            <option key={v} value={v}>{l}</option>
          ))}
        </Select>
      </Field>
    );
  }

  // Число
  return (
    <Field label={label} hint={hint} htmlFor={id}>
      <Input
        id={id}
        type="number"
        min={f.min}
        max={f.max}
        placeholder={optional ? "не знаю / ещё не было" : undefined}
        value={value === undefined ? (optional ? "" : String(f.default)) : String(value)}
        onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
      />
    </Field>
  );
}

/* ------------------------------------------------------------------------ */
function ResultView({
  result, onAgain,
}: {
  result: SurveyResult;
  onAgain: () => void;
}) {
  const p = result.probability;
  const outOf100 = Math.round(p * 100);

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <Card className="p-6">
        <p className="text-xs uppercase tracking-wide text-faint">
          Оценка вероятности
        </p>
        <div className="mt-2 flex flex-wrap items-baseline gap-3">
          <span className="nums text-4xl font-semibold tracking-tight">
            {pct(p)}
          </span>
          <Badge className={BAND_STYLE[result.risk_band]}>
            {BAND_LABEL[result.risk_band]} риск
          </Badge>
        </div>
        <div className="mt-4 h-2 overflow-hidden rounded-full bg-raised">
          <div
            className={cn("h-full rounded-full transition-all", BAND_BAR[result.risk_band])}
            style={{ width: `${Math.max(2, p * 100)}%` }}
          />
        </div>
        <p className="mt-4 text-sm text-muted">
          Из ста студентов, ответивших примерно так же, не сдают около{" "}
          <span className="nums font-semibold text-ink">{outOf100}</span>.
          Это <em>не</em> означает, что не сдадите именно вы.
        </p>
        <p className="mt-3 border-t border-line pt-3 text-xs text-faint">
          {result.level_note}
          {result.level === "L2" ? (
            <>
              {" "}Если укажете баллы за контрольные, оценка станет точнее.
            </>
          ) : null}
        </p>
      </Card>

      <section>
        <h2 className="mb-2 text-sm font-semibold">Что повлияло на оценку</h2>
        <FactorList factors={result.top_factors} />
      </section>

      {result.comparison.length ? (
        <section>
          <h2 className="mb-2 text-sm font-semibold">Сравнение с потоком</h2>
          <Card className="divide-y divide-line">
            {result.comparison.map((c) => (
              <div
                key={c.feature}
                className="flex items-baseline justify-between gap-4 px-4 py-2.5 text-sm"
              >
                <span className="text-muted">{c.label}</span>
                <span className="nums text-right">
                  <span className="font-medium">{c.you_label}</span>
                  <span className="text-faint"> · обычно {c.typical_label}</span>
                </span>
              </div>
            ))}
          </Card>
        </section>
      ) : null}

      {/* Этика — обязательная часть экрана, а не мелкий шрифт внизу */}
      <section
        aria-labelledby="ethics-h"
        className="rounded-xl border border-accent/25 bg-accent-soft/40 p-5"
      >
        <h2 id="ethics-h" className="text-sm font-semibold">
          Что это значит и чего это не значит
        </h2>
        <ul className="mt-2 space-y-2 text-sm text-muted">
          <li>
            <span className="font-medium text-ink">Это не оценка и не диагноз.</span>{" "}
            {result.disclaimer.not_a_decision}
          </li>
          <li>
            <span className="font-medium text-ink">Это вероятность, а не судьба.</span>{" "}
            {result.disclaimer.not_a_verdict}
          </li>
          <li>
            <span className="font-medium text-ink">Модель знает мало.</span>{" "}
            {result.disclaimer.data} Она видит анкету, а не вас.
          </li>
          <li>
            <span className="font-medium text-ink">Связь — не причина.</span>{" "}
            Факторы выше показывают, на кого вы похожи статистически. Они не
            говорят, что нужно «починить».
          </li>
          <li>
            <span className="font-medium text-ink">Что полезно сделать.</span>{" "}
            {result.disclaimer.action}
          </li>
        </ul>
        <p className="mt-3 text-2xs text-faint">{result.disclaimer.privacy}</p>
      </section>

      <Button onClick={onAgain}>Пройти заново</Button>
    </div>
  );
}

function SurveySkeleton() {
  return (
    <div className="mx-auto max-w-2xl space-y-4" aria-busy="true">
      <div className="skeleton h-8 w-48" />
      <div className="skeleton h-4 w-full max-w-md" />
      <div className="skeleton h-72 w-full" />
    </div>
  );
}
