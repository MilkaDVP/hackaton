import * as React from "react";
import type { Verification } from "@/lib/types";

/**
 * Режим проверки — неприметная плашка внизу, а не блок на пол-экрана.
 * Куратору он не нужен: это для того, кто принёс файл с известными итогами
 * и хочет посмотреть, насколько модель угадала.
 */
export function VerificationStrip({ v }: { v: Verification }) {
  const [open, setOpen] = React.useState(false);
  const c = v.confusion;

  return (
    <section className="rounded-xl border border-line bg-raised/40">
      <button
        onClick={() => setOpen((x) => !x)}
        aria-expanded={open}
        className="w-full px-4 py-2.5 text-left text-xs text-muted transition-colors hover:text-ink"
      >
        В файле есть фактические итоги —{" "}
        <span className="underline underline-offset-4">
          {open ? "скрыть качество на них" : "показать качество на них"}
        </span>
      </button>

      {open ? (
        <div className="border-t border-line px-4 py-3 text-xs text-muted">
          {c ? (
            <p className="nums">
              Модель нашла <span className="font-semibold text-ink">{c.tp}</span> из{" "}
              {c.tp + c.fn} тех, кто действительно не сдал. Зря позвано {c.fp}.
            </p>
          ) : null}
          {v.in_sample ? (
            <p className="mt-1.5 text-2xs text-faint">
              Это обучающая выборка: модель видела этих студентов, поэтому
              качество здесь завышено. Честные цифры — на странице «Как работает
              модель».
            </p>
          ) : (
            <p className="mt-1.5 text-2xs text-faint">
              Колонка с итогами использовалась только для сверки и в модель
              не передавалась.
            </p>
          )}
        </div>
      ) : null}
    </section>
  );
}
