import * as React from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import type { Schema } from "@/lib/types";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { CohortPage } from "@/pages/CohortPage";
import { SurveyPage } from "@/pages/SurveyPage";
import { ModelPage } from "@/pages/ModelPage";

// «Как работает модель» намеренно не в шапке: куратору она не нужна,
// а по прямой ссылке /model страница по-прежнему открывается — её удобно
// давать тем, кто спрашивает про внутренности.
const NAV = [
  { to: "/cohort", label: "Список" },
  { to: "/survey", label: "Анкета" },
];

export default function App() {
  const [schema, setSchema] = React.useState<Schema | null>(null);
  const [theme, setTheme] = React.useState<"light" | "dark">(() =>
    (localStorage.getItem("theme") as "light" | "dark") ??
    (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light"),
  );

  React.useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  React.useEffect(() => {
    api.schema().then(setSchema).catch(() => setSchema(null));
  }, []);

  return (
    <div className="min-h-dvh">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2 focus:text-sm"
      >
        К основному содержимому
      </a>

      <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur">
        <div className="mx-auto flex max-w-content items-center gap-4 px-4 py-3 sm:px-6">
          <span className="text-sm font-semibold tracking-tight">Риск незачёта</span>
          <nav className="flex gap-1" aria-label="Основная навигация">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) =>
                  cn(
                    "rounded-lg px-2.5 py-1.5 text-sm transition-colors",
                    isActive
                      ? "bg-accent-soft font-medium text-accent"
                      : "text-muted hover:bg-raised hover:text-ink",
                  )
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="ml-auto rounded-lg px-2.5 py-1.5 text-sm text-muted transition-colors hover:bg-raised hover:text-ink"
            aria-label={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </div>
      </header>

      <main id="main" className="mx-auto max-w-content px-4 py-6 sm:px-6 sm:py-8">
        <Routes>
          <Route path="/" element={<Navigate to="/cohort" replace />} />
          <Route path="/cohort" element={<CohortPage schema={schema} />} />
          <Route path="/survey" element={<SurveyPage schema={schema} />} />
          <Route path="/model" element={<ModelPage schema={schema} />} />
          <Route path="*" element={<Navigate to="/cohort" replace />} />
        </Routes>
      </main>

      <footer className="mx-auto max-w-content px-4 pb-10 pt-4 text-2xs text-faint sm:px-6">
        Оценка вероятности, а не приговор. Загруженные файлы и ответы анкеты
        не сохраняются.
      </footer>
    </div>
  );
}
