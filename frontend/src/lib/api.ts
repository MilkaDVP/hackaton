import type {
  ApiError, BatchResult, ModelInfo, Schema, SurveyResult,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class RequestError extends Error {
  hint?: string;
  details?: Record<string, unknown>;
  status: number;

  constructor(status: number, e: ApiError) {
    super(e.message);
    this.status = status;
    this.hint = e.hint;
    this.details = e.details;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (res.ok) return (await res.json()) as T;
  let payload: ApiError = { message: `Ошибка ${res.status}` };
  try {
    const body = await res.json();
    if (body?.error) payload = body.error;
  } catch {
    /* тело не JSON — оставляем общее сообщение */
  }
  throw new RequestError(res.status, payload);
}

export const api = {
  schema: () => fetch(`${BASE}/schema`).then(handle<Schema>),

  modelInfo: () => fetch(`${BASE}/model-info`).then(handle<ModelInfo>),

  health: () => fetch(`${BASE}/health`).then(handle<Record<string, unknown>>),

  batch(file: File, level?: string) {
    const fd = new FormData();
    fd.append("file", file);
    if (level) fd.append("level", level);
    return fetch(`${BASE}/predict/batch`, { method: "POST", body: fd })
      .then(handle<BatchResult>);
  },

  demo(level?: string) {
    const fd = new FormData();
    if (level) fd.append("level", level);
    return fetch(`${BASE}/predict/demo`, { method: "POST", body: fd })
      .then(handle<BatchResult>);
  },

  single(answers: Record<string, unknown>) {
    return fetch(`${BASE}/predict/single`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answers }),
    }).then(handle<SurveyResult>);
  },

  async export(rows: unknown[], format: "csv" | "xlsx") {
    const res = await fetch(`${BASE}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows, format }),
    });
    if (!res.ok) return handle(res);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = format === "xlsx" ? "risk.xlsx" : "risk.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    return null;
  },
};
