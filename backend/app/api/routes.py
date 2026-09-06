"""HTTP-слой. Файлы обрабатываются в памяти и никуда не сохраняются."""
from __future__ import annotations

import io
import logging
import platform
from typing import Any, Literal

import numpy as np
import pandas as pd
import sklearn
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.errors import AppError
from app.core.ratelimit import check_rate
from app.services import loader
from app.services.predictor import (
    BAND_RU,
    choose_level,
    evaluate_against_actual,
    predict_frame,
    registry,
)
from riskml import schema
from riskml.pipeline import GRADE_COLS, TARGET_SOURCE

log = logging.getLogger("risk.api")
router = APIRouter(prefix="/api")

ID_CANDIDATES = ("id", "ID", "student_id", "Id", "№", "номер")


# --------------------------------------------------------------------------
@router.get("/health")
def health() -> dict:
    return {
        "status": "ok" if registry.ready else "degraded",
        "model_loaded": registry.ready,
        "error": registry.error,
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "model_versions": registry.meta.get("versions") if registry.ready else None,
    }


@router.get("/schema")
def get_schema() -> dict:
    """Фронтенд строит анкету и карточку студента отсюда, а не из хардкода."""
    return schema.as_json()


@router.get("/model-info")
def model_info() -> dict:
    if not registry.ready:
        raise AppError("Модель не загружена.", status_code=503, hint=registry.error)
    m = registry.meta
    return {
        "trained_at": m["trained_at"],
        "versions": m["versions"],
        "training_data": m["training_data"],
        "metrics": m["metrics"],
        "threshold": m["threshold"],
        "risk_bands": m["risk_bands"],
        "drop_features": m["drop_features"],
        "ensemble_members": m["ensemble_members"],
        "importances": {lvl: m["importances"][lvl][:10] for lvl in ("L1", "L2")},
        "levels": {
            "L1": {"name": "Конец семестра",
                   "uses": "все признаки, включая оценки за две контрольные",
                   "caveat": "Точнее, но помогать уже поздно — предсказывает оценку по оценкам."},
            "L2": {"name": "Начало семестра",
                   "uses": "анкета, пропуски, прошлые незачёты; без оценок по предмету",
                   "caveat": "Основная модель: менее точна, зато оставляет время что-то сделать."},
        },
    }


# --------------------------------------------------------------------------
def _pick_id_column(df: pd.DataFrame) -> str | None:
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _required_for(level: str) -> list[str]:
    req = list(schema.as_json()["required_l2"])
    return req + GRADE_COLS if level == "L1" else req


def _run(df: pd.DataFrame, level_override: str | None, threshold: float | None,
         file_info: dict) -> dict:
    level, level_note = choose_level(df, level_override)
    loader.validate_columns(df, _required_for(level))

    thr = registry.threshold if threshold is None else float(threshold)
    if not 0.0 <= thr <= 1.0:
        raise AppError("Порог должен быть числом от 0 до 1.", status_code=422)

    id_col = _pick_id_column(df)
    out = predict_frame(df, level, thr, id_column=id_col)

    has_actual = any("actual" in r for r in out["rows"])
    payload: dict[str, Any] = {
        "level": level,
        "level_note": level_note,
        "threshold": thr,
        "default_threshold": registry.threshold,
        "capacity": registry.capacity,
        "risk_bands": registry.bands,
        "band_labels": BAND_RU,
        "file": file_info,
        "data_quality": loader.data_quality(df, _required_for(level)),
        "summary": out["summary"],
        "rows": out["rows"],
        "id_column": id_col,
        "privacy": "Файл обработан в памяти и никуда не сохранён.",
    }
    if has_actual:
        payload["verification"] = evaluate_against_actual(out["rows"], thr)
    return payload


@router.post("/predict/batch")
async def predict_batch(
    request: Request,
    file: UploadFile = File(...),
    level: str | None = Form(None),
    threshold: float | None = Form(None),
) -> dict:
    check_rate(request)
    raw = await file.read()
    df, info = loader.read_table(raw, file.filename or "upload.csv")
    log.info("файл принят: %s строк=%s разделитель=%r",
             info.get("filename"), info.get("n_rows"), info.get("delimiter"))
    return _run(df, level, threshold, info)


@router.post("/predict/demo")
async def predict_demo(request: Request, level: str | None = Form(None)) -> dict:
    """Демо на приложенном student-por.csv — чтобы попробовать без своего файла."""
    check_rate(request)
    path = settings.demo_data
    if not path.exists():
        raise AppError("Демо-данные недоступны на сервере.", status_code=404)
    df, info = loader.read_table(path.read_bytes(), path.name)
    info["demo"] = True
    # Демо показывает ОСНОВНУЮ модель — начало семестра. Автовыбор дал бы L1
    # (в файле есть G1/G2), а L1 на собственной обучающей выборке упирается
    # в единицу: весь список был бы «100 %», что и бесполезно, и нечестно.
    out = _run(df, level or "L2", None, info)
    if not level:
        out["level_note"] = (
            "Демо показывает основную модель — прогноз начала семестра. "
            "Оценки за контрольные в файле есть, но намеренно не используются: "
            "модель конца семестра на собственной обучающей выборке даёт "
            "сплошные 100 %.")
    # Демо-файл — это ТА ЖЕ выборка, на которой модель обучалась. Метрики режима
    # проверки здесь заведомо завышены, и умолчать об этом нельзя.
    if out.get("verification"):
        out["verification"]["in_sample"] = True
        out["verification"]["note"] = (
            "Внимание: это обучающая выборка. Модель видела этих студентов, "
            "поэтому качество здесь завышено. Честные метрики — на странице «О модели».")
    return out


# --------------------------------------------------------------------------
class SurveyRequest(BaseModel):
    answers: dict[str, Any] = Field(default_factory=dict)


@router.post("/predict/single")
async def predict_single(request: Request, body: SurveyRequest) -> dict:
    check_rate(request)
    required = schema.as_json()["required_l2"]
    answers = dict(body.answers)

    # G3 — источник целевой переменной, в признаки не идёт никогда.
    for forbidden in (TARGET_SOURCE, "no_pass"):
        answers.pop(forbidden, None)

    # Оценки за контрольные необязательны. Есть обе — берём модель конца
    # семестра: она заметно точнее (ROC-AUC 0.971 против 0.848).
    grades = {}
    for g in GRADE_COLS:
        v = answers.pop(g, None)
        if v not in (None, ""):
            try:
                iv = int(float(v))
            except (TypeError, ValueError):
                raise AppError(f"Балл «{g}» должен быть числом от 0 до 20.",
                               status_code=422) from None
            if not 0 <= iv <= 20:
                raise AppError(f"Балл «{g}» должен быть от 0 до 20.", status_code=422)
            grades[g] = iv
    level = "L1" if len(grades) == len(GRADE_COLS) else "L2"

    # Анкета короткая: спрашиваем только то, что реально влияет. Всё остальное
    # подставляем типичным значением обучающей выборки — и честно сообщаем,
    # сколько полей заполнено за пользователя.
    bg = registry.background
    filled, row_in = [], {}
    for c in required:
        v = answers.get(c)
        if v in (None, ""):
            b = bg.get(c, {})
            v = b.get("median") if b.get("kind") == "numeric" else b.get("mode")
            filled.append(c)
        row_in[c] = v
    row_in.update(grades)

    answered = [c for c in required if c not in filled] + list(grades)
    df = pd.DataFrame([row_in])
    out = predict_frame(df, level, registry.threshold, explain_only=answered)
    row = out["rows"][0]

    bg = registry.background
    comparison = []
    for f in ("failures", "absences", "studytime", "goout"):
        if f in answered and bg.get(f, {}).get("kind") == "numeric":
            comparison.append({
                "feature": f,
                "label": (schema.SELF_WORDING.get(f, (None, None))[0]
                          or schema.BY_NAME[f]["label"]),
                "you": float(pd.to_numeric(df[f].iloc[0], errors="coerce")),
                "typical": bg[f]["median"],
                "you_label": str(schema.decode(f, df[f].iloc[0])),
                "typical_label": str(schema.decode(f, bg[f]["median"])),
            })

    return {
        "probability": row["probability"],
        "risk_band": row["risk_band"],
        "band_label": BAND_RU[row["risk_band"]],
        "top_factors": row["top_factors"],
        "comparison": comparison,
        "level": level,
        "level_note": ("Учтены баллы за обе контрольные — это более точная оценка."
                       if level == "L1" else
                       "Оценки за контрольные не указаны, поэтому это прогноз "
                       "начала семестра — он менее точен."),
        "defaults_used": len(filled),
        "cohort": {
            "n": registry.meta["training_data"]["n_rows"],
            "fail_rate": registry.meta["training_data"]["positive_rate"],
        },
        "threshold": registry.threshold,
        "disclaimer": {
            "not_a_verdict": (
                f"Вероятность {row['probability']:.0%} означает: из ста студентов "
                f"с похожими ответами примерно {row['probability']*100:.0f} не сдают. "
                "Это не предсказание лично про вас."),
            "not_a_decision": ("Результат не является оценкой, диагнозом или основанием "
                               "для каких-либо решений о вас."),
            "data": (f"Модель обучена на {registry.meta['training_data']['n_rows']} "
                     "студентах одного потока по одному предмету."),
            "action": "Самое полезное, что можно сделать, — поговорить с куратором.",
            "privacy": "Ответы обработаны в памяти, никуда не сохранены и никому не отправлены.",
        },
    }


# --------------------------------------------------------------------------
class ExportRow(BaseModel):
    rows: list[dict]
    format: Literal["csv", "xlsx"] = "csv"
    threshold: float | None = None


@router.post("/export")
async def export(request: Request, body: ExportRow):
    check_rate(request)
    if not body.rows:
        raise AppError("Нечего выгружать.", status_code=422)

    flat = []
    for r in body.rows:
        base = {
            "rank": r.get("rank"),
            "id": r.get("id"),
            "probability": r.get("probability"),
            "risk_band": BAND_RU.get(r.get("risk_band"), r.get("risk_band")),
            "in_shortlist": r.get("in_shortlist"),
        }
        for k, v in (r.get("features") or {}).items():
            base[k] = v
        tf = r.get("top_factors") or []
        base["top_factors"] = "; ".join(
            f"{'+' if f['direction']=='up' else '-'}{f['label']}={f['value_label']}"
            for f in tf)
        if r.get("actual"):
            base["actual_G3"] = r["actual"].get("G3")
            base["actual_no_pass"] = r["actual"].get("no_pass")
        flat.append(base)

    df = pd.DataFrame(flat)
    if body.format == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            df.to_excel(w, index=False, sheet_name="Риск незачёта")
        buf.seek(0)
        return StreamingResponse(
            buf, media_type=("application/vnd.openxmlformats-officedocument."
                             "spreadsheetml.sheet"),
            headers={"Content-Disposition": 'attachment; filename="risk.xlsx"'})

    text = df.to_csv(index=False, sep=";")
    return StreamingResponse(
        io.BytesIO(text.encode("utf-8-sig")), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="risk.csv"'})
