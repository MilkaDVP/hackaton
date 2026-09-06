"""Чтение загруженного файла: определение кодировки, разделителя, валидация колонок.

Разделитель `;` вместо `,` — главные грабли этого датасета, поэтому определяем
его автоматически и с явным фолбэком, а не полагаемся на дефолт pandas.
"""
from __future__ import annotations

import csv
import difflib
import io
import logging

import pandas as pd

from app.core.config import settings
from app.core.errors import AppError

log = logging.getLogger("risk.loader")

ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "latin-1")
DELIMITERS = (";", ",", "\t", "|")


def _decode(raw: bytes) -> str:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise AppError("Не удалось определить кодировку файла.",
                   hint="Сохраните файл в UTF-8 или Windows-1251.")


def sniff_delimiter(text: str) -> str:
    """csv.Sniffer с фолбэком: на коротких файлах он ошибается, поэтому
    подстраховываемся подсчётом кандидатов в первой строке."""
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        header = text.splitlines()[0] if text.splitlines() else ""
        counts = {d: header.count(d) for d in DELIMITERS}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ";"


def read_table(raw: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    """Возвращает (таблица, сведения о том, как её прочитали)."""
    if not raw:
        raise AppError("Файл пустой.", hint="Загрузите файл с данными студентов.")
    if len(raw) > settings.max_upload_bytes:
        raise AppError(
            f"Файл больше {settings.max_upload_bytes // (1024*1024)} МБ.",
            status_code=413,
            hint="Разделите выгрузку на части или уберите лишние колонки.")

    name = (filename or "").lower()
    info: dict = {"filename": filename}

    if name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(raw))
        except ImportError as e:
            raise AppError("Чтение XLSX недоступно на сервере.",
                           hint="Сохраните файл как CSV.") from e
        except Exception as e:
            raise AppError("Не удалось прочитать Excel-файл.",
                           hint="Проверьте, что файл не повреждён.") from e
        info |= {"format": "xlsx", "delimiter": None, "encoding": None}
    else:
        text = _decode(raw)
        if not text.strip():
            raise AppError("Файл пустой.", hint="Загрузите файл с данными студентов.")
        delim = sniff_delimiter(text)
        try:
            df = pd.read_csv(io.StringIO(text), sep=delim)
        except Exception as e:
            raise AppError("Не удалось разобрать файл как таблицу.",
                           hint="Ожидается CSV с заголовком в первой строке.") from e
        info |= {"format": "csv", "delimiter": delim,
                 "encoding": "utf-8" if text is not None else None}

    df.columns = [str(c).strip() for c in df.columns]

    if df.empty:
        raise AppError("В файле нет ни одной строки данных.",
                       hint="Проверьте, что под заголовком есть строки.")
    if len(df) > settings.max_rows:
        raise AppError(f"В файле больше {settings.max_rows} строк.",
                       status_code=413,
                       hint="Загрузите файл меньшего размера.")
    if len(df.columns) < 2:
        raise AppError(
            "В файле только одна колонка — скорее всего, не угадан разделитель.",
            hint="Сохраните файл с разделителем «;» или «,».")

    info["n_rows"] = int(len(df))
    info["n_cols"] = int(len(df.columns))
    return df, info


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Не хватает колонок — 422 со списком и подсказкой про опечатки."""
    have = set(df.columns)
    missing = [c for c in required if c not in have]
    if not missing:
        return

    suggestions = {}
    for col in missing:
        near = difflib.get_close_matches(col, list(df.columns), n=1, cutoff=0.75)
        if not near:
            near = [c for c in df.columns if c.lower() == col.lower()]
        if near:
            suggestions[col] = near[0]

    hint = "Проверьте заголовки — регистр и написание должны совпадать."
    if suggestions:
        pairs = ", ".join(f"«{v}» → «{k}»" for k, v in suggestions.items())
        hint = f"Похоже на опечатку: {pairs}"

    raise AppError(
        f"В файле не хватает обязательных колонок ({len(missing)} из {len(required)}).",
        status_code=422, hint=hint,
        details={"missing": missing, "suggestions": suggestions,
                 "found_columns": list(df.columns)})


def data_quality(df: pd.DataFrame, cols: list[str]) -> dict:
    """Пропуски не выбрасываем (импьютация внутри пайплайна), но показываем."""
    present = [c for c in cols if c in df.columns]
    na = df[present].isna()
    per_col = {c: int(n) for c, n in na.sum().items() if n > 0}
    return {
        "rows_with_missing": int(na.any(axis=1).sum()),
        "columns_with_missing": per_col,
        "note": ("Пропуски заполняются медианой/модой внутри пайплайна — "
                 "строки не выбрасываются."),
    }
