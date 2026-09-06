"""Загрузка артефактов, предсказание и персональные объяснения."""
from __future__ import annotations

import json
import logging
import threading
import time

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.errors import AppError
from riskml import schema
from riskml.pipeline import GRADE_COLS, TARGET_SOURCE

log = logging.getLogger("risk.predictor")

#: Сколько признаков участвует в персональном объяснении. Ограничение
#: нужно и по скорости, и по смыслу: вклад `nursery` пользователю не интересен.
EXPLAIN_TOP_K = 8

#: Выше этого числа строк объяснения считаются только для самых рискующих —
#: остальным куратор всё равно не станет открывать карточку.
EXPLAIN_MAX_ROWS = 2000


class Registry:
    """Модели и метаданные, загруженные один раз при старте."""

    def __init__(self) -> None:
        self.models: dict[str, object] = {}
        self.meta: dict = {}
        self.ready = False
        self.error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        d = settings.artifacts_dir
        try:
            meta_path = d / "metadata.json"
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for level in ("L1", "L2"):
                self.models[level] = joblib.load(d / f"model_{level.lower()}.joblib")
            self.ready = True
            log.info("модели загружены из %s (обучены %s)", d,
                     self.meta.get("trained_at"))
        except FileNotFoundError as e:
            self.error = f"артефакт не найден: {e.filename}"
            log.error("%s — запустите `make train`", self.error)
        except Exception as e:  # noqa: BLE001
            # Чаще всего это несовпадение версии sklearn между обучением и рантаймом
            self.error = f"не удалось загрузить модель: {type(e).__name__}: {e}"
            log.exception("ошибка загрузки артефактов")

    def require(self, level: str):
        if not self.ready:
            raise AppError("Модель не загружена.", status_code=503,
                           hint=self.error or "Проверьте артефакты в riskml/artifacts.")
        return self.models[level]

    # --- производные из метаданных ---
    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"]["default"])

    @property
    def capacity(self) -> int:
        return int(self.meta["threshold"]["capacity"])

    @property
    def bands(self) -> dict:
        return self.meta["risk_bands"]

    @property
    def background(self) -> dict:
        return self.meta["background"]

    def top_features(self, level: str) -> list[str]:
        imp = self.meta["importances"][level]
        names = [d["feature"] for d in imp if d["feature"] not in GRADE_COLS or level == "L1"]
        return names[:EXPLAIN_TOP_K]

    def importance_map(self, level: str) -> dict:
        return {d["feature"]: d for d in self.meta["importances"][level]}


registry = Registry()


# --------------------------------------------------------------------------
def choose_level(df: pd.DataFrame, override: str | None) -> tuple[str, str]:
    """Автовыбор уровня: есть G1 и G2 -> L1 (конец семестра), иначе L2."""
    has_grades = all(c in df.columns for c in GRADE_COLS)
    if override in ("L1", "L2"):
        if override == "L1" and not has_grades:
            raise AppError(
                "Для прогноза конца семестра нужны колонки G1 и G2, а их нет в файле.",
                status_code=422,
                hint="Уберите выбор уровня — тогда сработает модель начала семестра.")
        level = override
        reason = "уровень выбран вручную"
    else:
        level = "L1" if has_grades else "L2"
        reason = ("в файле есть G1 и G2" if has_grades
                  else "в файле нет оценок за контрольные")
    note = ("Прогноз конца семестра: модель видит обе контрольные. "
            "Точнее, но помогать уже поздно."
            if level == "L1" else
            "Прогноз начала семестра: модель не использует оценки по предмету. "
            "Менее точен, зато остаётся время что-то сделать.")
    return level, f"{reason}. {note}"


def band_of(p: float, bands: dict) -> str:
    if p >= bands["high_min"]:
        return "high"
    if p <= bands["low_max"]:
        return "low"
    return "medium"


BAND_RU = {"high": "высокий", "medium": "средний", "low": "низкий"}


# --------------------------------------------------------------------------
def _background_frame(df: pd.DataFrame, feature: str) -> pd.Series:
    """Значение признака «как у типичного студента» — медиана или мода обучающей выборки."""
    bg = registry.background.get(feature)
    if bg is None:
        return df[feature]
    if bg["kind"] == "numeric":
        return pd.Series(bg["median"], index=df.index)
    return pd.Series(bg["mode"], index=df.index)


def _contradicts_global(feature: str, value, effect: float, imp: dict) -> bool:
    """Локальный эффект противоречит общей закономерности по выборке?

    Модель нелинейна, и у отдельного студента признак может работать не в ту
    сторону, что «в среднем». Это честное поведение модели, но показывать его
    без пометки нельзя — читается как ошибка.
    """
    if abs(effect) < 1e-3 or value is None or pd.isna(value):
        return False
    d = imp.get(feature, {}).get("direction") or {}
    bg = registry.background.get(feature) or {}

    if d.get("kind") == "numeric" and bg.get("kind") == "numeric":
        sign = d.get("sign", 0)
        if not sign:
            return False
        try:
            delta_from_typical = float(value) - float(bg["median"])
        except (TypeError, ValueError):
            return False
        if abs(delta_from_typical) < 1e-9:
            return False
        # Ожидаемое направление: выше медианы * (признак повышает риск) -> вверх
        expected_up = (delta_from_typical > 0) == (sign > 0)
        return expected_up != (effect > 0)

    if d.get("kind") == "categorical":
        v = str(value)
        if v == d.get("worst"):
            return effect < 0      # худшая категория должна повышать риск
        if v == d.get("best"):
            return effect > 0      # лучшая — понижать
    return False


def explain(df: pd.DataFrame, level: str, base_proba: np.ndarray,
            only: list[str] | None = None) -> list[list[dict]]:
    """Вклад каждого признака: насколько изменится вероятность, если заменить
    его значение на типичное для потока.

    Все возмущения склеиваются в ОДИН вызов predict_proba. Это принципиально
    для скорости: у модели большая фиксированная накладная стоимость вызова
    (649 строк — 1.3 мс/строку, 7788 строк — 0.27 мс/строку), поэтому восемь
    отдельных прогонов стоили бы ~8 с, а один склеенный — около полутора.

    SHAP здесь не используется осознанно: TreeExplainer на калиброванном
    ансамбле из 866-деревного ExtraTrees + SVM в требование «до 3 секунд»
    не укладывается.
    """
    model = registry.require(level)
    feats = [f for f in registry.top_features(level) if f in df.columns]
    if only is not None:
        # Объясняем только то, что пользователь реально указал. Показывать
        # «повлияло образование отца», когда это значение подставлено за него
        # по умолчанию, — прямая дезинформация.
        keep = set(only)
        feats = [f for f in feats if f in keep]
    n = len(df)

    # На больших файлах объясняем только самых рискующих.
    if n > EXPLAIN_MAX_ROWS:
        idx = np.argsort(-base_proba)[:EXPLAIN_MAX_ROWS]
        sub = df.iloc[idx]
        sub_base = base_proba[idx]
    else:
        idx = np.arange(n)
        sub, sub_base = df, base_proba

    if not feats or len(sub) == 0:
        return [[] for _ in range(n)]

    probes = []
    for f in feats:
        p = sub.copy()
        p[f] = _background_frame(sub, f)
        probes.append(p)
    stacked = pd.concat(probes, ignore_index=True)

    try:
        flat = model.predict_proba(stacked)[:, 1]
    except Exception:  # noqa: BLE001
        log.exception("объяснения не посчитаны")
        return [[] for _ in range(n)]

    m = len(sub)
    # >0 -> признак толкает риск ВВЕРХ относительно типичного значения
    contrib = {f: sub_base - flat[i * m:(i + 1) * m] for i, f in enumerate(feats)}

    imp = registry.importance_map(level)
    out: list[list[dict]] = [[] for _ in range(n)]
    for pos, row_i in enumerate(idx):
        items = []
        for f, delta in contrib.items():
            d = float(delta[pos])
            raw = sub[f].iloc[pos]
            items.append({
                "feature": f,
                "label": schema.BY_NAME.get(f, {}).get("label", f),
                "value": None if pd.isna(raw) else (
                    raw.item() if hasattr(raw, "item") else raw),
                "value_label": ("—" if pd.isna(raw) else str(schema.decode(f, raw))),
                "effect": round(d, 4),
                "direction": "up" if d > 0 else "down",
                "note": imp.get(f, {}).get("direction", {}).get("text", ""),
                "not_a_lever": f == "school",
                # Эффект ниже порога шума: у «типичного» студента все признаки
                # близки к медиане, и сдвигать вероятность им нечем. Это не
                # ошибка — но и подавать такой фактор как значимый нельзя.
                "weak": abs(d) < 1e-3,
                # У этого студента признак работает не в ту сторону, что обычно.
                # Модель нелинейна и местами немонотонна — умалчивать об этом
                # нельзя, иначе объяснение выглядит просто неверным.
                "unusual": _contradicts_global(f, raw, d, imp),
            })
        items.sort(key=lambda x: -abs(x["effect"]))
        out[int(row_i)] = items[:5]
    return out


def predict_frame(df: pd.DataFrame, level: str, threshold: float,
                  id_column: str | None = None,
                  explain_only: list[str] | None = None) -> dict:
    """Предсказание по таблице + сводка по файлу."""
    model = registry.require(level)
    t0 = time.time()

    # G3 в признаки не попадает НИКОГДА. Если она есть — только как факт для сверки.
    actual = None
    if TARGET_SOURCE in df.columns:
        actual = pd.to_numeric(df[TARGET_SOURCE], errors="coerce")
    features = df.drop(columns=[c for c in (TARGET_SOURCE, "no_pass") if c in df.columns])

    proba = model.predict_proba(features)[:, 1]
    # стабильная сортировка: при равных вероятностях порядок задаётся строкой файла,
    # иначе ранг «прыгал» бы между запусками и расходился с порядком в таблице
    order = np.argsort(-proba, kind="stable")
    rank = np.empty(len(proba), dtype=int)
    rank[order] = np.arange(1, len(proba) + 1)

    factors = explain(features, level, proba, only=explain_only)
    bands = registry.bands

    ids = (df[id_column].astype(str).tolist() if id_column and id_column in df.columns
           else [str(i) for i in range(1, len(df) + 1)])

    # В карточку отдаём только известные схеме признаки: лишние колонки из файла
    # игнорируются молча, показывать пользователю поля без подписей незачем.
    known_cols = [c for c in features.columns if c in schema.BY_NAME]

    rows = []
    for i in range(len(df)):
        rec = {c: (None if pd.isna(df[c].iloc[i]) else
                   (df[c].iloc[i].item() if hasattr(df[c].iloc[i], "item") else df[c].iloc[i]))
               for c in known_cols}
        item = {
            "id": ids[i],
            "row": i,
            "probability": round(float(proba[i]), 4),
            "risk_band": band_of(float(proba[i]), bands),
            "rank": int(rank[i]),
            "in_shortlist": bool(proba[i] >= threshold),
            "features": rec,
            "top_factors": factors[i],
        }
        if actual is not None and not pd.isna(actual.iloc[i]):
            item["actual"] = {"G3": float(actual.iloc[i]),
                              "no_pass": bool(actual.iloc[i] < 10)}
        rows.append(item)

    hist, edges = np.histogram(proba, bins=20, range=(0, 1))
    summary = {
        "n_rows": int(len(df)),
        "mean_probability": round(float(proba.mean()), 4),
        "median_probability": round(float(np.median(proba)), 4),
        "in_shortlist": int((proba >= threshold).sum()),
        "bands": {b: int(sum(1 for p in proba if band_of(float(p), bands) == b))
                  for b in ("high", "medium", "low")},
        "histogram": {"counts": hist.tolist(),
                      "edges": [round(float(e), 3) for e in edges]},
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
    return {"rows": rows, "summary": summary}


def evaluate_against_actual(rows: list[dict], threshold: float) -> dict | None:
    """Режим проверки: в файле был G3, поэтому можно измерить качество.

    Это отдельный, явно помеченный режим — не рабочий режим прогноза.
    """
    from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

    pairs = [(r["probability"], r["actual"]["no_pass"]) for r in rows if "actual" in r]
    if len(pairs) < 10:
        return None
    p = np.array([a for a, _ in pairs])
    y = np.array([int(b) for _, b in pairs])
    if len(np.unique(y)) < 2:
        return {"note": "В файле все студенты одного класса — метрики не определены.",
                "n": len(y)}
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_fail": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "threshold": threshold,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": round(float(tp / (tp + fp)), 4) if (tp + fp) else None,
        "recall": round(float(tp / (tp + fn)), 4) if (tp + fn) else None,
        "note": ("Режим проверки: в файле была колонка G3 с фактическим итогом. "
                 "В признаки она не передавалась."),
    }
