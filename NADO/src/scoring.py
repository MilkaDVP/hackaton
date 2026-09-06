"""Скоринг риска незачёта.

Единственный источник правды для подготовки признаков и финальной модели.
Ноутбук `risk.ipynb` импортирует эти функции, будущий веб-интерфейс должен
делать то же самое, а не переписывать логику заново.

Рабочий уровень — «уровень 2», без `G1` и `G2`: ранжирование строится
по анкете начала года, до того как написаны обе контрольные.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
)

TARGET_COL = "G3"
PASS_MARK = 10
LEVEL2_DROP = ["G1", "G2"]
RANDOM_STATE = 42
BLEND_WEIGHTS = (0.4, 0.3, 0.3)

# Три фактора из блока 15 ноутбука: исходные колонки, а не one-hot имена.
TOP_FACTORS = ["failures", "school", "higher"]

FACTOR_RU = {
    "failures": "прошлые незачёты",
    "school": "учебное заведение",
    "higher": "планирует учиться дальше",
}

MODEL_FILE = "blend.joblib"


def prep(df: pd.DataFrame, drop_g12: bool):
    """Возвращает X (get_dummies, drop_first=True, float) и y.

    drop_g12=False — уровень 1 (с оценками за контрольные G1, G2).
    drop_g12=True  — уровень 2 (без них), рабочий вариант.

    G3 выбрасывается всегда: из неё сделана целевая переменная.
    Если G3 в таблице нет (боевой режим, оценок ещё не существует),
    y возвращается как None.
    """
    to_drop = [TARGET_COL] + (LEVEL2_DROP if drop_g12 else [])
    features = df.drop(columns=[c for c in to_drop if c in df.columns])
    X = pd.get_dummies(features, drop_first=True).astype(float)
    y = None
    if TARGET_COL in df.columns:
        y = (df[TARGET_COL] < PASS_MARK).astype(int)
    return X, y


def to_ranks(scores) -> np.ndarray:
    """Переводит оценки риска в ранги на отрезке (0, 1].

    Шкалы у леса, деревьев и регрессора несопоставимы, а порядок сопоставим —
    поэтому смешиваем ранги, а не сырые числа.
    """
    scores = np.asarray(scores, dtype=float)
    return rankdata(scores) / len(scores)


RF_PARAMS = {"min_samples_leaf": 5, "max_features": 0.4, "max_depth": None}


def make_members(random_state: int = RANDOM_STATE, rf_params: dict | None = None) -> dict:
    """Три участника бленда, ровно в той конфигурации, что победила в блоке 12.

    rf_params позволяет ноутбуку передать победителя сетки из блока 9,
    чтобы сохранённая модель не разошлась с измеренной.
    """
    rf_params = dict(RF_PARAMS if rf_params is None else rf_params)
    return {
        "rf": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            **rf_params,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "reg": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def fit_blend(
    df: pd.DataFrame,
    weights: tuple = BLEND_WEIGHTS,
    random_state: int = RANDOM_STATE,
    rf_params: dict | None = None,
) -> dict:
    """Обучает финальный бленд на всей переданной таблице.

    Классификаторы учатся на бинарной цели, регрессор — на непрерывной G3.
    Возвращает словарь, который умеют читать predict_risk и explain.
    """
    X, y = prep(df, drop_g12=True)
    assert TARGET_COL not in X.columns, "G3 просочилась в признаки"
    if y is None:
        raise ValueError("для обучения нужна колонка G3")

    g3 = df[TARGET_COL].to_numpy(dtype=float)
    members = make_members(random_state, rf_params)
    members["rf"].fit(X, y)
    members["et"].fit(X, y)
    members["reg"].fit(X, g3)

    return {
        "columns": list(X.columns),
        "members": members,
        "weights": tuple(weights),
        "top_factors": list(TOP_FACTORS),
        "n_train": int(len(df)),
        "base_rate": float(y.mean()),
        "random_state": int(random_state),
        "rf_params": dict(RF_PARAMS if rf_params is None else rf_params),
    }


def save_model(model: dict, path: str | Path = "models/") -> Path:
    """Кладёт модель в каталог path, создавая его при необходимости."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    target = path / MODEL_FILE
    joblib.dump(model, target, compress=3)
    return target


def load_model(path: str | Path = "models/") -> dict:
    """Читает модель из каталога path (или прямо из .joblib-файла)."""
    path = Path(path)
    if path.is_dir():
        path = path / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} не найден — выполните блок 21 ноутбука risk.ipynb"
        )
    return joblib.load(path)


def _design_matrix(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Готовит признаки новой таблицы и выравнивает их по колонкам обучения."""
    X, _ = prep(df, drop_g12=True)
    return X.reindex(columns=model["columns"], fill_value=0.0)


def predict_risk(df: pd.DataFrame, model: dict) -> np.ndarray:
    """Оценка риска незачёта для каждой строки df: ранг в диапазоне (0, 1].

    1.0 — самый рискованный студент переданной группы, близко к 0 — самый
    спокойный. Это ранг ВНУТРИ переданной таблицы, а не вероятность: подавайте
    сюда всю группу целиком, а не по одному человеку.
    """
    X = _design_matrix(df, model)
    w = model["weights"]
    m = model["members"]
    score = (
        w[0] * to_ranks(m["rf"].predict_proba(X)[:, 1])
        + w[1] * to_ranks(m["et"].predict_proba(X)[:, 1])
        + w[2] * to_ranks(-m["reg"].predict(X))
    )
    return to_ranks(score)


def explain(df: pd.DataFrame, i: int, model: dict) -> dict:
    """Значения трёх главных признаков для строки с позицией i в df.

    i — порядковый номер строки (0 … len(df)-1), не метка индекса.
    Возвращает ранг риска, место в упорядочении и значения признаков.
    """
    if not 0 <= i < len(df):
        raise IndexError(f"i={i} вне диапазона 0..{len(df) - 1}")

    risk = predict_risk(df, model)
    order = np.argsort(-risk)
    place = int(np.where(order == i)[0][0]) + 1
    row = df.iloc[i]

    factors = {}
    for col in model["top_factors"]:
        if col in df.columns:
            value = row[col]
            factors[col] = {
                "название": FACTOR_RU.get(col, col),
                "значение": value.item() if hasattr(value, "item") else value,
            }

    return {
        "позиция": int(i),
        "риск": float(risk[i]),
        "место_в_очереди": place,
        "размер_группы": int(len(df)),
        "факторы": factors,
        "оговорка": (
            "Значения признаков описывают статистическую связь с целевой "
            "переменной и не являются утверждением о причинности."
        ),
    }
