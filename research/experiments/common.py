"""Общий модуль для черновых экспериментов.

Код отсюда переносится в solution.ipynb как есть, поэтому он написан
в «ноутбучном» стиле: без классов-обёрток, всё явно.
"""
from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
DATA_POR = "student-por.csv"
DATA_MAT = "student-mat.csv"

# --------------------------------------------------------------------------
# Колонки
# --------------------------------------------------------------------------
YESNO = ["schoolsup", "famsup", "paid", "activities",
         "nursery", "higher", "internet", "romantic"]

# бинарные не-yes/no: колонка -> значение, которое кодируем единицей
BIN2 = {"school": "GP", "sex": "F", "address": "U",
        "famsize": "GT3", "Pstatus": "T"}

CAT_MULTI = ["Mjob", "Fjob", "reason", "guardian"]

NUM_BASE = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
            "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]

GRADES = ["G1", "G2"]
TARGET_SRC = "G3"


def _find(path: str) -> str:
    """Данные лежат в data/ в корне репозитория (сами скрипты — в
    research/experiments/). Ищем и там, и рядом со скриптом."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    for cand in (os.path.join(root, "data", path),
                 os.path.join(os.getcwd(), "data", path),
                 os.path.join(here, path), path):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"не найден {path}: положите данные в data/")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(_find(path), sep=";")
    df["no_pass"] = (df["G3"] < 10).astype(int)
    return df


# --------------------------------------------------------------------------
# Инженерия признаков — строго построчная (stateless).
# Никакой статистики по выборке здесь не считается, поэтому утечка
# между фолдами невозможна по построению. Всё, что требует fit
# (кодирование категорий, скейлинг, импьютация), живёт в ColumnTransformer.
# --------------------------------------------------------------------------
def engineer(X: pd.DataFrame, level: str = "L2", fe: bool = True,
             exclude: tuple = (), keep_only: tuple | None = None) -> pd.DataFrame:
    """level: 'L1' = с G1 и G2, 'L1a' = только G1, 'L2' = без обеих.

    fe=False   -> только базовые признаки (для таблицы «до / после»).
    exclude    -> имена признаков, которые надо выбросить (для ablation).
    keep_only  -> оставить только эти колонки (после построения всех).
    """
    d = pd.DataFrame(index=X.index)

    for c in NUM_BASE:
        d[c] = pd.to_numeric(X[c], errors="coerce")
    for c in YESNO:
        d[c] = (X[c].astype(str).str.strip() == "yes").astype(int)
    for c, pos in BIN2.items():
        d[f"{c}_{pos}"] = (X[c].astype(str).str.strip() == pos).astype(int)
    for c in CAT_MULTI:
        d[c] = X[c].astype(str)

    if level == "L1":
        d["G1"], d["G2"] = X["G1"].astype(float), X["G2"].astype(float)
    elif level == "L1a":
        d["G1"] = X["G1"].astype(float)

    if not fe:
        return _finish(d, exclude, keep_only)

    # --- общие для всех уровней ---
    d["alc_total"] = d["Dalc"] + d["Walc"]                     # суммарный алкоголь
    d["alc_weekday_share"] = d["Dalc"] / (d["alc_total"] + 1e-9)
    d["n_support"] = d["schoolsup"] + d["famsup"] + d["paid"]  # сколько видов поддержки
    d["any_support"] = (d["n_support"] > 0).astype(int)
    d["abs_per_study"] = d["absences"] / d["studytime"]        # пропуски на час самоподготовки
    d["log_absences"] = np.log1p(d["absences"])
    d["abs_zero"] = (d["absences"] == 0).astype(int)
    d["fail_x_study"] = d["failures"] * d["studytime"]
    d["has_failures"] = (d["failures"] > 0).astype(int)
    d["parent_edu_max"] = d[["Medu", "Fedu"]].max(axis=1)
    d["parent_edu_mean"] = d[["Medu", "Fedu"]].mean(axis=1)
    d["age_over_17"] = (d["age"] > 17).astype(int)             # второгодники
    d["goout_x_alc"] = d["goout"] * d["alc_total"]
    d["study_minus_free"] = d["studytime"] - d["freetime"]
    d["no_higher"] = 1 - d["higher"]
    d["risk_count"] = (d["has_failures"] + d["no_higher"]
                       + (d["absences"] > 8).astype(int)
                       + (d["studytime"] <= 1).astype(int))
    # бины с ФИКСИРОВАННЫМИ границами -> stateless
    d["abs_bin"] = pd.cut(d["absences"], [-1, 0, 2, 6, 12, 1e9], labels=False).astype(float)
    d["age_bin"] = pd.cut(d["age"], [0, 16, 17, 18, 1e9], labels=False).astype(float)

    # --- динамика оценок (только там, где оценки доступны) ---
    if level == "L1":
        d["G_diff"] = d["G2"] - d["G1"]                        # динамика между контрольными
        d["G_mean"] = (d["G1"] + d["G2"]) / 2
        d["G_min"] = d[["G1", "G2"]].min(axis=1)
        d["G_proj"] = d["G2"] + (d["G2"] - d["G1"])            # линейный прогноз на G3
        d["G1_fail"] = (d["G1"] < 10).astype(int)
        d["G2_fail"] = (d["G2"] < 10).astype(int)
        d["G_both_fail"] = d["G1_fail"] * d["G2_fail"]
        d["G_declining"] = (d["G_diff"] < 0).astype(int)
    elif level == "L1a":
        d["G1_fail"] = (d["G1"] < 10).astype(int)
        d["G1_margin"] = d["G1"] - 10

    return _finish(d, exclude, keep_only)


def _finish(d, exclude, keep_only):
    if keep_only is not None:
        d = d[[c for c in d.columns if c in set(keep_only)]]
    if exclude:
        d = d.drop(columns=[c for c in exclude if c in d.columns])
    return d


# инженерные (не-базовые) признаки по уровням — для ablation
ENG_COMMON = ["alc_total", "alc_weekday_share", "n_support", "any_support",
              "abs_per_study", "log_absences", "abs_zero", "fail_x_study",
              "has_failures", "parent_edu_max", "parent_edu_mean", "age_over_17",
              "goout_x_alc", "study_minus_free", "no_higher", "risk_count",
              "abs_bin", "age_bin"]
ENG_L1 = ["G_diff", "G_mean", "G_min", "G_proj", "G1_fail", "G2_fail",
          "G_both_fail", "G_declining"]


def eng_cols(level: str):
    return ENG_COMMON + (ENG_L1 if level == "L1" else
                         ["G1_fail", "G1_margin"] if level == "L1a" else [])


# какие инженерные признаки порождены какой исходной колонкой —
# нужно для честной drop-column importance на уровне ИСХОДНЫХ переменных
DERIVED = {
    "absences":  ["abs_per_study", "log_absences", "abs_zero", "abs_bin", "risk_count"],
    "studytime": ["abs_per_study", "fail_x_study", "study_minus_free", "risk_count"],
    "failures":  ["fail_x_study", "has_failures", "risk_count"],
    "Dalc":      ["alc_total", "alc_weekday_share", "goout_x_alc"],
    "Walc":      ["alc_total", "alc_weekday_share", "goout_x_alc"],
    "goout":     ["goout_x_alc"],
    "higher":    ["no_higher", "risk_count"],
    "schoolsup": ["n_support", "any_support"],
    "famsup":    ["n_support", "any_support"],
    "paid":      ["n_support", "any_support"],
    "Medu":      ["parent_edu_max", "parent_edu_mean"],
    "Fedu":      ["parent_edu_max", "parent_edu_mean"],
    "age":       ["age_over_17", "age_bin"],
    "freetime":  ["study_minus_free"],
    "G1":        ["G_diff", "G_mean", "G_min", "G_proj", "G1_fail",
                  "G_both_fail", "G_declining", "G1_margin"],
    "G2":        ["G_diff", "G_mean", "G_min", "G_proj", "G2_fail",
                  "G_both_fail", "G_declining"],
}


def derived_of(raw: str) -> tuple:
    """Все инженерные колонки, которые надо убрать вместе с исходной `raw`."""
    own = f"{raw}_{BIN2[raw]}" if raw in BIN2 else raw
    return tuple([own] + DERIVED.get(raw, []))


def make_fe(level: str, fe: bool = True, exclude: tuple = (),
            keep_only: tuple | None = None):
    return FunctionTransformer(
        partial(engineer, level=level, fe=fe, exclude=tuple(exclude),
                keep_only=keep_only),
        feature_names_out=None, validate=False)


# --------------------------------------------------------------------------
# Препроцессор: OHE для номинальных, скейлинг (опционально) для остальных
# --------------------------------------------------------------------------
def make_prep(scale: bool = True, ohe: bool = True):
    cat_sel = make_column_selector(dtype_include=object)
    num_sel = make_column_selector(dtype_exclude=object)

    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))

    cat_enc = OneHotEncoder(handle_unknown="ignore", drop=None,
                            sparse_output=False) if ohe else \
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        [("num", Pipeline(num_steps), num_sel),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("enc", cat_enc)]), cat_sel)],
        remainder="drop", verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_pipe(model, level="L2", scale=True, fe=True, exclude=(), keep_only=None):
    """Полный пайплайн: FE -> препроцессинг -> модель. Всё фитится на train-фолде."""
    return Pipeline([
        ("fe", make_fe(level, fe, exclude, keep_only)),
        ("prep", make_prep(scale=scale)),
        ("clf", model),
    ])


# --------------------------------------------------------------------------
# Оценка
# --------------------------------------------------------------------------
def cv_obj(n_splits=5, n_repeats=5, seed=SEED):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)


def evaluate(pipe, X, y, cv=None, n_jobs=-1):
    """Возвращает dict со средним и std по ROC-AUC и PR-AUC."""
    cv = cv or cv_obj()
    res = cross_validate(pipe, X, y, cv=cv,
                         scoring=["roc_auc", "average_precision"],
                         n_jobs=n_jobs, error_score="raise")
    return {
        "roc_auc": res["test_roc_auc"].mean(),
        "roc_auc_std": res["test_roc_auc"].std(),
        "pr_auc": res["test_average_precision"].mean(),
        "pr_auc_std": res["test_average_precision"].std(),
        "_roc_raw": res["test_roc_auc"],
        "_pr_raw": res["test_average_precision"],
    }


def fmt(r: dict) -> str:
    return (f"ROC-AUC {r['roc_auc']:.3f} ± {r['roc_auc_std']:.3f}   "
            f"PR-AUC {r['pr_auc']:.3f} ± {r['pr_auc_std']:.3f}")


def get_Xy(df: pd.DataFrame):
    """Признаки = всё, кроме G3 и no_pass. G1/G2 отсекаются позже, в engineer()."""
    X = df.drop(columns=[TARGET_SRC, "no_pass"])
    y = df["no_pass"].to_numpy()
    return X, y
