"""Пайплайн модели — дословная копия из risk.ipynb (разделы 4 и 5.2).

ВАЖНО: этот модуль обязан быть импортируем и в скрипте обучения, и в бэкенде.
`joblib` пиклит `functools.partial(engineer, ...)` по ссылке на модуль, поэтому
при загрузке артефакта Python ищет именно `riskml.pipeline.engineer`. Если модуль
переименовать или переместить, готовые модели перестанут загружаться.

Ничего в этом файле не «улучшается» относительно ноутбука: продукт обязан
воспроизводить те же числа.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

SEED = 42

# --------------------------------------------------------------------------
# Колонки исходного датасета
# --------------------------------------------------------------------------
YESNO = ["schoolsup", "famsup", "paid", "activities",
         "nursery", "higher", "internet", "romantic"]
BIN2 = {"school": "GP", "sex": "F", "address": "U", "famsize": "GT3", "Pstatus": "T"}
CAT_MULTI = ["Mjob", "Fjob", "reason", "guardian"]
NUM_BASE = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
            "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]

#: Колонки, которые обязаны быть во входном файле для модели уровня L2.
REQUIRED_L2 = NUM_BASE + YESNO + list(BIN2) + CAT_MULTI
#: Дополнительно нужны для L1.
GRADE_COLS = ["G1", "G2"]
#: Никогда не признак — из неё сделана целевая переменная.
TARGET_SOURCE = "G3"


def engineer(X, level="L2", fe=True, exclude=(), keep_only=None):
    """Построчная инженерия признаков. Никакой статистики по выборке -> утечки нет."""
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

    if fe:
        d["alc_total"] = d["Dalc"] + d["Walc"]
        d["alc_weekday_share"] = d["Dalc"] / (d["alc_total"] + 1e-9)
        d["n_support"] = d["schoolsup"] + d["famsup"] + d["paid"]
        d["any_support"] = (d["n_support"] > 0).astype(int)
        d["abs_per_study"] = d["absences"] / d["studytime"]
        d["log_absences"] = np.log1p(d["absences"])
        d["abs_zero"] = (d["absences"] == 0).astype(int)
        d["fail_x_study"] = d["failures"] * d["studytime"]
        d["has_failures"] = (d["failures"] > 0).astype(int)
        d["parent_edu_max"] = d[["Medu", "Fedu"]].max(axis=1)
        d["parent_edu_mean"] = d[["Medu", "Fedu"]].mean(axis=1)
        d["age_over_17"] = (d["age"] > 17).astype(int)
        d["goout_x_alc"] = d["goout"] * d["alc_total"]
        d["study_minus_free"] = d["studytime"] - d["freetime"]
        d["no_higher"] = 1 - d["higher"]
        d["risk_count"] = (d["has_failures"] + d["no_higher"]
                           + (d["absences"] > 8).astype(int)
                           + (d["studytime"] <= 1).astype(int))
        # границы бинов ФИКСИРОВАНЫ по смыслу, не по квантилям выборки
        d["abs_bin"] = pd.cut(d["absences"], [-1, 0, 2, 6, 12, 1e9], labels=False).astype(float)
        d["age_bin"] = pd.cut(d["age"], [0, 16, 17, 18, 1e9], labels=False).astype(float)

        if level == "L1":
            d["G_diff"] = d["G2"] - d["G1"]
            d["G_mean"] = (d["G1"] + d["G2"]) / 2
            d["G_min"] = d[["G1", "G2"]].min(axis=1)
            d["G_proj"] = d["G2"] + (d["G2"] - d["G1"])
            d["G1_fail"] = (d["G1"] < 10).astype(int)
            d["G2_fail"] = (d["G2"] < 10).astype(int)
            d["G_both_fail"] = d["G1_fail"] * d["G2_fail"]
            d["G_declining"] = (d["G_diff"] < 0).astype(int)
        elif level == "L1a":
            d["G1_fail"] = (d["G1"] < 10).astype(int)
            d["G1_margin"] = d["G1"] - 10

    if keep_only is not None:
        d = d[[c for c in d.columns if c in set(keep_only)]]
    if exclude:
        d = d.drop(columns=[c for c in exclude if c in d.columns])
    return d


ENG_COMMON = ["alc_total", "alc_weekday_share", "n_support", "any_support",
              "abs_per_study", "log_absences", "abs_zero", "fail_x_study",
              "has_failures", "parent_edu_max", "parent_edu_mean", "age_over_17",
              "goout_x_alc", "study_minus_free", "no_higher", "risk_count",
              "abs_bin", "age_bin"]
ENG_L1 = ["G_diff", "G_mean", "G_min", "G_proj", "G1_fail", "G2_fail",
          "G_both_fail", "G_declining"]


def eng_cols(level):
    return ENG_COMMON + (ENG_L1 if level == "L1" else
                         ["G1_fail", "G1_margin"] if level == "L1a" else [])


# --------------------------------------------------------------------------
# Пайплайн
# --------------------------------------------------------------------------
def make_fe(level, fe=True, exclude=(), keep_only=None):
    return FunctionTransformer(partial(engineer, level=level, fe=fe,
                                       exclude=tuple(exclude), keep_only=keep_only),
                               validate=False)


def make_prep(scale=True):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(num_steps), make_column_selector(dtype_exclude=object)),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
          make_column_selector(dtype_include=object))],
        remainder="drop", verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_pipe(model, level="L2", scale=True, fe=True, exclude=(), keep_only=None):
    """FE -> препроцессинг -> модель. Всё, что требует fit, фитится на train-фолде."""
    return Pipeline([("fe", make_fe(level, fe, exclude, keep_only)),
                     ("prep", make_prep(scale=scale)),
                     ("clf", model)])


def cv_obj(n_splits=5, n_repeats=5, seed=SEED):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def evaluate(pipe, X, y, cv=None, n_jobs=-1):
    cv = cv if cv is not None else cv_obj()
    r = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "average_precision"],
                       n_jobs=n_jobs, error_score="raise")
    return {"roc_auc": r["test_roc_auc"].mean(), "roc_auc_std": r["test_roc_auc"].std(),
            "pr_auc": r["test_average_precision"].mean(),
            "pr_auc_std": r["test_average_precision"].std()}


# --------------------------------------------------------------------------
# Финальный ансамбль (раздел 5.2 и 6.2 ноутбука)
# --------------------------------------------------------------------------
ET_TUNED = {"n_estimators": 866, "max_features": 0.1267, "min_samples_leaf": 3,
            "min_samples_split": 13, "criterion": "gini", "class_weight": "balanced"}
RF_TUNED = {"n_estimators": 774, "max_features": 0.3124, "min_samples_leaf": 9,
            "min_samples_split": 18, "class_weight": "balanced_subsample"}
LR_TUNED = {"C": 0.0292}
SVM_TUNED = {"C": 0.3149, "gamma": 0.0711, "class_weight": "balanced"}

#: Состав ансамбля различается по уровням: параметры SVM подбирались на
#: пространстве признаков L2 и на L1 тянут ансамбль вниз (раздел 6.2 ноутбука).
FINAL_MEMBERS = {"L1": ["et", "rf", "lr"], "L1a": ["et", "rf", "lr"],
                 "L2": ["et", "lr", "svm"]}


def member(name, level, exclude):
    if name == "et":
        return make_pipe(ExtraTreesClassifier(random_state=SEED, n_jobs=1, **ET_TUNED),
                         level, scale=False, exclude=exclude)
    if name == "rf":
        return make_pipe(RandomForestClassifier(random_state=SEED, n_jobs=1, **RF_TUNED),
                         level, scale=False, exclude=exclude)
    if name == "lr":
        return make_pipe(LogisticRegression(max_iter=8000, class_weight="balanced",
                                            penalty="l1", solver="liblinear",
                                            random_state=SEED, **LR_TUNED),
                         level, scale=True, exclude=exclude)
    if name == "svm":
        return make_pipe(SVC(probability=True, random_state=SEED, **SVM_TUNED),
                         level, scale=True, exclude=exclude)
    raise ValueError(f"неизвестный член ансамбля: {name}")


def final_model(level, exclude):
    """Мягкое голосование трёх разнородных моделей — как в разделе 6.2 ноутбука."""
    return VotingClassifier([(k, member(k, level, exclude)) for k in FINAL_MEMBERS[level]],
                            voting="soft")


def get_Xy(df):
    """Признаки = всё, кроме G3 и no_pass. G1/G2 отсекаются позже, в engineer()."""
    X = df.drop(columns=[c for c in (TARGET_SOURCE, "no_pass") if c in df.columns])
    y = (df[TARGET_SOURCE] < 10).astype(int).to_numpy()
    return X, y
