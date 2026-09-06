"""Эксперимент 4: тюнинг базовых моделей + ансамбли. Уровень L2 — главный.

Протокол:
  * поиск гиперпараметров — RandomizedSearchCV на ВНУТРЕННЕЙ CV;
  * честная оценка — ВЛОЖЕННАЯ (nested) CV: внешний RepeatedStratifiedKFold,
    внутри каждого внешнего фолда поиск запускается заново.
"""
import sys, time, json, os
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform

from common import (load, get_Xy, make_pipe, evaluate, cv_obj, fmt, SEED, DATA_POR)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "L2"
N_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 40

df = load(DATA_POR)
X, y = get_Xy(df)

SPACES = {
    "ExtraTrees": (
        ExtraTreesClassifier(random_state=SEED, n_jobs=1, class_weight="balanced"), False,
        {"clf__n_estimators": randint(300, 900),
         "clf__max_features": uniform(0.05, 0.55),
         "clf__min_samples_leaf": randint(1, 12),
         "clf__min_samples_split": randint(2, 20),
         "clf__criterion": ["gini", "entropy"],
         "clf__class_weight": ["balanced", "balanced_subsample"]}),
    "СлучайныйЛес": (
        RandomForestClassifier(random_state=SEED, n_jobs=1, class_weight="balanced"), False,
        {"clf__n_estimators": randint(300, 900),
         "clf__max_features": uniform(0.05, 0.55),
         "clf__min_samples_leaf": randint(1, 12),
         "clf__min_samples_split": randint(2, 20),
         "clf__criterion": ["gini", "entropy"],
         "clf__class_weight": ["balanced", "balanced_subsample"]}),
    "ЛогрегL1": (
        LogisticRegression(max_iter=8000, class_weight="balanced", penalty="l1",
                           solver="liblinear", random_state=SEED), True,
        {"clf__C": loguniform(1e-3, 10)}),
    "SVM_RBF": (
        SVC(class_weight="balanced", random_state=SEED, probability=False), True,
        {"clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-4, 1e-1)}),
    "HistGB": (
        HistGradientBoostingClassifier(random_state=SEED), False,
        {"clf__learning_rate": loguniform(0.01, 0.3),
         "clf__max_iter": randint(100, 500),
         "clf__max_leaf_nodes": randint(3, 31),
         "clf__min_samples_leaf": randint(5, 40),
         "clf__l2_regularization": loguniform(1e-3, 10),
         "clf__max_features": uniform(0.3, 0.7)}),
    "kNN": (
        KNeighborsClassifier(), True,
        {"clf__n_neighbors": randint(5, 80),
         "clf__weights": ["uniform", "distance"],
         "clf__p": [1, 2]}),
}

inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
outer = cv_obj(5, 3, SEED)          # 15 внешних фолдов — компромисс по времени

results, best_params = [], {}
for name, (model, scale, space) in SPACES.items():
    pipe = make_pipe(model, level=LEVEL, scale=scale)
    search = RandomizedSearchCV(pipe, space, n_iter=N_ITER, scoring="roc_auc",
                                cv=inner, random_state=SEED, n_jobs=-1, refit=True)
    t = time.time()
    # 1) честная вложенная оценка
    nested = cross_validate(search, X, y, cv=outer,
                            scoring=["roc_auc", "average_precision"], n_jobs=1)
    # 2) сам поиск на всех данных — чтобы достать параметры для ноутбука
    search.fit(X, y)
    bp = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    best_params[name] = bp
    row = dict(model=name,
               nested_roc=nested["test_roc_auc"].mean(), nested_roc_sd=nested["test_roc_auc"].std(),
               nested_pr=nested["test_average_precision"].mean(), nested_pr_sd=nested["test_average_precision"].std(),
               inner_best=search.best_score_, sec=time.time() - t)
    results.append(row)
    print(f"{name:14s} nested ROC {row['nested_roc']:.3f} ± {row['nested_roc_sd']:.3f} | "
          f"nested PR {row['nested_pr']:.3f} ± {row['nested_pr_sd']:.3f} | "
          f"внутр.best {search.best_score_:.3f} | {row['sec']:.0f}s")
    print(f"{'':14s} {bp}")

pd.DataFrame(results).to_csv(os.path.join(HERE, f"out_04_tune_{LEVEL}.csv"),
                             index=False, encoding="utf-8")
with open(os.path.join(HERE, f"out_04_params_{LEVEL}.json"), "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=False, indent=2)
print("\nсохранено:", f"out_04_tune_{LEVEL}.csv", f"out_04_params_{LEVEL}.json")
print("\nРАЗРЫВ внутренний-vs-nested показывает, насколько оптимистичен best_score_:")
for r in results:
    print(f"  {r['model']:14s} {r['inner_best']:.3f} -> {r['nested_roc']:.3f} "
          f"(оптимизм {r['inner_best']-r['nested_roc']:+.3f})")
