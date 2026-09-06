"""Эксперимент 1: зоопарк моделей на обоих уровнях."""
import sys, io, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from common import (load, get_Xy, make_pipe, evaluate, cv_obj, fmt, SEED, DATA_POR)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

HAS = {}
try:
    from lightgbm import LGBMClassifier; HAS["lgbm"] = True
except Exception: HAS["lgbm"] = False
try:
    from xgboost import XGBClassifier; HAS["xgb"] = True
except Exception: HAS["xgb"] = False
try:
    from catboost import CatBoostClassifier; HAS["cat"] = True
except Exception: HAS["cat"] = False
print("опциональные библиотеки:", HAS)


def zoo():
    m = {
        "константа (нижняя граница)": (DummyClassifier(strategy="prior"), True),
        "логрег L2":       (LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=SEED), True),
        "логрег L1":       (LogisticRegression(max_iter=5000, class_weight="balanced", penalty="l1", solver="liblinear", C=0.3, random_state=SEED), True),
        "логрег elastic":  (LogisticRegression(max_iter=5000, class_weight="balanced", penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.3, random_state=SEED), True),
        "случайный лес":   (RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=1), False),
        "ExtraTrees":      (ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=1), False),
        "HistGB":          (HistGradientBoostingClassifier(random_state=SEED, max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0), False),
        "GradientBoosting":(GradientBoostingClassifier(random_state=SEED, n_estimators=200, learning_rate=0.05, max_depth=3), False),
        "SVM RBF":         (SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=SEED), True),
        "kNN (sanity)":    (KNeighborsClassifier(n_neighbors=25, weights="distance"), True),
        "GaussianNB":      (GaussianNB(), True),
    }
    if HAS["lgbm"]:
        m["LightGBM"] = (LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                        class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=1), False)
    if HAS["xgb"]:
        m["XGBoost"] = (XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                      random_state=SEED, n_jobs=1, scale_pos_weight=5.5), False)
    if HAS["cat"]:
        m["CatBoost"] = (CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4,
                                            verbose=0, random_seed=SEED,
                                            allow_writing_files=False,
                                            auto_class_weights="Balanced"), False)
    return m


def main():
    df = load(DATA_POR)
    X, y = get_Xy(df)
    cv = cv_obj(5, 5, SEED)
    rows = []
    for level in ["L1", "L2"]:
        print(f"\n{'='*70}\nУРОВЕНЬ {level}\n{'='*70}")
        for name, (model, scale) in zoo().items():
            t = time.time()
            try:
                r = evaluate(make_pipe(model, level=level, scale=scale), X, y, cv=cv)
            except Exception as e:
                print(f"{name:28s} ОШИБКА: {e}"); continue
            rows.append(dict(level=level, model=name, **{k: v for k, v in r.items() if not k.startswith("_")},
                             sec=time.time() - t))
            print(f"{name:28s} {fmt(r)}  [{time.time()-t:.1f}s]")
    out = pd.DataFrame(rows)
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_01_models.csv")
    out.to_csv(p, index=False, encoding="utf-8")
    print("\nсохранено ->", p)


if __name__ == "__main__":
    main()
