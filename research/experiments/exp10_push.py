"""Эксперимент 10: последняя честная попытка поднять L2.

Тюним оставшихся членов ансамбля на чищеных признаках, потом проверяем
итоговую конфигурацию ВЛОЖЕННОЙ CV — чтобы отчётное число не было
результатом подгонки под ту же кросс-валидацию.
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from common import load, get_Xy, make_pipe, evaluate, cv_obj, fmt, SEED, DATA_POR

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.feature_selection import RFECV

HERE = os.path.dirname(os.path.abspath(__file__))
df = load(DATA_POR)
X, y = get_Xy(df)
CV = cv_obj(5, 5, SEED)
inner = StratifiedKFold(5, shuffle=True, random_state=SEED)

DROP_L2 = ("fail_x_study", "n_support", "any_support", "log_absences", "parent_edu_max",
           "study_minus_free", "age_bin", "age_over_17", "abs_bin", "goout_x_alc",
           "abs_zero", "abs_per_study")
ET_TUNED = dict(n_estimators=866, max_features=0.1267, min_samples_leaf=3,
                min_samples_split=13, criterion="gini", class_weight="balanced")

# ---------------------------------------------------------------- тюним RF и SVM
SPACES = {
    "RF": (RandomForestClassifier(random_state=SEED, n_jobs=1), False,
           {"clf__n_estimators": randint(300, 900),
            "clf__max_features": uniform(0.05, 0.5),
            "clf__min_samples_leaf": randint(1, 10),
            "clf__min_samples_split": randint(2, 20),
            "clf__class_weight": ["balanced", "balanced_subsample"]}),
    "SVM": (SVC(probability=True, random_state=SEED), True,
            {"clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-4, 1e-1),
             "clf__class_weight": ["balanced"]}),
}
FOUND = {}
for name, (m, sc, sp) in SPACES.items():
    t0 = time.time()
    s = RandomizedSearchCV(make_pipe(m, "L2", scale=sc, exclude=DROP_L2), sp, n_iter=30,
                           scoring="roc_auc", cv=inner, n_jobs=-1, random_state=SEED)
    s.fit(X, y)
    FOUND[name] = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                   for k, v in s.best_params_.items()}
    print(f"{name}: внутр. best {s.best_score_:.4f}  [{time.time()-t0:.0f}s]  {FOUND[name]}")

RF_T = lambda: RandomForestClassifier(random_state=SEED, n_jobs=1, **FOUND["RF"])
SV_T = lambda: SVC(probability=True, random_state=SEED, **FOUND["SVM"])
ET_T = lambda: ExtraTreesClassifier(random_state=SEED, n_jobs=1, **ET_TUNED)
LR_T = lambda: LogisticRegression(max_iter=8000, class_weight="balanced", penalty="l1",
                                  solver="liblinear", C=0.0292, random_state=SEED)

def M(keys):
    all_m = {"et": make_pipe(ET_T(), "L2", scale=False, exclude=DROP_L2),
             "rf": make_pipe(RF_T(), "L2", scale=False, exclude=DROP_L2),
             "lr": make_pipe(LR_T(), "L2", scale=True, exclude=DROP_L2),
             "svm": make_pipe(SV_T(), "L2", scale=True, exclude=DROP_L2)}
    return [(k, all_m[k]) for k in keys]

CAND = {
    "ET+LR": VotingClassifier(M(["et", "lr"]), voting="soft"),
    "ET+RF+LR": VotingClassifier(M(["et", "rf", "lr"]), voting="soft"),
    "ET+RF+LR+SVM": VotingClassifier(M(["et", "rf", "lr", "svm"]), voting="soft"),
    "ET+LR+SVM": VotingClassifier(M(["et", "lr", "svm"]), voting="soft"),
    "ET+RF+LR (веса 2,1,1)": VotingClassifier(M(["et", "rf", "lr"]), voting="soft", weights=[2, 1, 1]),
}
rows = []
for name, ens in CAND.items():
    t0 = time.time()
    r = evaluate(ens, X, y, cv=CV)
    rows.append(dict(конфиг=name, roc=r["roc_auc"], roc_sd=r["roc_auc_std"],
                     pr=r["pr_auc"], pr_sd=r["pr_auc_std"]))
    print(f"{name:26s} {fmt(r)}  [{time.time()-t0:.0f}s]")

T = pd.DataFrame(rows).sort_values("roc", ascending=False)
print("\n" + T.round(4).to_string(index=False))
T.to_csv(os.path.join(HERE, "out_10_push.csv"), index=False, encoding="utf-8")
with open(os.path.join(HERE, "out_10_params.json"), "w", encoding="utf-8") as f:
    json.dump({"ET": ET_TUNED, **FOUND, "LR": {"C": 0.0292}, "drop": list(DROP_L2)},
              f, ensure_ascii=False, indent=2)
print("\nпараметры сохранены -> out_10_params.json")
