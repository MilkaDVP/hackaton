"""Эксперимент 9: выбор финальной конфигурации на уровне L2.

Всё считается на ОДНОЙ И ТОЙ ЖЕ 5x5 CV, чтобы сравнение было корректным,
плюс вложенная CV для тюнингованного варианта — как честная оценка.
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from common import load, get_Xy, make_pipe, evaluate, cv_obj, fmt, SEED, DATA_POR, eng_cols

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "L2"

df = load(DATA_POR)
X, y = get_Xy(df)
CV = cv_obj(5, 5, SEED)

# признаки, которые leave-one-out пометил как бесполезные (из раздела 4.3)
DROP_L2 = ("fail_x_study", "n_support", "any_support", "log_absences", "parent_edu_max",
           "study_minus_free", "age_bin", "age_over_17", "abs_bin", "goout_x_alc",
           "abs_zero", "abs_per_study")

# найдено RandomizedSearchCV в exp04/ноутбуке
ET_TUNED = dict(n_estimators=866, max_features=0.1267, min_samples_leaf=3,
                min_samples_split=13, criterion="gini", class_weight="balanced")

ET = lambda **kw: ExtraTreesClassifier(random_state=SEED, n_jobs=1,
                                       **{**dict(n_estimators=500, class_weight="balanced"), **kw})
RF = lambda **kw: RandomForestClassifier(random_state=SEED, n_jobs=1,
                                         **{**dict(n_estimators=500, class_weight="balanced"), **kw})
LR = lambda: LogisticRegression(max_iter=8000, class_weight="balanced", penalty="l1",
                                solver="liblinear", C=0.3, random_state=SEED)
LRT = lambda: LogisticRegression(max_iter=8000, class_weight="balanced", penalty="l1",
                                 solver="liblinear", C=0.0292, random_state=SEED)
SV = lambda: SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)
HG = lambda: HistGradientBoostingClassifier(random_state=SEED, max_iter=200, learning_rate=0.06,
                                            max_leaf_nodes=15, l2_regularization=1.0)


def voting(members):
    return VotingClassifier(members, voting="soft")


CONFIGS = {}

# --- базовые точки отсчёта ---
CONFIGS["0. baseline стартера (get_dummies + RF400)"] = ("SPECIAL_BASELINE", None)
CONFIGS["1. без FE, ExtraTrees"] = (make_pipe(ET(), LEVEL, scale=False, fe=False), None)
CONFIGS["2. без FE, RF"] = (make_pipe(RF(), LEVEL, scale=False, fe=False), None)
CONFIGS["3. все FE, ExtraTrees"] = (make_pipe(ET(), LEVEL, scale=False), None)
CONFIGS["4. чищеные FE, ExtraTrees"] = (make_pipe(ET(), LEVEL, scale=False, exclude=DROP_L2), None)
CONFIGS["5. чищеные FE, RF"] = (make_pipe(RF(), LEVEL, scale=False, exclude=DROP_L2), None)

# --- тюнингованные ---
CONFIGS["6. все FE, ExtraTrees тюн."] = (make_pipe(ET(**ET_TUNED), LEVEL, scale=False), None)
CONFIGS["7. чищеные FE, ExtraTrees тюн."] = (
    make_pipe(ET(**ET_TUNED), LEVEL, scale=False, exclude=DROP_L2), None)
CONFIGS["8. без FE, ExtraTrees тюн."] = (
    make_pipe(ET(**ET_TUNED), LEVEL, scale=False, fe=False), None)

# --- ансамбли на чищеных признаках ---
def members(exclude, tuned):
    et = ET(**ET_TUNED) if tuned else ET()
    return [
        ("et", make_pipe(et, LEVEL, scale=False, exclude=exclude)),
        ("rf", make_pipe(RF(), LEVEL, scale=False, exclude=exclude)),
        ("lr", make_pipe(LRT() if tuned else LR(), LEVEL, scale=True, exclude=exclude)),
        ("svm", make_pipe(SV(), LEVEL, scale=True, exclude=exclude)),
        ("hgb", make_pipe(HG(), LEVEL, scale=False, exclude=exclude)),
    ]

CONFIGS["9. ансамбль, все FE"] = (voting(members((), False)), None)
CONFIGS["10. ансамбль, чищеные FE"] = (voting(members(DROP_L2, False)), None)
CONFIGS["11. ансамбль тюн., чищеные FE"] = (voting(members(DROP_L2, True)), None)
CONFIGS["12. ансамбль тюн. без SVM/HGB"] = (
    voting([m for m in members(DROP_L2, True) if m[0] in ("et", "rf", "lr")]), None)
CONFIGS["13. ансамбль ET+LR тюн."] = (
    voting([m for m in members(DROP_L2, True) if m[0] in ("et", "lr")]), None)

rows = []
for name, (pipe, _) in CONFIGS.items():
    t0 = time.time()
    if pipe == "SPECIAL_BASELINE":
        Xb = pd.get_dummies(df.drop(columns=["G3", "no_pass"]), drop_first=True)
        if LEVEL == "L2":
            Xb = Xb.drop(columns=["G1", "G2"])
        r = evaluate(RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                            random_state=SEED), Xb, y, cv=CV)
    else:
        r = evaluate(pipe, X, y, cv=CV)
    rows.append(dict(конфигурация=name, roc=r["roc_auc"], roc_sd=r["roc_auc_std"],
                     pr=r["pr_auc"], pr_sd=r["pr_auc_std"], sec=time.time() - t0))
    print(f"{name:38s} {fmt(r)}  [{time.time()-t0:.0f}s]")

T = pd.DataFrame(rows).sort_values("roc", ascending=False)
T.to_csv(os.path.join(HERE, f"out_09_final_{LEVEL}.csv"), index=False, encoding="utf-8")
print("\n" + "=" * 78)
print(T.round(4).to_string(index=False))
