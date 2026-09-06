"""Эксперимент 3: что делать с G3 == 0 (15 студентов). Sensitivity analysis."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from common import load, get_Xy, make_pipe, evaluate, cv_obj, fmt, SEED, DATA_POR

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict, StratifiedKFold

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = load(DATA_POR)
z = df[df.G3 == 0]
print(f"Студентов с G3 == 0: {len(z)}\n")
print(z[["G1", "G2", "G3", "absences", "studytime", "failures", "higher", "schoolsup"]].to_string())

print("\n--- сравнение групп ---")
grp = pd.DataFrame({
    "G3==0 (n=%d)" % len(z): z[["G1", "G2", "absences", "failures", "studytime", "goout"]].mean(),
    "незачёт, G3>0 (n=%d)" % ((df.no_pass == 1) & (df.G3 > 0)).sum():
        df[(df.no_pass == 1) & (df.G3 > 0)][["G1", "G2", "absences", "failures", "studytime", "goout"]].mean(),
    "зачёт (n=%d)" % (df.no_pass == 0).sum():
        df[df.no_pass == 0][["G1", "G2", "absences", "failures", "studytime", "goout"]].mean(),
}).round(2)
print(grp.to_string())

print("\nиз них имели проходной балл на 2-й контрольной (G2>=10):", int((z.G2 >= 10).sum()))
print("имели G2 == 0 (уже пропали к 2-й контрольной):", int((z.G2 == 0).sum()))
print("имели G1 == 0:", int((z.G1 == 0).sum()))

# ---------------------------------------------------------------- метрики
models = {
    "логрег": (LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=SEED), True),
    "HistGB": (HistGradientBoostingClassifier(random_state=SEED, max_iter=200, learning_rate=0.06,
                                              max_leaf_nodes=15, l2_regularization=1.0), False),
    "лес":    (RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=1), False),
}

print("\n" + "=" * 78)
print("МЕТРИКИ: с 15 студентами против без них")
print("=" * 78)
rows = []
for variant, d in [("оставляем (n=%d)" % len(df), df),
                   ("убираем (n=%d)" % (len(df) - len(z)), df[df.G3 > 0].reset_index(drop=True))]:
    X, y = get_Xy(d)
    for level in ["L1", "L2"]:
        for mname, (m, sc) in models.items():
            r = evaluate(make_pipe(m, level=level, scale=sc), X, y, cv=cv_obj(5, 5, SEED))
            rows.append(dict(variant=variant, level=level, model=mname,
                             roc=r["roc_auc"], roc_sd=r["roc_auc_std"],
                             pr=r["pr_auc"], pr_sd=r["pr_auc_std"],
                             pos_rate=y.mean()))
            print(f"{variant:22s} {level:3s} {mname:8s} {fmt(r)}  (доля незач. {y.mean():.3f})")

t = pd.DataFrame(rows)
t.to_csv("out_03_g3zero.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- топ-3
print("\n" + "=" * 78)
print("МЕНЯЕТСЯ ЛИ ТОП-3 (permutation importance, уровень L2)")
print("=" * 78)
for variant, d in [("оставляем", df), ("убираем", df[df.G3 > 0].reset_index(drop=True))]:
    X, y = get_Xy(d)
    pipe = make_pipe(RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1), level="L2")
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=SEED)
    pipe.fit(Xtr, ytr)
    r = permutation_importance(pipe, Xte, yte, n_repeats=30, random_state=SEED, scoring="roc_auc", n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=Xte.columns).sort_values(ascending=False)
    print(f"{variant:12s} -> {list(imp.head(5).index)}")
    print(f"{'':12s}    {imp.head(5).round(4).to_dict()}")
