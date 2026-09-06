"""Эксперимент 6: перенос на математику + ловушка «382 студента»."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from common import (load, get_Xy, make_pipe, evaluate, cv_obj, fmt, SEED,
                    DATA_POR, DATA_MAT)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

pd.set_option("display.width", 220)

KEY = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
       "Mjob", "Fjob", "reason", "nursery", "internet"]

por, mat = load(DATA_POR), load(DATA_MAT)
print(f"por {por.shape}  доля незачётов {por.no_pass.mean():.4f}")
print(f"mat {mat.shape}  доля незачётов {mat.no_pass.mean():.4f}")

# ---------------------------------------------------------------- ловушка
print("\n" + "=" * 78)
print("ЛОВУШКА: сколько на самом деле общих студентов?")
print("=" * 78)

naive = por.merge(mat, on=KEY, suffixes=("_por", "_mat"))
print(f"pd.merge(por, mat, on=KEY) даёт строк: {len(naive)}   <- та самая «382»")

kp = por[KEY].apply(tuple, axis=1)
km = mat[KEY].apply(tuple, axis=1)
cp, cm = kp.value_counts(), km.value_counts()
common = set(cp.index) & set(cm.index)
print(f"уникальных ключей в por: {kp.nunique()} (строк {len(por)})")
print(f"уникальных ключей в mat: {km.nunique()} (строк {len(mat)})")
print(f"ОБЩИХ уникальных ключей: {len(common)}")
print(f"строк por, попавших в пересечение: {kp.isin(common).sum()}")
print(f"строк mat, попавших в пересечение: {km.isin(common).sum()}")
print(f"неуникальных ключей: в por {int((cp > 1).sum())}, в mat {int((cm > 1).sum())}")
print(f"проверка декартова произведения: sum(cp*cm) = "
      f"{sum(int(cp[k]) * int(cm[k]) for k in common)}")
print("\nВЫВОД: 382 — артефакт декартова произведения при мёрдже по неуникальному ключу.")
print(f"Реально пересекаются {len(common)} студентов; в математике вне пересечения "
      f"остаётся всего {len(mat) - int(km.isin(common).sum())} человек.")

por_ov = kp.isin(common).to_numpy()
mat_ov = km.isin(common).to_numpy()

# --------------------------------------------- один человек, два предмета
uniq = {k for k in common if cp[k] == 1 and cm[k] == 1}
pi = por[kp.isin(uniq)].copy(); pi["_k"] = kp[kp.isin(uniq)]
mi = mat[km.isin(uniq)].copy(); mi["_k"] = km[km.isin(uniq)]
pair = pi.merge(mi, on="_k", suffixes=("_p", "_m"))
print("\n" + "=" * 78)
print(f"ОДИН ЧЕЛОВЕК — ДВА ПРЕДМЕТА (n={len(pair)} с однозначным сопоставлением)")
print("=" * 78)
print(f"corr(G3_por, G3_mat) = {pair.G3_p.corr(pair.G3_m):.3f}")
ct = pd.crosstab(pair.no_pass_p, pair.no_pass_m)
ct.index.name = "незачёт por"; ct.columns.name = "незачёт mat"
print(ct.to_string())
both = ((pair.no_pass_p == 1) & (pair.no_pass_m == 1)).sum()
only_m = ((pair.no_pass_p == 0) & (pair.no_pass_m == 1)).sum()
print(f"\nнезачёт по mat: {int(pair.no_pass_m.sum())}, из них уже валили por: {int(both)}"
      f" ({both / max(pair.no_pass_m.sum(),1):.0%})")
print(f"«внезапные» незачёты по mat (por сдан): {int(only_m)}")
print(f"P(незачёт mat | незачёт por) = "
      f"{both / max((pair.no_pass_p==1).sum(),1):.2f}   "
      f"P(незачёт mat | зачёт por) = {only_m / max((pair.no_pass_p==0).sum(),1):.2f}")

# ---------------------------------------------------------------- перенос
def fit_eval(train_df, test_df, level, model, scale, tag):
    Xtr, ytr = get_Xy(train_df)
    Xte, yte = get_Xy(test_df)
    p = make_pipe(model, level=level, scale=scale)
    p.fit(Xtr, ytr)
    s = p.predict_proba(Xte)[:, 1]
    return dict(tag=tag, level=level, n_train=len(train_df), n_test=len(test_df),
                roc=roc_auc_score(yte, s), pr=average_precision_score(yte, s),
                base=yte.mean())

MODEL = lambda: HistGradientBoostingClassifier(random_state=SEED, max_iter=200,
                                               learning_rate=0.06, max_leaf_nodes=15,
                                               l2_regularization=1.0)
LOGREG = lambda: LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0,
                                    random_state=SEED)

por_clean = por[~por_ov].reset_index(drop=True)     # por без пересечения
mat_clean = mat[~mat_ov].reset_index(drop=True)     # mat без пересечения

print("\n" + "=" * 78)
print("ПЕРЕНОС por -> mat: наивная оценка против честной")
print("=" * 78)
rows = []
for level in ["L1", "L2"]:
    for mname, mk, sc in [("HistGB", MODEL, False), ("логрег", LOGREG, True)]:
        rows.append(dict(model=mname, **fit_eval(por, mat, level, mk(), sc,
                                                 "НАИВНО: весь por -> весь mat")))
        rows.append(dict(model=mname, **fit_eval(por_clean, mat, level, mk(), sc,
                                                 "ЧЕСТНО: por без пересечения -> весь mat")))
        rows.append(dict(model=mname, **fit_eval(por_clean, mat_clean, level, mk(), sc,
                                                 "ОЧЕНЬ ЧЕСТНО: por без перес. -> mat без перес.")))
        rows.append(dict(model=mname, **fit_eval(por, mat[mat_ov].reset_index(drop=True),
                                                 level, mk(), sc,
                                                 "ЗАГРЯЗНЕНО: весь por -> только пересечение mat")))
t = pd.DataFrame(rows)[["model", "level", "tag", "n_train", "n_test", "base", "roc", "pr"]]
print(t.round(3).to_string(index=False))
t.to_csv("out_06_transfer.csv", index=False, encoding="utf-8")

# внутренняя CV на самой математике — «потолок» для сравнения
print("\nдля сравнения — CV внутри математики (обучение и тест на mat):")
Xm, ym = get_Xy(mat)
for level in ["L1", "L2"]:
    r = evaluate(make_pipe(MODEL(), level=level, scale=False), Xm, ym, cv=cv_obj(5, 5, SEED))
    print(f"  mat CV {level}: {fmt(r)}")
