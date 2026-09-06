"""Эксперимент 2: вклад инженерных признаков. Таблица «до / после» + leave-one-out."""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from common import (load, get_Xy, make_pipe, evaluate, cv_obj, fmt, SEED,
                    DATA_POR, eng_cols)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "L2"

df = load(DATA_POR)
X, y = get_Xy(df)
cv = cv_obj(5, 3, SEED)

ET = lambda: ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=SEED, n_jobs=1)
LR = lambda: LogisticRegression(max_iter=8000, class_weight="balanced",
                                penalty="l1", solver="liblinear", C=0.3, random_state=SEED)
MODELS = [("ExtraTrees", ET, False), ("ЛогрегL1", LR, True)]

print(f"=== УРОВЕНЬ {LEVEL}: до / после инженерии признаков ===")
base = {}
for mn, mk, sc in MODELS:
    r0 = evaluate(make_pipe(mk(), level=LEVEL, scale=sc, fe=False), X, y, cv=cv, n_jobs=-1)
    r1 = evaluate(make_pipe(mk(), level=LEVEL, scale=sc, fe=True), X, y, cv=cv, n_jobs=-1)
    base[mn] = r1
    print(f"{mn:12s} без FE: {fmt(r0)}")
    print(f"{mn:12s} с  FE: {fmt(r1)}   Δroc {r1['roc_auc']-r0['roc_auc']:+.4f}  "
          f"Δpr {r1['pr_auc']-r0['pr_auc']:+.4f}")

print(f"\n=== leave-one-out по инженерным признакам ({LEVEL}) ===")
print("Δ = метрика БЕЗ признака минус метрика СО всеми. Δ<0 -> признак полезен.")
rows = []
cols = eng_cols(LEVEL)
for c in cols:
    rec = {"feature": c}
    for mn, mk, sc in MODELS:
        r = evaluate(make_pipe(mk(), level=LEVEL, scale=sc, fe=True, exclude=(c,)),
                     X, y, cv=cv, n_jobs=-1)
        rec[f"{mn}_droc"] = r["roc_auc"] - base[mn]["roc_auc"]
        rec[f"{mn}_dpr"] = r["pr_auc"] - base[mn]["pr_auc"]
    rec["mean_droc"] = np.mean([rec[f"{m[0]}_droc"] for m in MODELS])
    rows.append(rec)
    print(f"  {c:20s} " + "  ".join(
        f"{m[0]}: Δroc {rec[f'{m[0]}_droc']:+.4f} Δpr {rec[f'{m[0]}_dpr']:+.4f}" for m in MODELS))

t = pd.DataFrame(rows).sort_values("mean_droc", ascending=False)
t.to_csv(os.path.join(HERE, f"out_02_features_{LEVEL}.csv"), index=False, encoding="utf-8")
print("\n--- кандидаты на удаление (без них метрика РАСТЁТ, mean_droc > 0) ---")
print(t[t.mean_droc > 0][["feature", "mean_droc"]].round(4).to_string(index=False))

# --- проверяем удаление всей группы бесполезных сразу ---
drop = tuple(t[t.mean_droc > 0].feature)
if drop:
    print(f"\n=== убираем разом {len(drop)} признаков: {list(drop)} ===")
    for mn, mk, sc in MODELS:
        r = evaluate(make_pipe(mk(), level=LEVEL, scale=sc, fe=True, exclude=drop),
                     X, y, cv=cv, n_jobs=-1)
        print(f"{mn:12s} {fmt(r)}   Δroc {r['roc_auc']-base[mn]['roc_auc']:+.4f}  "
              f"Δpr {r['pr_auc']-base[mn]['pr_auc']:+.4f}")
