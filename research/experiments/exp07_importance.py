"""Эксперимент 7: устойчивость топ-3 по трём осям — сид, модель, метод.

Важно: permutation importance считается на ИСХОДНЫХ колонках (до FE).
Перемешивая, например, `absences`, мы автоматически ломаем и все производные
от неё признаки — то есть меряем полный вклад ПЕРЕМЕННОЙ, а не одной колонки.
"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from common import (load, get_Xy, make_pipe, evaluate, cv_obj, SEED, DATA_POR,
                    derived_of)

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "L2"
df = load(DATA_POR)
X, y = get_Xy(df)

FAMILIES = {
    "ExtraTrees": (lambda s: ExtraTreesClassifier(n_estimators=600, class_weight="balanced",
                                                  random_state=s, n_jobs=-1), False),
    "ЛогрегL1":   (lambda s: LogisticRegression(max_iter=8000, class_weight="balanced",
                                                penalty="l1", solver="liblinear", C=0.3,
                                                random_state=s), True),
    "HistGB":     (lambda s: HistGradientBoostingClassifier(random_state=s, max_iter=250,
                                                            learning_rate=0.06, max_leaf_nodes=15,
                                                            l2_regularization=1.0), False),
}
SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------- ось 1+2
print(f"=== {LEVEL}: permutation importance, {len(SEEDS)} сидов x {len(FAMILIES)} моделей ===")
recs = []
for fam, (mk, sc) in FAMILIES.items():
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=s)
        pipe = make_pipe(mk(s), level=LEVEL, scale=sc).fit(Xtr, ytr)
        r = permutation_importance(pipe, Xte, yte, n_repeats=25, random_state=s,
                                   scoring="roc_auc", n_jobs=-1)
        for col, val in zip(X.columns, r.importances_mean):
            recs.append(dict(family=fam, seed=s, method="permutation", feature=col, imp=val))
    print(f"  {fam}: готово")

P = pd.DataFrame(recs)
piv = P.pivot_table(index="feature", columns="family", values="imp", aggfunc="mean")
piv["среднее"] = piv.mean(axis=1)
piv = piv.sort_values("среднее", ascending=False)
print("\n--- средняя permutation importance (по 5 сидам) ---")
print(piv.head(10).round(4).to_string())

print("\n--- ТОП-3 в каждом отдельном запуске (видно, шатается ли тройка) ---")
for fam in FAMILIES:
    for s in SEEDS:
        top = (P[(P.family == fam) & (P.seed == s)]
               .nlargest(3, "imp").feature.tolist())
        print(f"  {fam:12s} seed={s}: {top}")

# ---------------------------------------------------------------- ось 3
print(f"\n=== метод 2: drop-column importance ({LEVEL}, 5x2 CV) ===")
cvd = cv_obj(5, 2, SEED)
et = lambda: ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=SEED, n_jobs=1)
full = evaluate(make_pipe(et(), level=LEVEL), X, y, cv=cvd, n_jobs=-1)
print(f"полная модель: ROC-AUC {full['roc_auc']:.4f}")
drop_rows = []
cands = [c for c in X.columns if c not in ("G1", "G2")] if LEVEL == "L2" else list(X.columns)
for col in cands:
    r = evaluate(make_pipe(et(), level=LEVEL, exclude=derived_of(col)), X, y, cv=cvd, n_jobs=-1)
    drop_rows.append(dict(feature=col, drop_loss=full["roc_auc"] - r["roc_auc"]))
D = pd.DataFrame(drop_rows).sort_values("drop_loss", ascending=False)
print(D.head(10).round(4).to_string(index=False))

# ---------------------------------------------------------------- метод 3
print(f"\n=== метод 3: коэффициенты логрегрессии (направление влияния) ===")
lr_pipe = make_pipe(LogisticRegression(max_iter=8000, class_weight="balanced",
                                       penalty="l1", solver="liblinear", C=0.3,
                                       random_state=SEED), level=LEVEL, scale=True).fit(X, y)
names = lr_pipe.named_steps["prep"].get_feature_names_out()
co = pd.Series(lr_pipe.named_steps["clf"].coef_[0], index=names)
co = co[co != 0].sort_values()
print("СНИЖАЮТ риск:"); print(co.head(8).round(3).to_string())
print("ПОВЫШАЮТ риск:"); print(co.tail(8).round(3).to_string())

# ---------------------------------------------------------------- SHAP
try:
    import shap
    print(f"\n=== метод 4: SHAP ({LEVEL}) ===")
    from sklearn.model_selection import train_test_split as tts
    Xtr, Xte, ytr, yte = tts(X, y, test_size=.3, stratify=y, random_state=SEED)
    pipe = make_pipe(et(), level=LEVEL).fit(Xtr, ytr)
    Zte = pipe.named_steps["prep"].transform(pipe.named_steps["fe"].transform(Xte))
    ex = shap.TreeExplainer(pipe.named_steps["clf"])
    sv = ex.shap_values(Zte)
    sv = sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv
    sh = pd.Series(np.abs(sv).mean(axis=0), index=Zte.columns).sort_values(ascending=False)
    print(sh.head(12).round(4).to_string())
    sh.to_csv(os.path.join(HERE, f"out_07_shap_{LEVEL}.csv"), encoding="utf-8")
except Exception as e:
    print("SHAP недоступен:", e)

# ---------------------------------------------------------------- бутстреп
print(f"\n=== бутстреп-CI для важностей (200 ресэмплов, ExtraTrees, {LEVEL}) ===")
rng = np.random.default_rng(SEED)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=SEED)
pipe = make_pipe(et(), level=LEVEL).fit(Xtr, ytr)
base_s = pipe.predict_proba(Xte)[:, 1]
top_feats = piv.head(8).index.tolist()
boot = {f: [] for f in top_feats}
Xte_r = Xte.reset_index(drop=True)
for b in range(200):
    idx = rng.integers(0, len(Xte_r), len(Xte_r))
    if len(np.unique(yte[idx])) < 2:
        continue
    a0 = roc_auc_score(yte[idx], base_s[idx])
    for f in top_feats:
        Xp = Xte_r.iloc[idx].copy()
        Xp[f] = rng.permutation(Xp[f].to_numpy())
        boot[f].append(a0 - roc_auc_score(yte[idx], pipe.predict_proba(Xp)[:, 1]))
B = pd.DataFrame({f: pd.Series(v).describe(percentiles=[.025, .5, .975]) for f, v in boot.items()}).T
print(B[["mean", "2.5%", "50%", "97.5%"]].round(4).sort_values("mean", ascending=False).to_string())

# ---------------------------------------------------------------- агрегация
print(f"\n=== АГРЕГАЦИЯ РАНГОВ по всем конфигурациям ({LEVEL}) ===")
P["rank"] = P.groupby(["family", "seed"])["imp"].rank(ascending=False)
agg = P.groupby("feature")["rank"].agg(["mean", "std", "min", "max"]).sort_values("mean")
agg.columns = ["средний ранг", "std", "лучший", "худший"]
print(agg.head(10).round(2).to_string())

piv.to_csv(os.path.join(HERE, f"out_07_perm_{LEVEL}.csv"), encoding="utf-8")
D.to_csv(os.path.join(HERE, f"out_07_drop_{LEVEL}.csv"), index=False, encoding="utf-8")
agg.to_csv(os.path.join(HERE, f"out_07_ranks_{LEVEL}.csv"), encoding="utf-8")
B.to_csv(os.path.join(HERE, f"out_07_boot_{LEVEL}.csv"), encoding="utf-8")
print("\nсохранено.")
