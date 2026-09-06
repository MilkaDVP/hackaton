"""Эксперимент 8: осмысленный порог + калибровка вероятностей."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from common import load, get_Xy, make_pipe, SEED, DATA_POR

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (precision_recall_curve, confusion_matrix, brier_score_loss,
                             roc_auc_score, average_precision_score, f1_score)

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "L2"
df = load(DATA_POR)
X, y = get_Xy(df)

ET = lambda: ExtraTreesClassifier(n_estimators=600, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)

raw_pipe = make_pipe(ET(), level=LEVEL)
cal_pipe = make_pipe(CalibratedClassifierCV(ET(), method="isotonic", cv=5),
                     level=LEVEL)

cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
p_raw = cross_val_predict(raw_pipe, X, y, cv=cv, method="predict_proba")[:, 1]
p_cal = cross_val_predict(cal_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

print(f"=== {LEVEL}: out-of-fold вероятности (10-fold) ===")
for nm, p in [("без калибровки", p_raw), ("изотоническая", p_cal)]:
    print(f"{nm:16s} ROC-AUC {roc_auc_score(y,p):.3f}  PR-AUC {average_precision_score(y,p):.3f}"
          f"  Brier {brier_score_loss(y,p):.4f}")

print("\n--- калибровочная кривая (10 бинов) ---")
for nm, p in [("без калибровки", p_raw), ("изотоническая", p_cal)]:
    ft, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    print(f"{nm:16s} предсказано {np.round(mp,3)}")
    print(f"{'':16s} факт.доля  {np.round(ft,3)}")

P = p_cal
# ------------------------------------------------- 1) порог через цену ошибки
print("\n=== подход 1: цена ошибки ===")
print("C_FN = во сколько раз пропустить студента дороже, чем зря позвать одного")
rows = []
for k in [1, 2, 3, 5, 8, 10, 15, 20]:
    best = None
    for t in np.linspace(0.01, 0.99, 197):
        pred = (P >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        cost = k * fn + fp
        if best is None or cost < best[0]:
            best = (cost, t, tp, fp, fn, tn)
    cost, t, tp, fp, fn, tn = best
    rows.append(dict(C_FN=k, threshold=round(t, 3), вызвано=tp + fp, поймано=tp,
                     пропущено=fn, зря=fp,
                     precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), cost=cost))
    print(f"  C_FN={k:2d} -> порог {t:.3f} | вызвано {tp+fp:3d} | поймано {tp:3d}/{tp+fn} "
          f"| зря {fp:3d} | precision {tp/max(tp+fp,1):.2f} recall {tp/max(tp+fn,1):.2f}")
pd.DataFrame(rows).to_csv(os.path.join(HERE, f"out_08_cost_{LEVEL}.csv"), index=False, encoding="utf-8")

# ------------------------------------------------- 2) порог через ёмкость
print("\n=== подход 2: ёмкость куратора (топ-K по вероятности) ===")
order = np.argsort(-P)
rows = []
for K in [20, 30, 40, 50, 60, 80, 100, 130]:
    sel = order[:K]
    tp = int(y[sel].sum())
    thr = P[order[K - 1]]
    rows.append(dict(K=K, threshold=round(float(thr), 3), поймано=tp,
                     precision=tp / K, recall=tp / y.sum()))
    print(f"  K={K:3d} -> порог {thr:.3f} | поймано {tp:3d}/{int(y.sum())} "
          f"| precision@{K} {tp/K:.2f} | recall@{K} {tp/y.sum():.2f}")
pd.DataFrame(rows).to_csv(os.path.join(HERE, f"out_08_capacity_{LEVEL}.csv"), index=False, encoding="utf-8")

# ------------------------------------------------- 3) F1 / F2 для сравнения
print("\n=== для сравнения: пороги по F1 и F2 ===")
prec, rec, thr = precision_recall_curve(y, P)
for beta, nm in [(1, "F1"), (2, "F2 (recall важнее)")]:
    f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    i = int(np.nanargmax(f[:-1]))
    print(f"  {nm:20s} порог {thr[i]:.3f}  precision {prec[i]:.2f}  recall {rec[i]:.2f}  {nm.split()[0]}={f[i]:.3f}")

print(f"\n=== матрица ошибок при пороге 0.5 (для контраста) ===")
print(confusion_matrix(y, (P >= 0.5).astype(int)))
print(f"поймано {int(((P>=0.5)&(y==1)).sum())} из {int(y.sum())} незачётов")
np.save(os.path.join(HERE, f"oof_{LEVEL}.npy"), np.vstack([P, y]))
