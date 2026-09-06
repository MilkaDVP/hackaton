"""Обучение и экспорт артефактов. Логика — из risk.ipynb, без изменений.

    python -m riskml.train [--data data/student-por.csv] [--out riskml/artifacts]

Скрипт детерминирован и идемпотентен: при одинаковом входе даёт побайтово
сравнимые предсказания. Если метрики разойдутся с ноутбуком больше чем
на TOLERANCE, скрипт падает — молча выпускать разъехавшуюся модель нельзя.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from riskml import schema
from riskml.pipeline import (
    GRADE_COLS,
    SEED,
    TARGET_SOURCE,
    cv_obj,
    eng_cols,
    evaluate,
    final_model,
    get_Xy,
    make_pipe,
)

TOLERANCE = 0.01          # допустимое расхождение метрик с ноутбуком
CAPACITY_DEFAULT = 40     # ёмкость куратора из раздела 8.4 ноутбука

#: Ожидаемый результат leave-one-out отбора (раздел 4.3). Пересчитывается ниже;
#: расхождение означает, что пайплайн разъехался с ноутбуком.
EXPECTED_DROP = {
    "parent_edu_mean", "log_absences", "study_minus_free", "any_support",
    "alc_total", "alc_weekday_share", "fail_x_study", "goout_x_alc",
    "n_support", "age_over_17", "parent_edu_max", "abs_zero", "abs_bin",
    "abs_per_study",
}


# Windows при перенаправлении вывода берёт кодировку консоли (cp1251) и падает
# на любом не-латинском символе — например на «Δ» в отчёте о сверке. Тогда
# обучение отрабатывает целиком и умирает на печати, не сохранив артефакты.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8")


def log(msg=""):
    print(msg, flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
def compute_drop_features(X, y) -> tuple:
    """Leave-one-out отбор инженерных признаков — раздел 4.3 ноутбука.

    Пересчитывается, а не хардкодится: так артефакт всегда согласован
    с текущим кодом пайплайна.
    """
    def et():
        return ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=SEED, n_jobs=1)

    def lr():
        return LogisticRegression(max_iter=8000, class_weight="balanced",
                                  penalty="l1", solver="liblinear", C=0.3,
                                  random_state=SEED)

    models = [("ExtraTrees", et, False), ("ЛогрегL1", lr, True)]
    cv = cv_obj(5, 2, SEED)

    base = {mn: evaluate(make_pipe(mk(), level="L2", scale=sc), X, y, cv=cv)
            for mn, mk, sc in models}
    rows = []
    for c in eng_cols("L2"):
        deltas = []
        for mn, mk, sc in models:
            r = evaluate(make_pipe(mk(), level="L2", scale=sc, exclude=(c,)), X, y, cv=cv)
            deltas.append(r["roc_auc"] - base[mn]["roc_auc"])
        rows.append((c, float(np.mean(deltas))))
    # Δ > 0 -> без признака метрика РАСТЁТ -> признак только мешает
    drop = tuple(c for c, d in sorted(rows, key=lambda r: r[1]) if d > 0)
    return drop, rows


def importances(model, X, y, level, n_repeats=8):
    """Важность исходных переменных + направление влияния."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y,
                                          random_state=SEED)
    fitted = model.fit(Xtr, ytr)
    r = permutation_importance(fitted, Xte, yte, n_repeats=n_repeats,
                               random_state=SEED, scoring="roc_auc", n_jobs=-1)
    out = []
    for col, imp in zip(X.columns, r.importances_mean, strict=False):
        if col == TARGET_SOURCE:
            continue
        out.append({"feature": col, "importance": float(imp),
                    "direction": direction_of(col, X, y)})
    out.sort(key=lambda d: -d["importance"])
    return out


def direction_of(col, X, y):
    """В какую сторону признак двигает риск — считается по данным, не по памяти."""
    s = X[col]
    if pd.api.types.is_numeric_dtype(s):
        hi, lo = float(s[y == 1].mean()), float(s[y == 0].mean())
        if abs(hi - lo) < 1e-9:
            return {"kind": "numeric", "text": "различий между группами нет",
                    "mean_fail": hi, "mean_pass": lo, "sign": 0}
        sign = 1 if hi > lo else -1
        word = "больше — выше риск" if sign > 0 else "больше — ниже риск"
        return {"kind": "numeric", "text": word, "mean_fail": hi,
                "mean_pass": lo, "sign": sign}
    rates = {str(k): float(v) for k, v in pd.Series(y).groupby(s.astype(str).values).mean().items()}
    if not rates:
        return {"kind": "categorical", "text": "", "rates": {}}
    worst = max(rates, key=rates.get)
    best = min(rates, key=rates.get)
    return {"kind": "categorical", "rates": rates, "worst": worst, "best": best,
            "text": (f"чаще не сдают: «{schema.decode(col, worst)}» ({rates[worst]:.0%}), "
                     f"реже: «{schema.decode(col, best)}» ({rates[best]:.0%})")}


def background_stats(X):
    """Медианы/моды обучающей выборки — фон для объяснений и сравнения «со средним»."""
    stats = {}
    for c in X.columns:
        if c in (TARGET_SOURCE, "no_pass"):
            continue
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            stats[c] = {"kind": "numeric", "median": float(s.median()),
                        "mean": float(s.mean()), "p25": float(s.quantile(.25)),
                        "p75": float(s.quantile(.75))}
        else:
            stats[c] = {"kind": "categorical", "mode": str(s.mode().iloc[0]),
                        "freq": {str(k): float(v) for k, v in s.value_counts(normalize=True).items()}}
    return stats


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/student-por.csv")
    ap.add_argument("--out", default="riskml/artifacts")
    ap.add_argument("--reference", default="research/results.json",
                    help="метрики ноутбука для сверки")
    ap.add_argument("--quick", action="store_true",
                    help="пропустить 5x5 CV (только для отладки, метрики не сверяются)")
    args = ap.parse_args()

    t0 = time.time()
    data_path = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log(f"данные: {data_path}")
    df = pd.read_csv(data_path, sep=";")
    X, y = get_Xy(df)
    log(f"строк {len(df)}, незачётов {int(y.sum())} ({y.mean():.1%})")
    assert TARGET_SOURCE not in X.columns, "УТЕЧКА: G3 в признаках"

    # ---------------------------------------------- отбор признаков
    log("\nleave-one-out отбор инженерных признаков (раздел 4.3)...")
    drop, loo_rows = compute_drop_features(X, y)
    log(f"убрано {len(drop)}: {', '.join(drop)}")
    if set(drop) != EXPECTED_DROP:
        extra, missing = set(drop) - EXPECTED_DROP, EXPECTED_DROP - set(drop)
        log(f"  ВНИМАНИЕ: список разошёлся с ноутбуком. лишние={extra} пропали={missing}")

    # ---------------------------------------------- метрики
    metrics = {}
    if not args.quick:
        log("\n5x5 CV (как в ноутбуке)...")
        for level in ["L2", "L1"]:
            m = evaluate(final_model(level, drop), X, y, cv=cv_obj(5, 5, SEED))
            metrics[level] = {k: float(v) for k, v in m.items()}
            log(f"  {level}: ROC-AUC {m['roc_auc']:.4f} ± {m['roc_auc_std']:.4f}   "
                f"PR-AUC {m['pr_auc']:.4f} ± {m['pr_auc_std']:.4f}")

        ref_path = Path(args.reference)
        if ref_path.exists():
            ref = json.loads(ref_path.read_text(encoding="utf-8"))
            problems = []
            for level, key in [("L1", "level1"), ("L2", "level2")]:
                for ours, theirs in [("roc_auc", "roc"), ("pr_auc", "pr")]:
                    d = abs(metrics[level][ours] - ref[key][theirs])
                    mark = "ok" if d <= TOLERANCE else "РАСХОЖДЕНИЕ"
                    log(f"  сверка {level} {ours}: наше {metrics[level][ours]:.4f} "
                        f"против ноутбука {ref[key][theirs]:.4f}  Δ={d:.4f}  {mark}")
                    if d > TOLERANCE:
                        problems.append(f"{level}.{ours}: Δ={d:.4f}")
            if problems:
                raise SystemExit(
                    "метрики разошлись с ноутбуком больше чем на "
                    f"{TOLERANCE}: {'; '.join(problems)}")
        else:
            log(f"  {ref_path} не найден — сверка с ноутбуком пропущена")

    # ---------------------------------------------- порог по ёмкости
    log("\nпорог по ёмкости куратора (раздел 8)...")
    cal_l2 = CalibratedClassifierCV(final_model("L2", drop), method="isotonic", cv=3)
    oof = cross_val_predict(cal_l2, X, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    order = np.argsort(-oof)
    threshold = float(np.round(oof[order[CAPACITY_DEFAULT - 1]], 3))
    caught = int(y[order[:CAPACITY_DEFAULT]].sum())
    log(f"  K={CAPACITY_DEFAULT} -> порог {threshold}; "
        f"поймано {caught}/{int(y.sum())}, precision@K {caught/CAPACITY_DEFAULT:.2f}")

    # ---------------------------------------------- финальные модели
    log("\nобучение финальных моделей на всех данных...")
    models = {}
    for level in ["L2", "L1"]:
        m = CalibratedClassifierCV(final_model(level, drop), method="isotonic", cv=3)
        m.fit(X, y)
        path = out / f"model_{level.lower()}.joblib"
        joblib.dump(m, path, compress=3)
        models[level] = m
        log(f"  {level} -> {path} ({path.stat().st_size/1e6:.1f} МБ)")

    # ---------------------------------------------- важности
    log("\nважности признаков...")
    imp = {}
    for level in ["L2", "L1"]:
        imp[level] = importances(final_model(level, drop), X, y, level)
        top = ", ".join(d["feature"] for d in imp[level][:3])
        log(f"  {level} топ-3: {top}")

    # ---------------------------------------------- метаданные
    meta = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "training_data": {
            "file": data_path.name,
            "sha256": sha256(data_path),
            "n_rows": int(len(df)),
            "n_positive": int(y.sum()),
            "positive_rate": float(y.mean()),
        },
        "seed": SEED,
        "drop_features": list(drop),
        "loo": [{"feature": c, "delta_roc": d} for c, d in loo_rows],
        "ensemble_members": {k: v for k, v in
                             __import__("riskml.pipeline", fromlist=["x"]).FINAL_MEMBERS.items()
                             if k in ("L1", "L2")},
        "metrics": metrics,
        "threshold": {
            "default": threshold,
            "capacity": CAPACITY_DEFAULT,
            "caught_at_capacity": caught,
            "note": "порог выведен из ёмкости куратора, а не выбран как 0.5",
        },
        "risk_bands": {
            "low_max": float(round(y.mean(), 3)),
            "high_min": threshold,
            "note": "низкий — ниже средней доли незачётов; высокий — от порога шортлиста",
        },
        "importances": imp,
        "background": background_stats(X),
        "expected_columns": {
            "required_l2": schema.as_json()["required_l2"],
            "grades": GRADE_COLS,
            "never_a_feature": [TARGET_SOURCE, "no_pass"],
        },
        "schema": schema.as_json(),
    }
    (out / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\nметаданные -> {out/'metadata.json'}")
    log(f"готово за {time.time()-t0:.0f} с")


if __name__ == "__main__":
    sys.exit(main())
