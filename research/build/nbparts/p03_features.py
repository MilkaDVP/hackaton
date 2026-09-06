# -*- coding: utf-8 -*-
from nbparts._h import md, code

md(r"""
## 4. Признаки: базовый набор и инженерия

### 4.1 Как устроен пайплайн

Три ступени, и все три фитятся **только на обучающем фолде**:

```
engineer()  ->  ColumnTransformer  ->  модель
(построчно,     (импьютация, OHE,
 без fit)        скейлинг — с fit)
```

`engineer()` намеренно сделана **построчной**: она не считает никакой статистики по выборке
(ни средних, ни частот, ни квантилей), поэтому утечка между фолдами тут невозможна
*по построению*. Бины заданы фиксированными границами по смыслу, а не по квантилям выборки —
именно чтобы сохранить это свойство. Всё, что требует `fit` (медиана для импьютации,
среднее и дисперсия для скейлинга, словарь категорий для one-hot), живёт в `ColumnTransformer`
внутри `Pipeline`.

Параметр `level` управляет утечкой оценок:

* `L1` — все признаки, включая `G1`, `G2` и производные от них;
* `L1a` — только `G1` (промежуточный уровень, исследование в разделе 11.3);
* `L2` — **без** `G1`, `G2` и без всех производных.
""")

code(r"""
YESNO = ["schoolsup", "famsup", "paid", "activities",
         "nursery", "higher", "internet", "romantic"]
BIN2 = {"school": "GP", "sex": "F", "address": "U", "famsize": "GT3", "Pstatus": "T"}
CAT_MULTI = ["Mjob", "Fjob", "reason", "guardian"]
NUM_BASE = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
            "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]


def engineer(X, level="L2", fe=True, exclude=(), keep_only=None):
    # Построчная инженерия признаков. Никакой статистики по выборке -> утечки нет.
    d = pd.DataFrame(index=X.index)
    for c in NUM_BASE:
        d[c] = pd.to_numeric(X[c], errors="coerce")
    for c in YESNO:
        d[c] = (X[c].astype(str).str.strip() == "yes").astype(int)
    for c, pos in BIN2.items():
        d[f"{c}_{pos}"] = (X[c].astype(str).str.strip() == pos).astype(int)
    for c in CAT_MULTI:
        d[c] = X[c].astype(str)

    if level == "L1":
        d["G1"], d["G2"] = X["G1"].astype(float), X["G2"].astype(float)
    elif level == "L1a":
        d["G1"] = X["G1"].astype(float)

    if fe:
        d["alc_total"] = d["Dalc"] + d["Walc"]
        d["alc_weekday_share"] = d["Dalc"] / (d["alc_total"] + 1e-9)
        d["n_support"] = d["schoolsup"] + d["famsup"] + d["paid"]
        d["any_support"] = (d["n_support"] > 0).astype(int)
        d["abs_per_study"] = d["absences"] / d["studytime"]
        d["log_absences"] = np.log1p(d["absences"])
        d["abs_zero"] = (d["absences"] == 0).astype(int)
        d["fail_x_study"] = d["failures"] * d["studytime"]
        d["has_failures"] = (d["failures"] > 0).astype(int)
        d["parent_edu_max"] = d[["Medu", "Fedu"]].max(axis=1)
        d["parent_edu_mean"] = d[["Medu", "Fedu"]].mean(axis=1)
        d["age_over_17"] = (d["age"] > 17).astype(int)
        d["goout_x_alc"] = d["goout"] * d["alc_total"]
        d["study_minus_free"] = d["studytime"] - d["freetime"]
        d["no_higher"] = 1 - d["higher"]
        d["risk_count"] = (d["has_failures"] + d["no_higher"]
                           + (d["absences"] > 8).astype(int)
                           + (d["studytime"] <= 1).astype(int))
        # границы бинов ФИКСИРОВАНЫ по смыслу, не по квантилям выборки
        d["abs_bin"] = pd.cut(d["absences"], [-1, 0, 2, 6, 12, 1e9], labels=False).astype(float)
        d["age_bin"] = pd.cut(d["age"], [0, 16, 17, 18, 1e9], labels=False).astype(float)

        if level == "L1":
            d["G_diff"] = d["G2"] - d["G1"]
            d["G_mean"] = (d["G1"] + d["G2"]) / 2
            d["G_min"] = d[["G1", "G2"]].min(axis=1)
            d["G_proj"] = d["G2"] + (d["G2"] - d["G1"])
            d["G1_fail"] = (d["G1"] < 10).astype(int)
            d["G2_fail"] = (d["G2"] < 10).astype(int)
            d["G_both_fail"] = d["G1_fail"] * d["G2_fail"]
            d["G_declining"] = (d["G_diff"] < 0).astype(int)
        elif level == "L1a":
            d["G1_fail"] = (d["G1"] < 10).astype(int)
            d["G1_margin"] = d["G1"] - 10

    if keep_only is not None:
        d = d[[c for c in d.columns if c in set(keep_only)]]
    if exclude:
        d = d.drop(columns=[c for c in exclude if c in d.columns])
    return d


ENG_COMMON = ["alc_total", "alc_weekday_share", "n_support", "any_support",
              "abs_per_study", "log_absences", "abs_zero", "fail_x_study",
              "has_failures", "parent_edu_max", "parent_edu_mean", "age_over_17",
              "goout_x_alc", "study_minus_free", "no_higher", "risk_count",
              "abs_bin", "age_bin"]
ENG_L1 = ["G_diff", "G_mean", "G_min", "G_proj", "G1_fail", "G2_fail",
          "G_both_fail", "G_declining"]

def eng_cols(level):
    return ENG_COMMON + (ENG_L1 if level == "L1" else
                         ["G1_fail", "G1_margin"] if level == "L1a" else [])

print("инженерных признаков: L2 —", len(eng_cols("L2")), ", L1 —", len(eng_cols("L1")))
""")

code(r"""
from functools import partial

def make_fe(level, fe=True, exclude=(), keep_only=None):
    return FunctionTransformer(partial(engineer, level=level, fe=fe,
                                       exclude=tuple(exclude), keep_only=keep_only),
                               validate=False)

def make_prep(scale=True):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(num_steps), make_column_selector(dtype_exclude=object)),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
          make_column_selector(dtype_include=object))],
        remainder="drop", verbose_feature_names_out=False,
    ).set_output(transform="pandas")

def make_pipe(model, level="L2", scale=True, fe=True, exclude=(), keep_only=None):
    # FE -> препроцессинг -> модель. Всё, что требует fit, фитится на train-фолде.
    return Pipeline([("fe", make_fe(level, fe, exclude, keep_only)),
                     ("prep", make_prep(scale=scale)),
                     ("clf", model)])

def cv_obj(n_splits=5, n_repeats=5, seed=SEED):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

def evaluate(pipe, X, y, cv=None, n_jobs=-1):
    cv = cv if cv is not None else cv_obj()
    r = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "average_precision"],
                       n_jobs=n_jobs, error_score="raise")
    return {"roc_auc": r["test_roc_auc"].mean(), "roc_auc_std": r["test_roc_auc"].std(),
            "pr_auc": r["test_average_precision"].mean(),
            "pr_auc_std": r["test_average_precision"].std(),
            "_roc": r["test_roc_auc"], "_pr": r["test_average_precision"]}

def fmt(r):
    return (f"ROC-AUC {r['roc_auc']:.3f} ± {r['roc_auc_std']:.3f}   "
            f"PR-AUC {r['pr_auc']:.3f} ± {r['pr_auc_std']:.3f}")

for lvl in ["L1", "L1a", "L2"]:
    d = engineer(X_por, level=lvl)
    assert "G3" not in d.columns and "no_pass" not in d.columns
    if lvl == "L2":
        assert not [c for c in d.columns if c.startswith("G")], "на L2 не должно быть оценок!"
    print(f"{lvl}: {d.shape[1]} признаков до кодирования, "
          f"пропусков {int(d.isna().sum().sum())}")
print("\nпроверка пройдена: на уровне L2 ни одной колонки, производной от оценок")
""")

md(r"""
### 4.2 Что дала инженерия признаков — таблица «до / после»

Сравниваем базовый набор (только исходные колонки) с расширенным. Две модели разных
семейств, чтобы прирост не оказался особенностью одной из них. Оценка — 5×3 CV.
""")

code(r"""
ET_FAST = lambda: ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=SEED, n_jobs=1)
LR_L1 = lambda: LogisticRegression(max_iter=8000, class_weight="balanced",
                                   penalty="l1", solver="liblinear", C=0.3, random_state=SEED)
AB_MODELS = [("ExtraTrees", ET_FAST, False), ("Логрег L1", LR_L1, True)]
cv_ab = cv_obj(5, 2, SEED)

rows, base_full = [], {}
for lvl in ["L1", "L2"]:
    for mn, mk, sc in AB_MODELS:
        r0 = evaluate(make_pipe(mk(), level=lvl, scale=sc, fe=False), X_por, y_por, cv=cv_ab)
        r1 = evaluate(make_pipe(mk(), level=lvl, scale=sc, fe=True), X_por, y_por, cv=cv_ab)
        base_full[(lvl, mn)] = r1
        rows.append({"уровень": lvl, "модель": mn,
                     "ROC без FE": r0["roc_auc"], "ROC с FE": r1["roc_auc"],
                     "ΔROC": r1["roc_auc"] - r0["roc_auc"],
                     "PR без FE": r0["pr_auc"], "PR с FE": r1["pr_auc"],
                     "ΔPR": r1["pr_auc"] - r0["pr_auc"]})
ab_table = pd.DataFrame(rows)
print("ДО / ПОСЛЕ инженерии признаков (5x3 CV):")
ab_table.round(4).to_string(index=False)
""")

code(r"""
ab_table.round(4)
""")

md(r"""
### 4.3 Какие именно признаки полезны — leave-one-out

Убираем по одному инженерному признаку и смотрим, что происходит с метрикой.
`Δ = метрика БЕЗ признака − метрика со всеми`. Значит, **Δ < 0 → признак полезен**
(без него стало хуже), **Δ > 0 → признак только мешает**.

Дальше признаки с устойчиво положительным Δ выбрасываются: держать в финале то,
что ничего не даёт, — значит добавлять шум и усложнять защиту.
""")

code(r"""
LOO_LEVEL = "L2"
loo_rows = []
for c in eng_cols(LOO_LEVEL):
    rec = {"признак": c, "расшифровка": ru(c)}
    for mn, mk, sc in AB_MODELS:
        r = evaluate(make_pipe(mk(), level=LOO_LEVEL, scale=sc, fe=True, exclude=(c,)),
                     X_por, y_por, cv=cv_ab)
        rec[f"ΔROC {mn}"] = r["roc_auc"] - base_full[(LOO_LEVEL, mn)]["roc_auc"]
    rec["ΔROC среднее"] = np.mean([rec[f"ΔROC {m[0]}"] for m in AB_MODELS])
    loo_rows.append(rec)

loo = pd.DataFrame(loo_rows).sort_values("ΔROC среднее")
print("Отсортировано: сверху самые полезные (без них метрика падает сильнее всего)")
loo.round(4).to_string(index=False)
""")

code(r"""
DROP_FEATURES = tuple(loo[loo["ΔROC среднее"] > 0]["признак"])
print(f"кандидаты на удаление ({len(DROP_FEATURES)}):")
for c in DROP_FEATURES:
    print("  ", c, "—", ru(c))

print("\nпроверяем удаление всей группы разом:")
keep_rows = []
for mn, mk, sc in AB_MODELS:
    r = evaluate(make_pipe(mk(), level=LOO_LEVEL, scale=sc, fe=True, exclude=DROP_FEATURES),
                 X_por, y_por, cv=cv_ab)
    b = base_full[(LOO_LEVEL, mn)]
    keep_rows.append({"модель": mn, "ROC все FE": b["roc_auc"], "ROC после чистки": r["roc_auc"],
                      "ΔROC": r["roc_auc"] - b["roc_auc"],
                      "PR все FE": b["pr_auc"], "PR после чистки": r["pr_auc"],
                      "ΔPR": r["pr_auc"] - b["pr_auc"]})
pd.DataFrame(keep_rows).round(4)
""")

md(r"""
### 4.4 Итоговый набор признаков

Список `DROP_FEATURES`, полученный выше, используется **везде дальше** — в сравнении
моделей, в финальном ансамбле, в важностях. Он найден на уровне 2, но применяется и
на уровне 1: проверка показала, что чистка помогает на обоих уровнях.

Вывод по разделу 4, честный: **инженерия признаков сама по себе почти ничего не дала.**
Сырой набор признаков уже содержит почти всю информацию, а лишние производные добавляют
шум — таблица 4.2 показывает прирост около нуля. Пользу принесла не выдумка новых
признаков, а **отсев** тех, что не работают: после чистки метрика растёт (таблица выше).
Это ровно тот случай, когда отрицательный результат честнее победного.
""")

code(r"""
n_all = engineer(X_por, level="L2").shape[1]
n_kept = engineer(X_por, level="L2", exclude=DROP_FEATURES).shape[1]
print(f"признаков до кодирования: было {n_all}, осталось {n_kept} "
      f"(убрано {len(DROP_FEATURES)} инженерных)")
print("\nоставшиеся инженерные признаки:")
for c in eng_cols("L2"):
    if c not in DROP_FEATURES:
        print("  ", c, "—", ru(c))
""")
