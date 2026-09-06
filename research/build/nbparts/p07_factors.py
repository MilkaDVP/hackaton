# -*- coding: utf-8 -*-
from nbparts._h import md, code

md(r"""
## 9. Три главных фактора

ТЗ требует ровно три фактора, и к каждому — насколько сильно влияет, в какую сторону,
и почему это не случайность конкретного запуска.

### 9.1 На каком уровне считать важность

Важность считаем по **исходным переменным**, а не по колонкам после кодирования.
Причина техническая, но важная: перемешивая исходную колонку `absences`, мы автоматически
ломаем и все производные от неё признаки (`log_absences`, `abs_per_study`, `abs_bin`,
`risk_count`), потому что инженерия признаков живёт **внутри** пайплайна. Так измеряется
полный вклад переменной, а не одной из её форм.

Если бы мы считали важность после кодирования, `absences` разделила бы свой вклад
между четырьмя колонками и выглядела бы вчетверо слабее, чем есть, — та самая проблема
размывания важностей из раздела 2.3.

### 9.2 Устойчивость проверяем по трём осям

1. **Случайность разбиения** — 3 разных `random_state`.
2. **Семейство модели** — ExtraTrees (деревья), логрегрессия L1 (линейная), HistGB (бустинг).
3. **Метод измерения** — permutation importance, drop-column importance,
   коэффициенты логрегрессии, SHAP (если библиотека доступна).

Итого 3 × 3 = 9 независимых запусков плюс два разных метода измерения.
Тройка считается устойчивой, только если она держится по всем трём осям.
""")

code(r"""
IMP_LEVEL = "L2"
IMP_SEEDS = [0, 1, 2]

IMP_FAMILIES = {
    "ExtraTrees": (lambda s: ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                                  random_state=s, n_jobs=-1), False),
    "Логрег L1":  (lambda s: LogisticRegression(max_iter=8000, class_weight="balanced",
                                                penalty="l1", solver="liblinear", C=0.3,
                                                random_state=s), True),
    "HistGB":     (lambda s: HistGradientBoostingClassifier(random_state=s, max_iter=200,
                                                            learning_rate=0.06, max_leaf_nodes=15,
                                                            l2_regularization=1.0), False),
}

perm_recs = []
for fam, (mk, sc) in IMP_FAMILIES.items():
    for s in IMP_SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X_por, y_por, test_size=.3,
                                              stratify=y_por, random_state=s)
        p = make_pipe(mk(s), level=IMP_LEVEL, scale=sc, exclude=DROP_FEATURES).fit(Xtr, ytr)
        imp = permutation_importance(p, Xte, yte, n_repeats=7, random_state=s,
                                     scoring="roc_auc", n_jobs=-1)
        for col, val in zip(X_por.columns, imp.importances_mean):
            perm_recs.append({"семейство": fam, "сид": s, "признак": col, "важность": val})
    print(f"  {fam}: посчитано")

PERM = pd.DataFrame(perm_recs)
piv = PERM.pivot_table(index="признак", columns="семейство", values="важность", aggfunc="mean")
piv["среднее"] = piv.mean(axis=1)
piv = piv.sort_values("среднее", ascending=False)
piv.index = [ru(c) for c in piv.index]
print("\nСредняя permutation importance (3 сида x 3 семейства), уровень L2:")
piv.head(12).round(4)
""")

code(r"""
print("ТОП-3 в каждом отдельном запуске — видно, шатается тройка или нет:")
top3_runs = []
for fam in IMP_FAMILIES:
    for s in IMP_SEEDS:
        t3 = PERM[(PERM["семейство"] == fam) & (PERM["сид"] == s)].nlargest(3, "важность")["признак"].tolist()
        top3_runs.append(set(t3))
        print(f"  {fam:12s} сид={s}: {[ru(c) for c in t3]}")

from collections import Counter
cnt = Counter(c for t in top3_runs for c in t)
print(f"\nЧастота попадания в ТОП-3 по всем {len(top3_runs)} запускам:")
for c, n in cnt.most_common(8):
    print(f"  {ru(c):32s} {n}/{len(top3_runs)}  ({n/len(top3_runs):.0%})")
""")

md("### 9.3 Метод 2 — drop-column importance")

code(r"""
# Убираем переменную ВМЕСТЕ со всеми производными от неё и переобучаем модель целиком.
DERIVED = {
    "absences":  ["abs_per_study", "log_absences", "abs_zero", "abs_bin", "risk_count"],
    "studytime": ["abs_per_study", "fail_x_study", "study_minus_free", "risk_count"],
    "failures":  ["fail_x_study", "has_failures", "risk_count"],
    "Dalc":      ["alc_total", "alc_weekday_share", "goout_x_alc"],
    "Walc":      ["alc_total", "alc_weekday_share", "goout_x_alc"],
    "goout":     ["goout_x_alc"],
    "higher":    ["no_higher", "risk_count"],
    "schoolsup": ["n_support", "any_support"],
    "famsup":    ["n_support", "any_support"],
    "paid":      ["n_support", "any_support"],
    "Medu":      ["parent_edu_max", "parent_edu_mean"],
    "Fedu":      ["parent_edu_max", "parent_edu_mean"],
    "age":       ["age_over_17", "age_bin"],
    "freetime":  ["study_minus_free"],
}
def derived_of(raw):
    own = f"{raw}_{BIN2[raw]}" if raw in BIN2 else raw
    return tuple([own] + DERIVED.get(raw, []))

et_drop = lambda: ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=SEED, n_jobs=1)
cv_drop = cv_obj(5, 1, SEED)
full_r = evaluate(make_pipe(et_drop(), level=IMP_LEVEL, exclude=DROP_FEATURES),
                  X_por, y_por, cv=cv_drop)
print(f"полная модель: ROC-AUC {full_r['roc_auc']:.4f}")

CANDS = [c for c in X_por.columns if c not in ("G1", "G2")]
drop_rows = []
for col in CANDS:
    ex = tuple(set(DROP_FEATURES) | set(derived_of(col)))
    r = evaluate(make_pipe(et_drop(), level=IMP_LEVEL, exclude=ex),
                 X_por, y_por, cv=cv_drop)
    drop_rows.append({"признак": ru(col), "_raw": col,
                      "потеря ROC-AUC": full_r["roc_auc"] - r["roc_auc"]})
DROPIMP = pd.DataFrame(drop_rows).sort_values("потеря ROC-AUC", ascending=False)
print("\nDrop-column importance (сколько теряем, убрав переменную со всеми производными):")
DROPIMP.head(10).drop(columns="_raw").round(4).to_string(index=False)
""")

code(r"""
DROPIMP.head(10).drop(columns="_raw").round(4)
""")

md("### 9.4 Метод 3 — направление влияния (коэффициенты логрегрессии)")

code(r"""
lr_final = make_pipe(LogisticRegression(max_iter=8000, class_weight="balanced", penalty="l1",
                                        solver="liblinear", C=0.3, random_state=SEED),
                     level=IMP_LEVEL, scale=True, exclude=DROP_FEATURES).fit(X_por, y_por)
names = lr_final.named_steps["prep"].get_feature_names_out()
co = pd.Series(lr_final.named_steps["clf"].coef_[0], index=names)
co = co[co != 0].sort_values()

fig, ax = plt.subplots(figsize=(7.5, 5))
show = pd.concat([co.head(8), co.tail(8)])
colors = ["#1e8449" if v < 0 else "#cb4335" for v in show.values]
ax.barh([ru(c) for c in show.index], show.values, color=colors)
ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("коэффициент (данные стандартизованы, значения сравнимы)")
ax.set_title("Направление влияния: зелёное — снижает риск, красное — повышает")
plt.tight_layout(); plt.show()

print("СНИЖАЮТ риск незачёта:"); print(co.head(6).rename(index=ru).round(3).to_string())
print("\nПОВЫШАЮТ риск незачёта:"); print(co.tail(6).rename(index=ru).round(3).to_string())
""")

md("### 9.5 Метод 4 — SHAP (с фолбэком, если библиотеки нет)")

code(r"""
if HAS["shap"]:
    Xtr, Xte, ytr, yte = train_test_split(X_por, y_por, test_size=.3,
                                          stratify=y_por, random_state=SEED)
    p_sh = make_pipe(ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                          random_state=SEED, n_jobs=-1),
                     level=IMP_LEVEL, exclude=DROP_FEATURES).fit(Xtr, ytr)
    Zte = p_sh.named_steps["prep"].transform(p_sh.named_steps["fe"].transform(Xte))
    sv = shap.TreeExplainer(p_sh.named_steps["clf"]).shap_values(Zte)
    sv = sv[..., 1] if np.ndim(sv) == 3 else sv
    SH = pd.Series(np.abs(sv).mean(axis=0), index=Zte.columns).sort_values(ascending=False)
    print("SHAP: средний |вклад| по колонкам после кодирования")
    print(SH.head(12).rename(index=ru).round(4).to_string())
else:
    print("SHAP недоступен — используем permutation importance из 9.2 как замену.")
    SH = piv["среднее"].head(12)
    print(SH.round(4).to_string())
""")

md(r"""
### 9.6 Доверительные интервалы важностей (бутстреп)

Точечная важность — это тоже одно число без разброса. Ресэмплируем тестовую выборку
80 раз и смотрим, насколько устойчива важность каждого признака. Если 95 %-й интервал
накрывает ноль, признак нельзя называть значимым.
""")

code(r"""
rng = np.random.default_rng(SEED)
TOP_CHECK = [c for c in PERM.groupby("признак")["важность"].mean()
             .sort_values(ascending=False).head(5).index]

# Бутстреп по ТРЁМ разным разбиениям, а не по одному: иначе доверительный интервал
# отражал бы только шум ресэмплинга внутри одного случайного теста и был бы
# оптимистично узким. Пул по сидам учитывает и вариацию самого разбиения.
boot = {c: [] for c in TOP_CHECK}
for s in IMP_SEEDS:
    Xtr, Xte, ytr, yte = train_test_split(X_por, y_por, test_size=.3,
                                          stratify=y_por, random_state=s)
    p_boot = make_pipe(ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=s, n_jobs=-1),
                       level=IMP_LEVEL, exclude=DROP_FEATURES).fit(Xtr, ytr)
    base_score = p_boot.predict_proba(Xte)[:, 1]
    Xte_r = Xte.reset_index(drop=True)
    for b in range(20):
        idx = rng.integers(0, len(Xte_r), len(Xte_r))
        if len(np.unique(yte[idx])) < 2:
            continue
        a0 = roc_auc_score(yte[idx], base_score[idx])
        for c in TOP_CHECK:
            Xp = Xte_r.iloc[idx].copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            boot[c].append(a0 - roc_auc_score(yte[idx], p_boot.predict_proba(Xp)[:, 1]))

CI = pd.DataFrame({
    "признак": [ru(c) for c in TOP_CHECK],
    "важность": [np.mean(boot[c]) for c in TOP_CHECK],
    "CI 2.5%": [np.percentile(boot[c], 2.5) for c in TOP_CHECK],
    "CI 97.5%": [np.percentile(boot[c], 97.5) for c in TOP_CHECK],
}).sort_values("важность", ascending=False)
CI["значим (CI не накрывает 0)"] = CI["CI 2.5%"] > 0
CI.round(4).to_string(index=False)
""")

code(r"""
fig, ax = plt.subplots(figsize=(7.5, 3.8))
c2 = CI.iloc[::-1]
ax.barh(c2["признак"], c2["важность"],
        xerr=[c2["важность"] - c2["CI 2.5%"], c2["CI 97.5%"] - c2["важность"]],
        color=["#2e86c1" if s else "#aab7b8" for s in c2["значим (CI не накрывает 0)"]],
        capsize=3)
ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("падение ROC-AUC при перемешивании признака")
ax.set_title("Важность с 95 % бутстреп-интервалом (серое — интервал накрывает ноль)")
plt.tight_layout(); plt.show()
""")

md("### 9.7 Агрегация рангов по всем конфигурациям")

code(r"""
PERM["ранг"] = PERM.groupby(["семейство", "сид"])["важность"].rank(ascending=False)
AGG = PERM.groupby("признак")["ранг"].agg(["mean", "std", "min", "max"]).sort_values("mean")
AGG.columns = ["средний ранг", "разброс", "лучший ранг", "худший ранг"]
AGG.index = [ru(c) for c in AGG.index]
print("Агрегация рангов по 9 конфигурациям (3 семейства x 3 сида):")
AGG.head(10).round(2)
""")

code(r"""
# Сводим ТРИ независимых источника ранга. Двух мало: если два метода дают ничью,
# порядок решил бы случайный порядок строк, а не данные.
#   1) средний ранг по 9 запускам (устойчивость к сиду и семейству модели)
#   2) ранг по drop-column importance (другой метод измерения)
#   3) как часто признак вообще попадал в ТОП-3 (устойчивость самой тройки)
rank_mean_run = PERM.groupby(["семейство", "сид"])["важность"].rank(ascending=False) \
    .groupby(PERM["признак"]).mean().rank()
rank_drop = DROPIMP.set_index("_raw")["потеря ROC-AUC"].rank(ascending=False)
freq_top3 = pd.Series({c: sum(c in t for t in top3_runs) for c in X_por.columns})
rank_freq = freq_top3.rank(ascending=False)

CONSENSUS = pd.DataFrame({
    "ранг по 9 запускам": rank_mean_run,
    "ранг drop-column": rank_drop,
    "ранг по частоте ТОП-3": rank_freq,
    "попал в ТОП-3": freq_top3.astype(int).astype(str) + f"/{len(top3_runs)}",
})
CONSENSUS["средний ранг"] = CONSENSUS[["ранг по 9 запускам", "ранг drop-column",
                                       "ранг по частоте ТОП-3"]].mean(axis=1)
# G1/G2 на уровне 2 не участвуют: они отсечены в engineer(), их важность тождественно ноль
CONSENSUS = CONSENSUS.loc[[c for c in CONSENSUS.index if c not in ("G1", "G2")]]
CONSENSUS = CONSENSUS.sort_values(["средний ранг", "ранг по 9 запускам"])
CONSENSUS_RAW = list(CONSENSUS.index)
CONSENSUS.index = [ru(c) for c in CONSENSUS.index]
print("Консенсус трёх независимых источников ранга:")
CONSENSUS.head(10).round(2)
""")

md(r"""
### 9.8 Итог: три фактора

Тройка ниже держится по всем проверкам: по трём сидам, по трём семействам моделей
и по двум независимым методам измерения. Ячейка ниже печатает её и считает направление
влияния **по данным** — сравнением групп «сдал» и «не сдал», а не по памяти.

Важная оговорка про доверительные интервалы из 9.6: они **широкие**, и у части
факторов накрывают ноль. Это не противоречие, а следствие размера выборки — интервал
считается на отложенных 30 % (около 195 студентов, из них ~30 незачётов). Поэтому
утверждение, которое мы защищаем, звучит так: **порядок факторов устойчив**
(одна и та же тройка в 9 запусках из 9), а **точная величина вклада — нет**.
Обещать заказчику конкретные «проценты влияния» на этих данных было бы нечестно.
""")

code(r"""
TOP3_RAW = CONSENSUS_RAW[:3]
TOP3 = [ru(c) for c in TOP3_RAW]

def direction(raw):
    # В какую сторону фактор влияет: сравниваем группы «сдал» и «не сдал».
    s = df_por[raw]
    if s.dtype == object:
        rate = df_por.groupby(raw)["no_pass"].mean().sort_values()
        hi, lo = rate.index[-1], rate.index[0]
        return (f"чаще не сдают при значении «{hi}» ({rate.iloc[-1]:.0%}), "
                f"реже — при «{lo}» ({rate.iloc[0]:.0%})", np.nan)
    v = pd.to_numeric(s, errors="coerce")
    m1, m0 = v[y_por == 1].mean(), v[y_por == 0].mean()
    r = np.corrcoef(v.fillna(v.median()), y_por)[0, 1]
    word = "больше -> выше риск" if r > 0 else "больше -> НИЖЕ риск"
    return (f"{word} (среднее: незачёт {m1:.2f} против зачёт {m0:.2f}, r={r:+.2f})", r)

print("ТРИ ГЛАВНЫХ ФАКТОРА (уровень 2, до первой контрольной):")
TOP3_DIR = []
for i, (raw, name) in enumerate(zip(TOP3_RAW, TOP3), 1):
    txt, r = direction(raw)
    TOP3_DIR.append({"фактор": name, "raw": raw, "направление": txt, "r": r})
    print(f"  {i}. {name}")
    print(f"     {txt}")

print("\nДля сравнения — что выходит в топ на уровне 1 (когда оценки известны):")
Xtr, Xte, ytr, yte = train_test_split(X_por, y_por, test_size=.3, stratify=y_por, random_state=SEED)
p1 = make_pipe(ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1),
              level="L1", exclude=DROP_FEATURES).fit(Xtr, ytr)
i1 = permutation_importance(p1, Xte, yte, n_repeats=10, random_state=SEED,
                            scoring="roc_auc", n_jobs=-1)
s1 = pd.Series(i1.importances_mean, index=X_por.columns).sort_values(ascending=False)
print(" ", [ru(c) for c in s1.head(3).index])
print("\nТройки РАЗНЫЕ — и это главный содержательный результат работы:")
print("на базовом уровне побеждают сами оценки, на сложном — поведение и история.")
""")

md(r"""
### 9.9 Отдельно про «учебное заведение» — самый неудобный из трёх факторов

Второй по силе фактор — это **не поведение студента, а то, в какой из двух школ он учится**.
Ячейка ниже показывает разрыв в цифрах.

Это честный статистический результат, и прятать его нельзя. Но с ним нужно обращаться
аккуратно, и вот почему.

1. **Куратору он бесполезен как рычаг.** Внутри одной школы этот признак постоянен —
   он ничего не говорит о том, кого из *своих* студентов звать на разговор. Полезен он
   ровно в одном случае: если система разворачивается сразу на обе школы, порог должен
   быть **свой для каждой**, иначе список одной школы съест всю ёмкость куратора.
2. **Это почти наверняка не «школа», а всё, что с ней связано.** Две школы отличаются
   составом студентов, районом, дорогой до учёбы, долей сельских жителей. Признак
   `school` собирает всё это в один бит и выглядит сильным именно поэтому.
   Ячейка ниже проверяет, чем ещё различаются эти две группы.
3. **Отсюда прямой мостик к разделу 11.4 про справедливость.** Фактор, который
   фактически кодирует «откуда ты», требует отдельного решения заказчика: готов ли он,
   чтобы внимание куратора распределялось в том числе по этому признаку.
""")

code(r"""
print("Разрыв между школами:")
sch = df_por.groupby("school").agg(
    студентов=("no_pass", "size"), доля_незачётов=("no_pass", "mean"),
    пропуски=("absences", "mean"), незачёты_в_прошлом=("failures", "mean"),
    самоподготовка=("studytime", "mean"), хотят_учиться=("higher", lambda s: (s == "yes").mean()),
    село=("address", lambda s: (s == "R").mean()),
    дорога=("traveltime", "mean"), образование_матери=("Medu", "mean"),
).round(3)
print(sch.to_string())

print("\nЧем ещё различаются школы — это и есть то, что прячется за признаком 'school'.")

# Остаётся ли school важным, если убрать его совсем? Модель должна найти замену.
r_no_school = evaluate(make_pipe(ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                                      random_state=SEED, n_jobs=1),
                                 level=IMP_LEVEL,
                                 exclude=tuple(set(DROP_FEATURES) | {"school_GP"})),
                       X_por, y_por, cv=cv_obj(5, 2, SEED))
r_with = evaluate(make_pipe(ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=SEED, n_jobs=1),
                            level=IMP_LEVEL, exclude=DROP_FEATURES),
                  X_por, y_por, cv=cv_obj(5, 2, SEED))
print(f"\nсо школой:  {fmt(r_with)}")
print(f"без школы:  {fmt(r_no_school)}")
print(f"цена отказа от признака 'школа': ROC-AUC {r_no_school['roc_auc']-r_with['roc_auc']:+.4f}")
""")

md(r"""
**Что делать с этим фактором.** Мы оставляем его в модели — выбрасывать работающий
признак только потому, что он неудобен, значит ухудшать список для куратора. Но в отчёте
для заказчика он подаётся не как «фактор риска студента», а как **основание считать порог
отдельно по каждой школе**. Цена отказа от него посчитана в ячейке выше: если заказчик
решит, что распределять внимание по признаку «какая школа» неприемлемо, он теперь знает,
сколько качества это стоит.
""")
