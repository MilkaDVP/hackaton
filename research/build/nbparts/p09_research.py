# -*- coding: utf-8 -*-
from nbparts._h import md, code

md(r"""
## 11. Собственные исследования

### 11.1 Кривая обучения: упрёмся ли мы в потолок, если дадут больше данных

Практический вопрос заказчику: стоит ли собирать данные за прошлые годы. Если кривая
вышла на плато — не стоит, качество упирается в природу задачи, а не в объём выборки.
""")

code(r"""
from sklearn.model_selection import learning_curve

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, level in zip(axes, ["L1", "L2"]):
    sizes, tr_sc, te_sc = learning_curve(
        make_pipe(ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=SEED, n_jobs=1),
                  level=level, exclude=DROP_FEATURES),
        X_por, y_por, cv=cv_obj(5, 1, SEED), scoring="roc_auc",
        train_sizes=np.linspace(0.15, 1.0, 7), n_jobs=-1, random_state=SEED)
    ax.plot(sizes, te_sc.mean(1), "o-", color="#cb4335", label="на новых данных (CV)")
    ax.fill_between(sizes, te_sc.mean(1) - te_sc.std(1), te_sc.mean(1) + te_sc.std(1),
                    alpha=.2, color="#cb4335")
    ax.plot(sizes, tr_sc.mean(1), "s--", color="#2e86c1", label="на обучающих данных")
    ax.set_xlabel("размер обучающей выборки"); ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Уровень {'1 (с оценками)' if level=='L1' else '2 (без оценок)'}")
    ax.legend(fontsize=8)
    print(f"{level}: при {int(sizes[0])} строках {te_sc.mean(1)[0]:.3f}, "
          f"при {int(sizes[-1])} строках {te_sc.mean(1)[-1]:.3f}, "
          f"прирост за последнюю четверть выборки "
          f"{te_sc.mean(1)[-1] - te_sc.mean(1)[-3]:+.4f}")
plt.tight_layout(); plt.show()
""")

md(r"""
### 11.2 Сколько «стоит» отказ от оценок и когда модель окупается

Переводим разницу между уровнями из «единиц AUC» в понятную величину: сколько студентов
в зоне риска мы найдём в списке фиксированной длины `K = 40`.
""")

code(r"""
rows = []
for level in ["L1", "L2"]:
    p = cross_val_predict(final_model(level), X_por, y_por,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          method="predict_proba")[:, 1]
    o = np.argsort(-p)
    tp = int(y_por[o[:CAPACITY]].sum())
    rows.append({"уровень": "1 — с оценками" if level == "L1" else "2 — без оценок",
                 "ROC-AUC": roc_auc_score(y_por, p),
                 f"поймано в топ-{CAPACITY}": tp,
                 f"точность@{CAPACITY}": tp / CAPACITY,
                 f"полнота@{CAPACITY}": tp / y_por.sum()})
rows.append({"уровень": "случайный отбор", "ROC-AUC": 0.5,
             f"поймано в топ-{CAPACITY}": CAPACITY * y_por.mean(),
             f"точность@{CAPACITY}": y_por.mean(), f"полнота@{CAPACITY}": CAPACITY / len(y_por)})
COST = pd.DataFrame(rows)
print(f"Что даёт список из {CAPACITY} человек на потоке {len(y_por)} студентов:")
COST.round(3).to_string(index=False)
""")

code(r"""
COST.round(3)
""")

md(r"""
### 11.3 Промежуточный уровень: прогноз после первой контрольной

Между «до семестра» и «конец семестра» есть третья точка — сразу после первой контрольной.
Куратор ещё успевает вмешаться, а сигнал уже сильно богаче. Проверяем, сколько качества
даёт один только `G1`.
""")

code(r"""
LEVELS3 = [("L2", "до первой контрольной"), ("L1a", "после первой контрольной"),
           ("L1", "после второй контрольной")]
rows = []
for lvl, nm in LEVELS3:
    r = evaluate(final_model(lvl), X_por, y_por, cv=cv_obj(5, 2, SEED))
    rows.append({"момент семестра": nm, "ROC-AUC": r["roc_auc"], "±": r["roc_auc_std"],
                 "PR-AUC": r["pr_auc"], "± ": r["pr_auc_std"]})
    print(f"{nm:28s} {fmt(r)}")
TIMING = pd.DataFrame(rows)
""")

code(r"""
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(range(3), TIMING["ROC-AUC"], "o-", color="#2e86c1", lw=2)
ax.errorbar(range(3), TIMING["ROC-AUC"], yerr=TIMING["±"], fmt="none",
            ecolor="#2e86c1", capsize=4)
ax.set_xticks(range(3))
ax.set_xticklabels(["до первой\nконтрольной", "после первой\nконтрольной",
                    "после второй\nконтрольной"])
ax.set_ylabel("ROC-AUC")
ax.set_title("Чем позже прогноз, тем он точнее — и тем меньше времени что-то менять")
for i, (v, s) in enumerate(zip(TIMING["ROC-AUC"], TIMING["±"])):
    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 10), ha="center")
plt.tight_layout(); plt.show()
""")

md(r"""
### 11.4 Справедливость: нет ли перекоса по подгруппам

Модель, которая хорошо работает в среднем, может систематически ошибаться на какой-то
группе. Смотрим качество и долю ложных тревог отдельно по полу, городу/селу и школе.
""")

code(r"""
P_prod = P_cal                        # калиброванные out-of-fold вероятности, уровень 2
pred_prod = (P_prod >= THRESHOLD).astype(int)

fair_rows = []
for col, nm in [("sex", "пол"), ("address", "город/село"), ("school", "школа")]:
    for val, sub in df_por.groupby(col):
        m = df_por[col].to_numpy() == val
        if m.sum() < 30 or len(np.unique(y_por[m])) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(y_por[m], pred_prod[m], labels=[0, 1]).ravel()
        fair_rows.append({
            "признак": nm, "группа": f"{val}", "n": int(m.sum()),
            "доля незачётов": y_por[m].mean(),
            "ROC-AUC": roc_auc_score(y_por[m], P_prod[m]),
            "доля попавших в список": pred_prod[m].mean(),
            "полнота": tp / max(tp + fn, 1),
            "точность": tp / max(tp + fp, 1),
            "ложные тревоги (FPR)": fp / max(fp + tn, 1)})
FAIR = pd.DataFrame(fair_rows)
FAIR.round(3).to_string(index=False)
""")

code(r"""
FAIR.round(3)
""")

code(r"""
print("Разброс между группами внутри каждого признака:")
for nm, g in FAIR.groupby("признак"):
    print(f"  {nm:12s} ROC-AUC {g['ROC-AUC'].min():.3f}–{g['ROC-AUC'].max():.3f} "
          f"| полнота {g['полнота'].min():.2f}–{g['полнота'].max():.2f} "
          f"| FPR {g['ложные тревоги (FPR)'].min():.2f}–{g['ложные тревоги (FPR)'].max():.2f}")
""")

md(r"""
### 11.5 Анализ ошибок: кого модель уверенно относит не туда

Смотрим на два хвоста: студентов, которых модель уверенно считала благополучными,
а они не сдали, и наоборот. Что у них общего — это подсказка, чего модели не хватает.
""")

code(r"""
err = df_por.copy()
err["p"] = P_prod
worst_fn = err[(err.no_pass == 1)].nsmallest(10, "p")   # пропущенные незачёты
worst_fp = err[(err.no_pass == 0)].nlargest(10, "p")    # ложные тревоги
show_cols = ["p", "G1", "G2", "G3", "absences", "failures", "studytime", "higher", "schoolsup"]

print("=== ПРОПУЩЕННЫЕ: не сдали, а модель была спокойна ===")
print(worst_fn[show_cols].round(3).to_string())
print("\n=== ЛОЖНЫЕ ТРЕВОГИ: модель била в набат, а человек сдал ===")
print(worst_fp[show_cols].round(3).to_string())

print("\nЧем пропущенные отличаются от остальных незачётов:")
cmp_cols = ["absences", "failures", "studytime", "goout", "age", "G1", "G2"]
print(pd.DataFrame({
    "пропущенные (n=10)": worst_fn[cmp_cols].mean(),
    "все незачёты": err[err.no_pass == 1][cmp_cols].mean(),
    "все зачёты": err[err.no_pass == 0][cmp_cols].mean(),
}).round(2).rename(index=ru).to_string())
""")

md(r"""
### 11.6 Что не сработало: регрессионный трюк

Гипотеза: вместо классификации предсказывать сам балл `G3` регрессией, а класс получать
порогом. Утечки здесь нет — `G3` используется только как **таргет**. Иногда на таких
данных это даёт лучшее ранжирование, потому что регрессия видит порядок, а не только
«сдал / не сдал». Проверяем.
""")

code(r"""
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold as SKF

reg_rows = []
for level in ["L1", "L2"]:
    aucs, prs = [], []
    for tr, te in cv_obj(5, 1, SEED).split(X_por, y_por):
        reg = make_pipe(ExtraTreesRegressor(n_estimators=400, random_state=SEED, n_jobs=1),
                        level=level, scale=False, exclude=DROP_FEATURES)
        reg.fit(X_por.iloc[tr], df_por["G3"].to_numpy()[tr])   # таргет — балл, не класс
        score = -reg.predict(X_por.iloc[te])                    # ниже балл -> выше риск
        aucs.append(roc_auc_score(y_por[te], score))
        prs.append(average_precision_score(y_por[te], score))
    reg_rows.append({"уровень": level, "подход": "регрессия G3 -> порог",
                     "ROC-AUC": np.mean(aucs), "±": np.std(aucs),
                     "PR-AUC": np.mean(prs), "± ": np.std(prs)})
    reg_rows.append({"уровень": level, "подход": "прямая классификация (наш финал)",
                     "ROC-AUC": FINAL[level]["roc_auc"], "±": FINAL[level]["roc_auc_std"],
                     "PR-AUC": FINAL[level]["pr_auc"], "± ": FINAL[level]["pr_auc_std"]})
REG = pd.DataFrame(reg_rows)
REG.round(3).to_string(index=False)
""")

code(r"""
REG.round(3)
""")

md(r"""
### 11.7 Что не сработало: SMOTE и другие способы борьбы с дисбалансом

`class_weight='balanced'` уже стоит во всех моделях. Проверяем, даёт ли что-то сверх этого
синтетическая генерация примеров. Важно: SMOTE обязан быть **внутри пайплайна**, иначе
синтетические примеры протекут в тестовый фолд и метрика вырастет на пустом месте.
""")

code(r"""
imb_rows = [{"подход": "class_weight='balanced' (наш выбор)",
             "ROC-AUC": FINAL["L2"]["roc_auc"], "±": FINAL["L2"]["roc_auc_std"],
             "PR-AUC": FINAL["L2"]["pr_auc"], "± ": FINAL["L2"]["pr_auc_std"]}]

et_plain = ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=1)
r = evaluate(make_pipe(et_plain, level="L2", exclude=DROP_FEATURES), X_por, y_por, cv=cv_obj(5, 2, SEED))
imb_rows.append({"подход": "без взвешивания вообще", "ROC-AUC": r["roc_auc"],
                 "±": r["roc_auc_std"], "PR-AUC": r["pr_auc"], "± ": r["pr_auc_std"]})

if HAS["imblearn"]:
    smote_pipe = ImbPipeline([
        ("fe", make_fe("L2")), ("prep", make_prep(scale=False)),
        ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
        ("clf", ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=1))])
    r = evaluate(smote_pipe, X_por, y_por, cv=cv_obj(5, 2, SEED))
    imb_rows.append({"подход": "SMOTE внутри пайплайна", "ROC-AUC": r["roc_auc"],
                     "±": r["roc_auc_std"], "PR-AUC": r["pr_auc"], "± ": r["pr_auc_std"]})
else:
    print("imbalanced-learn недоступен — строка про SMOTE пропущена (фолбэк).")

IMB = pd.DataFrame(imb_rows)
IMB.round(3).to_string(index=False)
""")

code(r"""
IMB.round(3)
""")
