# -*- coding: utf-8 -*-
from nbparts._h import md, code

md(r"""
## 6. Уровень 1 — прогноз с учётом обеих контрольных

Модель конца семестра. Практическая ценность ограничена (менять что-то уже поздно),
но это верхняя граница качества и хорошая проверка, что пайплайн вообще работает.

### 6.1 Что считаем базой

Чтобы «побили baseline» было утверждением, а не словами, воспроизводим стартовый подход
как есть: `pd.get_dummies` по всему датафрейму + случайный лес, без инженерии признаков.
Считаем его на том же протоколе (5×5 CV), что и наши модели.
""")

code(r"""
X_base = pd.get_dummies(df_por.drop(columns=["G3", "no_pass"]), drop_first=True)
baseline_rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=SEED)

BASE_COLS = {"L1": list(X_base.columns),
             "L2": [c for c in X_base.columns if c not in ("G1", "G2")]}
BASELINE = {}
for level in ["L1", "L2"]:
    BASELINE[level] = evaluate(baseline_rf, X_base[BASE_COLS[level]], y_por, cv=CV_MAIN)
    print(f"BASELINE (стартер, {level}): {fmt(BASELINE[level])}")
""")

md(r"""
### 6.2 Финальная модель

По таблице раздела 5.2 берём **мягкое голосование трёх разнородных моделей**. Обоснование —
не «у неё самое большое число», а устойчивость: модели разной природы (деревья, линейная,
ядровая) ошибаются по-разному, поэтому усреднение снижает разброс между фолдами.

**Состав ансамбля различается по уровням, и это не произвол.** Параметры членов
подбирались на пространстве признаков уровня 2 (без оценок). На уровне 1 пространство
другое — там доминируют `G1` и `G2`, — и SVM с ядром, настроенным под уровень 2,
туда не переносится: в таблице 5.2 видно, что на L1 он тянет ансамбль вниз.
Поэтому на уровне 1 третьим членом идёт случайный лес. Честная альтернатива —
подбирать параметры SVM отдельно под каждый уровень, но выигрыш этого не стоит:
уровень 1 и так у потолка.
""")

code(r"""
FINAL_MEMBERS = {"L1": ["et", "rf", "lr"], "L1a": ["et", "rf", "lr"],
                 "L2": ["et", "lr", "svm"]}

def final_model(level):
    # Финальная модель: мягкое голосование трёх разнородных моделей.
    return voting(FINAL_MEMBERS[level], level)

FINAL = {}
for level in ["L1", "L2"]:
    t0 = time.time()
    FINAL[level] = evaluate(final_model(level), X_por, y_por, cv=CV_MAIN)
    print(f"ФИНАЛ {level}: {fmt(FINAL[level])}   [{time.time()-t0:.0f}s]")
print(f"\nвложенная CV всей процедуры (L2, раздел 5.3): "
      f"ROC-AUC {NESTED_ENS['roc']:.3f} ± {NESTED_ENS['roc_sd']:.3f}  "
      f"PR-AUC {NESTED_ENS['pr']:.3f} ± {NESTED_ENS['pr_sd']:.3f}")
""")

code(r"""
summary = pd.DataFrame([
    {"уровень": "1 — с G1, G2", "что": "ориентир ТЗ", "ROC-AUC": 0.97, "±": np.nan,
     "PR-AUC": np.nan, "± ": np.nan},
    {"уровень": "1 — с G1, G2", "что": "baseline стартера", "ROC-AUC": BASELINE["L1"]["roc_auc"],
     "±": BASELINE["L1"]["roc_auc_std"], "PR-AUC": BASELINE["L1"]["pr_auc"],
     "± ": BASELINE["L1"]["pr_auc_std"]},
    {"уровень": "1 — с G1, G2", "что": "наша модель", "ROC-AUC": FINAL["L1"]["roc_auc"],
     "±": FINAL["L1"]["roc_auc_std"], "PR-AUC": FINAL["L1"]["pr_auc"],
     "± ": FINAL["L1"]["pr_auc_std"]},
    {"уровень": "2 — без G1, G2", "что": "ориентир ТЗ", "ROC-AUC": 0.83, "±": np.nan,
     "PR-AUC": np.nan, "± ": np.nan},
    {"уровень": "2 — без G1, G2", "что": "baseline стартера", "ROC-AUC": BASELINE["L2"]["roc_auc"],
     "±": BASELINE["L2"]["roc_auc_std"], "PR-AUC": BASELINE["L2"]["pr_auc"],
     "± ": BASELINE["L2"]["pr_auc_std"]},
    {"уровень": "2 — без G1, G2", "что": "наша модель", "ROC-AUC": FINAL["L2"]["roc_auc"],
     "±": FINAL["L2"]["roc_auc_std"], "PR-AUC": FINAL["L2"]["pr_auc"],
     "± ": FINAL["L2"]["pr_auc_std"]},
    {"уровень": "2 — без G1, G2", "что": "наша модель, вложенная CV",
     "ROC-AUC": NESTED_ENS["roc"], "±": NESTED_ENS["roc_sd"],
     "PR-AUC": NESTED_ENS["pr"], "± ": NESTED_ENS["pr_sd"]},
])
print("СВОДКА ПО ДВУМ УРОВНЯМ (5x5 CV; последняя строка — вложенная CV)")
summary.round(3).to_string(index=False)
""")

code(r"""
summary.round(3)
""")

md(r"""
### 6.3 ROC и PR кривые обоих уровней

Кривые строятся по **out-of-fold** предсказаниям (каждый студент предсказан моделью,
которая его не видела), а не на обучающей выборке.
""")

code(r"""
OOF = {}
for level in ["L1", "L2"]:
    OOF[level] = cross_val_predict(final_model(level), X_por, y_por,
                                   cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                   method="predict_proba")[:, 1]
    print(f"{level}: OOF ROC-AUC {roc_auc_score(y_por, OOF[level]):.3f}  "
          f"PR-AUC {average_precision_score(y_por, OOF[level]):.3f}")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for level, c in [("L1", "#2e86c1"), ("L2", "#cb4335")]:
    lbl = "уровень 1 (с оценками)" if level == "L1" else "уровень 2 (без оценок)"
    fpr, tpr, _ = roc_curve(y_por, OOF[level])
    axes[0].plot(fpr, tpr, color=c, label=f"{lbl}: AUC {roc_auc_score(y_por, OOF[level]):.3f}")
    pr, rc, _ = precision_recall_curve(y_por, OOF[level])
    axes[1].plot(rc, pr, color=c, label=f"{lbl}: PR-AUC {average_precision_score(y_por, OOF[level]):.3f}")

axes[0].plot([0, 1], [0, 1], "k--", lw=.8, label="случайная модель")
axes[0].set_xlabel("доля ложных тревог (FPR)"); axes[0].set_ylabel("доля пойманных (TPR)")
axes[0].set_title("ROC-кривая"); axes[0].legend(fontsize=8)

axes[1].axhline(y_por.mean(), ls="--", c="k", lw=.8, label=f"случайная модель ({y_por.mean():.3f})")
axes[1].set_xlabel("полнота (recall)"); axes[1].set_ylabel("точность (precision)")
axes[1].set_title("Precision-Recall кривая"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
**Вывод по разделу 6.** Уровень 1 работает почти идеально, и это ожидаемо: `G1` и `G2` —
это оценки по тому же предмету, то есть модель по двум измерениям величины предсказывает
третье. Практической ценности мало — к моменту, когда обе контрольные написаны, у куратора
почти не осталось времени. Главный интерес — в разделе 7.
""")

# ==========================================================================
md(r"""
## 7. Уровень 2 — прогноз до первой контрольной (главный раздел)

Здесь модель видит только то, что известно **в начале семестра**: анкету, историю прошлых
незачётов, пропуски, самоподготовку. Никаких оценок по предмету.

Метрика ниже, и это нормально — зато у куратора остаётся время что-то сделать.
Именно эта модель и есть продукт.
""")

code(r"""
print("=== УРОВЕНЬ 2: сравнение с ориентирами ===")
print(f"ориентир ТЗ:        ROC-AUC ≈ 0.830")
print(f"baseline стартера:  {fmt(BASELINE['L2'])}")
print(f"наша модель:        {fmt(FINAL['L2'])}")
print()
d_roc = FINAL["L2"]["roc_auc"] - BASELINE["L2"]["roc_auc"]
d_pr = FINAL["L2"]["pr_auc"] - BASELINE["L2"]["pr_auc"]
print(f"прирост к baseline: ROC-AUC {d_roc:+.4f}   PR-AUC {d_pr:+.4f}")
""")

md(r"""
### 7.1 Прирост статистически значим или это шум?

25 оценок кросс-валидации у двух моделей посчитаны на **одних и тех же** разбиениях,
поэтому корректно сравнивать их попарно. Считаем разности по фолдам и смотрим,
насколько уверенно они лежат по одну сторону от нуля.
""")

code(r"""
from scipy import stats

paired = {}
for level in ["L1", "L2"]:
    r_base = BASELINE[level]           # уже посчитан в 6.1 на тех же фолдах
    diff = FINAL[level]["_roc"] - r_base["_roc"]
    t_stat, p_val = stats.ttest_rel(FINAL[level]["_roc"], r_base["_roc"])
    w_stat, p_w = stats.wilcoxon(FINAL[level]["_roc"], r_base["_roc"])
    paired[level] = dict(mean_diff=diff.mean(), sd=diff.std(),
                         wins=int((diff > 0).sum()), n=len(diff), p_t=p_val, p_w=p_w)
    print(f"{level}: средняя разность ROC-AUC {diff.mean():+.4f} ± {diff.std():.4f} | "
          f"наша модель выиграла в {int((diff>0).sum())} из {len(diff)} фолдов | "
          f"p (парный t) = {p_val:.2g}, p (Wilcoxon) = {p_w:.2g}")
""")

md(r"""
### 7.2 Разброс между фолдами

Одно число без разброса ничего не значит. Показываем все 25 оценок.
Видно, что распределения baseline и нашей модели заметно перекрываются —
именно поэтому прирост надо проверять парным тестом, а не сравнением средних.
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, level in zip(axes, ["L1", "L2"]):
    r_base = BASELINE[level]
    ax.boxplot([r_base["_roc"], FINAL[level]["_roc"]],
               tick_labels=["baseline\nстартера", "наша\nмодель"], patch_artist=True,
               showmeans=True)
    for i, vals in enumerate([r_base["_roc"], FINAL[level]["_roc"]], start=1):
        ax.scatter(np.random.default_rng(SEED).normal(i, .04, len(vals)), vals,
                   s=9, alpha=.5, color="#34495e", zorder=3)
    ax.set_title(f"Уровень {'1 (с оценками)' if level=='L1' else '2 (без оценок)'}")
    ax.set_ylabel("ROC-AUC по фолду")
fig.suptitle("Разброс качества по 25 фолдам кросс-валидации", y=1.03)
plt.tight_layout(); plt.show()
""")

md(r"""
### 7.3 Честный итог по обоим уровням

Ниже ячейка сравнивает достигнутое с ориентирами ТЗ и печатает вердикт по каждому
пункту — без округлений в свою пользу.
""")

code(r"""
GOALS = {"L1": 0.97, "L2": 0.83}
print("СРАВНЕНИЕ С ОРИЕНТИРАМИ ТЗ")
print("=" * 72)
for level in ["L1", "L2"]:
    g, f, b = GOALS[level], FINAL[level], BASELINE[level]
    print(f"\nУровень {level} (ориентир ТЗ ROC-AUC ~ {g:.2f})")
    print(f"  baseline стартера : {b['roc_auc']:.4f} ± {b['roc_auc_std']:.4f}"
          f"   PR-AUC {b['pr_auc']:.4f}")
    print(f"  наша модель       : {f['roc_auc']:.4f} ± {f['roc_auc_std']:.4f}"
          f"   PR-AUC {f['pr_auc']:.4f}")
    print(f"  ориентир ТЗ       : {'ДОСТИГНУТ' if f['roc_auc'] >= g else 'НЕ достигнут'}")
    print(f"  прирост к baseline: ROC {f['roc_auc']-b['roc_auc']:+.4f}, "
          f"PR {f['pr_auc']-b['pr_auc']:+.4f}"
          f"  (выиграно фолдов {paired[level]['wins']}/{int(paired[level]['n'])}, "
          f"p={paired[level]['p_t']:.3f})")

print("\n" + "=" * 72)
print(f"Вложенная CV всей процедуры (L2): {NESTED_ENS['roc']:.4f} ± {NESTED_ENS['roc_sd']:.4f}")
print("Это оценка без оптимизма — то, чего стоит ждать на новом потоке.")
""")

md(r"""
**Что достигнуто, а что нет — прямо.**

* **Уровень 2 (главный).** Ориентир ТЗ (≈ 0.83) достигнут, baseline стартера побит
  и по ROC-AUC, и по PR-AUC. PR-AUC вырос заметно сильнее ROC-AUC — а для задачи
  «дать куратору чистый список» важнее именно он.
* **Уровень 1.** Ориентир ТЗ достигнут. Прирост к baseline по ROC-AUC здесь почти
  нулевой, и это не недоработка, а **потолок**: при ROC-AUC около 0.97 оставшиеся
  ошибки — это студенты, которых по двум контрольным отличить невозможно в принципе.
  Осмысленный прирост на этом уровне виден только по PR-AUC.
* **Чего не достигли.** Планка ROC-AUC ≥ 0.86 на уровне 2 **не взята**. Лучшее, что
  удалось получить честно, — значение в таблице выше. Дальнейший рост на этих данных
  упирается в объём выборки (649 строк, 100 незачётов) — см. кривую обучения в 11.1.
  Поднять число можно было бы подбором по той же кросс-валидации, по которой мы
  отчитываемся, но это была бы подгонка, а не качество, и вложенная CV её бы вскрыла.
""")
