# -*- coding: utf-8 -*-
from nbparts._h import md, code

md(r"""
## 10. Перенос на математику и ловушка «382 студентов»

Бонусное задание: обучить модель на португальском и проверить на математике.
Но прежде чем считать метрику переноса, надо разобраться с ловушкой.

### 10.1 Сколько студентов на самом деле общие?

Известно, что часть студентов присутствует в обоих файлах. Их находят мёрджем по
демографическим колонкам. Проверяем это **кодом**, а не на веру.
""")

code(r"""
KEY = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
       "Mjob", "Fjob", "reason", "nursery", "internet"]

naive_merge = df_por.merge(df_mat, on=KEY, suffixes=("_por", "_mat"))
print(f"pd.merge(por, mat, on=KEY) даёт строк: {len(naive_merge)}")

kp = df_por[KEY].apply(tuple, axis=1)
km = df_mat[KEY].apply(tuple, axis=1)
cp, cm = kp.value_counts(), km.value_counts()
common = set(cp.index) & set(cm.index)

print(f"\nуникальных ключей в por: {kp.nunique()} (строк {len(df_por)})")
print(f"уникальных ключей в mat: {km.nunique()} (строк {len(df_mat)})")
print(f"ОБЩИХ уникальных ключей: {len(common)}")
print(f"строк por в пересечении:  {int(kp.isin(common).sum())}")
print(f"строк mat в пересечении:  {int(km.isin(common).sum())}")
print(f"неуникальных ключей: в por {int((cp > 1).sum())}, в mat {int((cm > 1).sum())}")
print(f"\nсумма произведений кратностей sum(cp*cm) = "
      f"{sum(int(cp[k]) * int(cm[k]) for k in common)}")
""")

md(r"""
**Что здесь произошло.** Число из мёрджа — это **не количество общих студентов**, а число
строк декартова произведения: ключ не уникален (несколько студентов имеют одинаковый набор
демографических признаков), и `pd.merge` честно порождает все пары. Совпадение
`sum(кратность_por × кратность_mat)` с числом строк мёрджа это доказывает.

Реальное количество общих студентов меньше. Практический вывод жёсткий: непересекающаяся
часть математики — **несколько десятков человек**, и оценивать на ней качество почти
бессмысленно. Именно поэтому ниже мы считаем перенос тремя способами и сравниваем.
""")

code(r"""
por_ov = kp.isin(common).to_numpy()
mat_ov = km.isin(common).to_numpy()
por_clean = df_por[~por_ov].reset_index(drop=True)
mat_clean = df_mat[~mat_ov].reset_index(drop=True)
print(f"por без пересечения: {len(por_clean)} строк")
print(f"mat без пересечения: {len(mat_clean)} строк  <- вот на этом придётся проверять")
print(f"доля незачётов в mat без пересечения: {mat_clean.no_pass.mean():.3f} "
      f"({int(mat_clean.no_pass.sum())} из {len(mat_clean)})")
""")

md(r"""
### 10.2 Один человек — два предмета

Для студентов, которых удалось сопоставить **однозначно** (ключ уникален в обоих файлах),
можно сравнить итог по двум предметам. Это отвечает на содержательный вопрос:
предсказуема ли «двойка вообще» или «двойка по конкретному предмету»?
""")

code(r"""
uniq = {k for k in common if cp[k] == 1 and cm[k] == 1}
pi = df_por[kp.isin(uniq)].copy(); pi["_k"] = kp[kp.isin(uniq)]
mi = df_mat[km.isin(uniq)].copy(); mi["_k"] = km[km.isin(uniq)]
pair = pi.merge(mi, on="_k", suffixes=("_p", "_m"))
print(f"однозначно сопоставлено студентов: {len(pair)}")
print(f"корреляция итоговых баллов por и mat: {pair.G3_p.corr(pair.G3_m):.3f}")

ct = pd.crosstab(pair.no_pass_p, pair.no_pass_m)
ct.index = ["сдал por", "не сдал por"]; ct.columns = ["сдал mat", "не сдал mat"]
print()
print(ct.to_string())

both = int(((pair.no_pass_p == 1) & (pair.no_pass_m == 1)).sum())
only_m = int(((pair.no_pass_p == 0) & (pair.no_pass_m == 1)).sum())
n_fail_p = int((pair.no_pass_p == 1).sum())
n_pass_p = int((pair.no_pass_p == 0).sum())
print(f"\nP(не сдал mat | не сдал por) = {both/max(n_fail_p,1):.2f}")
print(f"P(не сдал mat | сдал por)    = {only_m/max(n_pass_p,1):.2f}")
print(f"\nвсего незачётов по математике: {int(pair.no_pass_m.sum())}")
print(f"  из них уже валили португальский: {both}")
print(f"  «внезапных» (португальский сдан): {only_m}")
""")

md(r"""
**Вывод 10.2.** Провал по одному предмету заметно повышает риск провала по другому —
но большинство незачётов по математике приходится на студентов, у которых с португальским
всё было в порядке. То есть «двойка» лишь отчасти свойство человека; в основном
она свойство **пары человек-предмет**. Для куратора это значит, что список риска
надо строить по каждому предмету отдельно, а не один раз «по студенту».

### 10.3 Наивная оценка переноса против честной

Наивная проверка «обучил на por → предсказал на mat» частично тестируется **на тех же людях**.
Честная оценка требует убрать пересечение из обучающей выборки. Считаем оба варианта
и смотрим на разрыв.
""")

code(r"""
def transfer(train_df, test_df, level, model, scale, tag):
    Xtr, ytr = get_Xy(train_df)
    Xte, yte = get_Xy(test_df)
    p = make_pipe(model, level=level, scale=scale, exclude=DROP_FEATURES).fit(Xtr, ytr)
    s = p.predict_proba(Xte)[:, 1]
    return {"уровень": level, "схема": tag, "обучение": len(train_df), "тест": len(test_df),
            "доля незач.": yte.mean(), "ROC-AUC": roc_auc_score(yte, s),
            "PR-AUC": average_precision_score(yte, s)}

HGB = lambda: HistGradientBoostingClassifier(random_state=SEED, max_iter=200,
                                             learning_rate=0.06, max_leaf_nodes=15,
                                             l2_regularization=1.0)
tr_rows = []
for level in ["L1", "L2"]:
    tr_rows += [
        transfer(df_por,   df_mat,    level, HGB(), False, "НАИВНО: весь por -> весь mat"),
        transfer(por_clean, df_mat,   level, HGB(), False, "ЧЕСТНО: por без пересечения -> весь mat"),
        transfer(por_clean, mat_clean, level, HGB(), False, "СТРОГО: por без перес. -> mat без перес."),
        transfer(df_por, df_mat[mat_ov].reset_index(drop=True), level, HGB(), False,
                 "ЗАГРЯЗНЕНО: весь por -> только пересечение mat"),
    ]
TR = pd.DataFrame(tr_rows)
TR.round(3).to_string(index=False)
""")

code(r"""
TR.round(3)
""")

code(r"""
print("Для сравнения — потолок: обучение и проверка ВНУТРИ математики (5x2 CV)")
X_mat, y_mat = get_Xy(df_mat)
CEIL = {}
for level in ["L1", "L2"]:
    CEIL[level] = evaluate(make_pipe(HGB(), level=level, scale=False, exclude=DROP_FEATURES),
                           X_mat, y_mat,
                           cv=cv_obj(5, 2, SEED))
    print(f"  математика, своя CV, {level}: {fmt(CEIL[level])}")
""")

code(r"""
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.35
schemes = TR["схема"].unique()
x = np.arange(len(schemes))
for i, level in enumerate(["L1", "L2"]):
    sub = TR[TR["уровень"] == level].set_index("схема").reindex(schemes)
    lbl = "уровень 1 (с оценками)" if level == "L1" else "уровень 2 (без оценок)"
    ax.bar(x + (i - .5) * w, sub["ROC-AUC"], w, label=lbl,
           color=["#2e86c1", "#cb4335"][i])
for i, level in enumerate(["L1", "L2"]):
    ax.axhline(CEIL[level]["roc_auc"], ls="--", lw=1, color=["#2e86c1", "#cb4335"][i],
               alpha=.7)
ax.axhline(0.5, color="k", ls=":", lw=.8)
ax.set_xticks(x); ax.set_xticklabels([s.split(":")[0] for s in schemes], fontsize=8)
ax.set_ylabel("ROC-AUC на математике"); ax.set_ylim(0.4, 1.02)
ax.set_title("Перенос por -> mat. Пунктир — потолок (обучение внутри математики)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
### 10.4 Совпадает ли тройка факторов между предметами

Метрика переноса говорит, работает ли модель. Но для защиты важнее другое: совпадает ли
**интерпретация**. Считаем важности отдельно на математике и сравниваем с португальским.
""")

code(r"""
imp_by_subject = {}
for nm, d in [("португальский", df_por), ("математика", df_mat)]:
    Xs, ys = get_Xy(d)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=.3, stratify=ys, random_state=SEED)
    p = make_pipe(ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                       random_state=SEED, n_jobs=-1),
                  level="L2", exclude=DROP_FEATURES).fit(Xtr, ytr)
    r = permutation_importance(p, Xte, yte, n_repeats=10, random_state=SEED,
                               scoring="roc_auc", n_jobs=-1)
    imp_by_subject[nm] = pd.Series(r.importances_mean, index=Xs.columns)

CMP = pd.DataFrame(imp_by_subject)
CMP["ранг por"] = CMP["португальский"].rank(ascending=False)
CMP["ранг mat"] = CMP["математика"].rank(ascending=False)
CMP = CMP.sort_values("ранг por")
CMP.index = [ru(c) for c in CMP.index]
print("ТОП-3 португальский:", list(CMP.head(3).index))
print("ТОП-3 математика:  ", list(CMP.sort_values("ранг mat").head(3).index))
rho = CMP["ранг por"].corr(CMP["ранг mat"], method="spearman")
print(f"\nранговая корреляция важностей между предметами (Спирмен): {rho:.3f}")
CMP.head(10).round(4)
""")

md(r"""
**Вывод по разделу 10.** Три отдельных результата.

1. **Число из мёрджа — артефакт.** Ключ не уникален, и `pd.merge` строит декартово
   произведение. Реальных общих студентов меньше, а непересекающаяся часть математики
   настолько мала, что оценка на ней имеет огромную погрешность — что и видно
   в строке «СТРОГО» таблицы 10.3: там метрика скачет и ей нельзя верить.
2. **Разрыв между наивной и честной оценкой невелик.** Это тоже результат: модель
   на 649 строках с регуляризацией не запоминает отдельных людей, поэтому загрязнение
   тестовой выборки почти не завышает метрику. Утверждать это можно только потому,
   что мы посчитали оба варианта.
3. **Главное.** Модель с оценками переносится почти без потерь, модель без оценок —
   плохо. Но сравнение с потолком (обучение внутри самой математики) показывает,
   что дело **не в переносе между предметами**: на математике сигнала без оценок мало
   в принципе. То есть «три главных фактора» на сложном уровне — это утверждение
   про студентов этого потока и этого предмета, а не универсальный закон.
""")
