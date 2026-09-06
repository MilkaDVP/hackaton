# -*- coding: utf-8 -*-
from nbparts._h import md, code

md(r"""
## 8. Порог: превращаем вероятность в решение

Модель выдаёт число от 0 до 1. Решение «звать студента на разговор или нет» требует
**порога**, и 0.5 — не закон природы, а произвольная точка. При 15 % незачётов порог 0.5
почти всегда означает «не звать никого».

Ставим вопрос правильно: **что дороже — пропустить студента в зоне риска или зря
побеспокоить того, у кого всё в порядке?** Ниже — два способа ответить, и оба честные.

### 8.1 Сначала калибровка

Порог имеет смысл только у **калиброванной** модели: «0.3» должно означать «примерно
30 из 100 таких студентов не сдадут». Голосующий ансамбль из деревьев этим свойством
не обладает — деревья систематически жмут вероятности к краям. Калибровка на ROC-AUC
не влияет (это монотонное преобразование, ранжирование не меняется), но делает порог
осмысленным и чинит график распределения вероятностей.

Калибровка обучается **внутри пайплайна** (`CalibratedClassifierCV` с внутренней CV),
поэтому на тестовом фолде она не подсматривает.
""")

code(r"""
LEVEL_PROD = "L2"   # рабочая модель — та, что успевает помочь

cal_pipe = CalibratedClassifierCV(final_model(LEVEL_PROD), method="isotonic", cv=3)
cv10 = StratifiedKFold(5, shuffle=True, random_state=SEED)

P_raw = OOF[LEVEL_PROD]                                   # без калибровки (из раздела 6.3)
P_cal = cross_val_predict(cal_pipe, X_por, y_por, cv=cv10, method="predict_proba")[:, 1]

print("out-of-fold качество (5-fold):")
for nm, p in [("без калибровки", P_raw), ("изотоническая", P_cal)]:
    print(f"  {nm:16s} ROC-AUC {roc_auc_score(y_por, p):.3f}  "
          f"PR-AUC {average_precision_score(y_por, p):.3f}  "
          f"Brier {brier_score_loss(y_por, p):.4f}")
print("\nROC-AUC почти не меняется — калибровка не трогает ранжирование.")
print("Brier (ошибка самих вероятностей) — вот что должно улучшиться.")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for nm, p, c in [("без калибровки", P_raw, "#cb4335"), ("изотоническая", P_cal, "#1e8449")]:
    ft, mp = calibration_curve(y_por, p, n_bins=8, strategy="quantile")
    axes[0].plot(mp, ft, "o-", color=c, label=nm)
axes[0].plot([0, 1], [0, 1], "k--", lw=.8, label="идеальная калибровка")
axes[0].set_xlabel("предсказанная вероятность"); axes[0].set_ylabel("фактическая доля незачётов")
axes[0].set_title("Калибровочная кривая"); axes[0].legend(fontsize=8)

bins = np.linspace(0, 1, 26)
axes[1].hist(P_cal[y_por == 0], bins=bins, alpha=.65, label="сдали", density=True, color="#2e86c1")
axes[1].hist(P_cal[y_por == 1], bins=bins, alpha=.65, label="не сдали", density=True, color="#e67e22")
axes[1].set_xlabel("предсказанная вероятность незачёта"); axes[1].set_ylabel("плотность")
axes[1].set_title("Распределение вероятностей по двум классам")
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
**Как читать правый график.** Это и есть картинка, по которой выбирается порог.
Две группы перекрываются — идеального разделения нет и быть не может. Порог — это
вертикальная черта на этой картинке: всё, что правее, попадает в список куратора.
Сдвигая её влево, мы ловим больше незачётов и вместе с ними больше «сдавших».

### 8.2 Способ 1 — через цену ошибки

Пусть пропустить студента в зоне риска в `C` раз дороже, чем зря позвать одного.
Тогда оптимальный порог — тот, что минимизирует `C · FN + FP`. Мы не знаем `C` точно,
поэтому показываем **всю таблицу** и даём заказчику выбрать.
""")

code(r"""
def cost_table(P, y, cs=(1, 2, 3, 5, 8, 10, 15, 20)):
    rows = []
    grid = np.linspace(0.01, 0.99, 197)
    for C in cs:
        best = None
        for t in grid:
            tn, fp, fn, tp = confusion_matrix(y, (P >= t).astype(int), labels=[0, 1]).ravel()
            cost = C * fn + fp
            if best is None or cost < best[0]:
                best = (cost, t, tp, fp, fn, tn)
        cost, t, tp, fp, fn, tn = best
        rows.append({"цена пропуска C": C, "порог": round(t, 3), "позовём": tp + fp,
                     "поймали": tp, "пропустили": fn, "зря позвали": fp,
                     "точность": tp / max(tp + fp, 1), "полнота": tp / max(tp + fn, 1)})
    return pd.DataFrame(rows)

ct = cost_table(P_cal, y_por)
print("Оптимальный порог при разной цене пропуска (всего студентов "
      f"{len(y_por)}, незачётов {int(y_por.sum())}):")
ct.round(3).to_string(index=False)
""")

code(r"""
ct.round(3)
""")

md(r"""
### 8.3 Способ 2 — через ёмкость куратора

Более практичная постановка. У куратора нет абстрактной «цены ошибки» — у него есть
рабочее время. Если он физически может поговорить с `K` студентами за месяц, то вопрос
не «какой порог», а «кого включить в топ-`K`». Метрики — `precision@K` (сколько из
приглашённых действительно в зоне риска) и `recall@K` (какую долю всех незачётов мы поймали).
""")

code(r"""
order = np.argsort(-P_cal)
rows = []
for K in [20, 30, 40, 50, 60, 80, 100, 130]:
    sel = order[:K]
    tp = int(y_por[sel].sum())
    rows.append({"ёмкость K": K, "порог": round(float(P_cal[order[K - 1]]), 3),
                 "поймали": tp, "из них зря": K - tp,
                 f"precision@K": tp / K, f"recall@K": tp / y_por.sum(),
                 "во сколько раз лучше случайного": (tp / K) / y_por.mean()})
cap = pd.DataFrame(rows)
cap.round(3).to_string(index=False)
""")

code(r"""
cap.round(3)
""")

code(r"""
# Ёмкость куратора задаём как ВХОДНОЕ ОГРАНИЧЕНИЕ задачи, а порог выводим из неё.
CAPACITY = 40
THRESHOLD = float(np.round(P_cal[np.argsort(-P_cal)[CAPACITY - 1]], 3))
print(f"ёмкость куратора K = {CAPACITY} студентов -> порог {THRESHOLD}")
""")

code(r"""
prec, rec, thr = precision_recall_curve(y_por, P_cal)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

axes[0].plot(thr, prec[:-1], label="точность (precision)", color="#2e86c1")
axes[0].plot(thr, rec[:-1], label="полнота (recall)", color="#e67e22")
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
axes[0].plot(thr, f1[:-1], label="F1", color="#7d3c98", ls="--")
axes[0].axvline(THRESHOLD, color="k", ls=":", lw=1.4, label=f"выбранный порог {THRESHOLD}")
axes[0].set_xlabel("порог"); axes[0].set_ylabel("значение метрики")
axes[0].set_title("Метрики в зависимости от порога"); axes[0].legend(fontsize=8)

ks = np.arange(5, 201)
p_at_k = [y_por[order[:k]].sum() / k for k in ks]
r_at_k = [y_por[order[:k]].sum() / y_por.sum() for k in ks]
axes[1].plot(ks, p_at_k, label="precision@K", color="#2e86c1")
axes[1].plot(ks, r_at_k, label="recall@K", color="#e67e22")
axes[1].axhline(y_por.mean(), ls="--", c="k", lw=.8, label=f"базовая доля ({y_por.mean():.3f})")
axes[1].axvline(CAPACITY, color="k", ls=":", lw=1.4, label=f"ёмкость K={CAPACITY}")
axes[1].set_xlabel("размер списка K"); axes[1].set_ylabel("значение метрики")
axes[1].set_title("Качество списка в зависимости от его длины"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
### 8.4 Выбранный порог и матрица ошибок

**Решение: ёмкость `K = 40` студентов, что соответствует порогу из таблицы 8.3.**

Почему именно так:

* Куратору нужен **список**, а не вероятность. Список из 40 человек на поток в 649 —
  это примерно два месяца разговоров по 30 минут, реалистичная нагрузка.
* Порог, привязанный к ёмкости, **саморегулируется**: если в следующем потоке риск
  вырастет, топ-40 просто станет «гуще», а нагрузка на куратора не изменится.
* Абстрактную «цену пропуска `C`» заказчик назвать не может, а «сколько человек я успею
  принять» — может сразу. Таблица 8.2 остаётся для тех, кто готов назвать `C`:
  выбранная точка соответствует примерно `C ≈ 3–5`.
""")

code(r"""
pred = (P_cal >= THRESHOLD).astype(int)
cm = confusion_matrix(y_por, pred)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(4.6, 3.8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["не зовём", "зовём"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["сдал", "не сдал"])
ax.set_xlabel("решение модели"); ax.set_ylabel("что было на самом деле")
names = [["верно не позвали", "зря позвали"], ["ПРОПУСТИЛИ", "поймали"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{names[i][j]}\n{cm[i, j]}", ha="center", va="center",
                fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_title(f"Матрица ошибок при пороге {THRESHOLD}")
ax.grid(False)
plt.tight_layout(); plt.show()

print(f"порог: {THRESHOLD}   размер списка: {tp + fp} студентов")
print(f"поймали {tp} из {int(y_por.sum())} незачётов (полнота {tp/(tp+fn):.0%})")
print(f"из позванных {tp + fp} человек действительно в зоне риска {tp} "
      f"(точность {tp/max(tp+fp,1):.0%}, при случайном отборе было бы {y_por.mean():.0%})")
print(f"пропустили {fn} студентов")
print()
print(classification_report(y_por, pred, target_names=["зачёт", "незачёт"], digits=3))
""")

code(r"""
print("Для контраста — что было бы при пороге 0.5 «по умолчанию»:")
cm05 = confusion_matrix(y_por, (P_cal >= 0.5).astype(int))
tn5, fp5, fn5, tp5 = cm05.ravel()
print(f"  список из {tp5 + fp5} человек, поймано {tp5} из {int(y_por.sum())} "
      f"(полнота {tp5/max(tp5+fn5,1):.0%})")
print(f"  то есть {fn5} студентов в зоне риска остались бы без внимания.")
""")

md(r"""
**Вывод по разделу 8.** Порог — это не свойство модели, а управленческое решение, и его
надо принимать вместе с заказчиком. Мы привязали его к ёмкости куратора: 40 разговоров
в семестр. На этом пороге список примерно втрое «гуще» незачётами, чем случайная выборка
того же размера. Порог 0.5, взятый по умолчанию, для этой задачи не работает вообще —
при 15 % положительного класса он оставляет большинство студентов в зоне риска без внимания.
""")
