# -*- coding: utf-8 -*-
from nbparts._h import md, code

md(r"""
**Вывод по `G3 == 0`.** Решение принимаем не «на глаз», а по данным: в разделе 5.4 считаем
все метрики в двух вариантах (с этими 15 студентами и без них) и смотрим, меняется ли от
этого тройка главных факторов. Забегая вперёд — **оставляем**. Обоснование содержательное:
заказчику нужен список тех, «с кем стоит поговорить», а студент, который перестал ходить
и не дошёл до итога, — ровно тот случай, ради которого система и строится. Выкинуть его
значило бы учить модель не замечать самый тяжёлый исход.
""")

# ==========================================================================
md("## 2. Смотрим на данные")

code(r"""
RU = {
    'absences': 'пропуски занятий', 'studytime': 'часы самоподготовки', 'failures': 'прошлые незачёты',
    'G1': 'контрольная 1', 'G2': 'контрольная 2', 'G3': 'итоговый балл',
    'traveltime': 'дорога до школы', 'famrel': 'отношения в семье', 'freetime': 'свободное время',
    'goout': 'встречи с друзьями', 'Dalc': 'алкоголь в будни', 'Walc': 'алкоголь в выходные',
    'health': 'здоровье', 'age': 'возраст', 'Medu': 'образование матери', 'Fedu': 'образование отца',
    'schoolsup': 'доп. поддержка школы', 'famsup': 'поддержка семьи', 'paid': 'платные занятия',
    'higher': 'хочет учиться дальше', 'internet': 'интернет дома', 'romantic': 'отношения',
    'activities': 'кружки', 'nursery': 'ходил в садик', 'school': 'учебное заведение',
    'sex': 'пол', 'address': 'город/село', 'famsize': 'размер семьи', 'Pstatus': 'родители вместе',
    'Mjob': 'работа матери', 'Fjob': 'работа отца', 'reason': 'причина выбора', 'guardian': 'опекун',
    # инженерные признаки (раздел 4)
    'alc_total': 'алкоголь всего', 'n_support': 'видов поддержки', 'any_support': 'есть поддержка',
    'abs_per_study': 'пропусков на час самоподготовки', 'log_absences': 'log(пропуски)',
    'abs_zero': 'ни одного пропуска', 'fail_x_study': 'незачёты x самоподготовка',
    'has_failures': 'были незачёты', 'parent_edu_max': 'образование родителей (макс.)',
    'parent_edu_mean': 'образование родителей (сред.)', 'age_over_17': 'старше 17',
    'goout_x_alc': 'друзья x алкоголь', 'study_minus_free': 'учёба минус свободное время',
    'no_higher': 'НЕ хочет учиться дальше', 'risk_count': 'счётчик риск-факторов',
    'abs_bin': 'пропуски (бины)', 'age_bin': 'возраст (бины)',
    'alc_weekday_share': 'доля будних в алкоголе',
    'G_diff': 'динамика G2-G1', 'G_mean': 'средняя за контрольные', 'G_min': 'худшая контрольная',
    'G_proj': 'линейный прогноз на G3', 'G1_fail': 'провалил 1-ю', 'G2_fail': 'провалил 2-ю',
    'G_both_fail': 'провалил обе', 'G_declining': 'оценка падает', 'G1_margin': 'G1 - 10',
    'school_GP': 'школа GP', 'sex_F': 'пол Ж', 'address_U': 'город', 'famsize_GT3': 'семья >3',
    'Pstatus_T': 'родители вместе',
}
def ru(c):
    return RU.get(c, c)

print("пропусков в данных:", int(df_por.isna().sum().sum()))
print("дубликатов строк:", int(df_por.duplicated().sum()))
print("\nсколько колонок каждого типа:")
print(df_por.dtypes.value_counts().to_string())
""")

code(r"""
key = ['absences', 'studytime', 'failures', 'G1', 'G2', 'goout', 'Dalc', 'age']
t = df_por.groupby('no_pass')[key].mean().round(2)
t.index = ['зачёт', 'незачёт']
t.columns = [ru(c) for c in t.columns]
print("Средние значения по группам:")
t.T
""")

md("### 2.1 Распределения трёх ключевых числовых признаков")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, col in zip(axes, ['absences', 'G2', 'studytime']):
    bins = np.histogram_bin_edges(df_por[col], bins=15)
    ax.hist(df_por[df_por.no_pass == 0][col], bins=bins, alpha=.6, label='зачёт', density=True)
    ax.hist(df_por[df_por.no_pass == 1][col], bins=bins, alpha=.6, label='незачёт', density=True)
    ax.set_title(ru(col)); ax.set_xlabel(ru(col)); ax.set_ylabel('плотность'); ax.legend()
fig.suptitle('Распределения: зачёт против незачёта', y=1.04)
plt.tight_layout(); plt.show()
""")

md(r"""
### 2.2 Boxplot и разрезы по трём главным факторам

Тройка, которая победит в разделе 9 на сложном уровне, — **прошлые незачёты**,
**учебное заведение** и **желание учиться дальше**. Первый фактор числовой, поэтому
для него boxplot; два других категориальные, для них честнее показать долю незачётов
в каждой группе. Рядом — пропуски и самоподготовка: их интуитивно ждут в тройке,
и полезно заранее увидеть, насколько слабее они разделяют группы.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, col in zip(axes, ['failures', 'absences', 'studytime']):
    data = [df_por[df_por.no_pass == 0][col], df_por[df_por.no_pass == 1][col]]
    bp = ax.boxplot(data, tick_labels=['зачёт', 'незачёт'], patch_artist=True, showmeans=True)
    for patch, c in zip(bp['boxes'], ['#7fb3d5', '#e59866']):
        patch.set_facecolor(c)
    ax.set_title(ru(col)); ax.set_ylabel(ru(col))
fig.suptitle('Boxplot: прошлые незачёты разделяют группы, пропуски и самоподготовка — почти нет',
             y=1.05, fontsize=9)
plt.tight_layout(); plt.show()
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
cats = [('higher', ['no', 'yes'], ['не хочет', 'хочет'], 'Планирует учиться дальше'),
        ('school', ['MS', 'GP'], ['MS', 'GP'], 'Учебное заведение'),
        ('failures', [0, 1, 2, 3], ['0', '1', '2', '3'], 'Прошлые незачёты')]
for ax, (col, order, labels, title) in zip(axes, cats):
    r = df_por.groupby(col)['no_pass'].agg(['mean', 'size']).reindex(order)
    ax.bar(labels, r['mean'].values, color='#e59866')
    ax.axhline(df_por.no_pass.mean(), ls='--', c='k', lw=.8,
               label=f'в среднем {df_por.no_pass.mean():.0%}')
    for i, (m, n) in enumerate(zip(r['mean'], r['size'])):
        if pd.notna(m):
            ax.text(i, m + .015, f"{m:.0%}\n(n={int(n)})", ha='center', fontsize=7.5)
    ax.set_ylabel('доля незачётов'); ax.set_title(title); ax.legend(fontsize=7)
    ax.set_ylim(0, min(1.0, max(r['mean'].dropna()) * 1.45))
fig.suptitle('Доля незачётов по группам — три главных фактора', y=1.06, fontsize=9)
plt.tight_layout(); plt.show()
""")

md("### 2.3 Матрица корреляций и мультиколлинеарность")

code(r"""
num = ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime',
       'goout','Dalc','Walc','health','absences','G1','G2','G3']
bin_map = {'schoolsup':'yes','famsup':'yes','paid':'yes','higher':'yes','romantic':'yes',
           'internet':'yes','activities':'yes','address':'U','sex':'F','school':'GP'}
C = df_por[num].copy()
for c, pos in bin_map.items():
    C[c] = (df_por[c] == pos).astype(int)
C['no_pass'] = df_por['no_pass']
corr = C.corr()

fig, ax = plt.subplots(figsize=(10.5, 8.5))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
labels = [ru(c) for c in corr.columns]
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]
        if abs(v) > .3 and i != j:
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=6)
plt.colorbar(im, fraction=.046)
ax.set_title('Матрица корреляций (числовые + закодированные бинарные)')
ax.grid(False)
plt.tight_layout(); plt.show()

print("Сильные пары признаков (|r| > 0.5, без G3 и no_pass):")
m = corr.drop(index=['G3','no_pass'], columns=['G3','no_pass']).abs()
for a in m.columns:
    for b in m.columns:
        if a < b and m.loc[a, b] > .5:
            print(f"  {ru(a):28s} - {ru(b):28s} r = {corr.loc[a,b]:+.2f}")
print("\nСильнее всего связаны с целевой переменной:")
print(corr['no_pass'].drop(['no_pass','G3']).abs().sort_values(ascending=False).head(10)
      .rename(index=ru).round(3).to_string())
""")

md(r"""
**Вывод по разделу 2.** Классы перекошены: незачётов 15.4 %, поэтому accuracy бесполезна,
а базовый уровень PR-AUC равен 0.154, а не 0.5. Пропусков и дубликатов в данных нет.
`G1` и `G2` — это одна и та же величина, измеренная дважды, и на уровне 1 они делят
важность между собой. `Dalc` и `Walc` тоже коррелируют, поэтому в разделе 4 сворачиваем
их в один суммарный признак. Для деревьев мультиколлинеарность не ломает качество, но
**размывает важности**: два похожих признака делят вклад пополам и оба выглядят слабее,
чем есть. Поэтому в разделе 9 важности считаются по **исходным переменным**, а не по
колонкам после кодирования.
""")

# ==========================================================================
md(r"""
## 3. Утечки: что нельзя пускать в признаки

1. **`G3`** — целевая переменная сделана прямо из него. Модель с `G3` внутри даст
   ROC-AUC ≈ 1.0 и будет бесполезна. Это главная ловушка датасета.
2. **`no_pass`** — та же величина в бинарном виде.
3. **Любые производные от них.** На уровне 2 вместе с `G1` и `G2` вычищаются и все
   инженерные признаки, построенные на оценках (`G_diff`, `G_mean`, `G_proj`, `G1_fail`, ...).
   За это отвечает параметр `level` функции `engineer()` — она просто не создаёт эти колонки.
4. **Утечка через препроцессинг.** Импьютация, скейлинг, кодирование категорий, отбор
   признаков и калибровка обязаны фититься **только на обучающем фолде**. Поэтому всё лежит
   внутри `Pipeline`, а не применяется к полному `X` до кросс-валидации.
5. **Утечка через подбор.** Гиперпараметры, подобранные по той же CV, по которой отчитываемся,
   завышают метрику. Протокол честной оценки — в разделе 5.3.

Ниже — проверка, что `G3` действительно не попадает в признаки, и демонстрация масштаба
утечки, если правило нарушить.
""")

code(r"""
FEATURE_BLACKLIST = ["G3", "no_pass"]

def get_Xy(d):
    X = d.drop(columns=FEATURE_BLACKLIST)
    y = d["no_pass"].to_numpy()
    return X, y

X_por, y_por = get_Xy(df_por)
print("колонок на входе:", X_por.shape[1], "| строк:", X_por.shape[0])
assert not (set(FEATURE_BLACKLIST) & set(X_por.columns)), "УТЕЧКА: целевая переменная в признаках!"
print("проверка пройдена: G3 и no_pass в признаки не попадают")

leak_cv = cross_validate(
    Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
    df_por[["G3"]], y_por,
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")
print(f"\nЕСЛИ оставить один только G3 в признаках: ROC-AUC = {leak_cv['test_score'].mean():.4f}")
print("Формально идеально, практически бесполезно — модель просто пересказывает ответ.")
""")
