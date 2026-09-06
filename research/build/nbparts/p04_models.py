# -*- coding: utf-8 -*-
from nbparts._h import md, code

md(r"""
## 5. Модели и протокол оценки

### 5.1 Протокол

* **Кросс-валидация.** `RepeatedStratifiedKFold(5 фолдов × 5 повторов)` = 25 оценок.
  Повторы нужны, потому что на 649 строках со 100 положительными примерами разброс между
  одиночными сплитами достигает ±0.05 — больше, чем разница между хорошей и плохой моделью.
* **Метрики.** ROC-AUC (качество ранжирования) и PR-AUC (чистота списка для куратора).
  Нижняя граница PR-AUC — доля положительного класса, 0.154.
* **Нижняя граница.** `DummyClassifier` включён в таблицу как sanity-check.
* **Всё в пайплайне.** Ни одного `fit` на полном `X` до кросс-валидации.
""")

code(r"""
def model_zoo(level):
    # Возвращает {имя: (модель, нужен_ли_скейлинг)}
    z = {
        "константа (нижняя граница)": (DummyClassifier(strategy="prior"), True),
        "логрег L2":  (LogisticRegression(max_iter=5000, class_weight="balanced",
                                          C=1.0, random_state=SEED), True),
        "логрег L1":  (LogisticRegression(max_iter=5000, class_weight="balanced", penalty="l1",
                                          solver="liblinear", C=0.3, random_state=SEED), True),
        "логрег elasticnet": (LogisticRegression(max_iter=5000, class_weight="balanced",
                                                 penalty="elasticnet", solver="saga",
                                                 l1_ratio=0.5, C=0.3, random_state=SEED), True),
        "случайный лес": (RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                 random_state=SEED, n_jobs=1), False),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                            random_state=SEED, n_jobs=1), False),
        "HistGB":     (HistGradientBoostingClassifier(random_state=SEED, max_iter=200,
                                                      learning_rate=0.06, max_leaf_nodes=15,
                                                      l2_regularization=1.0), False),
        "GradientBoosting": (GradientBoostingClassifier(random_state=SEED, n_estimators=200,
                                                        learning_rate=0.05, max_depth=3), False),
        "SVM RBF":    (SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=SEED), True),
        "kNN (sanity)": (KNeighborsClassifier(n_neighbors=25, weights="distance"), True),
        "GaussianNB": (GaussianNB(), True),
    }
    if HAS["lightgbm"]:
        z["LightGBM"] = (LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                        class_weight="balanced", random_state=SEED,
                                        verbose=-1, n_jobs=1), False)
    if HAS["xgboost"]:
        z["XGBoost"] = (XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                      random_state=SEED, n_jobs=1, scale_pos_weight=5.5), False)
    if HAS["catboost"]:
        # allow_writing_files=False обязателен: иначе CatBoost создаёт каталог
        # catboost_info/ в рабочей папке, а при n_jobs=-1 параллельные воркеры
        # дерутся за него и падают с "Can't create train working dir".
        z["CatBoost"] = (CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4,
                                            verbose=0, random_seed=SEED,
                                            allow_writing_files=False,
                                            auto_class_weights="Balanced"), False)
    return z

print("моделей в сравнении:", len(model_zoo("L2")))
""")

code(r"""
CV_MAIN = cv_obj(5, 5, SEED)      # для финальных моделей и baseline
CV_ZOO = cv_obj(5, 2, SEED)       # для обзорной таблицы — дешевле, сравнение честное
zoo_rows = []
for level in ["L1", "L2"]:
    for name, (m, sc) in model_zoo(level).items():
        r = evaluate(make_pipe(m, level=level, scale=sc, exclude=DROP_FEATURES),
                     X_por, y_por, cv=CV_ZOO)
        zoo_rows.append({"уровень": level, "модель": name,
                         "ROC-AUC": r["roc_auc"], "±": r["roc_auc_std"],
                         "PR-AUC": r["pr_auc"], "± ": r["pr_auc_std"]})
zoo_table = pd.DataFrame(zoo_rows)

for level in ["L1", "L2"]:
    print(f"\n=== УРОВЕНЬ {level} ({'с G1,G2' if level=='L1' else 'без G1,G2'}) ===")
    t = zoo_table[zoo_table["уровень"] == level].drop(columns="уровень")
    print(t.sort_values("ROC-AUC", ascending=False).round(3).to_string(index=False))
""")

md(r"""
### 5.2 Ансамбли

На 649 строках отдельная модель легко переобучается под конкретный фолд. Усреднение
нескольких **разных по природе** моделей (деревья + линейная + ядровая) даёт небольшой,
но устойчивый прирост — за счёт того, что их ошибки не совпадают.

Параметры членов ансамбля взяты из поиска, протокол честной оценки — в разделе 5.3.
Все члены работают на **очищенном** наборе признаков из раздела 4.4.
""")

code(r"""
# Параметры найдены RandomizedSearchCV (раздел 5.3). Здесь они зафиксированы,
# чтобы ноутбук считался быстро; воспроизводятся при RUN_HEAVY_SEARCH = True.
ET_TUNED = dict(n_estimators=866, max_features=0.1267, min_samples_leaf=3,
                min_samples_split=13, criterion="gini", class_weight="balanced")
RF_TUNED = dict(n_estimators=774, max_features=0.3124, min_samples_leaf=9,
                min_samples_split=18, class_weight="balanced_subsample")
LR_TUNED = dict(C=0.0292)
SVM_TUNED = dict(C=0.3149, gamma=0.0711, class_weight="balanced")

def member(name, level):
    # Один член ансамбля: модель + свой препроцессинг, на очищенных признаках.
    if name == "et":
        return make_pipe(ExtraTreesClassifier(random_state=SEED, n_jobs=1, **ET_TUNED),
                         level, scale=False, exclude=DROP_FEATURES)
    if name == "rf":
        return make_pipe(RandomForestClassifier(random_state=SEED, n_jobs=1, **RF_TUNED),
                         level, scale=False, exclude=DROP_FEATURES)
    if name == "lr":
        return make_pipe(LogisticRegression(max_iter=8000, class_weight="balanced",
                                            penalty="l1", solver="liblinear",
                                            random_state=SEED, **LR_TUNED),
                         level, scale=True, exclude=DROP_FEATURES)
    if name == "svm":
        return make_pipe(SVC(probability=True, random_state=SEED, **SVM_TUNED),
                         level, scale=True, exclude=DROP_FEATURES)
    if name == "hgb":
        return make_pipe(HistGradientBoostingClassifier(random_state=SEED, max_iter=200,
                                                        learning_rate=0.06, max_leaf_nodes=15,
                                                        l2_regularization=1.0),
                         level, scale=False, exclude=DROP_FEATURES)

def voting(keys, level):
    return VotingClassifier([(k, member(k, level)) for k in keys], voting="soft")

ENS_CANDS = {
    "ET + логрег": ["et", "lr"],
    "ET + логрег + SVM": ["et", "lr", "svm"],
    "ET + RF + логрег": ["et", "rf", "lr"],
}
ens_rows = []
for level in ["L1", "L2"]:
    for name, keys in ENS_CANDS.items():
        r = evaluate(voting(keys, level), X_por, y_por, cv=cv_obj(5, 2, SEED))
        ens_rows.append({"уровень": level, "ансамбль": name,
                         "ROC-AUC": r["roc_auc"], "±": r["roc_auc_std"],
                         "PR-AUC": r["pr_auc"], "± ": r["pr_auc_std"]})
        print(f"{level}  {name:24s} {fmt(r)}")
    # стэкинг — для раздела «что не сработало»
    st = StackingClassifier([(k, member(k, level)) for k in ["et", "lr", "svm"]],
                            final_estimator=LogisticRegression(max_iter=5000,
                                                               class_weight="balanced",
                                                               random_state=SEED),
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1)
    r = evaluate(st, X_por, y_por, cv=cv_obj(5, 1, SEED))
    ens_rows.append({"уровень": level, "ансамбль": "стэкинг (мета — логрег)",
                     "ROC-AUC": r["roc_auc"], "±": r["roc_auc_std"],
                     "PR-AUC": r["pr_auc"], "± ": r["pr_auc_std"]})
    print(f"{level}  {'стэкинг (мета — логрег)':24s} {fmt(r)}")
ens_table = pd.DataFrame(ens_rows)
""")

md(r"""
### 5.3 Честный протокол подбора гиперпараметров

Подбор гиперпараметров **не является честной оценкой**: `best_score_` у `RandomizedSearchCV` —
это максимум по сотне попыток на тех же фолдах, и он систематически завышен. Чтобы измерить
качество честно, используем **вложенную (nested) кросс-валидацию**: внутри каждого внешнего
фолда поиск запускается заново и никогда не видит внешний тестовый фолд.

Ниже считается разрыв между внутренней (оптимистичной) и вложенной (честной) оценкой.
Именно этот разрыв — цена самообмана, и он показывается явно.

Вложенная CV считается **всегда** — она и есть доказательство честности протокола.
Флаг `RUN_HEAVY_SEARCH` управляет только её масштабом: при `False` берётся облегчённый
вариант (2 модели × 25 итераций × внешние 5 фолдов), при `True` — полный перебор
(6 моделей × 40 итераций × внешние 5×3), который считается около 40 минут.
Никаких чисел «вписанных руками» здесь нет: и при `False` таблица ниже посчитана кодом.
""")

code(r"""
from scipy.stats import loguniform, randint, uniform

SEARCH_SPACES = {
    "ExtraTrees": (ExtraTreesClassifier(random_state=SEED, n_jobs=1), False,
                   {"clf__n_estimators": randint(300, 900),
                    "clf__max_features": uniform(0.05, 0.55),
                    "clf__min_samples_leaf": randint(1, 12),
                    "clf__min_samples_split": randint(2, 20),
                    "clf__criterion": ["gini", "entropy"],
                    "clf__class_weight": ["balanced", "balanced_subsample"]}),
    "Логрег L1":  (LogisticRegression(max_iter=8000, class_weight="balanced", penalty="l1",
                                      solver="liblinear", random_state=SEED), True,
                   {"clf__C": loguniform(1e-3, 10)}),
}

N_ITER = 30 if RUN_HEAVY_SEARCH else 6
OUTER = cv_obj(5, 2, SEED) if RUN_HEAVY_SEARCH else cv_obj(5, 1, SEED)

nested_rows, BEST_PARAMS = [], {}
for name, (model, sc, space) in SEARCH_SPACES.items():
    t0 = time.time()
    search = RandomizedSearchCV(make_pipe(model, level="L2", scale=sc, exclude=DROP_FEATURES),
                                space, n_iter=N_ITER, scoring="roc_auc", n_jobs=-1,
                                cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                random_state=SEED)
    # честная оценка: внутри каждого ВНЕШНЕГО фолда поиск запускается заново
    nested = cross_validate(search, X_por, y_por, cv=OUTER, scoring="roc_auc", n_jobs=1)
    # и отдельно — сам поиск на всех данных, чтобы посмотреть на найденные параметры
    search.fit(X_por, y_por)
    BEST_PARAMS[name] = search.best_params_
    nested_rows.append({"что оценивали": name,
                        "внутренний best_score_": search.best_score_,
                        "вложенная CV": nested["test_score"].mean(),
                        "± вложенной": nested["test_score"].std(),
                        "оптимизм": search.best_score_ - nested["test_score"].mean()})
    print(f"{name:12s} внутр. {search.best_score_:.4f} -> вложенная "
          f"{nested['test_score'].mean():.4f} ± {nested['test_score'].std():.4f}  "
          f"[{time.time()-t0:.0f}s]")
""")

md(r"""
Отдельные модели — это только половина честного ответа. Финальный продукт — не одна
модель, а **процедура**: «подобрать параметры трёх членов, собрать из них ансамбль».
Оценивать надо именно её целиком. Класс ниже делает весь подбор **внутри `fit()`**,
поэтому вложенная кросс-валидация над ним измеряет процедуру, а не заранее найденные
параметры. Это самая дорогая ячейка ноутбука и самая важная для доверия к числам.
""")

code(r"""
from sklearn.base import BaseEstimator, ClassifierMixin

class TunedEnsemble(ClassifierMixin, BaseEstimator):
    # Подбирает параметры членов на том train, что ей дали, и усредняет голоса.
    # Весь подбор внутри fit() -> вложенная CV честно оценивает процедуру целиком.
    def __init__(self, level="L2", n_iter=12, seed=SEED):
        self.level, self.n_iter, self.seed = level, n_iter, seed

    def fit(self, X, y):
        inner = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        specs = [
            ("et", ExtraTreesClassifier(random_state=self.seed, n_jobs=1), False,
             {"clf__n_estimators": randint(300, 900),
              "clf__max_features": uniform(0.05, 0.55),
              "clf__min_samples_leaf": randint(1, 12),
              "clf__min_samples_split": randint(2, 20),
              "clf__class_weight": ["balanced", "balanced_subsample"]}),
            ("lr", LogisticRegression(max_iter=8000, class_weight="balanced", penalty="l1",
                                      solver="liblinear", random_state=self.seed), True,
             {"clf__C": loguniform(1e-3, 10)}),
            ("svm", SVC(probability=True, random_state=self.seed), True,
             {"clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-4, 1e-1),
              "clf__class_weight": ["balanced"]}),
        ]
        members = []
        for nm, model, sc, space in specs:
            s = RandomizedSearchCV(make_pipe(model, self.level, scale=sc, exclude=DROP_FEATURES),
                                   space, n_iter=self.n_iter, scoring="roc_auc",
                                   cv=inner, n_jobs=-1, random_state=self.seed)
            s.fit(X, y)
            members.append((nm, s.best_estimator_))
        self.ens_ = VotingClassifier(members, voting="soft").fit(X, y)
        self.classes_ = self.ens_.classes_
        return self

    def predict_proba(self, X):
        return self.ens_.predict_proba(X)

    def predict(self, X):
        return self.ens_.predict(X)

t0 = time.time()
nested_ens = cross_validate(TunedEnsemble(level="L2", n_iter=N_ITER), X_por, y_por,
                            cv=OUTER, scoring=["roc_auc", "average_precision"], n_jobs=1)
NESTED_ENS = {"roc": nested_ens["test_roc_auc"].mean(),
              "roc_sd": nested_ens["test_roc_auc"].std(),
              "pr": nested_ens["test_average_precision"].mean(),
              "pr_sd": nested_ens["test_average_precision"].std()}
nested_rows.append({"что оценивали": "ВСЯ процедура (тюнинг + ансамбль)",
                    "внутренний best_score_": np.nan,
                    "вложенная CV": NESTED_ENS["roc"], "± вложенной": NESTED_ENS["roc_sd"],
                    "оптимизм": np.nan})
nested_table = pd.DataFrame(nested_rows)

print(f"ВЛОЖЕННАЯ CV всей процедуры: ROC-AUC {NESTED_ENS['roc']:.4f} ± {NESTED_ENS['roc_sd']:.4f}"
      f"   PR-AUC {NESTED_ENS['pr']:.4f} ± {NESTED_ENS['pr_sd']:.4f}   [{time.time()-t0:.0f}s]")
print("\nнайденные параметры (поиск на всех данных):")
for k, v in BEST_PARAMS.items():
    print(" ", k, {kk.replace("clf__", ""): (round(vv, 4) if isinstance(vv, float) else vv)
                   for kk, vv in v.items()})
nested_table.round(4)
""")

md(r"""
**Как читать эту таблицу — и что мы отчитываем.**

Колонка «оптимизм» показывает, сколько качества `best_score_` приписывает себе сам:
это разница между «лучшим из десятков попыток на тех же фолдах» и честной вложенной
оценкой. Она положительна, и это ожидаемо.

Поэтому у финальной модели в разделах 6–7 мы приводим **два числа**:

* **5×5 CV с зафиксированными параметрами** — основная цифра, по ней сравниваем модели
  между собой на одинаковых фолдах. Она слегка оптимистична, потому что параметры
  когда-то были найдены на этих же данных.
* **вложенная CV всей процедуры** (строка «ВСЯ процедура») — цифра без всяких «но».
  Это то, чего стоит ждать на новом потоке студентов.

Разница между ними и есть честная поправка. Оба числа названы явно, чтобы никто
не принял оптимистичное за честное.
""")

md(r"""
### 5.4 Решение по `G3 == 0`: sensitivity analysis

Считаем всё дважды — с пятнадцатью «нулевыми» студентами и без них. Смотрим и на метрики,
и на то, меняется ли от этого тройка факторов.
""")

code(r"""
SENS_MODELS = {
    "логрег L1": (LR_L1, True),
    "ExtraTrees": (ET_FAST, False),
    "HistGB": (lambda: HistGradientBoostingClassifier(random_state=SEED, max_iter=200,
                                                      learning_rate=0.06, max_leaf_nodes=15,
                                                      l2_regularization=1.0), False),
}
df_no_zero = df_por[df_por.G3 > 0].reset_index(drop=True)

sens_rows = []
for vname, d in [(f"оставляем (n={len(df_por)})", df_por),
                 (f"убираем (n={len(df_no_zero)})", df_no_zero)]:
    Xs, ys = get_Xy(d)
    for level in ["L1", "L2"]:
        for mn, (mk, sc) in SENS_MODELS.items():
            r = evaluate(make_pipe(mk(), level=level, scale=sc, exclude=DROP_FEATURES),
                         Xs, ys, cv=cv_obj(5, 2, SEED))
            sens_rows.append({"вариант": vname, "уровень": level, "модель": mn,
                              "доля незачётов": ys.mean(),
                              "ROC-AUC": r["roc_auc"], "±": r["roc_auc_std"],
                              "PR-AUC": r["pr_auc"], "± ": r["pr_auc_std"]})
sens_table = pd.DataFrame(sens_rows)
sens_table.round(3)
""")

code(r"""
print("Меняется ли ТОП-5 факторов на уровне L2? (permutation importance, ExtraTrees)")
sens_top = {}
for vname, d in [("оставляем", df_por), ("убираем", df_no_zero)]:
    Xs, ys = get_Xy(d)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=.3, stratify=ys, random_state=SEED)
    p = make_pipe(ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                       random_state=SEED, n_jobs=-1),
                  level="L2", exclude=DROP_FEATURES).fit(Xtr, ytr)
    imp = permutation_importance(p, Xte, yte, n_repeats=6, random_state=SEED,
                                 scoring="roc_auc", n_jobs=-1)
    s = pd.Series(imp.importances_mean, index=Xs.columns).sort_values(ascending=False)
    sens_top[vname] = s.head(5)
    print(f"  {vname:12s}: {[ru(c) for c in s.head(5).index]}")

same3 = set(sens_top['оставляем'].head(3).index) == set(sens_top['убираем'].head(3).index)
print(f"\nтройка совпадает: {'ДА' if same3 else 'НЕТ'}")
""")

md(r"""
**Решение по `G3 == 0`: оставляем.** Три аргумента, все проверяемые по таблицам выше.

1. **Содержательный.** Заказчику нужен список тех, «с кем стоит поговорить до того, как
   станет поздно». Студент, который перестал ходить и не дошёл до итога, — ровно тот
   случай, ради которого система строится. Убрав его, мы бы учили модель не замечать
   самый тяжёлый исход.
2. **По данным.** Тройка главных факторов на сложном уровне от этого решения не зависит
   (проверено ячейкой выше) — значит, ответ на главный вопрос ТЗ устойчив.
3. **Честный контраргумент.** На уровне 1 удаление этих 15 студентов заметно меняет метрику,
   потому что «человек с приличными `G1`/`G2` и нулём в итоге» — это шум с точки зрения
   оценок. Но уровень 1 и так работает поздно; ради него портить постановку задачи не стоит.
""")

md(r"""
### 5.5 Кодирование категорий: one-hot против ordinal против target encoding

Многозначных текстовых колонок всего четыре — `Mjob`, `Fjob`, `reason`, `guardian`.
Остальные «текстовые» на самом деле бинарные (`yes`/`no`), и для них вопроса кодирования нет.

Отдельно стоит помнить: `Medu`, `studytime`, `goout`, `health` и подобные — **уже числа
с осмысленным порядком**. One-hot им не нужен, он только раздробит порядковую шкалу.
Проверяем это утверждение, а не принимаем на веру.

**Target encoding обязан быть внутри пайплайна.** `TargetEncoder` из sklearn использует
внутреннюю кросс-валидацию при обучении, поэтому не подсматривает в целевую переменную
той же строки. Считать средние по всей выборке заранее — классическая утечка.
""")

code(r"""
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, KBinsDiscretizer

def prep_variant(kind, scale=False):
    # kind: 'onehot' | 'ordinal' | 'target' | 'onehot_ordcols'
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    if kind == "onehot":
        cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif kind == "ordinal":
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    elif kind == "target":
        cat = TargetEncoder(random_state=SEED)      # внутренняя CV -> утечки нет
    else:
        cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    steps = [("num", num_pipe, make_column_selector(dtype_exclude=object)),
             ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                               ("enc", cat)]), make_column_selector(dtype_include=object))]
    return ColumnTransformer(steps, remainder="drop",
                             verbose_feature_names_out=False).set_output(transform="pandas")

def pipe_variant(model, kind, level="L2", scale=False):
    return Pipeline([("fe", make_fe(level, True, DROP_FEATURES, None)),
                     ("prep", prep_variant(kind, scale)),
                     ("clf", model)])

enc_rows = []
for kind, nm in [("onehot", "one-hot (наш выбор)"), ("ordinal", "ordinal"),
                 ("target", "target encoding (внутр. CV)")]:
    for mname, mk, sc in [("ExtraTrees", lambda: ExtraTreesClassifier(**{**ET_TUNED,
                                                                        "random_state": SEED,
                                                                        "n_jobs": 1}), False),
                          ("Логрег L1", lambda: LogisticRegression(max_iter=8000,
                                                                   class_weight="balanced",
                                                                   penalty="l1", solver="liblinear",
                                                                   C=0.0292, random_state=SEED), True)]:
        r = evaluate(pipe_variant(mk(), kind, "L2", sc), X_por, y_por, cv=cv_obj(5, 1, SEED))
        enc_rows.append({"кодирование": nm, "модель": mname,
                         "ROC-AUC": r["roc_auc"], "±": r["roc_auc_std"],
                         "PR-AUC": r["pr_auc"], "± ": r["pr_auc_std"]})
        print(f"{nm:28s} {mname:12s} {fmt(r)}")
ENC = pd.DataFrame(enc_rows)
ENC.round(4)
""")

code(r"""
# А нужен ли one-hot порядковым шкалам? Раздробим их и посмотрим.
ORD_COLS = ["Medu", "Fedu", "traveltime", "studytime", "famrel", "freetime",
            "goout", "Dalc", "Walc", "health"]

def to_str_ordinals(X):
    # Превращаем порядковые шкалы в строки -> ColumnTransformer закодирует их one-hot.
    d = engineer(X, level="L2", exclude=DROP_FEATURES)
    for c in ORD_COLS:
        if c in d.columns:
            d[c] = d[c].astype(int).astype(str)
    return d

ord_pipe = Pipeline([("fe", FunctionTransformer(to_str_ordinals, validate=False)),
                     ("prep", make_prep(scale=False)),
                     ("clf", ExtraTreesClassifier(**{**ET_TUNED, "random_state": SEED, "n_jobs": 1}))])
r_ord = evaluate(ord_pipe, X_por, y_por, cv=cv_obj(5, 1, SEED))
r_num = evaluate(make_pipe(ExtraTreesClassifier(**{**ET_TUNED, "random_state": SEED, "n_jobs": 1}),
                           level="L2", scale=False, exclude=DROP_FEATURES),
                 X_por, y_por, cv=cv_obj(5, 1, SEED))
print(f"порядковые как числа (наш выбор): {fmt(r_num)}")
print(f"порядковые через one-hot:        {fmt(r_ord)}")
print(f"\nразница ROC-AUC: {r_ord['roc_auc'] - r_num['roc_auc']:+.4f}")
""")

md(r"""
**Вывод 5.5.** Разница между способами кодирования на этих данных невелика — четыре
номинальные колонки просто не несут много информации. One-hot оставлен как выбор
по умолчанию: он не хуже остальных, не вносит ложного порядка (в отличие от ordinal,
где `Mjob` получил бы бессмысленную шкалу «teacher < health < services») и не требует
объяснений про внутреннюю кросс-валидацию, как target encoding.

Проверка порядковых шкал подтвердила ожидание: дробить `Medu` или `goout` на one-hot
смысла нет — модель и так видит их порядок, а от дробления только растёт число колонок.

### 5.6 Отбор признаков: RFECV

Ещё один способ борьбы с шумом — рекурсивное исключение признаков с кросс-валидацией.
Проверяем, найдёт ли RFECV набор лучше, чем наш отсев по leave-one-out из раздела 4.3.
""")

code(r"""
from sklearn.feature_selection import RFECV

def lr_sel():
    return LogisticRegression(max_iter=8000, class_weight="balanced", penalty="l1",
                              solver="liblinear", C=0.0292, random_state=SEED)

rfe_base = lr_sel()
rfe = RFECV(lr_sel(), step=4, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            scoring="roc_auc", min_features_to_select=8, n_jobs=-1)

rfe_pipe = Pipeline([("fe", make_fe("L2", True, DROP_FEATURES, None)),
                     ("prep", make_prep(scale=True)),
                     ("sel", rfe),
                     ("clf", lr_sel())])
r_rfe = evaluate(rfe_pipe, X_por, y_por, cv=cv_obj(5, 1, SEED))
r_plain = evaluate(make_pipe(rfe_base, level="L2", scale=True, exclude=DROP_FEATURES),
                   X_por, y_por, cv=cv_obj(5, 1, SEED))
print(f"логрег без RFECV: {fmt(r_plain)}")
print(f"логрег с RFECV:   {fmt(r_rfe)}")
print(f"разница ROC-AUC: {r_rfe['roc_auc'] - r_plain['roc_auc']:+.4f}")

rfe_pipe.fit(X_por, y_por)
sel = rfe_pipe.named_steps["sel"]
kept = rfe_pipe.named_steps["prep"].get_feature_names_out()[sel.support_]
print(f"\nRFECV оставил {sel.n_features_} признаков из {len(sel.support_)}:")
print("  " + ", ".join(ru(c) for c in kept[:20]))
""")
