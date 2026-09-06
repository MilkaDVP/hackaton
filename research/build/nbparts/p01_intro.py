# -*- coding: utf-8 -*-
from nbparts._h import md, code

# ==========================================================================
md(r"""
# Кто не получит зачёт — решение

**Задача.** По анкете и учебному поведению студента предсказать, кто не получит зачёт
(`G3 < 10`), и назвать **три фактора**, которые влияют сильнее всего.

**Два уровня.**

| Уровень | Признаки | Когда работает | Ориентир ТЗ |
|---|---|---|---|
| 1 — базовый | все, включая `G1`, `G2` | конец семестра | ROC-AUC ≈ 0.97 |
| 2 — сложный | **без `G1`, `G2`** | до первой контрольной | ROC-AUC ≈ 0.83 |

Уровень 2 — главный: именно он оставляет куратору время что-то сделать.

**Как читать метрики.** Везде `среднее ± стандартное отклонение` по
**повторяющейся** стратифицированной кросс-валидации (`RepeatedStratifiedKFold`, 5 фолдов × 5 повторов
= 25 оценок). Одиночный train/test-сплит на 649 строках даёт разброс ±0.05 и ничего не доказывает.

**Две метрики.** ROC-AUC — насколько хорошо модель *ранжирует* студентов.
PR-AUC — насколько чист будет список, который получит куратор. При 15 % незачётов
PR-AUC информативнее: у случайной модели он равен доле положительного класса (0.154), а не 0.5.
""")

md("## 0. Загрузка, версии, фиксация случайности")

code(r"""
import sys, os, time, json, warnings, platform
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn

t_start = time.time()
SEED = 42
np.random.seed(SEED)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 200)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.axisbelow": True})

print("Python     ", sys.version.split()[0], "|", platform.system())
for m in (np, pd, sklearn, matplotlib):
    print(f"{m.__name__:12s}", m.__version__)
""")

md(r"""
Опциональные библиотеки подключаются через `try/except`. Если их нет — ноутбук
не падает, а честно откатывается на `HistGradientBoostingClassifier` из sklearn
и на permutation importance вместо SHAP. Ниже печатается, что реально доступно.
""")

code(r"""
HAS = {}
try:
    from lightgbm import LGBMClassifier; HAS["lightgbm"] = True
except Exception: HAS["lightgbm"] = False
try:
    from xgboost import XGBClassifier;  HAS["xgboost"] = True
except Exception: HAS["xgboost"] = False
try:
    from catboost import CatBoostClassifier; HAS["catboost"] = True
except Exception: HAS["catboost"] = False
try:
    import shap; HAS["shap"] = True
except Exception: HAS["shap"] = False
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS["imblearn"] = True
except Exception: HAS["imblearn"] = False

print("доступные опциональные библиотеки:")
for k, v in HAS.items():
    print(f"  {k:10s} {'есть' if v else 'НЕТ — используется фолбэк'}")

# Тяжёлый перебор гиперпараметров выключен: его результат вписан в BEST_PARAMS ниже.
# Поставьте True, чтобы воспроизвести поиск (это занимает ~20 минут).
RUN_HEAVY_SEARCH = False
""")

code(r"""
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_validate,
                                     cross_val_predict, train_test_split, RandomizedSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             classification_report, precision_recall_curve, roc_curve,
                             brier_score_loss)

def data_path(name):
    # Данные лежат в data/ в корне репозитория. Ищем и оттуда, и из research/,
    # чтобы ноутбук запускался и из корня, и из своей папки.
    for cand in (Path("data") / name, Path("..") / "data" / name, Path(name)):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"не найден {name}: положите данные в data/")

df_por = pd.read_csv(data_path("student-por.csv"), sep=";")   # ВАЖНО: разделитель ';'
df_mat = pd.read_csv(data_path("student-mat.csv"), sep=";")
print("португальский:", df_por.shape, " математика:", df_mat.shape)
df_por.head(3)
""")

# ==========================================================================
md(r"""
## 1. Целевая переменная и разбор `G3 == 0`

Готовой колонки «зачёт / незачёт» в файле нет. Порог зачёта в португальской системе — 10 из 20.
""")

code(r"""
for d in (df_por, df_mat):
    d["no_pass"] = (d["G3"] < 10).astype(int)

print("ПОРТУГАЛЬСКИЙ")
print(df_por["no_pass"].value_counts().rename({0: "зачёт", 1: "незачёт"}).to_string())
print("доля незачётов:", round(df_por["no_pass"].mean(), 4))
print("\nМАТЕМАТИКА")
print(df_mat["no_pass"].value_counts().rename({0: "зачёт", 1: "незачёт"}).to_string())
print("доля незачётов:", round(df_mat["no_pass"].mean(), 4))
""")

md(r"""
### 1.1 Пятнадцать студентов с `G3 == 0`

`G3 == 0` — это почти наверняка не «написал работу на ноль», а «человек не дошёл до итога».
Вопрос, который надо решить явно: «бросил» и «не сдал» — одно событие или разные?
Смотрим на их оценки за контрольные и на пропуски.
""")

code(r"""
z = df_por[df_por.G3 == 0]
print(f"студентов с G3 == 0: {len(z)}\n")
print(z[["G1", "G2", "G3", "absences", "studytime", "failures", "higher", "schoolsup"]].to_string())

print("\nиз них имели проходной балл на 2-й контрольной (G2 >= 10):", int((z.G2 >= 10).sum()))
print("имели G2 == 0 (то есть пропали уже ко 2-й контрольной):", int((z.G2 == 0).sum()))
print("имели G1 == 0:", int((z.G1 == 0).sum()))
""")

code(r"""
cols = ["G1", "G2", "absences", "failures", "studytime", "goout"]
grp = pd.DataFrame({
    f"G3==0 (n={len(z)})": z[cols].mean(),
    f"незачёт, G3>0 (n={int(((df_por.no_pass==1)&(df_por.G3>0)).sum())})":
        df_por[(df_por.no_pass == 1) & (df_por.G3 > 0)][cols].mean(),
    f"зачёт (n={int((df_por.no_pass==0).sum())})":
        df_por[df_por.no_pass == 0][cols].mean(),
}).round(2)
print(grp.to_string())
""")
