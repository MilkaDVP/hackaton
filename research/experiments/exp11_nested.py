"""Эксперимент 11: ВЛОЖЕННАЯ CV всей процедуры «тюним членов -> собираем ансамбль».

Это и есть честное число для защиты: внутри каждого внешнего фолда подбор
запускается заново и никогда не видит внешний тест.
"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

from common import load, get_Xy, make_pipe, cv_obj, SEED, DATA_POR

DROP_L2 = ("fail_x_study", "n_support", "any_support", "log_absences", "parent_edu_max",
           "study_minus_free", "age_bin", "age_over_17", "abs_bin", "goout_x_alc",
           "abs_zero", "abs_per_study")

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 2


class TunedEnsemble(ClassifierMixin, BaseEstimator):
    """Тюнит трёх членов на ТОМ ЖЕ train, что ей дали, и усредняет их голоса.

    Так как весь подбор происходит внутри fit(), вложенная CV измеряет
    процедуру целиком, а не заранее подобранные параметры.
    """
    def __init__(self, level="L2", n_iter=20, seed=SEED):
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
        for name, model, sc, space in specs:
            s = RandomizedSearchCV(make_pipe(model, self.level, scale=sc, exclude=DROP_L2),
                                   space, n_iter=self.n_iter, scoring="roc_auc",
                                   cv=inner, n_jobs=-1, random_state=self.seed)
            s.fit(X, y)
            members.append((name, s.best_estimator_))
        self.ens_ = VotingClassifier(members, voting="soft")
        # члены уже обучены на этом же train; VotingClassifier переобучит их — это нормально
        self.ens_.fit(X, y)
        self.classes_ = self.ens_.classes_
        return self

    def predict_proba(self, X):
        return self.ens_.predict_proba(X)

    def predict(self, X):
        return self.ens_.predict(X)


df = load(DATA_POR)
X, y = get_Xy(df)

t0 = time.time()
res = cross_validate(TunedEnsemble(level="L2", n_iter=N_ITER), X, y,
                     cv=cv_obj(5, N_REP, SEED),
                     scoring=["roc_auc", "average_precision"], n_jobs=1, verbose=1)
roc, pr = res["test_roc_auc"], res["test_average_precision"]
print(f"\nВЛОЖЕННАЯ CV (внешние 5x{N_REP}, внутри {N_ITER} итераций поиска на члена):")
print(f"  ROC-AUC {roc.mean():.4f} ± {roc.std():.4f}")
print(f"  PR-AUC  {pr.mean():.4f} ± {pr.std():.4f}")
print(f"  время {time.time()-t0:.0f}s")

pd.DataFrame({"roc": roc, "pr": pr}).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_11_nested.csv"),
    index=False, encoding="utf-8")
