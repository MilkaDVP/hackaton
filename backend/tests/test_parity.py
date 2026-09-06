"""Проверка, что продукт считает ровно то же, что ноутбук.

Два независимых инварианта:

1. Артефакт == пайплайн, собранный из кода ноутбука и обученный тем же сидом.
   Расхождение < 1e-6. Это ловит любую попытку «улучшить» модель по дороге.

2. Метрики 5x5 CV из артефакта == метрики из results.json ноутбука, Δ <= 0.01.

Почему не сравниваем напрямую с числами из ноутбука по строкам: в ноутбуке
вероятности — это out-of-fold предсказания (`cross_val_predict`), где каждая
строка предсказана моделью, которая её не видела. Модель в продукте обучена
на всех данных, поэтому на тех же строках её вероятности ОБЯЗАНЫ отличаться —
это разные величины, а не расхождение реализаций.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from riskml.pipeline import final_model, get_Xy

TOL_PROBA = 1e-6
TOL_METRIC = 0.01


@pytest.fixture(scope="module")
def meta(root):
    return json.loads((root / "riskml" / "artifacts" / "metadata.json")
                      .read_text(encoding="utf-8"))


def test_artifact_matches_notebook_pipeline(root, por_df, meta):
    """Пересобираем модель из кода пайплайна и сверяем предсказания с артефактом."""
    X, y = get_Xy(por_df)
    drop = tuple(meta["drop_features"])

    rebuilt = CalibratedClassifierCV(final_model("L2", drop), method="isotonic", cv=3)
    rebuilt.fit(X, y)

    artifact = joblib.load(root / "riskml" / "artifacts" / "model_l2.joblib")

    p_new = rebuilt.predict_proba(X)[:, 1]
    p_old = artifact.predict_proba(X)[:, 1]

    diff = np.abs(p_new - p_old).max()
    assert diff < TOL_PROBA, (
        f"артефакт разошёлся с пайплайном ноутбука: max|Δp| = {diff:.2e}")


def test_drop_features_match_notebook(meta):
    """Список выброшенных признаков — тот же, что посчитал ноутбук (раздел 4.3)."""
    expected = {
        "parent_edu_mean", "log_absences", "study_minus_free", "any_support",
        "alc_total", "alc_weekday_share", "fail_x_study", "goout_x_alc",
        "n_support", "age_over_17", "parent_edu_max", "abs_zero", "abs_bin",
        "abs_per_study",
    }
    assert set(meta["drop_features"]) == expected


def test_metrics_match_results_json(root, meta):
    ref_path = root / "research" / "results.json"
    if not ref_path.exists():
        pytest.skip("results.json ноутбука недоступен")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    pairs = [("L1", "level1"), ("L2", "level2")]
    for level, key in pairs:
        for ours, theirs in (("roc_auc", "roc"), ("pr_auc", "pr")):
            d = abs(meta["metrics"][level][ours] - ref[key][theirs])
            assert d <= TOL_METRIC, f"{level}.{ours} разошлось на {d:.4f}"


def test_threshold_is_from_capacity_not_half(meta):
    assert meta["threshold"]["default"] != 0.5
    assert meta["threshold"]["capacity"] == 40


def test_no_target_in_expected_columns(meta):
    never = set(meta["expected_columns"]["never_a_feature"])
    assert "G3" in never and "no_pass" in never
    assert not (never & set(meta["expected_columns"]["required_l2"]))
