"""
Unit tests for src/train.py.

Covers:
  - compute_metrics: all keys, values in valid ranges, perfect classifier
  - train_logistic_regression: return shape, AUC ≥ chance, binary preds, proba [0,1]
  - train_xgboost: same guarantees
  - plot_roc_curve / plot_confusion_matrix: artifacts written to disk
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import (
    compute_metrics,
    train_logistic_regression,
    train_xgboost,
    plot_roc_curve,
    plot_confusion_matrix,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_preprocessed_data(
    n_train: int = 300,
    n_test: int = 80,
    n_features: int = 30,
    seed: int = 42,
):
    """Synthetic preprocessed feature matrices (already scaled + encoded)."""
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame(
        rng.standard_normal((n_train, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    X_test = pd.DataFrame(
        rng.standard_normal((n_test, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y_train = pd.Series(rng.integers(0, 2, n_train))
    y_test = pd.Series(rng.integers(0, 2, n_test))
    return X_train, X_test, y_train, y_test


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    EXPECTED_KEYS = {"roc_auc", "f1", "precision", "recall", "log_loss"}

    def test_returns_all_keys(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert self.EXPECTED_KEYS == set(metrics.keys())

    def test_auc_in_unit_interval(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.05, 0.1, 0.9, 0.95])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_log_loss_is_positive(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.6, 0.4])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert metrics["log_loss"] > 0


# ── train_logistic_regression ─────────────────────────────────────────────────

class TestTrainLogisticRegression:
    def test_returns_five_tuple(self):
        result = train_logistic_regression(*make_preprocessed_data())
        assert len(result) == 5

    def test_auc_at_least_chance(self):
        _, metrics, _, _, _ = train_logistic_regression(*make_preprocessed_data())
        assert metrics["roc_auc"] >= 0.35

    def test_predictions_are_binary(self):
        _, _, y_pred, _, _ = train_logistic_regression(*make_preprocessed_data())
        assert set(y_pred).issubset({0, 1})

    def test_probabilities_in_unit_interval(self):
        _, _, _, y_proba, _ = train_logistic_regression(*make_preprocessed_data())
        assert (y_proba >= 0.0).all() and (y_proba <= 1.0).all()

    def test_params_contain_expected_keys(self):
        _, _, _, _, params = train_logistic_regression(*make_preprocessed_data())
        assert "C" in params
        assert "class_weight" in params


# ── train_xgboost ─────────────────────────────────────────────────────────────

class TestTrainXGBoost:
    def test_returns_five_tuple(self):
        result = train_xgboost(*make_preprocessed_data())
        assert len(result) == 5

    def test_auc_at_least_chance(self):
        _, metrics, _, _, _ = train_xgboost(*make_preprocessed_data())
        assert metrics["roc_auc"] >= 0.35

    def test_predictions_are_binary(self):
        _, _, y_pred, _, _ = train_xgboost(*make_preprocessed_data())
        assert set(y_pred).issubset({0, 1})

    def test_probabilities_in_unit_interval(self):
        _, _, _, y_proba, _ = train_xgboost(*make_preprocessed_data())
        assert (y_proba >= 0.0).all() and (y_proba <= 1.0).all()

    def test_params_logged(self):
        _, _, _, _, params = train_xgboost(*make_preprocessed_data())
        assert "n_estimators" in params
        assert "learning_rate" in params
        assert "scale_pos_weight" in params

    def test_xgboost_outperforms_random(self):
        """On a linearly separable subset, XGBoost should clearly beat chance."""
        rng = np.random.default_rng(0)
        n = 400
        # Feature 0 perfectly correlated with label
        X = pd.DataFrame(rng.standard_normal((n, 10)), columns=[f"f{i}" for i in range(10)])
        y = (X["f0"] > 0).astype(int)
        X_tr, X_te = X.iloc[:300], X.iloc[300:]
        y_tr, y_te = y.iloc[:300], y.iloc[300:]
        _, metrics, _, _, _ = train_xgboost(X_tr, y_tr, X_te, y_te)
        assert metrics["roc_auc"] > 0.85


# ── Plot helpers ──────────────────────────────────────────────────────────────

class TestPlotHelpers:
    def test_roc_curve_file_created(self, tmp_path):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        path = plot_roc_curve(y_true, y_proba, "TestModel", str(tmp_path))
        assert os.path.isfile(path)
        assert path.endswith(".png")

    def test_confusion_matrix_file_created(self, tmp_path):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        path = plot_confusion_matrix(y_true, y_pred, "TestModel", str(tmp_path))
        assert os.path.isfile(path)
        assert path.endswith(".png")
