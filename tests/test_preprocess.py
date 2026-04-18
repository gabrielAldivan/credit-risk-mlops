"""
Unit tests for src/preprocess.py.

Covers:
  - engineer_features: new columns added, correct formula, no nulls
  - preprocess: output shape, no NaN, target removed, scaler reuse
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import engineer_features, preprocess, CATEGORICAL_COLS, NUMERICAL_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_sample_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Minimal German Credit-style DataFrame for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "duration":           rng.integers(6, 72, n),
        "credit_amount":      rng.integers(500, 10_000, n).astype(float),
        "installment_rate":   rng.integers(1, 5, n),
        "residence_since":    rng.integers(1, 5, n),
        "age":                rng.integers(20, 70, n),
        "existing_credits":   rng.integers(1, 4, n),
        "num_dependents":     rng.integers(1, 3, n),
        "checking_account":   rng.choice(["A11", "A12", "A13", "A14"], n),
        "credit_history":     rng.choice(["A30", "A31", "A32", "A33", "A34"], n),
        "purpose":            rng.choice(["A40", "A41", "A42", "A43", "A49"], n),
        "savings_account":    rng.choice(["A61", "A62", "A63", "A64", "A65"], n),
        "employment_since":   rng.choice(["A71", "A72", "A73", "A74", "A75"], n),
        "personal_status":    rng.choice(["A91", "A92", "A93", "A94"], n),
        "other_debtors":      rng.choice(["A101", "A102", "A103"], n),
        "property":           rng.choice(["A121", "A122", "A123", "A124"], n),
        "other_installments": rng.choice(["A141", "A142", "A143"], n),
        "housing":            rng.choice(["A151", "A152", "A153"], n),
        "job":                rng.choice(["A171", "A172", "A173", "A174"], n),
        "telephone":          rng.choice(["A191", "A192"], n),
        "foreign_worker":     rng.choice(["A201", "A202"], n),
        "target":             rng.integers(0, 2, n),
    })


# ── engineer_features ─────────────────────────────────────────────────────────

class TestEngineerFeatures:
    def test_adds_credit_per_month(self):
        df = make_sample_df(20)
        result = engineer_features(df)
        assert "credit_per_month" in result.columns

    def test_adds_age_group(self):
        df = make_sample_df(20)
        result = engineer_features(df)
        assert "age_group" in result.columns

    def test_credit_per_month_formula(self):
        df = make_sample_df(30)
        result = engineer_features(df)
        expected = df["credit_amount"] / df["duration"].clip(lower=1)
        pd.testing.assert_series_equal(
            result["credit_per_month"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_no_nan_in_engineered_cols(self):
        df = make_sample_df(50)
        result = engineer_features(df)
        assert result["credit_per_month"].isna().sum() == 0

    def test_age_groups_only_valid_labels(self):
        df = make_sample_df(100)
        result = engineer_features(df)
        valid = {"young", "adult", "middle", "senior"}
        actual = set(result["age_group"].dropna().astype(str).unique())
        assert actual.issubset(valid)

    def test_does_not_modify_original(self):
        df = make_sample_df(20)
        original_cols = list(df.columns)
        engineer_features(df)
        assert list(df.columns) == original_cols


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_target_removed_from_X(self):
        df = make_sample_df(50)
        X, _, _ = preprocess(df, fit_scaler=True)
        assert "target" not in X.columns

    def test_output_lengths_match(self):
        df = make_sample_df(50)
        X, y, _ = preprocess(df, fit_scaler=True)
        assert len(X) == len(y) == 50

    def test_target_is_binary(self):
        df = make_sample_df(50)
        _, y, _ = preprocess(df, fit_scaler=True)
        assert set(y.unique()).issubset({0, 1})

    def test_no_nan_in_feature_matrix(self):
        df = make_sample_df(60)
        X, _, _ = preprocess(df, fit_scaler=True)
        assert X.isna().sum().sum() == 0

    def test_numerical_cols_are_scaled(self):
        df = make_sample_df(200)
        X, _, _ = preprocess(df, fit_scaler=True)
        # After standard scaling, |mean| should be close to 0 and std close to 1
        for col in ["duration", "age"]:
            if col in X.columns:
                assert abs(X[col].mean()) < 0.5, f"{col} mean not near 0"
                assert 0.5 < X[col].std() < 2.0, f"{col} std far from 1"

    def test_scaler_reuse_aligns_column_count(self):
        df = make_sample_df(100)
        X_train, _, scaler = preprocess(df.iloc[:80].copy(), fit_scaler=True)
        X_test, _, _ = preprocess(df.iloc[80:].copy(), scaler=scaler, fit_scaler=False)
        assert X_train.shape[1] == X_test.shape[1], (
            "Train and test feature matrices must have the same number of columns"
        )

    def test_feature_count_is_positive(self):
        df = make_sample_df(40)
        X, _, _ = preprocess(df, fit_scaler=True)
        assert X.shape[1] > 10, "Encoded feature matrix should have many columns"
