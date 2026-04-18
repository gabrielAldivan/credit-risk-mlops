"""
Unit tests for src/monitor.py.

Covers:
  - simulate_production_data: shape, columns, drift injection
  - parse_drift_summary: correct key extraction from Evidently result dict
  - send_alert: prints to stdout, does not raise
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from monitor import simulate_production_data, parse_drift_summary, send_alert


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_reference_data(n: int = 500, n_num_features: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Minimal reference dataset that mirrors the expected structure used by monitor.py.
    Numeric features + target column.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.standard_normal((n, n_num_features)),
        columns=[f"feature_{i}" for i in range(n_num_features)],
    )
    df["target"] = rng.integers(0, 2, n)
    return df


# ── simulate_production_data ──────────────────────────────────────────────────

class TestSimulateProductionData:
    def test_output_has_200_rows(self):
        ref = make_reference_data()
        current = simulate_production_data(ref, drift=False)
        assert len(current) == 200

    def test_columns_match_reference(self):
        ref = make_reference_data()
        current = simulate_production_data(ref, drift=False)
        assert set(ref.columns) == set(current.columns)

    def test_no_drift_keeps_similar_distribution(self):
        ref = make_reference_data()
        current = simulate_production_data(ref, drift=False)
        num_cols = ref.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols[:5]:
            # Sampled data mean should stay close to reference mean
            diff = abs(current[col].mean() - ref[col].mean())
            assert diff < 2.0, f"Column '{col}' mean shifted too much without drift"

    def test_drift_shifts_feature_distributions(self):
        ref = make_reference_data()
        no_drift = simulate_production_data(ref, drift=False)
        with_drift = simulate_production_data(ref, drift=True)

        num_cols = with_drift.select_dtypes(include=[np.number]).columns.tolist()[:5]
        # Drifted data should be systematically shifted upward on multiple features
        shifted = sum(
            with_drift[col].mean() > no_drift[col].mean() + 0.3
            for col in num_cols
        )
        assert shifted >= 3, (
            f"Expected ≥3 features to shift with drift injection, got {shifted}"
        )

    def test_output_is_a_dataframe(self):
        ref = make_reference_data()
        current = simulate_production_data(ref, drift=False)
        assert isinstance(current, pd.DataFrame)


# ── parse_drift_summary ───────────────────────────────────────────────────────

class TestParseDriftSummary:
    def test_empty_metrics_returns_empty_dict(self):
        result = {"metrics": []}
        summary = parse_drift_summary(result)
        assert isinstance(summary, dict)
        assert len(summary) == 0

    def test_extracts_all_drift_keys(self):
        result = {
            "metrics": [{
                "metric": "DatasetDriftMetric",
                "result": {
                    "dataset_drift": True,
                    "number_of_drifted_columns": 7,
                    "number_of_columns": 20,
                    "share_of_drifted_columns": 0.35,
                },
            }]
        }
        summary = parse_drift_summary(result)
        assert summary["dataset_drift_detected"] is True
        assert summary["drifted_features"] == 7
        assert summary["total_features"] == 20
        assert summary["drift_share"] == pytest.approx(0.35)

    def test_no_drift_scenario(self):
        result = {
            "metrics": [{
                "metric": "DatasetDriftMetric",
                "result": {
                    "dataset_drift": False,
                    "number_of_drifted_columns": 0,
                    "number_of_columns": 20,
                    "share_of_drifted_columns": 0.0,
                },
            }]
        }
        summary = parse_drift_summary(result)
        assert summary["dataset_drift_detected"] is False
        assert summary["drift_share"] == pytest.approx(0.0)

    def test_ignores_unrelated_metrics(self):
        result = {
            "metrics": [
                {"metric": "DataDriftPreset", "result": {}},
                {
                    "metric": "DatasetDriftMetric",
                    "result": {
                        "dataset_drift": False,
                        "number_of_drifted_columns": 1,
                        "number_of_columns": 10,
                        "share_of_drifted_columns": 0.1,
                    },
                },
            ]
        }
        summary = parse_drift_summary(result)
        assert summary["total_features"] == 10


# ── send_alert ────────────────────────────────────────────────────────────────

class TestSendAlert:
    def test_does_not_raise(self):
        summary = {
            "drifted_features": 10,
            "total_features": 20,
            "drift_share": 0.5,
            "dataset_drift_detected": True,
        }
        send_alert(summary)  # must not raise

    def test_prints_alert_message(self, capsys):
        summary = {
            "drifted_features": 5,
            "total_features": 20,
            "drift_share": 0.25,
            "dataset_drift_detected": True,
        }
        send_alert(summary)
        captured = capsys.readouterr()
        assert "DRIFT ALERT" in captured.out
        assert "5/20" in captured.out

    def test_prints_retraining_recommendation(self, capsys):
        summary = {
            "drifted_features": 8,
            "total_features": 15,
            "drift_share": 0.53,
            "dataset_drift_detected": True,
        }
        send_alert(summary)
        captured = capsys.readouterr()
        assert "retraining" in captured.out.lower()
