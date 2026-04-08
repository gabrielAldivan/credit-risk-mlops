"""
Production drift monitoring using Evidently AI.

Compares a reference dataset (training baseline) against
a current production window to detect:
- Data drift (feature distribution shifts)
- Target drift (label distribution shift)
- Data quality issues (nulls, out-of-range values)

Outputs an HTML report and raises an alert if drift exceeds threshold.
"""
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
DRIFT_THRESHOLD = 0.3   # fraction of drifted features to trigger alert


def load_reference() -> pd.DataFrame:
    """Training data = reference (baseline) distribution."""
    X = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    y = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet")).squeeze()
    ref = X.copy()
    ref["target"] = y.values
    return ref


def simulate_production_data(reference: pd.DataFrame, drift: bool = False) -> pd.DataFrame:
    """
    Simulate a production batch.
    When drift=True, injects distribution shifts to test monitoring.
    """
    np.random.seed(42)
    n = 200
    current = reference.sample(n=n, replace=True, random_state=42).copy()

    if drift:
        # Simulate a portfolio shift: higher-risk applicants entering
        num_cols = current.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols[:5]:
            current[col] = current[col] + np.random.normal(1.5, 0.5, n)
        print("[SIMULATION] Drift injected into production batch.")
    else:
        print("[SIMULATION] No drift injected — production batch mirrors training distribution.")

    return current


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Generate Evidently drift report and return summary dict."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset(),
        DatasetDriftMetric(),
    ])

    report.run(reference_data=reference, current_data=current)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(REPORTS_DIR, f"drift_report_{timestamp}.html")
    report.save_html(html_path)
    print(f"Report saved: {html_path}")

    result = report.as_dict()
    return result, html_path


def parse_drift_summary(result: dict) -> dict:
    """Extract key drift indicators from Evidently report."""
    summary = {}
    for metric in result.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            r = metric.get("result", {})
            summary["dataset_drift_detected"] = r.get("dataset_drift", False)
            summary["drifted_features"] = r.get("number_of_drifted_columns", 0)
            summary["total_features"] = r.get("number_of_columns", 0)
            summary["drift_share"] = r.get("share_of_drifted_columns", 0.0)
            break
    return summary


def send_alert(summary: dict):
    """
    Alert when drift exceeds threshold.
    In production: replace with Teams/Slack webhook call.
    """
    msg = (
        f"[DRIFT ALERT] {summary['drifted_features']}/{summary['total_features']} features drifted "
        f"({summary['drift_share']:.1%}). "
        f"Dataset drift detected: {summary['dataset_drift_detected']}. "
        f"Consider retraining the model."
    )
    print(f"\n{'!'*60}")
    print(msg)
    print(f"{'!'*60}\n")

    # --- Production webhook (Teams example) ---
    # import requests
    # webhook_url = os.environ["TEAMS_WEBHOOK_URL"]
    # requests.post(webhook_url, json={"text": msg})


def run(inject_drift: bool = False):
    print("Loading reference dataset (training baseline)...")
    reference = load_reference()

    print("Simulating production batch...")
    current = simulate_production_data(reference, drift=inject_drift)

    print("Running Evidently drift report...")
    result, html_path = run_drift_report(reference, current)

    summary = parse_drift_summary(result)
    print(f"\nDrift Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if summary.get("drift_share", 0) > DRIFT_THRESHOLD or summary.get("dataset_drift_detected"):
        send_alert(summary)
    else:
        print("\n[OK] No significant drift detected. Model remains stable.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject-drift", action="store_true",
                        help="Inject artificial drift for testing")
    args = parser.parse_args()
    run(inject_drift=args.inject_drift)
