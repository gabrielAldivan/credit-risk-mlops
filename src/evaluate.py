"""
Model evaluation — loads the best registered model from MLflow
and generates a comprehensive evaluation report with SHAP explainability.

Artifacts produced:
  - precision_recall_curve.png
  - shap_summary.png          (beeswarm — global feature importance + direction)
  - shap_feature_importance.png  (bar chart — top-15 mean |SHAP|)
  - shap_waterfall_sample_<N>.png  (per-applicant explanation, high-risk + low-risk)
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")   # headless — no display needed in CI or Docker

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, precision_recall_curve,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_data():
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet")).squeeze()
    return X_test, y_test


# ── Model loading ─────────────────────────────────────────────────────────────

def load_production_model(model_name: str = "credit-risk-XGBoost", stage: str = "latest"):
    """Load pyfunc wrapper from MLflow registry (used for predict/predict_proba)."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{model_name}/{stage}"
    print(f"Loading model from: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def load_xgboost_native(model_name: str = "credit-risk-XGBoost", stage: str = "latest"):
    """Load raw XGBoost model — required for SHAP TreeExplainer (exact, not kernel)."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.xgboost.load_model(model_uri)


# ── Standard evaluation plots ─────────────────────────────────────────────────

def plot_precision_recall(y_true, y_proba, save_dir: str) -> str:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="darkorange")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(alpha=0.3)
    path = os.path.join(save_dir, "precision_recall_curve.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Saved: {path}")
    return path


# ── SHAP explainability ───────────────────────────────────────────────────────

def plot_shap_summary(shap_values: np.ndarray, X_test: pd.DataFrame, save_dir: str) -> str:
    """
    Beeswarm plot — shows each feature's impact on every prediction.
    Red = high feature value pushes toward bad credit; blue = low value.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    path = os.path.join(save_dir, "shap_summary.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_shap_bar(shap_values: np.ndarray, X_test: pd.DataFrame, save_dir: str) -> str:
    """
    Bar chart of mean |SHAP| per feature — top-15 global importance ranking.
    Answers: which features matter most across all predictions?
    """
    mean_abs = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns)
        .nlargest(15)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    mean_abs.plot(kind="barh", ax=ax, color="#1565C0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 Features — Global Importance (SHAP)")
    ax.grid(axis="x", alpha=0.3)
    path = os.path.join(save_dir, "shap_feature_importance.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_shap_waterfall(
    shap_values: np.ndarray,
    expected_value: float,
    X_test: pd.DataFrame,
    idx: int,
    save_dir: str,
    label: str = "",
) -> str:
    """
    Waterfall plot for a single applicant — shows how each feature
    pushes the prediction above or below the baseline expected value.
    """
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=expected_value,
        data=X_test.iloc[idx].values,
        feature_names=X_test.columns.tolist(),
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    suffix = f"_{label}" if label else ""
    path = os.path.join(save_dir, f"shap_waterfall_sample{suffix}_{idx}.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Saved: {path}")
    return path


def run_shap_analysis(
    model,
    X_test: pd.DataFrame,
    save_dir: str,
    log_to_mlflow: bool = False,
) -> np.ndarray:
    """
    Full SHAP pipeline using TreeExplainer (exact Shapley values for tree models).

    Produces:
      - summary beeswarm (global, all features)
      - bar chart top-15 (global ranking)
      - waterfall for the highest-risk applicant in the test set
      - waterfall for the lowest-risk applicant in the test set
    """
    print("\nRunning SHAP explainability analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    paths = [
        plot_shap_summary(shap_values, X_test, save_dir),
        plot_shap_bar(shap_values, X_test, save_dir),
    ]

    # Pick one high-risk and one low-risk example for individual explanations
    proba = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(proba))
    low_risk_idx = int(np.argmin(proba))

    paths.append(
        plot_shap_waterfall(shap_values, explainer.expected_value,
                            X_test, high_risk_idx, save_dir, label="high_risk")
    )
    paths.append(
        plot_shap_waterfall(shap_values, explainer.expected_value,
                            X_test, low_risk_idx, save_dir, label="low_risk")
    )

    if log_to_mlflow:
        for p in paths:
            mlflow.log_artifact(p)

    print(f"SHAP analysis complete — {len(paths)} plots saved to {save_dir}")
    return shap_values


# ── Main entrypoint ───────────────────────────────────────────────────────────

def run():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    X_test, y_test = load_test_data()

    xgb_model = None
    try:
        pyfunc_model = load_production_model()
        xgb_model = load_xgboost_native()
        y_proba = pyfunc_model.predict(X_test)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
    except Exception as exc:
        print(f"MLflow unavailable ({exc}). Training fresh model for evaluation.")
        sys.path.insert(0, os.path.dirname(__file__))
        from train import train_xgboost, load_data  # noqa: PLC0415
        X_train, _, y_train, _ = load_data()
        xgb_model, _, y_pred, y_proba, _ = train_xgboost(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Good Credit", "Bad Credit"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    plot_precision_recall(y_test, y_proba, ARTIFACTS_DIR)

    if xgb_model is not None:
        run_shap_analysis(xgb_model, X_test, ARTIFACTS_DIR)
    else:
        print("SHAP skipped — native XGBoost model not available.")

    print("\nEvaluation complete. Artifacts saved to:", ARTIFACTS_DIR)


if __name__ == "__main__":
    run()
