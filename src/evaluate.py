"""
Model evaluation — loads the best registered model from MLflow
and generates a comprehensive evaluation report.
"""
import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def load_test_data():
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet")).squeeze()
    return X_test, y_test


def load_production_model(model_name: str = "credit-risk-XGBoost", stage: str = "latest"):
    """Load model from MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{model_name}/{stage}"
    print(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def plot_precision_recall(y_true, y_proba, save_dir: str):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
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


def run():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    X_test, y_test = load_test_data()

    try:
        model = load_production_model()
        y_proba = model.predict(X_test)
        # MLflow pyfunc may return labels for classifiers — handle both cases
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
    except Exception as e:
        print(f"Could not load from MLflow ({e}). Running evaluation with a fresh XGBoost model.")
        from xgboost import XGBClassifier
        from train import train_xgboost, load_data
        X_train, X_test, y_train, y_test = load_data()
        model_raw, metrics, y_pred, y_proba, _ = train_xgboost(X_train, y_train, X_test, y_test)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["Good Credit", "Bad Credit"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    plot_precision_recall(y_test, y_proba, ARTIFACTS_DIR)
    print("\nEvaluation complete. Artifacts saved to:", ARTIFACTS_DIR)


if __name__ == "__main__":
    run()
