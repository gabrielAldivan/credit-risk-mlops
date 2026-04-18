"""
Model training with MLflow experiment tracking.

Trains two models:
- Logistic Regression (interpretability / regulatory requirement)
- XGBoost (performance), with optional Optuna hyperparameter search

All parameters, metrics, and artifacts are logged to MLflow.
Best model is registered to MLflow Model Registry.

Usage:
  python src/train.py              # fixed params (fast)
  python src/train.py --tune       # Optuna search, 50 trials (slower, better AUC)
  python src/train.py --tune --trials 100
"""
import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, log_loss, confusion_matrix, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "credit-risk-scoring"


def load_data():
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet")).squeeze()
    y_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet")).squeeze()
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
    }


def plot_roc_curve(y_true, y_proba, model_name: str, save_dir: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    path = os.path.join(save_dir, f"roc_curve_{model_name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Good", "Bad"]); ax.set_yticklabels(["Good", "Bad"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


def train_logistic_regression(X_train, y_train, X_test, y_test):
    params = {
        "C": 0.1,
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": 42,
    }
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    return model, metrics, y_pred, y_proba, params


def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost with fixed hyperparameters — fast baseline."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": neg / pos,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    return model, metrics, y_pred, y_proba, params


# ── Optuna hyperparameter search ──────────────────────────────────────────────

def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50) -> dict:
    """
    Bayesian hyperparameter search with Optuna (TPE sampler).

    Objective: maximise mean ROC-AUC on 5-fold stratified CV.
    Search space covers the parameters that most affect XGBoost performance
    on imbalanced credit-risk datasets.

    Returns the best params dict ready to pass to XGBClassifier.
    """
    import optuna  # noqa: PLC0415 — lazy import keeps startup fast when not tuning
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": neg / pos,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "scale_pos_weight": neg / pos,
        "eval_metric": "logloss",
        "random_state": 42,
    })

    print(f"\nOptuna finished — best CV ROC-AUC: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    return best


def train_xgboost_tuned(X_train, y_train, X_test, y_test, n_trials: int = 50):
    """XGBoost trained on Optuna-selected hyperparameters."""
    print(f"Running Optuna hyperparameter search ({n_trials} trials, 5-fold CV)...")
    params = tune_xgboost(X_train, y_train, n_trials=n_trials)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    return model, metrics, y_pred, y_proba, params


def run_experiment(model_name: str, train_fn, X_train, y_train, X_test, y_test):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        model, metrics, y_pred, y_proba, params = train_fn(X_train, y_train, X_test, y_test)

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log artifacts
        roc_path = plot_roc_curve(y_test, y_proba, model_name, ARTIFACTS_DIR)
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name, ARTIFACTS_DIR)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)

        # Log model
        if "xgb" in model_name.lower():
            mlflow.xgboost.log_model(model, artifact_path="model",
                                     registered_model_name=f"credit-risk-{model_name}")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     registered_model_name=f"credit-risk-{model_name}")

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow Run ID: {run_id}")

    return metrics


def run(tune: bool = False, n_trials: int = 50):
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training on {len(X_train)} samples, {X_train.shape[1]} features")

    print("\nTraining Logistic Regression...")
    lr_metrics = run_experiment(
        "LogisticRegression",
        train_logistic_regression,
        X_train, y_train, X_test, y_test,
    )

    if tune:
        print(f"\nTraining XGBoost (Optuna, {n_trials} trials)...")
        xgb_train_fn = lambda Xtr, ytr, Xte, yte: train_xgboost_tuned(  # noqa: E731
            Xtr, ytr, Xte, yte, n_trials=n_trials
        )
        xgb_metrics = run_experiment(
            "XGBoost-Tuned", xgb_train_fn, X_train, y_train, X_test, y_test
        )
    else:
        print("\nTraining XGBoost (fixed params — use --tune for Optuna search)...")
        xgb_metrics = run_experiment(
            "XGBoost",
            train_xgboost,
            X_train, y_train, X_test, y_test,
        )

    winner = "XGBoost" if xgb_metrics["roc_auc"] > lr_metrics["roc_auc"] else "LogisticRegression"
    print(f"\nBest model by ROC-AUC: {winner}")
    print("Check MLflow UI at http://localhost:5000 for full experiment details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search for XGBoost")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()
    run(tune=args.tune, n_trials=args.trials)
