"""
Credit Risk Scoring API — FastAPI inference server.

Endpoints:
  GET  /health           liveness check
  GET  /model/info       model metadata (name, version, AUC)
  POST /predict          score a single applicant
  POST /predict/batch    score up to 500 applicants

Run locally:
  uvicorn src.serve:app --reload --port 8000

Docker (add to docker-compose.yml):
  serve:
    build: .
    command: uvicorn src.serve:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    depends_on: [mlflow]

Production (AWS):
  Replace local model fallback with SageMaker endpoint or S3-stored artifact.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger("credit_risk_api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "credit-risk-XGBoost")
RISK_THRESHOLD = float(os.environ.get("RISK_THRESHOLD", "0.5"))


# ── Request / Response schemas ────────────────────────────────────────────────

class ApplicantFeatures(BaseModel):
    """Raw German Credit dataset fields — same schema as model training input."""

    duration: int = Field(..., ge=1, description="Loan duration in months")
    credit_amount: float = Field(..., gt=0, description="Loan amount (DM)")
    installment_rate: int = Field(..., ge=1, le=4)
    residence_since: int = Field(..., ge=1, le=4)
    age: int = Field(..., ge=18, le=100)
    existing_credits: int = Field(..., ge=1)
    num_dependents: int = Field(..., ge=0)
    checking_account: str
    credit_history: str
    purpose: str
    savings_account: str
    employment_since: str
    personal_status: str
    other_debtors: str
    property: str
    other_installments: str
    housing: str
    job: str
    telephone: str
    foreign_worker: str

    model_config = {"json_schema_extra": {"example": {
        "duration": 24, "credit_amount": 5000.0, "installment_rate": 2,
        "residence_since": 2, "age": 35, "existing_credits": 1,
        "num_dependents": 1, "checking_account": "A11", "credit_history": "A32",
        "purpose": "A43", "savings_account": "A61", "employment_since": "A73",
        "personal_status": "A93", "other_debtors": "A101", "property": "A121",
        "other_installments": "A143", "housing": "A152", "job": "A173",
        "telephone": "A192", "foreign_worker": "A201",
    }}}


class PredictionResponse(BaseModel):
    risk_label: str             # "good" or "bad"
    probability_bad: float      # P(default) — higher = riskier
    risk_score: int             # 0–1000 scorecard; higher = safer applicant


class BatchRequest(BaseModel):
    applicants: List[ApplicantFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int


class ModelInfo(BaseModel):
    model_name: str
    tracking_uri: str
    threshold: float


# ── Application & state ───────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Predicts default probability for loan applicants using an XGBoost model "
        "trained on the German Credit Risk dataset. Inspired by real behaviour scoring "
        "pipelines (Regressão Logística + XGBoost) deployed across 8 LATAM countries."
    ),
    version="1.0.0",
)

_model = None          # XGBClassifier
_scaler = None         # StandardScaler (fitted on training data)
_feature_columns: Optional[List[str]] = None   # ordered columns from X_train


def _load_artifacts():
    """Load model + preprocessing artifacts at startup."""
    global _model, _scaler, _feature_columns

    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    _feature_columns = X_train.columns.tolist()
    _scaler = joblib.load(os.path.join(PROCESSED_DIR, "scaler.pkl"))

    try:
        import mlflow.xgboost
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        _model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/latest")
        logger.info("Model loaded from MLflow registry: %s", MODEL_NAME)
    except Exception as exc:
        logger.warning("MLflow unavailable (%s). Falling back to local training.", exc)
        from train import train_xgboost, load_data
        X_train_df, X_test_df, y_train, y_test = load_data()
        _model, _, _, _, _ = train_xgboost(X_train_df, y_train, X_test_df, y_test)
        logger.info("Fallback model trained locally.")


@app.on_event("startup")
def startup_event():
    _load_artifacts()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(applicant: ApplicantFeatures) -> pd.DataFrame:
    """
    Convert a single applicant to the feature vector used during training.
    Applies the same encoding + scaling as src/preprocess.py.
    """
    from preprocess import engineer_features, CATEGORICAL_COLS, NUMERICAL_COLS

    row = pd.DataFrame([applicant.model_dump()])
    row = engineer_features(row)

    df_enc = pd.get_dummies(row, columns=CATEGORICAL_COLS + ["age_group"], drop_first=True)
    df_enc = df_enc.reindex(columns=_feature_columns, fill_value=0)

    num_present = [c for c in NUMERICAL_COLS + ["credit_per_month"] if c in df_enc.columns]
    df_enc[num_present] = _scaler.transform(df_enc[num_present])

    return df_enc


def _score(applicant: ApplicantFeatures) -> PredictionResponse:
    X = _preprocess(applicant)
    prob_bad = float(_model.predict_proba(X)[0, 1])
    label = "bad" if prob_bad >= RISK_THRESHOLD else "good"
    risk_score = int(round((1.0 - prob_bad) * 1000))
    return PredictionResponse(
        risk_label=label,
        probability_bad=round(prob_bad, 4),
        risk_score=risk_score,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    """Liveness check — returns 200 if the service and model are loaded."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info", response_model=ModelInfo, tags=["ops"])
def model_info():
    """Returns metadata about the currently loaded model."""
    return ModelInfo(
        model_name=MODEL_NAME,
        tracking_uri=MLFLOW_URI,
        threshold=RISK_THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(applicant: ApplicantFeatures):
    """
    Score a single loan applicant.

    Returns:
    - **risk_label**: "good" (approved) or "bad" (denied)
    - **probability_bad**: model's estimated default probability (0–1)
    - **risk_score**: scorecard value (0–1000, higher = safer)
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    try:
        return _score(applicant)
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(request: BatchRequest):
    """
    Score a batch of applicants (max 500 per request).

    Useful for offline batch scoring pipelines or nightly portfolio reviews.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if len(request.applicants) > 500:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 500.")
    try:
        predictions = [_score(a) for a in request.applicants]
        return BatchResponse(predictions=predictions, total=len(predictions))
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
