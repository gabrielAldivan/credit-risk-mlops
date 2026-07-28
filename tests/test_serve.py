"""
Smoke tests for the FastAPI serving layer (src/serve.py).

Exercises the /health, /predict and /predict/batch endpoints end-to-end with a
stubbed model (deterministic predict_proba) but the REAL preprocessing pipeline
(scaler + feature columns from src/preprocess.py output), so the request →
preprocess → score → response path is covered without needing MLflow.

Requires src/preprocess.py to have run first (produces X_train.parquet + scaler.pkl),
which is the case in CI (preprocessing step precedes pytest).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import serve  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


EXAMPLE = {
    "duration": 24,
    "credit_amount": 5000.0,
    "installment_rate": 2,
    "residence_since": 2,
    "age": 35,
    "existing_credits": 1,
    "num_dependents": 1,
    "checking_account": "A11",
    "credit_history": "A32",
    "purpose": "A43",
    "savings_account": "A61",
    "employment_since": "A73",
    "personal_status": "A93",
    "other_debtors": "A101",
    "property": "A121",
    "other_installments": "A143",
    "housing": "A152",
    "job": "A173",
    "telephone": "A192",
    "foreign_worker": "A201",
}


class _StubModel:
    """Deterministic stand-in for the XGBoost model."""

    def predict_proba(self, X):
        return np.array([[0.3, 0.7]] * len(X))


@pytest.fixture
def client(monkeypatch):
    import joblib
    import pandas as pd

    processed = serve.PROCESSED_DIR
    x_path = os.path.join(processed, "X_train.parquet")
    scaler_path = os.path.join(processed, "scaler.pkl")
    if not (os.path.exists(x_path) and os.path.exists(scaler_path)):
        pytest.skip("processed artifacts missing — run src/preprocess.py first")

    def fake_load():
        serve._feature_columns = pd.read_parquet(x_path).columns.tolist()
        serve._scaler = joblib.load(scaler_path)
        serve._model = _StubModel()

    monkeypatch.setattr(serve, "_load_artifacts", fake_load)
    with TestClient(serve.app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    assert "model_name" in r.json()


def test_predict_returns_valid_score(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_label"] in ("good", "bad")
    assert 0.0 <= body["probability_bad"] <= 1.0
    assert 0 <= body["risk_score"] <= 1000


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"applicants": [EXAMPLE, EXAMPLE]})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["predictions"]) == 2


def test_predict_rejects_invalid_age(client):
    bad = dict(EXAMPLE)
    bad["age"] = 5  # violates ge=18
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
