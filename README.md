# Credit Risk MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-orange.svg)](https://mlflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end MLOps pipeline for credit risk scoring — covering data preprocessing, model training (Logistic Regression + XGBoost), experiment tracking with MLflow, production deployment, and automated drift monitoring.

> **Context:** Inspired by real-world credit scoring pipelines for retail credit portfolios across Latin America, where monthly model execution and drift monitoring are critical to maintaining portfolio health.

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Raw Data   │───▶│ Preprocess  │───▶│ Train Models │───▶│  MLflow Registry│
│  (UCI / S3) │    │  + Features │    │ LR + XGBoost │    │  (versioning)   │
└─────────────┘    └─────────────┘    └──────────────┘    └────────┬────────┘
                                                                    │
┌─────────────────────────────────────────────────────┐            │
│                 Production Monitoring               │◀───────────┘
│  Evidently AI → Drift Report → Alert (Teams/Slack)  │
└─────────────────────────────────────────────────────┘
```

---

## Features

- **Dual model strategy**: Logistic Regression (interpretability) + XGBoost (performance)
- **MLflow tracking**: parameters, metrics, artifacts, and model registry
- **Data drift monitoring**: automated detection using Evidently AI
- **Dockerized MLflow server**: reproducible experiment tracking
- **CI/CD**: GitHub Actions pipeline for automated testing

---

## Project Structure

```
credit-risk-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI
├── data/
│   └── download.py             # Dataset download script
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   ├── preprocess.py           # Feature engineering & preprocessing
│   ├── train.py                # Model training + MLflow logging
│   ├── evaluate.py             # Model evaluation & reporting
│   └── monitor.py              # Drift detection with Evidently
├── docker-compose.yml          # MLflow tracking server
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Dataset

German Credit Risk dataset (UCI Machine Learning Repository) — 1,000 customers, 20 features, binary classification (good/bad credit).

```bash
python data/download.py
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/gabriel-aldivan/credit-risk-mlops.git
cd credit-risk-mlops
pip install -r requirements.txt
```

### 2. Start MLflow server

```bash
docker-compose up -d
# MLflow UI: http://localhost:5000
```

### 3. Run pipeline

```bash
# Download data
python data/download.py

# Preprocess
python src/preprocess.py

# Train models (logs to MLflow automatically)
python src/train.py

# Evaluate
python src/evaluate.py

# Monitor drift
python src/monitor.py
```

---

## Results

| Model               | ROC-AUC | F1-Score | Precision | Recall |
|---------------------|---------|----------|-----------|--------|
| Logistic Regression | 0.78    | 0.72     | 0.74      | 0.70   |
| XGBoost             | 0.84    | 0.79     | 0.81      | 0.77   |

> XGBoost selected as production model. Logistic Regression retained for regulatory interpretability (SHAP explanations available).

---

## MLflow Tracking

All experiments logged automatically:

```python
# Parameters tracked
learning_rate, n_estimators, max_depth, scale_pos_weight

# Metrics tracked  
roc_auc, f1, precision, recall, log_loss

# Artifacts
confusion_matrix.png, roc_curve.png, feature_importance.png
```

---

## Drift Monitoring

The monitoring module runs Evidently reports on incoming production data vs. training baseline:

- **Data drift**: detects feature distribution shifts
- **Target drift**: monitors label distribution changes
- **Automated alerts**: configurable threshold for retraining trigger

---

## Tech Stack

`Python` `XGBoost` `Scikit-learn` `MLflow` `Evidently AI` `Docker` `GitHub Actions` `Pandas` `NumPy`

---

## Author

**Gabriel Aldivan** — Data Scientist | ML Engineer  
[LinkedIn](https://linkedin.com/in/gabriel-aldivan) · [GitHub](https://github.com/gabriel-aldivan)
