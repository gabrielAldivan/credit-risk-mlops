"""
Preprocessing pipeline for German Credit Risk dataset.

Handles:
- Categorical encoding (one-hot)
- Numerical scaling (StandardScaler)
- Train/test split with stratification
- Persistence of preprocessed splits for downstream use
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "raw", "german_credit.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

CATEGORICAL_COLS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment_since", "personal_status", "other_debtors", "property",
    "other_installments", "housing", "job", "telephone", "foreign_worker"
]
NUMERICAL_COLS = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents"
]
TARGET_COL = "target"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional domain-relevant features."""
    df = df.copy()
    # Debt-to-income ratio proxy
    df["credit_per_month"] = df["credit_amount"] / df["duration"].clip(lower=1)
    # Age groups
    df["age_group"] = pd.cut(df["age"], bins=[0, 25, 35, 50, 100],
                              labels=["young", "adult", "middle", "senior"])
    return df


def preprocess(df: pd.DataFrame, scaler=None, fit_scaler: bool = True):
    """
    Encode categoricals, scale numericals.
    Returns (X, y, scaler).
    """
    df = engineer_features(df)

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS + ["age_group"], drop_first=True)
    df_encoded = df_encoded.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    num_cols_present = [c for c in NUMERICAL_COLS + ["credit_per_month"] if c in df_encoded.columns]

    if fit_scaler:
        scaler = StandardScaler()
        df_encoded[num_cols_present] = scaler.fit_transform(df_encoded[num_cols_present])
    else:
        df_encoded[num_cols_present] = scaler.transform(df_encoded[num_cols_present])

    return df_encoded, y, scaler


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = load_raw()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET_COL])
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Target distribution (train): {train_df[TARGET_COL].value_counts(normalize=True).to_dict()}")

    X_train, y_train, scaler = preprocess(train_df, fit_scaler=True)
    X_test, y_test, _ = preprocess(test_df, scaler=scaler, fit_scaler=False)

    # Align columns (test may lack some dummies)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Persist
    X_train.to_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"), index=False)
    X_test.to_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"), index=False)
    y_train.to_frame().to_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet"), index=False)
    y_test.to_frame().to_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet"), index=False)
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))

    print(f"Saved processed data to {PROCESSED_DIR}")
    print(f"Feature matrix shape: {X_train.shape}")


if __name__ == "__main__":
    run()
