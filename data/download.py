"""
Download the German Credit Risk dataset from UCI ML Repository.
Saves raw data to data/raw/german_credit.csv
"""
import os
import urllib.request
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose", "credit_amount",
    "savings_account", "employment_since", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property", "age", "other_installments",
    "housing", "existing_credits", "job", "num_dependents", "telephone",
    "foreign_worker", "target"
]


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, "german_credit.csv")

    if os.path.exists(out_path):
        print(f"Dataset already exists at {out_path}")
        return

    print("Downloading German Credit Risk dataset...")
    urllib.request.urlretrieve(URL, os.path.join(RAW_DIR, "german.data"))

    df = pd.read_csv(
        os.path.join(RAW_DIR, "german.data"),
        sep=" ",
        header=None,
        names=COLUMNS
    )
    # UCI encodes target as 1=good, 2=bad — convert to 0=good, 1=bad
    df["target"] = df["target"].map({1: 0, 2: 1})
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    download()
