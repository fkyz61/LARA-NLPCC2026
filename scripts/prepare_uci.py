"""Download and preprocess the UCI Census-Income dataset.

Two binary tasks per the paper:
    T1 = income > $50K
    T2 = never-married

Saves x/p/y train/val tensors to `out_dir` consumable by lara_trainer.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

UCI_URL_TRAIN = "https://archive.ics.uci.edu/static/public/117/data.csv"

COLUMNS = [
    "age", "class_of_worker", "industry_code", "occupation_code",
    "education", "wage_per_hour", "enrolled_in_edu", "marital_status",
    "major_industry", "major_occupation", "race", "hispanic_origin",
    "sex", "labor_union", "unemployment_reason", "employment_status",
    "capital_gains", "capital_losses", "dividends_from_stocks",
    "tax_filer_status", "previous_region", "previous_state",
    "household_status", "household_summary", "weight", "migration_msa",
    "migration_reg", "migration_within_reg", "live_one_year",
    "previous_sunbelt", "num_persons_employer", "family_under_18",
    "country_father", "country_mother", "country_self", "citizenship",
    "self_employed", "veteran_admin", "veteran_benefits", "weeks_worked",
    "year", "income",
]

CATEGORICAL = [
    "class_of_worker", "education", "enrolled_in_edu", "marital_status",
    "major_industry", "major_occupation", "race", "hispanic_origin",
    "sex", "labor_union", "unemployment_reason", "employment_status",
    "tax_filer_status", "previous_region", "previous_state",
    "household_status", "household_summary", "migration_msa",
    "migration_reg", "migration_within_reg", "live_one_year",
    "previous_sunbelt", "family_under_18", "country_father",
    "country_mother", "country_self", "citizenship", "self_employed",
    "veteran_admin",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = args.out_dir / "raw.csv"
    if not raw.exists():
        print(f"Downloading UCI Census-Income to {raw}")
        urlretrieve(UCI_URL_TRAIN, raw)

    df = pd.read_csv(raw, header=None, names=COLUMNS, skipinitialspace=True)
    df = df.dropna()

    # Two binary labels.
    y1 = (df["income"].str.strip() == "50000+.").astype(int).values
    y2 = (df["marital_status"].str.contains("Never married",
                                            case=False)).astype(int).values
    y = np.stack([y1, y2], axis=1)

    # Build feature matrix: one-hot categoricals + standardized continuous.
    cat = pd.get_dummies(df[CATEGORICAL], dummy_na=False).values.astype(np.float32)
    cont_cols = [c for c in df.columns
                 if c not in CATEGORICAL + ["income", "marital_status"]]
    cont = df[cont_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    cont = StandardScaler().fit_transform(cont).astype(np.float32)
    X = np.concatenate([cat, cont], axis=1)

    # User-context vector p_u: simple 16-d projection of age, sex (one-hot),
    # weeks_worked, capital_gains, capital_losses, dividends, etc.
    pu_cols = ["age", "weeks_worked", "capital_gains", "capital_losses",
               "dividends_from_stocks", "num_persons_employer", "weight",
               "wage_per_hour"]
    pu = df[pu_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    pu = StandardScaler().fit_transform(pu).astype(np.float32)
    # Pad to 16 dims.
    pu = np.concatenate([pu, np.zeros((pu.shape[0], 16 - pu.shape[1]),
                                       dtype=np.float32)], axis=1)

    x_tr, x_va, p_tr, p_va, y_tr, y_va = train_test_split(
        X, pu, y, test_size=0.4, random_state=42, stratify=y[:, 0]
    )
    # 60/20/20 -> here we keep 60/40 split; for val/test users can re-split.

    torch.save(torch.from_numpy(x_tr), args.out_dir / "x_train.pt")
    torch.save(torch.from_numpy(x_va), args.out_dir / "x_val.pt")
    torch.save(torch.from_numpy(p_tr), args.out_dir / "p_train.pt")
    torch.save(torch.from_numpy(p_va), args.out_dir / "p_val.pt")
    torch.save(torch.from_numpy(y_tr), args.out_dir / "y_train.pt")
    torch.save(torch.from_numpy(y_va), args.out_dir / "y_val.pt")

    print(f"Saved tensors to {args.out_dir}")
    print(f"Train: x={x_tr.shape} p={p_tr.shape} y={y_tr.shape}")
    print(f"Val:   x={x_va.shape} p={p_va.shape} y={y_va.shape}")


if __name__ == "__main__":
    main()
