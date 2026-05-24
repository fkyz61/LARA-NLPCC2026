"""Download and preprocess the UCI Census-Income (KDD) dataset (id=117).

Two binary tasks per the paper:
    T1 = income > $50K
    T2 = never-married

Uses the official `ucimlrepo` Python package to fetch the canonical
data, avoiding hard-coded URLs that may break.

Saves x/p/y train/val tensors to `out_dir` consumable by lara_trainer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch via the official UCI ML Repository client ───────────────
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise SystemExit(
            "ucimlrepo is required. Install with: pip install ucimlrepo"
        ) from e

    print("Fetching UCI Census-Income (KDD), dataset id 117 ...")
    bundle = fetch_ucirepo(id=117)
    df_x: pd.DataFrame = bundle.data.features
    df_y: pd.DataFrame = bundle.data.targets

    # Combine features and target so we can access the marital column.
    df = pd.concat([df_x, df_y], axis=1).dropna()
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")

    # ── Two binary labels (paper Section 4.1) ─────────────────────────
    # Target column name in ucimlrepo is "income".
    target_col = df_y.columns[0]
    y1 = (df[target_col].astype(str).str.strip() == "50000+.").astype(int).values

    marital_candidates = [c for c in df.columns
                          if "marital" in c.lower() or "marri" in c.lower()]
    if not marital_candidates:
        raise RuntimeError("Could not find a marital-status column in the dataset.")
    marital_col = marital_candidates[0]
    y2 = (df[marital_col].astype(str)
                    .str.contains("Never married", case=False, na=False)).astype(int).values
    y = np.stack([y1, y2], axis=1)

    drop_cols = {target_col, marital_col}
    feature_df = df.drop(columns=list(drop_cols))

    # ── Build feature matrix: one-hot categoricals + standardized continuous
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_df.columns if c not in cat_cols]

    cat = pd.get_dummies(feature_df[cat_cols], dummy_na=False).values.astype(np.float32)
    num = feature_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    num = StandardScaler().fit_transform(num).astype(np.float32)
    X = np.concatenate([cat, num], axis=1)
    print(f"Feature matrix: {X.shape}")

    # ── User-context vector p_u: 16-d projection over numeric columns ─
    # Pick up to 8 numeric columns, standardize, pad to 16-d to match the
    # `user_ctx_dim: 16` setting in configs/lara_uci.yaml.
    pu_src = num[:, :8] if num.shape[1] >= 8 else num
    pu_src = np.concatenate([pu_src,
                              np.zeros((pu_src.shape[0], 16 - pu_src.shape[1]),
                                       dtype=np.float32)], axis=1)

    # 60/40 train/val split (paper uses 60/20/20 with separate test;
    # for the public reproduction we keep a single 60/40 split).
    x_tr, x_va, p_tr, p_va, y_tr, y_va = train_test_split(
        X, pu_src, y, test_size=0.4, random_state=42, stratify=y[:, 0]
    )

    torch.save(torch.from_numpy(x_tr), args.out_dir / "x_train.pt")
    torch.save(torch.from_numpy(x_va), args.out_dir / "x_val.pt")
    torch.save(torch.from_numpy(p_tr), args.out_dir / "p_train.pt")
    torch.save(torch.from_numpy(p_va), args.out_dir / "p_val.pt")
    torch.save(torch.from_numpy(y_tr), args.out_dir / "y_train.pt")
    torch.save(torch.from_numpy(y_va), args.out_dir / "y_val.pt")

    print(f"Saved tensors to {args.out_dir}")
    print(f"Train: x={x_tr.shape} p={p_tr.shape} y={y_tr.shape}")
    print(f"Val:   x={x_va.shape} p={p_va.shape} y={y_va.shape}")
    print()
    print("NOTE: in_dim in configs/lara_uci.yaml may need to be set to "
          f"{X.shape[1]} (the actual one-hot dimension after preprocessing).")


if __name__ == "__main__":
    main()
