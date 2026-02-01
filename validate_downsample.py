#!/usr/bin/env python3
"""
Validate control downsampling math to explain n=477 in screening logs.

Expected workflow:
1. Load split files for seed=1
2. Check training set composition (incident/prevalent/controls)
3. Simulate 3-fold inner CV split
4. Show expected sample counts for screening
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    # Load split indices for seed=1
    splits_dir = Path("splits")
    scenario = "IncidentPlusPrevalent"
    seed = 1

    train_file = splits_dir / f"train_idx_{scenario}_seed{seed}.csv"
    val_file = splits_dir / f"val_idx_{scenario}_seed{seed}.csv"
    test_file = splits_dir / f"test_idx_{scenario}_seed{seed}.csv"

    if not train_file.exists():
        print(f"❌ Split file not found: {train_file}")
        print("Run: ced save-splits first")
        return

    # Load split indices (CSV files have 'idx' header)
    train_idx = pd.read_csv(train_file)["idx"].values
    val_idx = pd.read_csv(val_file)["idx"].values
    test_idx = pd.read_csv(test_file)["idx"].values

    # Load full dataset to check labels
    data_file = Path("data/Celiac_dataset_proteomics_w_demo.parquet")
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return

    df_full = pd.read_parquet(data_file)

    print("="*60)
    print("Split Composition (seed=1)")
    print("="*60)
    print(f"Train: {len(train_idx)} samples")
    print(f"Val:   {len(val_idx)} samples")
    print(f"Test:  {len(test_idx)} samples")
    print()

    # Check training set composition (use .loc since indices are actual row numbers)
    train_data = df_full.loc[train_idx]
    target_col = "CeD_comparison"
    train_labels = train_data[target_col].value_counts().to_dict()

    print("Training Set Composition:")
    print(f"  Controls:           {train_labels.get('Controls', 0)}")
    print(f"  Incident CeD:       {train_labels.get('Incident', 0)}")
    print(f"  Prevalent CeD:      {train_labels.get('Prevalent', 0)}")
    print(f"  Total:              {len(train_data)}")

    # Calculate case:control ratio
    n_cases = train_labels.get('Incident', 0) + train_labels.get('Prevalent', 0)
    n_controls = train_labels.get('Controls', 0)
    ratio = n_controls / n_cases if n_cases > 0 else 0
    print(f"  Case:Control Ratio: 1:{ratio:.1f}")
    print()

    # Simulate 3-fold inner CV (stratified)
    print("="*60)
    print("3-Fold Inner CV Simulation (for Optuna trials)")
    print("="*60)

    from sklearn.model_selection import StratifiedKFold

    y_train = train_data[target_col].values
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fold_sizes = []
    for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(skf.split(train_idx, y_train), 1):
        inner_train_data = train_data.iloc[inner_train_idx]
        inner_train_labels = inner_train_data[target_col].value_counts().to_dict()

        n_inner_train = len(inner_train_data)
        fold_sizes.append(n_inner_train)

        print(f"\nInner Fold {fold_idx}:")
        print(f"  Train samples: {n_inner_train} ({100*n_inner_train/len(train_data):.1f}% of {len(train_data)})")
        print(f"    Controls:    {inner_train_labels.get('Controls', 0)}")
        print(f"    Incident:    {inner_train_labels.get('Incident', 0)}")
        print(f"    Prevalent:   {inner_train_labels.get('Prevalent', 0)}")
        print(f"  Val samples:   {len(inner_val_idx)} ({100*len(inner_val_idx)/len(train_data):.1f}%)")

    print()
    print("="*60)
    print("Expected Screening Sample Counts")
    print("="*60)
    print(f"ScreeningTransformer will fit on each inner training fold:")
    for i, size in enumerate(fold_sizes, 1):
        print(f"  Inner fold {i}: n={size} samples")

    print()
    print("Comparison to HPC logs:")
    print("  Log shows: n=477, n=477, n=478")
    print(f"  Expected:  n={fold_sizes[0]}, n={fold_sizes[1]}, n={fold_sizes[2]}")

    diff = abs(fold_sizes[0] - 477)
    if diff < 5:
        print(f"\n✅ MATCH! (diff={diff} samples, likely due to split config differences)")
    else:
        print(f"\n❌ MISMATCH! (diff={diff} samples)")
        print("\nPossible reasons:")
        print("  1. Different split seed used in HPC run")
        print("  2. Different prevalent_train_frac or control downsampling")
        print("  3. Splits regenerated after HPC run started")

    print()
    print("="*60)
    print("Why 3 screening calls instead of 30?")
    print("="*60)
    print("With 10 Optuna trials × 3 inner folds = 30 potential fits,")
    print("you only see 3 screening calls because:")
    print()
    print("  • Trial 1:")
    print("    - Inner fold 1: n=477 [COMPUTE + CACHE]")
    print("    - Inner fold 2: n=477 [COMPUTE + CACHE]")
    print("    - Inner fold 3: n=478 [COMPUTE + CACHE]")
    print()
    print("  • Trials 2-10:")
    print("    - All 3 folds: [CACHE HIT] (same stratified split, deterministic)")
    print()
    print("The screening_cache uses (data_hash, method, top_n) as key.")
    print("Since StratifiedKFold is deterministic (random_state=42),")
    print("all trials use the same 3 inner fold splits → cache hits.")


if __name__ == "__main__":
    main()
