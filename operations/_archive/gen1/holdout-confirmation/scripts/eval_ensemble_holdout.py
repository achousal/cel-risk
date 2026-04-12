"""Evaluate ensemble on holdout by chaining base model predictions.

Replicates the data loading and filtering logic from eval-holdout
but feeds base model predictions through the ensemble meta-learner.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import TARGET_COL, get_positive_label, get_scenario_labels
from ced_ml.evaluation.holdout import load_holdout_indices


def main():
    run_id = sys.argv[1]  # e.g., "phase3_holdout"
    results_dir = Path(f"../results/run_{run_id}")
    holdout_idx_path = Path("../splits/HOLDOUT_idx_IncidentPlusPrevalent.csv")
    infile = Path("../data/Celiac_dataset_proteomics_w_demo.parquet")

    scenario = "IncidentPlusPrevalent"
    positive_label = get_positive_label(scenario)
    keep_labels = get_scenario_labels(scenario)

    # Load and filter data (matching eval-holdout logic)
    df_raw = read_proteomics_file(str(infile), validate=True)
    df_scenario = df_raw[df_raw[TARGET_COL].isin(keep_labels)].copy()

    holdout_result = load_holdout_indices(str(holdout_idx_path))
    holdout_idx = holdout_result.indices
    split_meta = holdout_result.metadata

    meta_num_cols = None
    if split_meta.get("row_filters"):
        meta_num_cols = split_meta["row_filters"].get("meta_num_cols_used")

    df_filtered, _ = apply_row_filters(df_scenario, meta_num_cols=meta_num_cols)
    df_filtered["y"] = (df_filtered[TARGET_COL] == positive_label).astype(int)

    # Extract holdout subset
    valid_idx = [i for i in holdout_idx if i < len(df_filtered)]
    df_holdout = df_filtered.iloc[valid_idx].copy()
    y_true = df_holdout["y"].values
    print(f"Holdout: {len(df_holdout)} samples, {y_true.sum()} positive")

    base_models = ["LR_EN", "LinSVM_cal", "RF", "XGBoost"]
    seeds = list(range(200, 210))
    all_results = []

    for seed in seeds:
        # Load ensemble
        ens_path = results_dir / "ENSEMBLE" / "splits" / f"split_seed{seed}" / "core" / "ENSEMBLE__final_model.joblib"
        ens_bundle = joblib.load(ens_path)
        ensemble = ens_bundle["model"]
        resolved_cols = ens_bundle.get("resolved_columns", {})

        protein_cols = resolved_cols.get("protein_cols", [])
        cat_cols = resolved_cols.get("categorical_metadata", [])
        meta_num_cols_feat = resolved_cols.get("numeric_metadata", [])
        feature_cols = protein_cols + cat_cols + meta_num_cols_feat

        X_holdout = df_holdout[feature_cols]

        # Get base model predictions on holdout
        preds_dict = {}
        for bm in base_models:
            bm_path = results_dir / bm / "splits" / f"split_seed{seed}" / "core" / f"{bm}__final_model.joblib"
            bm_bundle = joblib.load(bm_path)
            pipeline = bm_bundle["model"]
            proba = pipeline.predict_proba(X_holdout)[:, 1]
            preds_dict[bm] = proba

        # Ensemble prediction
        ens_proba = ensemble.predict_proba_from_base_preds(preds_dict)[:, 1]

        auroc = roc_auc_score(y_true, ens_proba)
        prauc = average_precision_score(y_true, ens_proba)
        brier = brier_score_loss(y_true, ens_proba)

        all_results.append({
            "seed": seed,
            "AUROC_holdout": auroc,
            "PR_AUC_holdout": prauc,
            "Brier_holdout": brier,
            "n_holdout": len(y_true),
            "n_positive": int(y_true.sum()),
        })
        print(f"  Seed {seed}: AUROC={auroc:.4f}, PR-AUC={prauc:.4f}, Brier={brier:.4f}")

    results_df = pd.DataFrame(all_results)
    outpath = results_dir / "ENSEMBLE" / "holdout" / "ensemble_holdout_summary.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outpath, index=False)

    mean_auroc = results_df["AUROC_holdout"].mean()
    std_auroc = results_df["AUROC_holdout"].std()
    se = std_auroc / np.sqrt(len(results_df))
    print(f"\nENSEMBLE Holdout: AUROC={mean_auroc:.3f} +/- {std_auroc:.3f} "
          f"[{mean_auroc - 1.96*se:.3f}-{mean_auroc + 1.96*se:.3f}]")
    print(f"Saved to: {outpath}")


if __name__ == "__main__":
    main()
