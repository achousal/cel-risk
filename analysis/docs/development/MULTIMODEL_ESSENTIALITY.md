# Multi-Model Essentiality Validation

## Overview

Extended within-panel essentiality validation to run for **all models** (not just the first model), enabling:
- **Per-model essentiality**: Cluster importance for each model independently
- **Cross-model aggregation**: Identify universally essential vs model-specific clusters
- **Uncertainty estimates**: Variability of importance across models

## Implementation Details

### New Function: `_run_multimodel_essentiality_validation()`

Location: `analysis/src/ced_ml/cli/consensus_panel.py`

**Purpose**: Run drop-column essentiality validation for all models and aggregate results.

**Input Parameters**:
- `model_dirs`: Dict of all available models (model_name -> aggregated_dir_path)
- `split_dirs`: List of all split directories for cross-validation seeds
- `df`, `df_train`: Data matrices
- `y_all`: Binary target vector
- `panel_features`: Consensus panel proteins
- `resolved_cols`: Metadata column information
- `scenario`: Data scenario (e.g., 'IncidentPlusPrevalent')
- `essentiality_dir`: Output directory
- `essentiality_corr_threshold`: Threshold for clustering features by correlation
- `include_brier`, `include_pr_auc`: Whether to compute secondary metrics

**Processing Steps**:

1. **Per-Model Evaluation**: For each model in `model_dirs`:
   - Load trained model from each split seed
   - Clone model architecture (preserves hyperparameters)
   - Refit on consensus panel proteins only
   - Run drop-column analysis to measure cluster importance
   - Aggregate across folds (mean, std)

2. **Cross-Model Aggregation**: Merge per-model results:
   - Compute mean delta-AUROC across all models per cluster
   - Compute standard deviation (uncertainty)
   - Identify min/max importance values
   - Count how many models evaluate each cluster

3. **Classification**:
   - **Universal clusters**: Important in >=50% of models (highest confidence)
   - **Model-specific clusters**: Important in <50% of models (model architecture differences)

### Output Files

Stored in `results/run_<RUN_ID>/consensus/essentiality/`:

#### Per-Model Results
- `per_model/essentiality_{MODEL_NAME}.csv`
  - Columns: cluster_id, n_features, mean_delta_auroc, std_delta_auroc, max_delta_auroc
  - One file per model

#### Cross-Model Aggregation
- `cross_model_essentiality.csv` (main analysis output)
  - Columns:
    - `cluster_id`: Cluster identifier
    - `n_features`: Number of proteins in cluster
    - `delta_auroc_{MODEL_1}`, `delta_auroc_{MODEL_2}`, ...: Per-model importance
    - `n_models_with_importance`: Count of models evaluating this cluster
    - `mean_delta_auroc_cross_model`: Average importance across models
    - `std_delta_auroc_cross_model`: Variability across models
    - `max_delta_auroc_cross_model`: Highest importance value
    - `min_delta_auroc_cross_model`: Lowest importance value
    - `is_universal`: Boolean (True if >=50% of models)

#### Summary Statistics
- `multimodel_essentiality_summary.json`
  - `validation_type`: "multimodel_within_panel"
  - `n_models`: Number of models evaluated
  - `models_used`: List of model names
  - `n_clusters`: Total number of clusters
  - `n_universal_clusters`: Clusters important in >50% of models
  - `n_model_specific_clusters`: Clusters important in <50% of models
  - `mean_delta_auroc_cross_model`: Average importance
  - `cross_model_std`: Average std dev across clusters
  - `max_delta_auroc_cross_model`: Maximum importance
  - `top_cluster_id`: ID of most important cluster
  - `top_cluster_delta_auroc`: Importance of top cluster
  - `top_cluster_n_models`: How many models rate it important
  - `top_cluster_is_universal`: Boolean
  - `per_model_summary`: Statistics for each model
    - Each model has: mean_delta_auroc, max_delta_auroc, n_clusters_evaluated

### Interpretation

**Universally Essential Clusters** (high importance in most/all models):
- Robust biomarkers for CeD risk prediction
- Likely represent consistent biological signals across model architectures
- High confidence for clinical interpretation

**Model-Specific Clusters** (high importance in only some models):
- Reveal model architecture differences
- May indicate non-linear relationships captured by tree-based models
- Could represent overfitting or model-specific artifacts

**Variability (std dev)**:
- Low std dev: Consistent importance across models -> robust clusters
- High std dev: Variable importance -> potential confounding or noise

## Example Output

```
Within-Panel Essentiality Validation (Multi-Model):
  Models evaluated: LR_EN, RF, XGBoost
  Clusters validated: 24
  Universal clusters (>50% models): 18
  Model-specific clusters: 6
  Mean delta AUROC (cross-model): +0.0245
  Cross-model std dev: 0.0087
  Max delta AUROC (cross-model): +0.0412
  Top cluster: cluster_5 (delta AUROC=+0.0412, found in 3/3 models)
    -> Universal cluster (found in >50% of models)

  Per-model summary:
    LR_EN: mean delta AUROC=+0.0268, max=+0.0412
    RF: mean delta AUROC=+0.0251, max=+0.0389
    XGBoost: mean delta AUROC=+0.0216, max=+0.0378
```

## Changes to `run_consensus_panel()`

The function now:
1. Passes `model_dirs` (all models) to essentiality validation
2. Calls `_run_multimodel_essentiality_validation()` instead of single-model validation
3. Produces aggregated essentiality summary

## Integration with Pipeline

Usage in consensus panel generation:

```bash
ced consensus-panel --run-id 20260127_115115 --run-essentiality
```

The multi-model essentiality runs automatically post-hoc after panel selection (not part of ranking logic).

## Key Differences from Previous Implementation

| Aspect | Previous | New |
|--------|----------|-----|
| **Models evaluated** | First model only | All available models |
| **Output files** | Single within_panel_essentiality.csv | Per-model CSVs + cross-model aggregation |
| **Interpretation** | Model-specific importance | Model-specific + cross-model consensus |
| **Uncertainty** | Single AUROC per cluster | Per-model + cross-model variability |
| **Cluster classification** | Ranked by importance | Universal vs model-specific |

## Testing Recommendations

1. **Syntax validation**: Already performed ✓
2. **Integration test**: Run on existing run with 3+ models
3. **Output validation**:
   - Verify CSV structure (columns, dtypes)
   - Check JSON summary completeness
   - Validate per-model results subset into cross-model
4. **Interpretation validation**:
   - Compare model-specific findings with model characteristics
   - Verify universal clusters have low std dev
   - Spot-check top clusters across models

## Future Enhancements

1. **Visualization**: Create heatmap of per-model importance (clusters x models)
2. **Reporting**: Add essentiality findings to consensus metadata report
3. **Thresholding**: Filter to universal clusters for stricter panel validation
4. **Bootstrap confidence intervals**: For cross-model statistics
