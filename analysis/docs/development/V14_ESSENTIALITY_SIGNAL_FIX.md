# V-14 Fix: Essentiality Signal Now Produced by optimize-panel

**Issue**: V-14 from PIPELINE_CORRECTNESS_AUDIT_AND_PLAN.md
**Status**: FIXED
**Date**: 2026-02-09
**Files Modified**: `analysis/src/ced_ml/cli/optimize_panel.py`

## Problem Statement

The consensus composite ranking (ADR-004) is designed to use three signals:
1. OOF importance (60% weight)
2. **Essentiality / drop-column (30% weight)**
3. Stability (10% weight)

However, the essentiality signal was never produced as a pre-consensus input. The consensus loader expected files at:
```
optimize_panel/essentiality/panel_{threshold}_essentiality.csv
```

But no command wrote these files. The drop-column code in `consensus_panel.py:609` was a **post-consensus artifact** (validating the final panel), not a **pre-consensus input** (ranking proteins before consensus).

Result: Consensus silently dropped the 30% essentiality weight, degrading to OOF + stability only (70% + 30% renormalized).

## Solution

### Implementation

Added drop-column essentiality validation to the end of `optimize-panel` command:

**New Function**: `run_drop_column_validation_for_panels()` (lines 293-427)
- Runs automatically after RFE aggregation completes
- For each recommended panel (95pct, 99pct, pareto):
  1. Extracts panel proteins from RFE curve
  2. Clusters proteins by correlation (same threshold as RFE)
  3. Runs drop-column validation across all seeds
  4. Aggregates results (mean ± std delta AUROC)
  5. Saves to `optimize_panel/essentiality/panel_{threshold}_essentiality.csv`

**Integration**: Lines 880-902
- Called at end of `run_optimize_panel_aggregated()`
- Wrapped in try-except to not fail the command on validation errors
- Logs progress and top essential features

### Output Format

Matches consensus loader expectations (`consensus_panel.py:212-247`):

| Column | Description | Required by Consensus? |
|--------|-------------|----------------------|
| `cluster_id` | Cluster identifier | No |
| `representative` | First protein in cluster | **Yes** (key for lookup) |
| `cluster_features` | Comma-separated protein list | No |
| `n_features_in_cluster` | Cluster size | No |
| `mean_delta_auroc` | Average importance | **Yes** (ranking score) |
| `std_delta_auroc` | Standard deviation | No |
| `min_delta_auroc` | Min across seeds | No |
| `max_delta_auroc` | Max across seeds | No |
| `n_folds` | Number of seeds | No |
| `n_errors` | Failed evaluations | No |

### Workflow Integration

```bash
# Before (essentiality signal missing):
ced optimize-panel --run-id <ID>  # RFE only, no drop-column
ced consensus-panel --run-id <ID>  # WARNING: essentiality signal absent

# After (complete pipeline):
ced optimize-panel --run-id <ID>  # RFE + drop-column essentiality
ced consensus-panel --run-id <ID>  # All three signals present (OOF + ess + stab)
```

### Output Structure

```
results/run_{ID}/{MODEL}/aggregated/optimize_panel/
├── essentiality/
│   ├── panel_95pct_essentiality.csv    # NEW - 30% weight in consensus
│   ├── panel_99pct_essentiality.csv    # NEW - 30% weight in consensus
│   └── panel_pareto_essentiality.csv   # NEW - 30% weight in consensus
├── panel_curve_aggregated.png
├── feature_ranking_aggregated.png
└── ...
```

## Verification

### Consensus Loader Check

The consensus loader (`load_model_essentiality` in `consensus_panel.py`) now finds:
1. Primary path: `optimize_panel/essentiality/panel_{threshold}_essentiality.csv` ✅
2. Columns: `representative` ✅ and `mean_delta_auroc` ✅

### Weight Normalization

When all three signals are present:
- OOF importance: 60% → 0.6
- Essentiality: 30% → 0.3
- Stability: 10% → 0.1

Previously (missing essentiality):
- OOF importance: 60% → 0.857 (60/70 renormalized)
- Stability: 10% → 0.143 (10/70 renormalized)
- **Essentiality: effectively 0%** ⚠️

## Testing

```bash
# Import validation
python -c "from ced_ml.cli.optimize_panel import run_drop_column_validation_for_panels; print('OK')"

# End-to-end test (run on existing RFE results)
ced optimize-panel --run-id <EXISTING_RUN_ID> --model LR_EN

# Expected output:
# 1. RFE aggregation completes
# 2. "Running drop-column essentiality validation..." header
# 3. For each threshold (95pct, 99pct, pareto):
#    - "Processing {threshold} panel (size={N})..."
#    - "Seed {i}: {M} clusters evaluated"
#    - "Saved essentiality results to .../panel_{threshold}_essentiality.csv"
#    - "Top 3 essential features: ..."

# Verify consensus now uses essentiality
ced consensus-panel --run-id <RUN_ID>
# Check logs for: "essentiality='yes'" instead of "essentiality='no'"
```

## Impact

### Before Fix
- Consensus ranking used only **70% of intended signal** (OOF + stability)
- Essentiality signal (30% weight) was silently absent
- Post-consensus drop-column was diagnostic only, not used for ranking

### After Fix
- Consensus ranking uses **100% of intended signal** (OOF + essentiality + stability)
- Drop-column essentiality computed at optimal panel sizes from RFE
- Pre-consensus ranking matches ADR-004 design

## Related Files

- **Modified**: `analysis/src/ced_ml/cli/optimize_panel.py`
- **Unchanged** (already correct):
  - `analysis/src/ced_ml/features/drop_column.py` (compute & aggregate functions)
  - `analysis/src/ced_ml/cli/consensus_panel.py` (loader expects this format)
  - `analysis/src/ced_ml/features/consensus/ranking.py` (processes essentiality signal)

## Notes

- Drop-column validation runs on the **target panel size** for each threshold (95pct, 99pct)
- Uses the same correlation clustering as RFE (corr_threshold, corr_method)
- Evaluates essentiality across **all available seeds** (same as RFE)
- Gracefully handles missing seeds or failed evaluations
- Does not fail `optimize-panel` if drop-column has errors (logs warning)

## Remaining Work

None. V-14 is now **RESOLVED**.

The essentiality signal is correctly produced and consumed by the consensus pipeline.
