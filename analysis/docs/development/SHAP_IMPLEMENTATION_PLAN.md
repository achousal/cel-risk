# SHAP Explainability Integration Plan

**Date**: 2026-02-14
**Status**: Planned (v3 -- revised with owner review + Codex audit)
**Priority**: P1 (High Impact)
**Effort**: 3-4 days
**Dependency**: `shap>=0.43.0` (optional extra)

---

## Context

Clinical ML requires interpretability -- clinicians need to understand *why* a patient is high-risk (which proteins drive the prediction). The pipeline currently has OOF permutation importance and model coefficients/Gini gain, but no local (per-sample) explanations. SHAP provides both local and global explanations with a solid theoretical foundation (Shapley values).

**Goal**: Add SHAP computation, aggregation, and visualization as an opt-in feature alongside existing OOF importance. Two modes: OOF SHAP (per-fold, unbiased) and final-model SHAP (test publication plots).

**Reviews incorporated**:
1. Owner review (10 points): SVM margin explanation, output scale metadata, TreeExplainer accuracy claims, OOF aggregation robustness, storage optimization, waterfall FN inclusion, background strategy
2. Codex audit (4 High, 4 Medium): incomplete model unwrapping, inconsistent final SHAP state, duplicated plot ownership, return signature breakage, path errors, additivity test underspecification

**Key design decisions from review**:
- Return type: replace 7-tuple with `NestedCVResult` dataclass
- OOF SHAP layer: always explain base estimator (unwrap all calibration wrappers)
- Final SHAP: compute BEFORE calibration wrapping (no unwrapping needed)
- Val SHAP: test only by default, val as opt-in config flag

---

## Explainer Strategy

| Model | Explainer | Output Scale | Accuracy | Notes |
|-------|-----------|-------------|----------|-------|
| XGBoost | `TreeExplainer` | configurable (default `"probability"`) | Exact (path-dependent) | `feature_perturbation="tree_path_dependent"` |
| RF | `TreeExplainer` | configurable (default `"probability"`) | Approximate (interventional assumes feature independence) | `feature_perturbation="interventional"`, requires background data |
| LR_EN | `LinearExplainer` | `"log_odds"` | Exact | Direct access to `coef_`/`intercept_` |
| LR_L1 | `LinearExplainer` | `"log_odds"` | Exact | Direct access to `coef_`/`intercept_` |
| LinSVM_cal | `LinearExplainer` | `"margin"` | Exact on margin | Unwrap `CalibratedClassifierCV` -> average `coef_`/`intercept_` across `calibrated_classifiers_`. Explain decision_function (margin), NOT calibrated probability. |
| Ensemble | Deferred | -- | -- | Base models use different feature subsets after feature selection; per-model SHAP only. |

### Unwrapping Rules

The pipeline has multiple calibration wrapping layers that SHAP must handle:

- **Per-fold calibration** (`strategy="per_fold"`): `_apply_per_fold_calibration()` (nested_cv.py:833) may wrap LR/RF/XGB in `CalibratedClassifierCV`. SHAP hook fires BEFORE this wrapping (at line 335, same point as OOF importance extraction).
- **OOFCalibratedModel** (`strategy="oof_posthoc"`): Final model wrapped at training_stage.py:281. Final SHAP is computed BEFORE wrapping (between `final_pipeline.fit()` at line 267 and `_apply_per_fold_calibration()` at line 270).
- **LinSVM_cal**: Always a `CalibratedClassifierCV` by construction (registry.py:350). Always unwrap: average `coef_`/`intercept_` from `calibrated_classifiers_` (mirrors importance.py:148-168 pattern).

### Why Margin for SVM, Not Probability

`CalibratedClassifierCV` applies a nonlinear (sigmoid/isotonic) calibration on top of `LinearSVC.decision_function`. SHAP additivity only holds on the native model output. For `LinearSVC`, that is the margin. Explaining calibrated probabilities breaks additivity and can produce misleading attributions (especially isotonic calibration, which creates flat regions). The existing `importance.py` already unwraps to get raw `coef_` -- SHAP follows the same pattern.

---

## Files to Create (3)

### 1. `analysis/src/ced_ml/config/shap_schema.py` -- Computation config

```python
from typing import Literal
from pydantic import BaseModel, Field


class SHAPConfig(BaseModel):
    """Configuration for SHAP explainability computation."""

    # Master toggles
    enabled: bool = False                    # Off by default; opt-in
    compute_oof_shap: bool = True            # Per-fold SHAP during CV
    compute_final_shap: bool = True          # Final model SHAP on test

    # Background data
    max_background_samples: int = Field(
        default=100, ge=10,
        description="Background set size for LinearExplainer / interventional TreeExplainer",
    )
    background_strategy: Literal["random_train", "controls_only", "stratified"] = Field(
        default="random_train",
        description=(
            "How to sample background data. "
            "'controls_only' is clinically meaningful: shows what pushes away from typical control."
        ),
    )

    # Tree explainer settings
    tree_feature_perturbation: Literal["interventional", "tree_path_dependent"] = Field(
        default="tree_path_dependent",
        description=(
            "TreeExplainer perturbation method. "
            "'tree_path_dependent' (exact for XGB), "
            "'interventional' (default for RF, approximate, assumes feature independence). "
            "RF always overrides to 'interventional' in the explainer factory."
        ),
    )
    tree_model_output: Literal["probability", "log_odds", "raw"] = Field(
        default="probability",
        description="TreeExplainer model_output parameter.",
    )

    # Evaluation limits
    max_eval_samples: int = Field(
        default=0, ge=0,
        description="Cap on samples to explain per fold (0 = all).",
    )

    # Storage
    save_raw_values: bool = Field(
        default=False,
        description="Save full per-sample SHAP matrix (large; opt-in).",
    )
    save_val_shap: bool = Field(
        default=False,
        description="Compute and save SHAP on validation set (default: test only).",
    )
    raw_dtype: Literal["float32", "float64"] = Field(
        default="float32",
        description="NumPy dtype for raw SHAP values (float32 halves storage).",
    )
    max_features_warn: int = Field(
        default=200,
        description="Warn if n_features exceeds this (pre-selection scenario).",
    )
    interaction_values: bool = False         # SHAP interactions (very expensive)

    # Waterfall plot sample selection
    n_waterfall_samples: int = Field(
        default=4, ge=0,
        description="Waterfall samples: highest-risk TP, FP, FN, near-threshold negative.",
    )
```

### 2. `analysis/src/ced_ml/features/shap_values.py` -- Core computation

```python
@dataclass
class SHAPFoldResult:
    """Result of SHAP computation for a single CV fold."""
    values: np.ndarray              # (n_samples, n_features)
    expected_value: float           # E[f(x)] baseline
    feature_names: list[str]        # after pipeline transformation
    shap_output_scale: str          # "margin" | "log_odds" | "probability"
    model_name: str
    explainer_type: str             # "TreeExplainer" | "LinearExplainer"
    repeat: int = -1                # set by caller
    outer_split: int = -1           # set by caller


@dataclass
class SHAPTestPayload:
    """Full SHAP result for final model on test (or val).
    Carries all metadata needed for persistence and plotting."""
    values: np.ndarray              # (n_samples, n_features), dtype per config
    expected_value: float
    feature_names: list[str]
    shap_output_scale: str
    model_name: str
    explainer_type: str
    split: str                      # "test" | "val"
    y_pred: np.ndarray | None = None
    y_true: np.ndarray | None = None


def get_model_matrix_and_feature_names(
    pipeline: Pipeline, X: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Transform X through pipeline preprocessing, return (X_transformed, feature_names).

    Steps:
    1. Apply 'pre' (ColumnTransformer) to get transformed matrix
    2. Apply 'sel' (KBest selector) mask if present
    3. Apply 'model_sel' mask if present
    4. Return transformed X and resolved feature names

    Reuses _get_final_feature_names() logic from importance.py:56-107.
    """


def _unwrap_calibrated_for_shap(clf) -> tuple[Any, str]:
    """Unwrap any calibration wrapper to get base estimator + output scale.

    Handles:
    - CalibratedClassifierCV with LinearSVC: average coef_/intercept_ across
      calibrated_classifiers_ (mirrors importance.py:148-168 pattern).
      Returns ((averaged_coef, averaged_intercept), "margin").
    - CalibratedClassifierCV with tree models: access base estimator.
      Returns (base_estimator, inferred_scale).
    - OOFCalibratedModel: access base_model (defensive; should not reach here
      if hook placement is correct).
    - Raw estimator: passthrough.
    """


def get_shap_explainer(
    clf, model_name: str, X_background: np.ndarray, config: SHAPConfig,
) -> tuple[shap.Explainer, str]:
    """Select and create SHAP explainer by model type.

    Returns (explainer, shap_output_scale).

    XGBoost -> TreeExplainer(clf,
        feature_perturbation=config.tree_feature_perturbation,
        model_output=config.tree_model_output)

    RF -> TreeExplainer(clf,
        feature_perturbation="interventional",  # always, regardless of config
        data=X_background,
        model_output=config.tree_model_output)

    LR_EN, LR_L1 -> LinearExplainer((coef_, intercept_), X_background)
        output_scale = "log_odds"

    LinSVM_cal -> _unwrap_calibrated_for_shap() ->
        LinearExplainer((averaged_coef, averaged_intercept), X_background)
        output_scale = "margin"
    """


def compute_shap_for_fold(
    fitted_model: Pipeline, model_name: str,
    X_val: pd.DataFrame, X_train: pd.DataFrame,
    config: SHAPConfig, random_state: int = 42,
) -> SHAPFoldResult:
    """SHAP on single CV fold.

    Steps:
    1. get_model_matrix_and_feature_names(pipeline, X_val) -> X_val_transformed, names
    2. Transform + sample background from X_train per config.background_strategy
    3. Warn if len(names) > config.max_features_warn
    4. Extract clf from pipeline (named_steps["clf"])
    5. get_shap_explainer(clf, ...) -> explainer, scale
    6. Compute SHAP values, cast to config.raw_dtype
    7. Return SHAPFoldResult with scale metadata
    """


def compute_final_shap(
    fitted_pipeline: Pipeline, model_name: str,
    X_eval: pd.DataFrame, y_eval: np.ndarray,
    X_train: pd.DataFrame, config: SHAPConfig,
    split: str = "test",
) -> SHAPTestPayload:
    """SHAP on final fitted model for test (or val) set.

    Same unwrapping logic as compute_shap_for_fold but returns SHAPTestPayload
    with y_pred and y_true for waterfall sample selection.
    """


def aggregate_fold_shap(fold_results: list[SHAPFoldResult]) -> pd.DataFrame:
    """Aggregate SHAP across CV folds.

    Mirrors aggregate_fold_importances() (importance.py:673-836) exactly:
    - Per fold: compute mean(|shap_values|) per feature
    - Cross-fold: mean, std, median, n_folds_nonzero of per-fold mean_abs_shap
    - Validates all folds share same shap_output_scale (warns if mixed)

    Output schema: [feature, mean_abs_shap, std_abs_shap, median_abs_shap, n_folds_nonzero]
    """


def select_waterfall_samples(
    y_pred_proba: np.ndarray, y_true: np.ndarray,
    threshold: float, n: int = 4,
) -> list[dict]:
    """Select clinically informative samples for waterfall plots.

    Selection (in priority order):
    1. Highest-risk true positive (TP) -- strongest signal case
    2. Highest-risk false positive (FP) -- what drives false alarms
    3. Highest-risk false negative (FN) -- what the model misses (clinically critical)
    4. Near-threshold negative -- borderline safe case

    Returns list of dicts: [{index, label, pred_proba, category}, ...]
    """
```

Design decisions:
- Unwrap Pipeline: transform X through preprocessing, explain classifier only
- Guard `import shap` with try/except; raise clear error if called without install
- All explainers produce `shap_output_scale` metadata to prevent silent cross-model comparison bugs
- LinSVM_cal: explain margin (decision_function), not calibrated probability
- float32 default dtype for raw values (halves storage)

### 3. `analysis/src/ced_ml/plotting/shap_plots.py` -- Visualization

```python
def plot_beeswarm(shap_values, X, feature_names, max_display=20, outpath=None):
    """Beeswarm plot showing feature impact distribution."""

def plot_bar_importance(shap_values, feature_names, max_display=20, outpath=None):
    """Global bar plot of mean |SHAP| per feature."""

def plot_waterfall(shap_values, sample_idx, X, feature_names, outpath=None):
    """Waterfall plot for a single sample showing feature contributions."""

def plot_dependence(shap_values, feature_name, X, feature_names, outpath=None):
    """Dependence plot showing feature value vs SHAP value."""

def generate_all_shap_plots(
    test_payload: SHAPTestPayload, oof_shap_df: pd.DataFrame | None,
    threshold: float, config, outdir: Path,
):
    """Orchestrator: generate all enabled SHAP plots -> outdir/shap/.

    Uses select_waterfall_samples() for waterfall plot sample selection.
    Gated on output config flags (plot_shap_summary, plot_shap_waterfall, etc.).
    """
```

---

## NestedCVResult Dataclass

Replace the 7-tuple return of `oof_predictions_with_nested_cv()` with a dataclass:

```python
@dataclass
class NestedCVResult:
    """Result of nested cross-validation with OOF predictions."""
    preds: np.ndarray
    elapsed_sec: float
    best_params_df: pd.DataFrame
    selected_proteins_df: pd.DataFrame
    oof_calibrator: OOFCalibrator | None
    nested_rfecv_result: NestedRFECVResult | None
    oof_importance_df: pd.DataFrame | None
    oof_shap_df: pd.DataFrame | None = None     # NEW: aggregated SHAP importance
```

**Blast radius** (8 unpacking sites to update):
- `cli/orchestration/training_stage.py:210-216` -- production caller
- `tests/models/test_training.py` -- lines 72, 143, 440, 479, 505, 524, 829 (7 test sites)
- `models/__init__.py` -- re-exports
- `models/training.py` -- re-exports

All switch from `a, b, c, d, e, f, g = func(...)` to `result = func(...)` then `result.preds`, etc.

---

## Files to Modify (15)

### A. Config Layer

**4. `analysis/src/ced_ml/config/features_schema.py`**
- Import `SHAPConfig` from `shap_schema.py`
- Add `shap: SHAPConfig = Field(default_factory=SHAPConfig)` to `FeatureConfig`

**5. `analysis/src/ced_ml/config/output_schema.py`**
- Add artifact/plot flags following existing pattern (`plot_roc`, `save_feature_importance`):
  - `save_shap_importance: bool = True` (save summary CSV when SHAP enabled)
  - `plot_shap_summary: bool = True` (bar + beeswarm)
  - `plot_shap_waterfall: bool = True`
  - `plot_shap_dependence: bool = True`

**6. `analysis/src/ced_ml/config/defaults.py`**
- Add SHAP defaults mirroring new `output_schema.py` flags

**7. `configs/training_config.yaml` + `configs/output_config.yaml`**
- Commented-out SHAP sections as documentation

### B. Training Pipeline

**8. `analysis/src/ced_ml/models/nested_cv.py`** (primary hook + dataclass)

Define `NestedCVResult` dataclass. Update return from tuple to dataclass.

SHAP hook at line 335 (after OOF importance block, BEFORE RFECV/calibration):
```python
# Line 334: end of OOF importance block

# --- OOF SHAP computation (if enabled) ---
if (shap_config := getattr(config.features, "shap", None)) \
        and shap_config.enabled \
        and shap_config.compute_oof_shap:
    from ced_ml.features.shap_values import compute_shap_for_fold
    try:
        fold_shap = compute_shap_for_fold(
            fitted_model, model_name,
            X_val=X.iloc[test_idx], X_train=X.iloc[train_idx],
            config=shap_config, random_state=random_state,
        )
        fold_shap.repeat = repeat_num
        fold_shap.outer_split = split_idx
        fold_shap_results.append(fold_shap)
    except Exception as e:
        logger.warning(f"Fold {split_idx}: SHAP failed: {e}")

# Line 336: RFECV block starts
```

After CV loop (~line 553): aggregate via `aggregate_fold_shap()`, assign to `NestedCVResult.oof_shap_df`.

**Note**: SHAP is computed on the pre-RFECV model (same model OOF importance uses). If RFECV retrains on fewer features, SHAP and OOF predictions come from different models. This matches the existing OOF importance behavior.

**9. `analysis/src/ced_ml/cli/orchestration/context.py`**
- Add `oof_shap_df: pd.DataFrame | None = None` to `TrainingContext`
- Add `test_shap_payload: SHAPTestPayload | None = None` (full payload for test)
- Add `val_shap_payload: SHAPTestPayload | None = None` (optional val payload)

**10. `analysis/src/ced_ml/cli/orchestration/training_stage.py`**
- Consume `NestedCVResult` (update unpacking to attribute access)
- Final SHAP computed BETWEEN `final_pipeline.fit()` (line 267) and `_apply_per_fold_calibration()` (line 270):

```python
# Line 267: final_pipeline.fit(ctx.X_train, ctx.y_train)

# --- Final-model SHAP on test (before calibration wrapping) ---
if shap_config and shap_config.enabled and shap_config.compute_final_shap:
    from ced_ml.features.shap_values import compute_final_shap
    ctx.test_shap_payload = compute_final_shap(
        final_pipeline, config.model, ctx.X_test, ctx.y_test,
        ctx.X_train, shap_config,
    )
    if shap_config.save_val_shap:
        ctx.val_shap_payload = compute_final_shap(
            final_pipeline, config.model, ctx.X_val, ctx.y_val,
            ctx.X_train, shap_config, split="val",
        )

# Line 270: final_pipeline = _apply_per_fold_calibration(...)
```

This avoids unwrapping entirely -- SHAP is computed on the raw fitted pipeline before any calibration layer is added.

**11. `analysis/src/ced_ml/cli/orchestration/persistence_stage.py`**
- Save per-split: `cv/oof_shap_importance__{model}.csv` (mirrors OOF importance pattern)
- Save per-split: `cv/shap_metadata__{model}.json` (output scale, explainer type, n_background, n_features)
- Optionally save: `cv/oof_shap_raw__{model}.parquet.gz` (if `save_raw_values`, float32)
- Save final: `shap/test_shap_values__{model}.parquet.gz` (float32)
- Optionally save: `shap/val_shap_values__{model}.parquet.gz` (if `save_val_shap`)
- **NO plot generation** (plots are in plotting_stage only)

**12. `analysis/src/ced_ml/cli/orchestration/plotting_stage.py`**
- Add SHAP plot calls gated on `output.plot_shap_*` and `output.save_plots`
- Uses `ctx.test_shap_payload` for beeswarm/waterfall/dependence
- Uses `ctx.oof_shap_df` for bar importance comparison
- **ONLY location for SHAP plot generation** (persistence_stage saves data only)

### C. Aggregation Layer (cross-split)

**13. `analysis/src/ced_ml/cli/aggregation/orchestrator.py`**
- Add `aggregate_shap_importance()` mirroring existing `aggregate_importance()` (~line 124)
- Reads `split_seedX/cv/oof_shap_importance__{model}.csv` from each split
- Aggregates to `aggregated/importance/oof_shap_importance__{model}.csv`

**14. `analysis/src/ced_ml/cli/aggregation/report_phase.py`**
- Call `aggregate_shap_importance()` in the per-model loop (~line 515) alongside OOF importance

### D. Output Structure

**15. `analysis/src/ced_ml/evaluation/reports.py`**
- Add `"shap": "shap"` to output directory structure dict (line 122)

### E. Dependency

**16. `analysis/pyproject.toml`**
- Add `shap>=0.43.0` as optional: `shap = ["shap>=0.43.0"]`
- Install via `pip install -e ".[shap]"`

### F. Test Updates (return signature migration)

**17. `analysis/tests/models/test_training.py`**
- Update 7 unpacking sites (lines 72, 143, 440, 479, 505, 524, 829) from tuple to `NestedCVResult` attribute access

**18. `analysis/src/ced_ml/models/__init__.py`** + **`analysis/src/ced_ml/models/training.py`**
- Export `NestedCVResult`

---

## Storage

Plot path is `plots/shap/` (matches reports.py:123 output dir structure).

| Artifact | Per-Split Path | Aggregated Path | Format |
|---|---|---|---|
| SHAP importance (global) | `cv/oof_shap_importance__{model}.csv` | `importance/oof_shap_importance__{model}.csv` | CSV |
| SHAP metadata | `cv/shap_metadata__{model}.json` | -- | JSON |
| Raw OOF SHAP (opt-in) | `cv/oof_shap_raw__{model}.parquet.gz` | -- | Parquet float32 |
| Test SHAP values | `shap/test_shap_values__{model}.parquet.gz` | -- | Parquet float32 |
| Val SHAP values (opt-in) | `shap/val_shap_values__{model}.parquet.gz` | -- | Parquet float32 |
| Plots | `plots/shap/*.png` | `aggregated/plots/shap/*.png` | PNG |

**Size estimate** (30 features post-selection): ~6 MB compressed per model for raw OOF, ~0.5 MB for test. Total for 5 models x 3 seeds: ~100 MB.

---

## Patterns to Reuse

| What | Source | Lines |
|------|--------|-------|
| Pipeline feature name resolution | importance.py `_get_final_feature_names()` | 56-107 |
| CalibratedClassifierCV unwrapping | importance.py `extract_linear_importance()` | 148-168 |
| Per-fold aggregation (mean/std/nonzero) | importance.py `aggregate_fold_importances()` | 673-836 |
| OOF importance hook placement | nested_cv.py OOF importance block | 299-334 |
| Persistence pattern (CSV save) | persistence_stage.py OOF importance save | 207-212 |
| Cross-split aggregation | orchestrator.py `aggregate_importance()` | 124-176 |
| Output dir structure | reports.py | 122-141 |

---

## Implementation Order

1. **NestedCVResult dataclass** -- nested_cv.py: define dataclass, update return. Update training_stage.py caller. Update __init__.py / training.py exports. Update 7 test unpacking sites in test_training.py. Run existing tests to verify no breakage.
2. **Config** -- shap_schema.py (new), wire into features_schema.py, output_schema.py, defaults.py
3. **Dependency** -- analysis/pyproject.toml optional extra
4. **Core module** -- shap_values.py: SHAPFoldResult, SHAPTestPayload, pipeline unwrapping, calibration unwrapping, explainer factory, fold computation, aggregation, waterfall selection
5. **Tests (core)** -- tests/features/test_shap_values.py: additivity (all models), SVM margin additivity, pipeline unwrapping, aggregation schema, waterfall FN
6. **CV hook** -- nested_cv.py: fold SHAP at line 335, aggregate after loop
7. **Final SHAP + context** -- training_stage.py (between fit and calibration), context.py (new fields)
8. **Persistence** -- persistence_stage.py: CSV + parquet + metadata JSON (no plots)
9. **Output dirs** -- reports.py: add "shap" key
10. **Aggregation** -- orchestrator.py, report_phase.py
11. **Plotting** -- shap_plots.py (new), plotting_stage.py (wire)
12. **Config YAML** -- commented-out sections
13. **Tests (persistence + integration)** -- test_shap_persistence.py, test_shap_integration.py

---

## Testing

### Unit (`analysis/tests/features/test_shap_values.py`)
- **Additivity per model type**: sum(shap) == model_output(x) - expected_value
- **SVM margin additivity**: sum(shap) == decision_function(x) - expected_value, NOT predict_proba. Catches regression if someone passes calibrated wrapper.
- **Per-fold calibration unwrapping**: CalibratedClassifierCV(LR) -> unwrap to base LR
- Pipeline unwrapping for all 5 model types
- Correct explainer selection per model type (Tree vs Linear)
- `aggregate_fold_shap()` output schema: `[feature, mean_abs_shap, std_abs_shap, median_abs_shap, n_folds_nonzero]`
- Mixed output scale warning when folds have different scales
- Waterfall selection includes FN (highest-risk false negative)

### NestedCVResult Migration (`analysis/tests/models/test_training.py`)
- Existing tests pass after tuple -> dataclass migration (7 sites)

### Persistence (`analysis/tests/cli/test_shap_persistence.py`)
- Mirror `test_oof_importance_persistence.py` pattern exactly
- Mock context with SHAP df -> `_save_cv_artifacts()` -> verify file exists + schema
- Metadata JSON written with `shap_output_scale`
- Verify no artifacts when SHAP disabled
- Verify plots NOT generated in persistence stage

### Aggregation (`analysis/tests/cli/test_shap_aggregation.py`)
- Multi-split SHAP CSVs -> `aggregate_shap_importance()` -> correct aggregation
- Fallback: no SHAP files -> graceful skip (no crash)

### Integration (`analysis/tests/features/test_shap_integration.py`)
- Synthetic data -> train -> SHAP -> plots on disk
- Config disabled -> no SHAP artifacts
- `shap` not installed -> clear ImportError with install instructions

Deterministic: fixed seeds, ~200 samples, ~20 features.

---

## Verification

```bash
# 1. Install
cd analysis && pip install -e ".[shap]"

# 2. Verify NestedCVResult migration (no SHAP yet)
pytest analysis/tests/models/test_training.py -v

# 3. Core SHAP + persistence tests
pytest analysis/tests/features/test_shap_values.py analysis/tests/cli/test_shap_persistence.py -v

# 4. Smoke: train with SHAP
ced train --model XGBoost --split-seed 0 \
  --set features.shap.enabled=true

# 5. Verify per-split artifacts
ls results/<run_id>/XGBoost/cv/oof_shap_importance__XGBoost.csv
ls results/<run_id>/XGBoost/cv/shap_metadata__XGBoost.json
ls results/<run_id>/XGBoost/plots/shap/

# 6. Aggregate across splits
ced aggregate-splits --run-id <run_id>
ls results/<run_id>/aggregated/importance/oof_shap_importance__XGBoost.csv

# 7. Full regression suite
pytest analysis/tests/ -v --timeout=300
```

---

## Design Decisions

1. **Ensemble SHAP**: Skipped -- base models use different feature subsets after feature selection. Per-model SHAP only.
2. **LinSVM_cal**: Explain margin (decision_function) via `LinearExplainer` on unwrapped `LinearSVC`, not calibrated probability. Calibration is a nonlinear post-transform that breaks SHAP additivity.
3. **Output scale metadata**: Every SHAP result carries `shap_output_scale` ("margin" | "log_odds" | "probability") to prevent silent cross-model comparison bugs.
4. **RF TreeExplainer**: Uses `feature_perturbation="interventional"` (approximate, assumes feature independence) rather than path-dependent. Avoids correlated-feature bias but requires background data.
5. **OOF aggregation**: Per-fold mean(|SHAP|) -> cross-fold mean + std + median + n_folds_nonzero. Median provides robustness to outlier folds.
6. **Waterfall selection**: Includes highest-risk FN (missed cases) -- clinically more important than median-risk positive.
7. **Storage**: CSV for summary statistics (small, human-readable), compressed Parquet float32 for raw SHAP matrices. Raw values opt-in. Metadata JSON sidecar with scale info.
8. **Val SHAP**: Test only by default. Val SHAP opt-in via `save_val_shap` flag to avoid storage explosion and presentation confusion.
9. **Plot ownership**: Plots generated ONLY in plotting_stage (Stage 7). Persistence stage (Stage 6) saves data artifacts only. No duplication.
10. **Return type**: `NestedCVResult` dataclass replaces 7-tuple for extensibility without future breakage.
11. **Backward compatibility**: Disabled by default (`shap.enabled: false`). Optional dependency. No changes to existing artifacts or behavior when SHAP is off.
12. **Background strategy**: `"controls_only"` option for clinical interpretability (explains what pushes a sample away from typical control population).
