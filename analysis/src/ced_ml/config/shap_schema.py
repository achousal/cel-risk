"""SHAP explainability configuration schema for CeD-ML pipeline."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SHAPConfig(BaseModel):
    """Configuration for SHAP explainability computation."""

    # Master toggles
    enabled: bool = False
    compute_oof_shap: bool = True
    compute_final_shap: bool = True

    # Background data
    max_background_samples: int = Field(
        default=100,
        ge=10,
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
            "TreeExplainer perturbation method.\n\n"
            "'tree_path_dependent' (XGB default, exact):\n"
            "  - Uses tree structure for exact Shapley values\n"
            "  - Only available for XGBoost/LightGBM\n\n"
            "'interventional' (RF default, approximate):\n"
            "  - Samples features from background, assumes independence\n"
            "  - CAVEAT (Aas et al. 2021): Under strong correlation (e.g., pathway proteins):\n"
            "    * May misattribute effects across correlated features\n"
            "    * Breaks conditional dependencies P(A|B) != P(A)\n"
            "    * Noisier attributions than path-dependent methods\n"
            "  - Use for hypothesis generation, NOT causal inference\n\n"
            "RF always overrides to 'interventional' in the explainer factory."
        ),
    )
    tree_model_output: Literal["auto", "raw", "probability"] = Field(
        default="auto",
        description=(
            "TreeExplainer model_output parameter. "
            "'auto' resolves to model-specific defaults (RF='probability', XGBoost='raw'). "
            "'raw' is log-odds for XGBoost binary classification (exact with tree_path_dependent); "
            "for sklearn RF, 'raw' is tree-internal scale (model-dependent, not guaranteed log-odds). "
            "'probability' is ONLY valid with feature_perturbation='interventional'. "
            "RF can use 'probability' since it always forces interventional perturbation."
        ),
    )

    # Evaluation limits
    max_eval_samples: int = Field(
        default=0,
        ge=0,
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
    interaction_values: bool = False

    # Background sensitivity analysis
    background_sensitivity_mode: bool = Field(
        default=False,
        description=(
            "Enable multi-background sensitivity analysis. "
            "Computes SHAP with N random background samples and reports rank stability "
            "(Spearman correlation). Useful for assessing attribution robustness under "
            "baseline choice. WARNING: N times slower (default N=3)."
        ),
    )
    n_background_replicates: int = Field(
        default=3,
        ge=2,
        le=10,
        description=(
            "Number of background samples for sensitivity mode "
            "(only used if background_sensitivity_mode=True)."
        ),
    )

    # Aggregation safety
    allow_mixed_scales: bool = Field(
        default=False,
        description=(
            "If False (default), aggregate_fold_shap() raises ValueError when folds "
            "have different shap_output_scale values. Set True only if you understand "
            "the implications of mixing scales in aggregation."
        ),
    )

    # Waterfall plot sample selection
    n_waterfall_samples: int = Field(
        default=4,
        ge=0,
        description="Waterfall samples: highest-risk TP, FP, FN, near-threshold negative.",
    )
    positive_label: int = Field(
        default=1,
        description=(
            "Target class label for SHAP normalization in binary classification. "
            "Used by _normalize_expected_value and _normalize_shap_values."
        ),
    )

    @model_validator(mode="after")
    def _validate_tree_explainer_combo(self) -> "SHAPConfig":
        """Reject tree_path_dependent + probability (invalid per SHAP docs)."""
        if (
            self.tree_feature_perturbation == "tree_path_dependent"
            and self.tree_model_output == "probability"
        ):
            raise ValueError(
                "tree_model_output='probability' requires "
                "tree_feature_perturbation='interventional', not 'tree_path_dependent'. "
                "SHAP TreeExplainer only supports 'raw' with tree_path_dependent."
            )
        return self
