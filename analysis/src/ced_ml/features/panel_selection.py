"""Deterministic panel size selection via statistical stopping rules.

Implements a three-rule decision procedure for selecting the smallest
protein panel that is statistically non-inferior to the full model:

    Rule 1: Bootstrap non-inferiority test (paired AUROC comparison)
    Rule 2: Cross-seed stability gate (coefficient of variation)
    Rule 3: Marginal contribution test (drop-column essentiality)

Usage:
    result = select_optimal_panel(
        curve=aggregated_pareto_curve,
        full_auroc_by_seed=[0.89, 0.88, 0.90, 0.87, 0.89],
        delta_primary=0.02,
    )
    print(result.selected_size, result.selected_proteins)

References:
    Wellek (2010). Testing Statistical Hypotheses of Equivalence
    and Noninferiority. Chapman & Hall.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ced_ml.metrics.discrimination import auroc

logger = logging.getLogger(__name__)


@dataclass
class NonInferiorityResult:
    """Result of a bootstrap non-inferiority test for AUROC.

    Attributes:
        rejected: True if H0 (inferiority) is rejected -> panel is non-inferior.
        p_value: One-sided p-value for the non-inferiority test.
        delta_estimate: Point estimate of AUROC(full) - AUROC(reduced).
        ci_upper: Upper bound of the one-sided (1-alpha) CI for the difference.
        delta_margin: Pre-specified non-inferiority margin.
        n_bootstrap: Number of bootstrap replicates used.
    """

    rejected: bool
    p_value: float
    delta_estimate: float
    ci_upper: float
    delta_margin: float
    n_bootstrap: int


@dataclass
class StabilityResult:
    """Result of cross-seed stability check.

    Attributes:
        stable: True if CV is below threshold.
        cv: Coefficient of variation (std/mean) of AUROC across seeds.
        mean_auroc: Mean AUROC across seeds.
        std_auroc: Standard deviation of AUROC across seeds.
        n_seeds: Number of seeds evaluated.
    """

    stable: bool
    cv: float
    mean_auroc: float
    std_auroc: float
    n_seeds: int


@dataclass
class PanelSizeDecision:
    """Decision record for a single panel size.

    Attributes:
        size: Panel size (number of proteins).
        stability: Cross-seed stability result.
        noninferiority: Bootstrap non-inferiority result (None if stability failed).
        accepted: Whether this size passed all applicable rules.
        rejection_reason: Why this size was rejected (empty if accepted).
    """

    size: int
    stability: StabilityResult
    noninferiority: NonInferiorityResult | None = None
    accepted: bool = False
    rejection_reason: str = ""


@dataclass
class PanelSelectionResult:
    """Final result of the deterministic panel selection procedure.

    Attributes:
        selected_size: Optimal panel size (primary delta). 0 if no panel passed.
        selected_proteins: Protein list for the selected panel.
        delta_used: Non-inferiority margin used for selection.
        sensitivity_size: Panel size at the sensitivity delta. 0 if none passed.
        sensitivity_proteins: Protein list for the sensitivity panel.
        n_passengers_removed: Number of non-essential proteins removed.
        decision_table: Full audit trail -- one PanelSizeDecision per evaluated k.
        full_model_auroc: Reference AUROC (mean across seeds).
    """

    selected_size: int = 0
    selected_proteins: list[str] = field(default_factory=list)
    delta_used: float = 0.02
    sensitivity_size: int = 0
    sensitivity_proteins: list[str] = field(default_factory=list)
    n_passengers_removed: int = 0
    decision_table: list[PanelSizeDecision] = field(default_factory=list)
    full_model_auroc: float = 0.0


def bootstrap_noninferiority_test(
    y_true: np.ndarray,
    y_pred_full: np.ndarray,
    y_pred_reduced: np.ndarray,
    delta: float = 0.02,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> NonInferiorityResult:
    """Test whether a reduced panel is non-inferior to the full model.

    Uses paired bootstrap resampling to test:
        H0: AUROC(full) - AUROC(reduced) > delta   (inferior)
        H1: AUROC(full) - AUROC(reduced) <= delta  (non-inferior)

    The test is one-sided: we reject H0 (declare non-inferiority) when
    the upper (1-alpha) percentile of the bootstrap difference distribution
    is <= delta.

    Args:
        y_true: True binary labels (0/1).
        y_pred_full: Predicted probabilities from the full model.
        y_pred_reduced: Predicted probabilities from the reduced panel model.
        delta: Non-inferiority margin (maximum acceptable AUROC loss).
        alpha: Significance level (one-sided).
        n_bootstrap: Number of bootstrap replicates.
        seed: Random seed for reproducibility.

    Returns:
        NonInferiorityResult with test outcome and statistics.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]

        if len(np.unique(y_b)) < 2:
            continue

        try:
            auc_full = auroc(y_b, y_pred_full[idx])
            auc_reduced = auroc(y_b, y_pred_reduced[idx])
            diffs.append(auc_full - auc_reduced)
        except Exception:
            continue

    if not diffs:
        return NonInferiorityResult(
            rejected=False,
            p_value=1.0,
            delta_estimate=np.nan,
            ci_upper=np.nan,
            delta_margin=delta,
            n_bootstrap=0,
        )

    diffs = np.array(diffs)
    delta_estimate = float(np.mean(diffs))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha)))

    # p-value: fraction of bootstrap diffs exceeding delta
    p_value = float(np.mean(diffs > delta))

    # Reject H0 (declare non-inferior) if ci_upper <= delta
    rejected = ci_upper <= delta

    return NonInferiorityResult(
        rejected=rejected,
        p_value=p_value,
        delta_estimate=delta_estimate,
        ci_upper=ci_upper,
        delta_margin=delta,
        n_bootstrap=len(diffs),
    )


def pooled_noninferiority_test(
    seed_data: list[dict[str, np.ndarray]],
    delta: float = 0.02,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> NonInferiorityResult:
    """Non-inferiority test pooling bootstrap replicates across seeds.

    Each seed contributes n_bootstrap/n_seeds replicates from its own
    validation set. The pooled distribution captures both within-seed
    sampling uncertainty and cross-seed variability.

    Args:
        seed_data: List of dicts, each with keys:
            "y_true": np.ndarray of true labels
            "y_pred_full": np.ndarray of full-model predictions
            "y_pred_reduced": np.ndarray of reduced-panel predictions
        delta: Non-inferiority margin.
        alpha: Significance level (one-sided).
        n_bootstrap: Total bootstrap replicates (split across seeds).
        seed: Random seed.

    Returns:
        NonInferiorityResult from the pooled bootstrap distribution.
    """
    n_seeds = len(seed_data)
    if n_seeds == 0:
        return NonInferiorityResult(
            rejected=False,
            p_value=1.0,
            delta_estimate=np.nan,
            ci_upper=np.nan,
            delta_margin=delta,
            n_bootstrap=0,
        )

    per_seed_boots = max(1, n_bootstrap // n_seeds)
    rng = np.random.default_rng(seed)
    all_diffs = []

    for sd in seed_data:
        y_true = sd["y_true"]
        y_full = sd["y_pred_full"]
        y_reduced = sd["y_pred_reduced"]
        n = len(y_true)

        for _ in range(per_seed_boots):
            idx = rng.choice(n, size=n, replace=True)
            y_b = y_true[idx]

            if len(np.unique(y_b)) < 2:
                continue

            try:
                auc_full = auroc(y_b, y_full[idx])
                auc_reduced = auroc(y_b, y_reduced[idx])
                all_diffs.append(auc_full - auc_reduced)
            except Exception:
                continue

    if not all_diffs:
        return NonInferiorityResult(
            rejected=False,
            p_value=1.0,
            delta_estimate=np.nan,
            ci_upper=np.nan,
            delta_margin=delta,
            n_bootstrap=0,
        )

    diffs = np.array(all_diffs)
    delta_estimate = float(np.mean(diffs))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha)))
    p_value = float(np.mean(diffs > delta))
    rejected = ci_upper <= delta

    return NonInferiorityResult(
        rejected=rejected,
        p_value=p_value,
        delta_estimate=delta_estimate,
        ci_upper=ci_upper,
        delta_margin=delta,
        n_bootstrap=len(diffs),
    )


def cross_seed_stability_check(
    auroc_by_seed: list[float],
    cv_threshold: float = 0.05,
) -> StabilityResult:
    """Check whether AUROC is stable across split seeds.

    Args:
        auroc_by_seed: AUROC values, one per seed.
        cv_threshold: Maximum acceptable coefficient of variation.

    Returns:
        StabilityResult indicating whether stability gate is passed.
    """
    values = np.array(auroc_by_seed, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return StabilityResult(
            stable=True,
            cv=0.0,
            mean_auroc=float(values[0]) if len(values) == 1 else 0.0,
            std_auroc=0.0,
            n_seeds=len(values),
        )

    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1))
    cv = std_val / mean_val if mean_val > 0 else float("inf")

    return StabilityResult(
        stable=cv < cv_threshold,
        cv=cv,
        mean_auroc=mean_val,
        std_auroc=std_val,
        n_seeds=len(values),
    )


def filter_passengers(
    essentiality_df,
    panel_proteins: list[str],
) -> tuple[list[str], list[str], int]:
    """Identify and remove passenger proteins using drop-column essentiality.

    A protein is a passenger if its drop-column delta-AUROC 95% CI includes 0
    (i.e., removing it does not significantly hurt performance).

    Args:
        essentiality_df: DataFrame with columns:
            "representative" or "cluster_features": protein identifier
            "mean_delta_auroc": mean AUROC drop when removed
            "std_delta_auroc": std of AUROC drop across seeds
            "n_folds": number of seeds/folds evaluated
        panel_proteins: Current panel protein list.

    Returns:
        Tuple of (essential_proteins, passenger_proteins, n_removed).
    """
    if essentiality_df is None or essentiality_df.empty:
        return list(panel_proteins), [], 0

    passengers = []
    essentials = []

    for _, row in essentiality_df.iterrows():
        protein = row.get("representative", "")
        mean_delta = row.get("mean_delta_auroc", 0.0)
        std_delta = row.get("std_delta_auroc", 0.0)
        n_folds = row.get("n_folds", 1)

        if n_folds < 2:
            essentials.append(protein)
            continue

        # 95% CI lower bound for delta_AUROC
        # Using t-distribution approximation for small n
        from scipy.stats import t as t_dist

        se = std_delta / np.sqrt(n_folds)
        ci_lower = mean_delta - t_dist.ppf(0.975, df=n_folds - 1) * se

        if ci_lower <= 0:
            passengers.append(protein)
            logger.info(
                f"  Passenger: {protein} (delta_AUROC={mean_delta:.4f}, "
                f"95% CI lower={ci_lower:.4f})"
            )
        else:
            essentials.append(protein)

    # Keep proteins that are in the panel and essential
    essential_set = set(essentials)
    passenger_set = set(passengers)

    essential_panel = [p for p in panel_proteins if p not in passenger_set]
    removed = [p for p in panel_proteins if p in passenger_set]

    # Proteins not in essentiality_df are kept (no evidence to remove)
    tested = essential_set | passenger_set
    untested = [p for p in panel_proteins if p not in tested]
    essential_panel = essential_panel + [p for p in untested if p not in essential_panel]

    return essential_panel, removed, len(removed)


def select_optimal_panel(
    curve: list[dict[str, Any]],
    full_auroc_by_seed: list[float],
    essentiality: dict[int, Any] | None = None,
    delta_primary: float = 0.02,
    delta_sensitivity: float = 0.01,
    cv_threshold: float = 0.05,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> PanelSelectionResult:
    """Select the optimal panel size using the three-rule decision procedure.

    Evaluates panel sizes from smallest to largest on the Pareto curve.
    The first size passing all three rules is selected.

    The curve must contain cross-seed AUROC values. Each curve point should
    have keys: "size", "auroc_val", "auroc_val_std", "n_seeds", "proteins".
    If per-seed AUROC values are available as "auroc_val_by_seed", those
    are used for the stability check; otherwise std-based approximation is used.

    Args:
        curve: Aggregated Pareto curve from RFE (sorted by size, any order).
        full_auroc_by_seed: AUROC of the full model per seed.
        essentiality: Optional dict mapping panel_size -> essentiality DataFrame.
            If provided, Rule 3 (passenger filtering) is applied.
        delta_primary: Primary non-inferiority margin.
        delta_sensitivity: Sensitivity analysis margin (stricter).
        cv_threshold: Max coefficient of variation for stability gate.
        alpha: Significance level for non-inferiority test.
        n_bootstrap: Bootstrap replicates for non-inferiority test.
        seed: Random seed.

    Returns:
        PanelSelectionResult with the selected panel and full decision audit.
    """
    if not curve:
        logger.warning("Empty Pareto curve, cannot select panel")
        return PanelSelectionResult()

    full_mean = float(np.mean(full_auroc_by_seed))
    sorted_curve = sorted(curve, key=lambda x: x["size"])

    decision_table: list[PanelSizeDecision] = []
    primary_candidate: PanelSizeDecision | None = None
    sensitivity_candidate: PanelSizeDecision | None = None

    for point in sorted_curve:
        size = point["size"]
        mean_auroc_k = point.get("auroc_val", 0.0)
        std_auroc_k = point.get("auroc_val_std", 0.0)
        n_seeds = point.get("n_seeds", 1)

        # -- Rule 2: Stability gate (checked first, cheaper) --
        if "auroc_val_by_seed" in point:
            seed_aurocs = point["auroc_val_by_seed"]
        else:
            # Approximate from mean and std (assume normal)
            seed_aurocs = [mean_auroc_k] * n_seeds
            if std_auroc_k > 0 and n_seeds > 1:
                # Reconstruct plausible seed values preserving mean and std
                rng = np.random.default_rng(seed + size)
                noise = rng.standard_normal(n_seeds)
                noise = noise - noise.mean()
                if np.std(noise) > 0:
                    noise = noise / np.std(noise) * std_auroc_k
                seed_aurocs = list(np.clip(mean_auroc_k + noise, 0, 1))

        stability = cross_seed_stability_check(seed_aurocs, cv_threshold)

        if not stability.stable:
            decision = PanelSizeDecision(
                size=size,
                stability=stability,
                accepted=False,
                rejection_reason=f"Unstable: CV={stability.cv:.4f} > {cv_threshold}",
            )
            decision_table.append(decision)
            logger.info(f"  k={size}: SKIP (unstable, CV={stability.cv:.4f})")
            continue

        # -- Rule 1: Non-inferiority test --
        # Use point estimate for non-inferiority (paired bootstrap requires
        # raw predictions which we may not have at aggregation time).
        # Fall back to normal approximation when raw predictions unavailable.
        delta_est = full_mean - mean_auroc_k

        # Approximate SE of the difference from cross-seed variability
        full_std = float(np.std(full_auroc_by_seed, ddof=1)) if len(full_auroc_by_seed) > 1 else 0.0
        # Combined SE (conservative: assume independent)
        se_diff = np.sqrt(full_std**2 + std_auroc_k**2) if n_seeds > 1 else 0.0

        # Normal approximation for CI upper bound
        from scipy.stats import norm

        z = norm.ppf(1 - alpha)
        ci_upper = delta_est + z * se_diff

        # p-value: P(delta > delta_margin) under normal approximation
        if se_diff > 0:
            z_stat = (delta_primary - delta_est) / se_diff
            p_value_primary = 1.0 - norm.cdf(z_stat)
        else:
            p_value_primary = 0.0 if delta_est <= delta_primary else 1.0

        ni_primary = NonInferiorityResult(
            rejected=ci_upper <= delta_primary,
            p_value=p_value_primary,
            delta_estimate=delta_est,
            ci_upper=ci_upper,
            delta_margin=delta_primary,
            n_bootstrap=0,  # 0 indicates normal approximation, not bootstrap
        )

        if not ni_primary.rejected:
            decision = PanelSizeDecision(
                size=size,
                stability=stability,
                noninferiority=ni_primary,
                accepted=False,
                rejection_reason=(
                    f"Non-inferior test failed: delta={delta_est:.4f}, "
                    f"CI_upper={ci_upper:.4f} > {delta_primary}"
                ),
            )
            decision_table.append(decision)
            logger.info(
                f"  k={size}: SKIP (inferior, delta={delta_est:.4f}, " f"CI_upper={ci_upper:.4f})"
            )
            continue

        # Passed Rules 1 and 2
        decision = PanelSizeDecision(
            size=size,
            stability=stability,
            noninferiority=ni_primary,
            accepted=True,
        )
        decision_table.append(decision)
        logger.info(
            f"  k={size}: PASS (delta={delta_est:.4f}, "
            f"CI_upper={ci_upper:.4f}, CV={stability.cv:.4f})"
        )

        if primary_candidate is None:
            primary_candidate = decision

        # Check sensitivity delta too
        if sensitivity_candidate is None:
            if ci_upper <= delta_sensitivity:
                sensitivity_candidate = decision

    # -- Build result --
    result = PanelSelectionResult(
        delta_used=delta_primary,
        full_model_auroc=full_mean,
        decision_table=decision_table,
    )

    if primary_candidate is not None:
        size = primary_candidate.size
        proteins = _get_proteins_for_size(curve, size)

        # -- Rule 3: Essentiality filter --
        n_removed = 0
        if essentiality and size in essentiality:
            proteins, removed, n_removed = filter_passengers(essentiality[size], proteins)
            if n_removed > 0:
                logger.info(f"  Removed {n_removed} passengers from k={size}: " f"{removed}")

        result.selected_size = len(proteins)
        result.selected_proteins = proteins
        result.n_passengers_removed = n_removed

    if sensitivity_candidate is not None:
        sens_size = sensitivity_candidate.size
        sens_proteins = _get_proteins_for_size(curve, sens_size)
        result.sensitivity_size = sens_size
        result.sensitivity_proteins = sens_proteins

    # Summary log
    if result.selected_size > 0:
        logger.info(
            f"\nOptimal panel: {result.selected_size} proteins "
            f"(delta={delta_primary}, full AUROC={full_mean:.4f})"
        )
        if result.sensitivity_size > 0:
            logger.info(
                f"Sensitivity panel: {result.sensitivity_size} proteins "
                f"(delta={delta_sensitivity})"
            )
    else:
        logger.warning(
            "No panel size passed all decision rules. " "Consider relaxing delta or cv_threshold."
        )

    return result


def _get_proteins_for_size(curve: list[dict[str, Any]], size: int) -> list[str]:
    """Extract protein list for a given panel size from the Pareto curve."""
    for point in curve:
        if point["size"] == size:
            return list(point.get("proteins", []))
    return []


def _to_native(val: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def decision_table_to_dict(
    result: PanelSelectionResult,
) -> dict[str, Any]:
    """Serialize PanelSelectionResult to a JSON-compatible dict.

    Used for saving the full audit trail to optimal_panel_selection.json.
    All numpy types are converted to native Python for JSON compatibility.
    """
    decisions = []
    for d in result.decision_table:
        entry: dict[str, Any] = {
            "size": _to_native(d.size),
            "accepted": _to_native(d.accepted),
            "rejection_reason": d.rejection_reason,
            "stability": {
                "stable": _to_native(d.stability.stable),
                "cv": _to_native(d.stability.cv),
                "mean_auroc": _to_native(d.stability.mean_auroc),
                "std_auroc": _to_native(d.stability.std_auroc),
                "n_seeds": _to_native(d.stability.n_seeds),
            },
        }
        if d.noninferiority is not None:
            entry["noninferiority"] = {
                "rejected": _to_native(d.noninferiority.rejected),
                "p_value": _to_native(d.noninferiority.p_value),
                "delta_estimate": _to_native(d.noninferiority.delta_estimate),
                "ci_upper": _to_native(d.noninferiority.ci_upper),
                "delta_margin": _to_native(d.noninferiority.delta_margin),
                "n_bootstrap": _to_native(d.noninferiority.n_bootstrap),
            }
        decisions.append(entry)

    return {
        "selected_size": _to_native(result.selected_size),
        "selected_proteins": list(result.selected_proteins),
        "delta_used": _to_native(result.delta_used),
        "sensitivity_size": _to_native(result.sensitivity_size),
        "sensitivity_proteins": list(result.sensitivity_proteins),
        "n_passengers_removed": _to_native(result.n_passengers_removed),
        "full_model_auroc": _to_native(result.full_model_auroc),
        "decisions": decisions,
    }
