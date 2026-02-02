"""
Metadata building utilities for plot annotations.

Provides functions to create rich, reproducible metadata lines
for plot annotations with run configuration details.

Usage Guide:
    - build_plot_metadata: Main builder for test/val plots during training
    - build_oof_metadata: Specialized builder for out-of-fold plots during training
    - build_aggregated_metadata: Builder for plots aggregating multiple splits

"""

from datetime import datetime


def build_plot_metadata(
    model: str,
    scenario: str,
    seed: int,
    train_prev: float,
    target_prev: float | None = None,
    cv_folds: int | None = None,
    cv_repeats: int | None = None,
    cv_scoring: str | None = None,
    n_features: int | None = None,
    feature_method: str | None = None,
    n_train: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    n_train_pos: int | None = None,
    n_val_pos: int | None = None,
    n_test_pos: int | None = None,
    n_train_controls: int | None = None,
    n_train_incident: int | None = None,
    n_train_prevalent: int | None = None,
    n_val_controls: int | None = None,
    n_val_incident: int | None = None,
    n_val_prevalent: int | None = None,
    n_test_controls: int | None = None,
    n_test_incident: int | None = None,
    n_test_prevalent: int | None = None,
    split_mode: str | None = None,
    optuna_enabled: bool = False,
    n_trials: int | None = None,
    n_iter: int | None = None,
    threshold_objective: str | None = None,
    prevalence_adjusted: bool = False,
    timestamp: bool = True,
    extra_lines: list[str] | None = None,
) -> list[str]:
    """
    Build enriched metadata lines for plot annotations.

    Creates a structured set of metadata lines that capture key aspects
    of the model training run for reproducibility and context.

    Args:
        model: Model identifier (e.g., "LR_EN", "XGBoost")
        scenario: Scenario name (e.g., "IncidentOnly")
        seed: Random seed used
        train_prev: Training set prevalence
        target_prev: Target prevalence for calibration (optional)
        cv_folds: Number of CV folds (optional)
        cv_repeats: Number of CV repeats (optional)
        cv_scoring: CV scoring metric (optional)
        n_features: Number of features selected (optional)
        feature_method: Feature selection method (optional)
        n_train: Training set size (optional)
        n_val: Validation set size (optional)
        n_test: Test set size (optional)
        n_train_pos: Number of positive cases in training set (optional)
        n_val_pos: Number of positive cases in validation set (optional)
        n_test_pos: Number of positive cases in test set (optional)
        n_train_controls: Number of control samples in training set (optional)
        n_train_incident: Number of incident cases in training set (optional)
        n_train_prevalent: Number of prevalent cases in training set (optional)
        n_val_controls: Number of control samples in validation set (optional)
        n_val_incident: Number of incident cases in validation set (optional)
        n_val_prevalent: Number of prevalent cases in validation set (optional)
        n_test_controls: Number of control samples in test set (optional)
        n_test_incident: Number of incident cases in test set (optional)
        n_test_prevalent: Number of prevalent cases in test set (optional)
        split_mode: Split mode ("development" or "holdout") (optional)
        optuna_enabled: Whether Optuna was used (default: False)
        n_trials: Number of Optuna trials (optional)
        n_iter: Number of RandomizedSearchCV iterations (optional)
        threshold_objective: Threshold selection objective (optional)
        prevalence_adjusted: Whether prevalence adjustment was applied (default: False)
        timestamp: Include generation timestamp (default: True)
        extra_lines: Additional custom metadata lines (optional)

    Returns:
        List of metadata strings suitable for plot annotation

    Example:
        >>> meta = build_plot_metadata(
        ...     model="LR_EN",
        ...     scenario="IncidentOnly",
        ...     seed=0,
        ...     train_prev=0.167,
        ...     target_prev=0.0034,
        ...     cv_folds=5,
        ...     cv_repeats=10,
        ...     n_train=1000,
        ...     n_train_pos=150,
        ...     n_features=200,
        ...     feature_method="hybrid"
        ... )
    """
    lines = []

    # Line 1: Core identifiers and split mode
    line1_parts = [f"Model: {model}", f"Scenario: {scenario}"]
    if split_mode:
        line1_parts.append(f"Split: {split_mode}")
    line1_parts.append(f"Seed: {seed}")
    lines.append(" | ".join(line1_parts))

    # Line 2: Sample sizes with category breakdown or positive counts
    size_parts = []

    # Helper to format sample size with breakdown
    def format_split_info(split_name, n_total, n_controls, n_incident, n_prevalent, n_pos):
        """Format sample info with category breakdown or fallback to pos count."""
        split_str = f"{split_name}: n={n_total}"
        breakdown = []

        # Check if we have category breakdown
        if n_controls is not None:
            breakdown.append(f"controls={n_controls}")
        if n_incident is not None:
            breakdown.append(f"incident={n_incident}")
        if n_prevalent is not None:
            breakdown.append(f"prevalent={n_prevalent}")

        # Add prevalence if we have categories and positive count
        if breakdown and n_pos is not None and n_total > 0:
            breakdown.append(f"prev={n_pos/n_total:.3f}")
        elif n_pos is not None and not breakdown:
            # Fallback to old format if no categories
            breakdown.append(f"pos={n_pos}")

        if breakdown:
            split_str += f" ({', '.join(breakdown)})"
        return split_str

    if n_train is not None:
        size_parts.append(
            format_split_info(
                "Train",
                n_train,
                n_train_controls,
                n_train_incident,
                n_train_prevalent,
                n_train_pos,
            )
        )

    if n_val is not None:
        size_parts.append(
            format_split_info(
                "Val", n_val, n_val_controls, n_val_incident, n_val_prevalent, n_val_pos
            )
        )

    if n_test is not None:
        size_parts.append(
            format_split_info(
                "Test",
                n_test,
                n_test_controls,
                n_test_incident,
                n_test_prevalent,
                n_test_pos,
            )
        )

    if size_parts:
        lines.append(" | ".join(size_parts))

    # Line 3: CV configuration and scoring
    line3_parts = []
    if cv_folds and cv_repeats:
        cv_str = f"CV: {cv_folds}-fold x {cv_repeats} repeats"
        line3_parts.append(cv_str)

    if optuna_enabled and n_trials:
        line3_parts.append(f"Optuna: {n_trials} trials")
    elif cv_scoring:
        line3_parts.append(f"Scoring: {cv_scoring}")
        if n_iter:
            line3_parts.append(f"n_iter={n_iter}")

    if line3_parts:
        lines.append(" | ".join(line3_parts))

    # Line 4: Features and prevalence
    line4_parts = []
    if n_features is not None:
        feat_str = f"Features: {n_features}"
        if feature_method:
            feat_str += f" ({feature_method})"
        line4_parts.append(feat_str)

    # Prevalence info
    prev_parts = [f"train={train_prev:.3f}"]
    if target_prev is not None:
        prev_parts.append(f"target={target_prev:.3f}")
    line4_parts.append(f"Prev: {', '.join(prev_parts)}")

    if line4_parts:
        lines.append(" | ".join(line4_parts))

    # Line 5: Advanced settings (if present)
    line5_parts = []
    if threshold_objective:
        line5_parts.append(f"Threshold: {threshold_objective}")

    if prevalence_adjusted:
        line5_parts.append("Prevalence-adjusted")

    if line5_parts:
        lines.append(" | ".join(line5_parts))

    # Timestamp
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"Generated: {timestamp_str}")

    # Extra custom lines
    if extra_lines:
        lines.extend(extra_lines)

    return lines


def build_oof_metadata(
    model: str,
    scenario: str,
    seed: int,
    cv_folds: int,
    cv_repeats: int,
    train_prev: float,
    n_train: int | None = None,
    n_train_pos: int | None = None,
    n_train_controls: int | None = None,
    n_train_incident: int | None = None,
    n_train_prevalent: int | None = None,
    n_features: int | None = None,
    feature_method: str | None = None,
    cv_scoring: str | None = None,
    extra_lines: list[str] | None = None,
) -> list[str]:
    """
    Build metadata for out-of-fold (OOF) plots.

    Specialized metadata builder for OOF predictions across CV repeats.

    Args:
        model: Model identifier
        scenario: Scenario name
        seed: Random seed
        cv_folds: Number of CV folds
        cv_repeats: Number of CV repeats
        train_prev: Training set prevalence
        n_train: Training set size (optional)
        n_train_pos: Number of positive cases in training set (optional)
        n_features: Number of features selected (optional)
        feature_method: Feature selection method (optional)
        cv_scoring: CV scoring metric (optional)
        extra_lines: Additional metadata lines (optional)

    Returns:
        List of metadata strings
    """
    return build_plot_metadata(
        model=model,
        scenario=scenario,
        seed=seed,
        train_prev=train_prev,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        cv_scoring=cv_scoring,
        n_train=n_train,
        n_train_pos=n_train_pos,
        n_train_controls=n_train_controls,
        n_train_incident=n_train_incident,
        n_train_prevalent=n_train_prevalent,
        n_features=n_features,
        feature_method=feature_method,
        timestamp=True,
        extra_lines=extra_lines,
    )


def build_aggregated_metadata(
    n_splits: int,
    split_seeds: list[int],
    sample_categories: dict[str, dict[str, int]] | None = None,
    timestamp: bool = True,
) -> list[str]:
    """
    Build metadata for aggregated plots across multiple splits.

    Args:
        n_splits: Number of splits aggregated
        split_seeds: List of seed values used
        sample_categories: Dict with test/val/train_oof sample counts by category
                          (e.g., {"test": {"controls": 1800, "incident": 36, ...}})
        timestamp: Include generation timestamp (default: True)

    Returns:
        List of metadata strings
    """
    lines = [f"Pooled from {n_splits} splits (seeds: {split_seeds})"]

    # Add sample category breakdown if provided
    if sample_categories:
        for split_name in ["train_oof", "val", "test"]:
            if split_name in sample_categories:
                counts = sample_categories[split_name]
                total = counts.get("total", 0)
                controls = counts.get("controls")
                incident = counts.get("incident")
                prevalent = counts.get("prevalent")

                # Format based on available data
                if controls is not None and incident is not None:
                    # Full breakdown
                    cat_str = (
                        f"{split_name.replace('_', ' ').title()}: "
                        f"n={total} "
                        f"(controls={controls}, incident={incident}, prevalent={prevalent})"
                    )
                elif controls is not None:
                    # Partial breakdown
                    cat_str = (
                        f"{split_name.replace('_', ' ').title()}: n={total} (controls={controls})"
                    )
                else:
                    # Just total
                    cat_str = f"{split_name.replace('_', ' ').title()}: n={total}"

                lines.append(cat_str)

    if timestamp:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"Generated: {timestamp_str}")

    return lines


def count_category_breakdown(df) -> dict[str, int]:
    """
    Count samples by category from prediction DataFrame.

    Args:
        df: DataFrame with 'category' column

    Returns:
        Dict with keys: 'controls', 'incident', 'prevalent', 'total'
        Returns empty dict if 'category' column not found.
    """
    if df is None or "category" not in df.columns:
        return {}

    counts = df["category"].value_counts().to_dict()
    return {
        "controls": counts.get("Controls", 0),
        "incident": counts.get("Incident", 0),
        "prevalent": counts.get("Prevalent", 0),
        "total": len(df),
    }
