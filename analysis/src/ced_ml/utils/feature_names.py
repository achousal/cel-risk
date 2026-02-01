"""Feature name manipulation utilities."""


def extract_protein_name(feature_name: str) -> str:
    """Extract original protein name from transformed feature name.

    Handles patterns:
    - num__{protein}_resid → {protein}_resid (strip prefix only)
    - num__{protein} → {protein} (strip prefix only)
    - {protein}_resid → {protein}_resid (no change if no prefix)
    - {protein} → {protein} (plain name, no change)

    This preserves the _resid suffix when present, matching the behavior
    needed by feature extraction functions that compare against protein_cols
    which include the _resid suffix.

    Args:
        feature_name: Feature name from fitted pipeline.

    Returns:
        Protein name with num__ prefix stripped but _resid suffix preserved.

    Example:
        >>> extract_protein_name("num__APOE_resid")
        'APOE_resid'
        >>> extract_protein_name("num__IL6")
        'IL6'
        >>> extract_protein_name("IL6_resid")
        'IL6_resid'
        >>> extract_protein_name("IL6")
        'IL6'
    """
    if feature_name.startswith("num__"):
        return feature_name[len("num__") :]
    return feature_name
