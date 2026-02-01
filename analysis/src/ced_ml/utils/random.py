"""
Random seed management for reproducibility.

Provides utilities for deterministic RNG seeding, including an optional
SEED_GLOBAL environment variable for single-threaded reproducibility debugging.

Naming Convention
-----------------
Parameter names for random state vary by context:

- **seed** (int): User-facing parameters in configs and CLI entrypoints.
  Used when accepting seed values from users or config files.
  Examples: bootstrap.py confidence_interval(..., seed=42)

- **random_state** (int): Parameters that directly pass to sklearn/xgboost.
  Used to maintain compatibility with scikit-learn API conventions.
  Examples: splits.py, model_selector.py, any sklearn wrappers.

- **rng** (np.random.RandomState): Internal RNG instances.
  Used when passing explicit RandomState objects between functions.
  Prefer this over global seeding for isolated reproducibility.
  Examples: create_rng(seed) -> rng, then pass rng to worker functions.

Migration Path
--------------
The global seeding function set_random_seed() is deprecated.
New code should use create_rng() to create isolated RNG instances.
Existing CLI callers are preserved for backward compatibility during migration.
"""

import logging
import os
import random
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def set_random_seed(seed: int):
    """
    Set random seed for all libraries.

    .. deprecated:: 2026-01-31
        This function uses global RNG state and is deprecated.
        Use create_rng(seed) to create isolated np.random.RandomState instances instead.
        Global seeding prevents reproducibility in parallel/concurrent contexts.

    Args:
        seed: Random seed value
    """
    warnings.warn(
        "set_random_seed() is deprecated and will be removed in a future version. "
        "Use create_rng(seed) to create an isolated np.random.RandomState instance instead. "
        "Global seeding via np.random.seed() prevents reproducibility in concurrent contexts.",
        DeprecationWarning,
        stacklevel=2,
    )
    random.seed(seed)
    np.random.seed(seed)

    # Sklearn doesn't have global seed, use random_state parameter
    # XGBoost uses numpy's random state


def create_rng(seed: int) -> np.random.RandomState:
    """
    Create an isolated NumPy RandomState instance.

    Prefer this over set_random_seed() for new code. Isolated RNG instances
    provide reproducibility without global state side effects, making code
    safe for parallel execution and easier to test.

    Args:
        seed: Random seed value (must be in range [0, 2^32-1])

    Returns:
        np.random.RandomState: Isolated RNG instance seeded with the given value

    Raises:
        ValueError: If seed is outside the valid range

    Examples:
        >>> rng = create_rng(42)
        >>> rng.random(3)
        array([0.37454012, 0.95071431, 0.73199394])

        >>> # Pass to downstream functions
        >>> def worker(rng):
        ...     return rng.choice([1, 2, 3], size=5, replace=True)
        >>> worker(create_rng(99))
        array([2, 3, 1, 1, 2])
    """
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} outside valid range [0, 2^32-1]")
    return np.random.RandomState(seed)


def apply_seed_global() -> int | None:
    """
    Check SEED_GLOBAL environment variable and apply global seeding if set.

    When SEED_GLOBAL is set to an integer value, seeds Python's random module
    and NumPy's legacy global RNG. This is intended for single-threaded
    reproducibility debugging only; production runs should use explicit
    per-component seeds via config files.

    Returns:
        The seed value applied, or None if SEED_GLOBAL was not set or invalid.

    Examples:
        >>> import os
        >>> os.environ["SEED_GLOBAL"] = "42"
        >>> seed = apply_seed_global()
        >>> seed
        42
        >>> del os.environ["SEED_GLOBAL"]
    """
    seed_str = os.environ.get("SEED_GLOBAL")
    if seed_str is None:
        return None

    seed_str = seed_str.strip()
    if not seed_str:
        return None

    try:
        seed = int(seed_str)
    except ValueError:
        logger.warning(
            "SEED_GLOBAL environment variable has non-integer value '%s'; ignoring.",
            seed_str,
        )
        return None

    if seed < 0 or seed > 2**32 - 1:
        logger.warning(
            "SEED_GLOBAL=%d out of valid range [0, 2^32-1]; ignoring.",
            seed,
        )
        return None

    set_random_seed(seed)
    logger.info("SEED_GLOBAL=%d applied (global RNG seeded for reproducibility).", seed)
    return seed


def get_cv_seed(base_seed: int, fold_idx: int, repeat_idx: int = 0) -> int:
    """
    Generate deterministic seed for CV fold.

    Args:
        base_seed: Base random seed
        fold_idx: Fold index (0-based)
        repeat_idx: Repeat index (0-based)

    Returns:
        Deterministic seed for this fold/repeat combination
    """
    return base_seed + (repeat_idx * 1000) + fold_idx
