"""Mathematical utility functions for numerical stability."""

import numpy as np

# Epsilon constants for numerical stability
EPSILON_LOGIT = 1e-7
EPSILON_LOGLOSS = 1e-15
EPSILON_PREVALENCE = 1e-9
EPSILON_BOUNDS = 1e-6

# Jeffreys prior constant
JEFFREYS_ALPHA = 0.5


def logit(p: np.ndarray, eps: float = EPSILON_LOGIT) -> np.ndarray:
    """Compute logit (log-odds) from probabilities.

    Args:
        p: Probabilities in [0, 1]
        eps: Epsilon for clipping to prevent log(0)

    Returns:
        Log-odds values
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def inv_logit(z: np.ndarray) -> np.ndarray:
    """Compute inverse logit (sigmoid) from log-odds.

    Args:
        z: Log-odds values

    Returns:
        Probabilities in [0, 1]
    """
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def jeffreys_smooth(k: float, n: float, alpha: float = JEFFREYS_ALPHA) -> float:
    """Apply Jeffreys prior smoothing to proportion estimate.

    Args:
        k: Number of successes
        n: Total trials
        alpha: Prior pseudo-count (default: 0.5 for Jeffreys)

    Returns:
        Smoothed proportion (k + alpha) / (n + 2 * alpha)
    """
    return (k + alpha) / (n + 2 * alpha)
