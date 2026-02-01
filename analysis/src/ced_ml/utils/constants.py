"""Statistical and numerical constants used across the pipeline."""

# Confidence interval parameters
CI_ALPHA = 0.05
CI_LOWER_PCT = 2.5
CI_UPPER_PCT = 97.5

# Bootstrap parameters
MIN_BOOTSTRAP_SAMPLES = 20

# Statistical thresholds
Z_CRITICAL_005 = 1.96

# Prevalence safety bounds for training
MIN_SAFE_PREVALENCE = 0.01
MAX_SAFE_PREVALENCE = 0.50

# Default target specificity for clinical screening
DEFAULT_TARGET_SPEC = 0.95
