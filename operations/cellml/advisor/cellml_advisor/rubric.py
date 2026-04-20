"""Fixed falsifier rubric (see rulebook/SCHEMA.md: "Fixed falsifier rubric" and "Per-protocol metric overrides")."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .models import Protocol, RubricOverride

ClaimType = Literal["direction", "equivalence", "dominance", "inconclusive"]


class RubricThresholds(BaseModel):
    """Active direction/equivalence thresholds for a single metric.

    Defaults from rulebook/SCHEMA.md: direction requires |delta| >= 0.02 AND CI
    excludes 0; equivalence requires |delta| < 0.01 AND CI subset of
    [-0.02, 0.02]. Dominance is computed at the axis level (handled by the
    tension detector), not per metric here.
    """

    direction_margin: float = 0.02
    equivalence_abs_max: float = 0.01
    equivalence_band: tuple[float, float] = (-0.02, 0.02)


def classify_claim(
    delta: float,
    ci_lo: float,
    ci_hi: float,
    overrides: RubricThresholds | None = None,
) -> ClaimType:
    """Classify an observed (delta, CI) under the fixed rubric.

    Returns one of direction | equivalence | dominance | inconclusive. When
    `overrides` is provided, per-metric thresholds replace the defaults; the
    CI exclusion/inclusion logic is invariant regardless of override
    (rulebook/SCHEMA.md: "What overrides cannot do"). Dominance is NOT
    emitted here — it is an axis-level aggregation computed by the caller.
    """
    raise NotImplementedError("scaffold: implement in v0.2")


def apply_overrides(default: RubricThresholds, protocol: Protocol) -> dict[str, RubricThresholds]:
    """Merge a protocol's `metric_overrides` onto the default thresholds.

    Returns a mapping metric_name -> RubricThresholds. Metrics without an
    override inherit `default`. Per SCHEMA, overrides affect only the declared
    metric and do NOT propagate to downstream gates.
    """
    raise NotImplementedError("scaffold: implement in v0.2")


def override_to_thresholds(o: RubricOverride) -> RubricThresholds:
    """Convert a protocol-frontmatter override into runtime thresholds."""
    raise NotImplementedError("scaffold: implement in v0.2")
