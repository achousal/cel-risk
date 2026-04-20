"""Detect and emit tensions when ledger predictions disagree with observed metrics."""

from __future__ import annotations

from pathlib import Path

from .models import Ledger, Observation, Rulebook, Tension


def detect_tensions(ledger: Ledger, observation: Observation, rulebook: Rulebook) -> list[Tension]:
    """Return tensions for every prediction whose observed claim type disagrees with its predicted claim type.

    Per prediction: locate the matching metric(s) in `observation` (same name +
    axis_slice), classify the observed delta under the rubric (using any active
    overrides from the ledger frontmatter), and emit a `Tension` when the
    observed claim type differs from the predicted claim type. Metric-specific
    rules from the rubric (AUROC/PR-AUC/Brier need CI; counts compare exactly)
    are honored.
    """
    raise NotImplementedError("scaffold: implement in v0.2")


def write_tensions_md(tensions: list[Tension], out_path: Path) -> None:
    """Write `tensions.md` at `out_path` with one entry per tension.

    Each entry includes: prediction id, predicted vs observed claim type, the
    metric and axis slice, delta with CI, and a pointer to the source ledger
    prediction. Routes to `projects/<name>/tensions/rule-vs-observation/` as
    candidate rulebook-update evidence per `rulebook/SCHEMA.md`.
    """
    raise NotImplementedError("scaffold: implement in v0.2")
