"""Pydantic v2 models mirroring rulebook and gate-artifact schemas (see rulebook/SCHEMA.md)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Rulebook entries
# -----------------------------------------------------------------------------


class Equation(BaseModel):
    """Rulebook equation entry. Mirrors `rulebook/equations/{slug}.md` frontmatter.

    Body sections (Statement, Derivation, Boundary conditions, Worked reference)
    are not modeled here; use a loader to surface them as strings.
    """

    type: Literal["equation"] = "equation"
    slug: str
    symbol: str
    depends_on: list[str] = Field(default_factory=list)
    computational_cost: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)


class CondensateEvidence(BaseModel):
    """Single evidence row for a condensate."""

    dataset: str
    gate: str | None = None
    n: int | None = None
    delta: str | None = None
    date: str | None = None
    source_gate: str | None = None


class Condensate(BaseModel):
    """Rulebook condensate entry. Mirrors `rulebook/condensates/{slug}.md` frontmatter."""

    type: Literal["condensate"] = "condensate"
    slug: str
    depends_on: list[str] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list)
    status: Literal["provisional", "established", "retired"] = "provisional"
    confirmations: int = 0
    evidence: list[CondensateEvidence] = Field(default_factory=list)
    falsifier: str | None = None


class RubricOverride(BaseModel):
    """Per-metric rubric override declared in a protocol's `metric_overrides`."""

    direction_margin: float
    equivalence_band: Annotated[list[float], Field(min_length=2, max_length=2)]
    justification: str


class Protocol(BaseModel):
    """Rulebook protocol entry. One playbook per factorial gate version.

    Mirrors `rulebook/protocols/{slug}.md` frontmatter. `informational` is a
    reserved flag for protocols that document rather than gate (no locks).
    """

    type: Literal["protocol"] = "protocol"
    slug: str
    gate: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    axes_explored: list[str] = Field(default_factory=list)
    axes_deferred: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    informational: bool = False
    metric_overrides: dict[str, RubricOverride] = Field(default_factory=dict)


class Rulebook(BaseModel):
    """Container aggregating all rulebook entries indexed by slug."""

    equations: dict[str, Equation] = Field(default_factory=dict)
    condensates: dict[str, Condensate] = Field(default_factory=dict)
    protocols: dict[str, Protocol] = Field(default_factory=dict)
    snapshot: str | None = None  # e.g. "rb-v0.1.0"


# -----------------------------------------------------------------------------
# Gate artifacts (projects/<name>/gates/<gate>/)
# -----------------------------------------------------------------------------


class Prediction(BaseModel):
    """One pre-registered prediction with its rubric claim type and criterion."""

    id: str
    statement: str
    claim_type: Literal["direction", "equivalence", "dominance", "inconclusive"]
    criterion: str


class Ledger(BaseModel):
    """PRE-run reasoning (ledger.md frontmatter + section presence)."""

    gate: str
    project: str
    rulebook_snapshot: str
    dataset_fingerprint: str
    created: str
    author: Literal["llm-advisor", "human", "retrospective"]
    active_overrides: dict[str, RubricOverride] = Field(default_factory=dict)
    # Body section presence (populated by ledger-loader; True iff header found).
    has_hypothesis: bool = False
    has_search_space: bool = False
    has_cited_entries: bool = False
    has_falsifier_criteria: bool = False
    has_predictions: bool = False
    has_risks: bool = False
    predictions: list[Prediction] = Field(default_factory=list)


class Metric(BaseModel):
    """A measured metric value with 95% bootstrap CI (when applicable)."""

    name: str  # e.g. "AUROC", "PRAUC", "Brier", "REL", "panel_size"
    value: float
    ci_lo: float | None = None
    ci_hi: float | None = None
    axis_slice: dict[str, str] = Field(default_factory=dict)  # e.g. {"model": "LR_EN"}


class Observation(BaseModel):
    """POST-run facts (observation.md). Produced by cellml-reduce, not the advisor."""

    gate: str
    project: str
    observed_at: str
    metrics: list[Metric] = Field(default_factory=list)


class Decision(BaseModel):
    """POST-run reasoning (decision.md frontmatter + section presence)."""

    gate: str
    observed_at: str
    predictions_held: list[str] = Field(default_factory=list)  # prediction ids
    predictions_failed: list[str] = Field(default_factory=list)
    actual_claim_type: Literal["direction", "equivalence", "dominance", "inconclusive"]
    locks: list[str] = Field(default_factory=list)
    predictions_for_next: list[str] = Field(default_factory=list)


class Tension(BaseModel):
    """Delta log entry: a prediction disagreed with the observed metric under the rubric."""

    prediction_id: str
    predicted_claim_type: Literal["direction", "equivalence", "dominance", "inconclusive"]
    observed_claim_type: Literal["direction", "equivalence", "dominance", "inconclusive"]
    metric: str
    axis_slice: dict[str, str] = Field(default_factory=dict)
    delta: float
    ci_lo: float | None = None
    ci_hi: float | None = None
    notes: str | None = None
