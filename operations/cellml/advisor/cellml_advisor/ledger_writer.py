"""Draft and validate gate `ledger.md` artifacts (pre-run reasoning)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .models import Rulebook


class ValidationError(BaseModel):
    """A schema or content problem found while validating a ledger.md file."""

    location: str  # e.g. "frontmatter.rulebook_snapshot" or "body.section.predictions"
    kind: str  # "missing_section" | "missing_field" | "dangling_link" | "rubric_violation"
    message: str


def draft_ledger(rulebook: Rulebook, project_dir: Path, gate: str) -> str:
    """Return a `ledger.md` string drafted from rulebook + prior gate state.

    Reads the protocol for `gate`, any prior-gate `decision.md` files under
    `project_dir/gates/` (for inherited locks), and writes frontmatter (gate,
    project, rulebook_snapshot, dataset_fingerprint, created, author,
    active_overrides) plus the 6 required body sections (Hypothesis,
    Search-space restriction, Cited rulebook entries, Falsifier criteria,
    Predictions with criteria, Risks & fallbacks). Each section contains a
    `<!-- LLM_FILL_HERE: <section> -->` marker and the cited condensate list
    from `protocol.depends_on`. Pure string return; caller writes to disk.
    """
    raise NotImplementedError("scaffold: implement in v0.2")


def validate_ledger(ledger_path: Path, rulebook: Rulebook) -> list[ValidationError]:
    """Validate a ledger.md file on disk against the schema and rulebook.

    Checks: all 6 required sections present; frontmatter fields complete;
    wiki-links in the body resolve to real rulebook slugs; declared falsifier
    criteria use only rubric claim types (direction | equivalence | dominance
    | inconclusive); active_overrides echo the protocol's metric_overrides.
    """
    raise NotImplementedError("scaffold: implement in v0.2")
