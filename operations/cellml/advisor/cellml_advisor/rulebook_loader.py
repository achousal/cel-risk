"""Load and validate the CellML rulebook (equations, condensates, protocols)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .models import Condensate, Rulebook


class LinkError(BaseModel):
    """A wiki-link integrity problem found during rulebook validation."""

    source_slug: str
    target: str
    kind: str  # "dangling" | "cycle" | "malformed"
    message: str


def load_rulebook(rulebook_dir: Path) -> Rulebook:
    """Load every equation, condensate, and protocol under `rulebook_dir`.

    Walks `rulebook_dir/{equations,condensates,protocols}/*.md`, parses YAML
    frontmatter, constructs the typed models, and returns a `Rulebook` keyed by
    slug. The snapshot field is read from the git tag at the rulebook HEAD when
    available; otherwise left None.
    """
    raise NotImplementedError("scaffold: implement in v0.2")


def validate_link_integrity(rulebook: Rulebook) -> list[LinkError]:
    """Walk `depends_on` across all entries and return dangling/cycle errors.

    A dangling link points to a slug not present in the loaded rulebook. A
    cycle is any non-trivial strongly connected component of size > 1 in the
    condensate/equation dependency graph. Protocols are DAG-leaves; cycles
    through protocols are reported as malformed.
    """
    raise NotImplementedError("scaffold: implement in v0.2")


def get_condensate(rulebook: Rulebook, slug: str) -> Condensate:
    """Return the condensate for `slug` or raise KeyError with a searchable message."""
    raise NotImplementedError("scaffold: implement in v0.2")
