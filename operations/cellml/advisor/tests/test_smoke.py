"""Smoke tests for the cellml-advisor scaffold."""


def test_imports() -> None:
    """Prove the package is importable and __version__ is exposed."""
    import cellml_advisor

    assert cellml_advisor.__version__ == "0.1.0"


# TODO(v0.2): rulebook_loader round-trip on rulebook/equations/
# TODO(v0.2): models — Equation/Condensate/Protocol parse real frontmatter
# TODO(v0.2): ledger_writer.draft_ledger emits all 6 required sections
# TODO(v0.2): ledger_writer.validate_ledger catches missing sections + dangling links
# TODO(v0.2): decision_writer.draft_decision pulls predictions and metrics
# TODO(v0.2): tension_detector.detect_tensions classifies under the rubric
# TODO(v0.2): rubric.classify_claim covers direction/equivalence/inconclusive
# TODO(v0.2): rubric.apply_overrides merges per-metric overrides
# TODO(v0.2): cli — each subcommand dispatches to the right module
