---
type: tension
severity: medium
category: data-quality
discovered: 2026-04-20
discovered_by: celiac-scaffold-agent
status: open
affects:
  - "operations/cellml/projects/celiac/gates/v6-ensemble/ledger.md"
  - "operations/cellml/projects/celiac/gates/v6-ensemble/decision.md"
references:
  - "docs/adr/ADR-007-oof-stacking-ensemble.md"
  - "operations/cellml/MASTER_PLAN.md"
  - "operations/cellml/DECISION_TREE_AUDIT.md"
---

# ADR-007 retains Accepted status and does not reflect the 2026-04-08 V6 informational restructure

## Contradiction

`MASTER_PLAN.md` and `DECISION_TREE_AUDIT.md` both record a 2026-04-08 architectural change that reframed the OOF stacking ensemble from an "always-active" decision (per original ADR-007) to a V6 **informational** post-tree comparison: stacking runs after V1-V5 lock a single-model winner, the ensemble is measured against the winner with a higher bar (`Δ > 0.02` AND full CI > 0), and a human makes the interpretability trade-off. The file `docs/adr/ADR-007-oof-stacking-ensemble.md` on disk does not reflect this restructure: its status is still a plain "Accepted" dated 2026-01-22, its body describes stacking as the normal path without the V6 gating criterion, and there is no note that the decision has been revisited or downgraded to informational.

## Evidence

- `docs/adr/ADR-007-oof-stacking-ensemble.md` line 3:
  "**Status:** Accepted | **Date:** 2026-01-22"
  Body (lines 5-29) describes OOF stacking as the production approach with no V6 / informational / higher-bar qualifiers. No mention of `δ=0.02` threshold or "human decides interpretability tradeoff".
- `operations/cellml/DECISION_TREE_AUDIT.md` §3.4 (line 163):
  "ADR-007 disables stacking. If the best clinical model is an ensemble of models, the factorial can't discover this."
- `operations/cellml/DECISION_TREE_AUDIT.md` disposition row (line 222):
  "P3: Ensemble exclusion | **Documented.** Added 'Design Constraint: Single-Model Scope' section to DESIGN.md. **ADR-007 updated.** | 2026-04-08 | Post-factorial follow-up if no single model dominates."
- `operations/cellml/DECISION_TREE_AUDIT.md` disposition row (line 224):
  "P3: Ensemble excluded | **Implemented as V6.** Post-tree informational comparison. Stacks non-dominated models from V2, compares vs locked single-model winner. Higher bar (δ=0.02, full CI > 0). Human decides."
- `operations/cellml/MASTER_PLAN.md` §"Decision Architecture" V6 block (lines 122-125):
  "V6: Ensemble Comparison (post-tree, informational) — Non-dominated models from V2 → stacking meta-learner. Higher bar: gain > δ (0.02) AND full CI > 0. Human decides interpretability tradeoff."
- `operations/cellml/MASTER_PLAN.md` §"ADR Map" (line 260):
  "ADR-007 | OOF stacking ensemble | **Partial** — V6 post-tree ensemble comparison (informational, not auto-selected)"

DECISION_TREE_AUDIT line 222 asserts "ADR-007 updated" as of 2026-04-08, but the ADR file on disk does not contain any text matching the V6 restructure. Either the audit entry is aspirational (the update was planned but never committed), or the update was committed to a different file.

## Status

This is a provenance / documentation-staleness issue. The file `docs/adr/ADR-007-oof-stacking-ensemble.md` does not match the behavior documented elsewhere. The severity is medium because:

- Operators reading ADR-007 in isolation will believe stacking is a production decision, not an informational post-tree step.
- The LLM advisor at V6 entry will cite a rulebook / ADR set whose ADR-007 contradicts MASTER_PLAN's V6 spec. If the advisor treats ADR-007 as authoritative, V6's ledger predictions will be framed as "stacking wins" vs "stacking loses" rather than "stacking gain > δ" vs "stacking gain ≤ δ" — a different claim type under the SCHEMA rubric.
- A factorial reviewer checking reproducibility will find that the committed ADR disagrees with the committed MASTER_PLAN, which is a provenance failure independent of which one is "right".

## Proposed resolution

Human should:

1. **Amend `docs/adr/ADR-007-oof-stacking-ensemble.md`** with a status update — either change status to "Revisited 2026-04-08" with a dated addendum block describing the V6 restructure (`δ=0.02`, full CI > 0, informational, human adjudicates), or add a "Superseded by V6 spec in MASTER_PLAN.md" note at the top. Leave the original 2026-01-22 decision body intact as historical record.
2. **Do not** edit MASTER_PLAN.md or DECISION_TREE_AUDIT.md — their V6 description is the current spec and the audit row is the change log.
3. Optionally, add a one-line cross-reference from the amended ADR-007 back to `operations/cellml/DECISION_TREE_AUDIT.md` §3.4 row so future readers can trace the restructure.

Do not attempt to modify the rulebook from this tension — ADR wording is project-level, not rulebook-level.

## Status

Open — awaiting human adjudication.
