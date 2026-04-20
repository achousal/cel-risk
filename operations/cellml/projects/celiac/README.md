# celiac

UK Biobank incident-celiac project. First cohort through the CellML decision
tree. V0 gate has already run; V1 main factorial is in preparation.

## Status

- **Active rulebook:** `rb-v0.0.0-unfinalized` (placeholder; will rebind to
  `rb-v0.1.0` once the ADR -> condensate migration completes)
- **Gates completed:** V0 (training strategy + control ratio) — ledger is
  retrospective; see `gates/v0-strategy/ledger.md` for caveats
- **Gates pending:** V1 recipe, V2 model, V3 imbalance (joint), V4 calibration,
  V5 confirmation, V6 ensemble (informational)

## Contents

- `dataset/` — fingerprint and documentation for the celiac dataset. The
  fingerprint is write-once; if inputs change, a new project slug is required.
- `gates/v0-strategy/` — V0 gate artifacts (ledger, observation, decision,
  tensions). Reconstructed retrospectively from `operations/cellml/DESIGN.md`,
  `MASTER_PLAN.md`, and `DECISION_TREE_AUDIT.md`.
- `tensions/` — three subfolders separating tension types. See
  `tensions/README.md`.
- `rulebook-snapshot/` — binds this project's current gate run to a specific
  rulebook tag via `hash.txt`. Currently pending the first tag.
- `archive/` — superseded gate artifacts.

## Upstream references

- Experiment index: `operations/cellml/MASTER_PLAN.md`
- Scientific design: `operations/cellml/DESIGN.md`
- V0-V5 tree tensions: `operations/cellml/DECISION_TREE_AUDIT.md`
- ADRs: `docs/adr/ADR-00{1,2,3}-*.md` (V0 axes)
- Rulebook: `operations/cellml/rulebook/`
