# Rulebook Changelog

## rb-v0.2.0 — 2026-04-21

Strategy C: V0 becomes imbalance-family probe gate.

### Changed
- `protocols/v0-strategy.md`: axis `control_ratio ∈ {1, 2, 5}` REPLACED by `imbalance_probe ∈ {none, downsample_5, cw_log}`. V0 outputs now lock `imbalance_family` (categorical: none | downsample | weight) instead of a numeric ratio level
- `protocols/v3-imbalance.md`: rewritten as branching protocol conditional on V0's `imbalance_family` lock. Within-family level refinement. Legacy 3×3 grid only runs in V0-Inconclusive fallback branch
- `condensates/nested-downsampling-composition.md`: multiplicative composition claim boundary-conditioned to V0-Inconclusive fallback only
- `condensates/v3-utility-provenance-chain.md`: substrate invariant now scoped to within-family runs

### Added
- `condensates/imbalance-two-stage-decision.md`: new meta-condensate justifying family-then-level cascade

### Affected
- V1-V6 protocols: no direct changes; V1 inherits new V0 lock structure (training_strategy + imbalance_family)
- Gate artifacts: celiac V0 ledger/decision/observation/tensions + V1 ledger updated to match new axis
