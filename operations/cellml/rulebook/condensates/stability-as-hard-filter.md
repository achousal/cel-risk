---
type: condensate
depends_on:
  - "[[equations/consensus-rank-aggregation]]"
  - "[[equations/nested-cv]]"
applies_to:
  - "per-model feature-selection pipelines that produce selection-frequency statistics across CV folds"
  - "panels destined for cross-model consensus aggregation"
  - "high-dimensional p > n settings where twin-feature artifacts are likely"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "hard stability filter (freq >= 0.90) removed inconsistently selected proteins before consensus; composite weighting (OOF + stability + essentiality) was rejected in ADR-004 for silent degradation when any signal is absent"
    date: "2026-02-09"
    source: "ADR-004 (Stage 2 per-model evidence; Alternatives section)"
falsifier: |
  Direction claim: per-model OOF-importance-only consensus (no stability filter)
  vs. stability-filtered-then-OOF consensus produces different top-N panels
  on the same dataset. Criterion: Jaccard(top-N) < 0.80 with 95% bootstrap CI
  excluding 0.80. If filtered and unfiltered consensus agree within |ΔJaccard|
  < 0.01 and 95% CI inside [−0.02, 0.02] (Equivalence) across >=3 datasets
  with p > n, the hard-filter requirement is weakened (OOF may be sufficient
  alone). Retire at 5 such datasets.
---

# Stability must be a hard pre-filter, not a weight in a composite score

## Claim

Selection-frequency stability — the fraction of CV folds in which a protein is retained by the per-model selector — MUST be applied as a hard pass/fail filter (threshold default 0.90) before OOF-importance ranking feeds the cross-model aggregator. It must NOT be combined with OOF importance and drop-column essentiality into a single weighted composite score. Composite scoring silently degrades when any one signal is absent or miscalibrated, because the missing signal's zero contribution is indistinguishable from a real zero. A hard filter fails loudly; a composite weight fails quietly.

## Mechanism

Drop-column essentiality (post-hoc $\Delta$AUROC per cluster on the final panel) and RFE rank (independent panel sizing via Pareto) are computed for different purposes — interpretation and sizing respectively — and on different units (clusters, not proteins). Coercing them into the same weighted sum as OOF importance and stability requires a weight schema; any weight schema is a tuning knob that the pipeline cannot ground in data without leakage. ADR-004 resolves the ambiguity by assigning each signal a distinct role:

- OOF importance: primary ranking input to [[equations/consensus-rank-aggregation]]
- Stability: hard filter (pass/fail before ranking)
- RFE: sizing (selects $N$ via Pareto, not order)
- Drop-column: post-hoc interpretation on the locked panel

This mapping is strictly ordered. Stability filter -> OOF ranking -> RRA aggregation across models -> top-$N$ panel via RFE-derived $N$ -> drop-column interpretation. Each stage has a single decision artifact; no stage weights multiple inputs.

## Actionable rule

- Stability threshold is a protocol lock, not a tunable. Default 0.90-0.95; declared before V1 runs.
- Proteins with stability frequency $< \tau$ are removed from the per-model ranked list BEFORE that list is passed to [[equations/consensus-rank-aggregation]].
- Do NOT emit "composite feature score" artifacts that blend OOF + stability + essentiality. If such an artifact is requested, redirect the consumer to the individual outputs.
- Drop-column essentiality runs AFTER the panel is locked; it is read-only for panel composition.
- RFE outputs are consumed only by the panel-sizing protocol, never by the ranking aggregator.

## Boundary conditions

- **Requires CV structure.** Stability frequency is only definable when the per-model selector runs inside [[equations/nested-cv]]. A single fit produces no frequency distribution.
- **Requires the selector to be stochastic or data-dependent.** A fully deterministic selector (e.g. a threshold on a fixed statistic with no resampling) has frequency 1.0 for every selected feature — the filter becomes trivial.
- **Threshold sensitivity.** Very high thresholds (e.g. 0.99) on small CV configurations can produce empty filtered lists. At $K_\text{out} \cdot R = 50$ folds (celiac default), $\tau = 0.90$ corresponds to selection in 45/50 folds; $\tau = 0.95$ corresponds to 47.5/50.
- **Twin-feature correction is separate.** Correlation clustering on the top-$N$ candidates of $s_i$ (ADR-004 Stage 3) handles the twin-feature problem at the cross-model level; it is not a substitute for stability filtering at the per-model level.

## Evidence

| Dataset | n | p | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | 2,920 | ADR-004 Alternatives section rejects composite weighting because "drop-column is interpretability, not ranking; stability is better as hard filter; silently degraded when signals absent" | Migrated from ADR-004 2026-02-09 |

## Related

- [[equations/consensus-rank-aggregation]] — consumes the filtered per-model ranking
- [[equations/nested-cv]] — provides the fold structure that makes stability frequency definable
- [[condensates/feature-selection-needs-model-gate]] — the pre-aggregation gate for models; this condensate is the pre-aggregation filter for proteins within a model
- ADR-004 (three-stage feature selection) — canonical source
