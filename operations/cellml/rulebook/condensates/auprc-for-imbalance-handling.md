---
type: condensate
depends_on:
  - "[[equations/brier-decomp]]"
  - "[[equations/case-control-ratio-downsampling]]"
applies_to:
  - "gates that explore training-time class-imbalance axes (weighting, within-training downsampling)"
  - "low-prevalence cohorts (pi < 0.05)"
  - "utility functions that combine discrimination with calibration at a single gate"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v3-imbalance
    delta: "pi ~ 0.00338; AUROC is prevalence-invariant by construction, so changes across the V3 weighting x downsampling grid reflect only rank-level changes; AUPRC varies with prevalence shifts and therefore exposes imbalance-handling differences that AUROC absorbs"
    date: "2026-04-20"
    source: "operations/cellml/DESIGN.md §V3; v3-imbalance protocol §2.3 (AUPRC choice rationale); T-V3.3 flagged AUPRC as not formalized in a rulebook equation"
falsifier: |
  If on >=3 cohorts, delta-AUPRC and delta-AUROC between imbalance cells
  correlate with r > 0.95 at the Direction-claim level, AUPRC offers no
  axis-unique information (Equivalence claim on the cross-metric). In that
  regime AUROC alone suffices and this condensate is weakened. If on >=5
  cohorts, retire.
---

# AUPRC is the correct discrimination metric for V3 imbalance adjudication

## Claim

Area Under Precision-Recall Curve (AUPRC) is the correct discrimination
metric for adjudicating class-imbalance-handling axes (training-time
weighting, within-training downsampling) at the V3 gate. AUROC is
insensitive to prevalence shifts by construction and therefore absorbs
exactly the variation that V3's axes induce; AUPRC varies with prevalence
and therefore exposes the differential performance that V3 needs to
decide on.

## Mechanism

AUROC integrates sensitivity against (1 - specificity). Both components
depend only on the score distribution within each class (cases and
controls separately), not on their mixing proportions. Scaling up controls
by `k` does not change either the case-score distribution or the
control-score distribution; it only changes how many controls there are.
Hence AUROC is invariant to prevalence.

AUPRC integrates precision against recall. Precision =
TP / (TP + FP) has a denominator that depends on the **mix** of cases and
controls at each threshold — FP is a count of controls misclassified,
which scales directly with control count. So AUPRC bakes prevalence into
its denominator. A change in the training prevalence (via V3's axes)
propagates into the score distribution the model produces on held-out
data, and AUPRC captures the downstream precision consequences that AUROC
cannot.

Concretely: V3 weighting changes the loss surface, which changes the
learned model's score distribution; V3 downsampling changes the training
prevalence (per [[equations/case-control-ratio-downsampling]] and the
composite composition in
[[condensates/nested-downsampling-composition]]), which changes the
learned score distribution in a prevalence-dependent way. Both effects
manifest in AUPRC; neither manifests (at rank level) in AUROC.

On low-prevalence cohorts the effect is amplified: when `pi << 1`, a small
absolute change in the false-positive rate translates into a large
relative change in precision, so AUPRC has high sensitivity to
imbalance-handling choices exactly where the choices matter most. Saito
and Rehmsmeier (2015, PLOS ONE) demonstrate this relationship empirically
and formally.

## Actionable rule

- V3's [[protocols/v3-imbalance]] uses **AUPRC** as the primary
  discrimination metric in the joint utility
  `U = 0.5 * AUPRC_norm + 0.5 * (1 - REL_norm)`.
- AUROC is logged as a secondary invariance check: if V3's `observation.md`
  shows `delta-AUROC` > 0.005 between cells at Direction claim, flag as
  an internal inconsistency — the axes should NOT be changing AUROC
  meaningfully if the model is behaving as expected; a large AUROC shift
  indicates a change in the rank-level score distribution that is not
  explained by imbalance handling alone.
- AUPRC MUST be computed on **prevalence-adjusted** predictions per
  [[condensates/downsample-requires-prevalence-adjustment]] — AUPRC on
  raw downsampled scores bakes in a training-prevalence artifact rather
  than the genuine imbalance-handling signal, and the two effects cannot
  be separated post-hoc.
- 95% bootstrap CI over 1000 seed-level paired resamples per `SCHEMA.md`
  metric rules.
- V3's `ledger.md` MUST cite this condensate alongside
  [[equations/brier-decomp]] (for the REL half of the utility) when
  declaring the utility function.

## Boundary conditions

- Applies under class imbalance, conventionally `pi < 0.05`. For
  balanced classes (`pi ~ 0.5`), AUPRC and AUROC converge — AUPRC
  collapses to a metric approximately equivalent to AUROC when the
  mixing ratio is 1:1, and the axis-unique information advantage
  disappears.
- Dominates AUROC most strongly for rare-event cohorts (`pi < 0.01`).
  The celiac cohort at `pi ~ 0.00338` is firmly in this regime.
- Does NOT apply when the gate's axes do not change the empirical class
  frequency fed to the loss function. For a gate that only changes
  post-hoc score mapping (V4 calibration), AUROC is invariant by
  construction AND AUPRC is approximately invariant (up to threshold
  discretization), so neither is the V4 discrimination adjudicator —
  that gate adjudicates on REL per
  [[condensates/calib-parsimony-order]].
- Does NOT apply to non-probabilistic outputs (e.g., binary decisions
  without score access). Both AUROC and AUPRC require ranked scores.
- The bootstrap behavior of AUPRC on very small positive counts (e.g.,
  n_cases < 50 in a bootstrap slice) has non-trivial tail behavior; see
  v3-imbalance T-V3.3 for the flag that a dedicated AUPRC-bootstrap
  equation is still needed.

## Evidence

| Dataset | n_case | n_control | prevalence | Axis varied | AUROC-delta across axis | AUPRC-delta across axis | Source gate |
|---|---|---|---|---|---|---|---|
| Celiac (UKBB) | 148 | 43662 | 0.00338 | weighting x downsampling (V3 3x3) | rank-invariant by construction | measurable differences with paired-bootstrap CI-separation | v3-imbalance protocol §2.3, DESIGN.md §V3 |

At the cohort's prevalence (~0.003), AUPRC sensitivity to imbalance-
handling is expected to be maximal; AUROC is expected to be near-
invariant across the V3 grid.

External reference (not a vault source, logged for auditability):
Saito, T. & Rehmsmeier, M. "The precision-recall plot is more informative
than the ROC plot when evaluating binary classifiers on imbalanced
datasets." PLOS ONE 10(3): e0118432 (2015).

## Related

- [[condensates/v3-utility-provenance-chain]] — shared invariant this claim participates in
- [[protocols/v3-imbalance]] — protocol that cited this gap (§2.3 AUPRC
  choice rationale, T-V3.3 "AUPRC not formalized in a rulebook equation"
  flags both a missing equation and a missing condensate; this file
  addresses the condensate; the equation remains a separate TODO)
- [[condensates/downsample-requires-prevalence-adjustment]] — AUPRC MUST
  be computed on prevalence-adjusted predictions, not raw downsampled
  scores
- [[condensates/nested-downsampling-composition]] — training prevalence
  used in the adjustment is the composite, not either axis alone
- [[equations/case-control-ratio-downsampling]] — grounds the prevalence
  formula that the adjustment consumes
- [[equations/brier-decomp]] — REL half of V3's utility; complements
  this condensate's discrimination half
