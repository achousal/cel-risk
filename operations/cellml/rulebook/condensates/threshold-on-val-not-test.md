---
type: condensate
depends_on: ["[[equations/fixed-spec-threshold]]"]
applies_to:
  - "any pipeline that selects a decision threshold on calibrated probabilities"
  - "three-way split designs (TRAIN / VAL / TEST)"
  - "screening and clinical operating-point reporting"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v5-confirmation
    delta: "ADR-009 locks threshold_source = VAL; threshold selection on TEST would leak operating-point optimism into the final sensitivity/specificity estimate"
    date: "2026-01-20"
    source: "ADR-009, operations/cellml/DESIGN.md V5, ADR-001 (three-way split)"
falsifier: |
  On bootstrap resamples (n_boot >= 1000) of archived completed projects whose
  TEST sets are no longer quarantined (i.e., already reported and locked in
  publications or archive), compute two statistics per resample: (a) oracle-
  threshold AUROC -- threshold selected on TEST, applied to TEST -- and (b)
  val-threshold AUROC -- threshold selected on VAL, applied to TEST. If the
  distribution of |AUROC_oracle - AUROC_val| converges within +/-0.005 with
  95% CI subset of [-0.01, 0.01] across at least 3 non-overlapping historical
  cohorts, this condensate is weakened (Equivalence claim per SCHEMA rubric).
  At 5 or more cohorts, retire.

  This simulation uses data that is already post-quarantine and is a retro-
  spective analysis; it does NOT break the quarantine of any live or in-progress
  project.
---

# Decision thresholds selected on TEST produce optimistically biased sensitivity at the reported operating point

## Claim

The decision threshold $\tau$ (including fixed-specificity thresholds, Youden's $J$, and max $F_1$) MUST be selected on the VAL split. Selecting $\tau$ on TEST leaks operating-point optimism: the TEST-set sensitivity reported at the TEST-selected $\tau$ is biased upward relative to deploy-time sensitivity, because $\tau$ was chosen to maximize performance on the very same data used for the final evaluation.

## Mechanism

Threshold selection is an optimization step: among candidate $\tau$ values, pick the one that best satisfies the objective (e.g., the one whose control-score quantile matches the target specificity, see [[equations/fixed-spec-threshold]]). Any data used as the optimization target is no longer a held-out estimate of the objective's value at the selected $\tau$. Reporting sensitivity at a $\tau$ selected on TEST is therefore reporting the **maximum** of a finite set of sensitivities, evaluated on the same data — a statistical overestimate.

The three-way split (ADR-001) exists specifically to break this circularity:
- **TRAIN**: biased (model was fit on it)
- **VAL**: unbiased for threshold selection (model was not fit on it, but threshold is chosen to optimize on it)
- **TEST**: unbiased for operating-point evaluation only if NEVER peeked at during selection

Steyerberg (2019, Ch. 11) documents this as a standard failure mode in clinical prediction modeling: "decision threshold optimism" inflates reported sensitivity by 3–10 percentage points on cohorts with $n_\text{case} \lesssim 200$.

## Actionable rule

- `ThresholdConfig.threshold_source` (see `analysis/src/ced_ml/config/schema.py:198-208`) MUST be set to `VAL`. `TEST` is disallowed at the schema level; pipelines that bypass this schema are hook-blocked at write time.
- V5 confirmation gate: the final sensitivity/specificity report uses $\tau_{\text{VAL}}$ selected on VAL and evaluated (never re-optimized) on TEST.
- Pre-registration: the threshold rule (fixed-spec at target 0.95, Youden, or max $F_1$) is declared in the V4 ledger; swapping rules after peeking at TEST is a search-space violation.
- Reporting: every sensitivity estimate must cite the split on which the threshold was selected AND the split on which sensitivity was measured. The two must differ.

## Boundary conditions

- Requires a three-way split with VAL $\ne$ TEST. Two-way splits (TRAIN / TEST) have no VAL slice for threshold selection; the only valid move in that design is to use OOF predictions on TRAIN for threshold selection.
- Requires VAL to be large enough that the target-quantile estimate is stable. For fixed-spec at 0.95, the rule-of-thumb is $n_\text{VAL,control} \ge 1000$ so that the 5th percentile is estimated over at least 50 rows. On the celiac cohort, $n_\text{VAL,control} = 10{,}916$ far exceeds this.
- Does NOT apply when the operating point is pre-registered (e.g., $\tau = 0.5$ specified in advance without any data-dependent selection). Pre-registered thresholds carry no selection optimism and can be reported directly on TEST.
- This condensate assumes only ONE threshold report. Reporting sensitivity at multiple thresholds for illustration purposes (e.g., an ROC curve) does not select an operating point; the risk arises only when a single $\tau$ is locked and cited as the deploy-time operating point.

## Evidence

| Dataset | n_VAL_cases | n_TEST_cases | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 37 | 37 | ADR-009 locks threshold_source = VAL for the cel-risk pipeline; TEST-selection was never executed (by design) so the empirical delta on this cohort is mechanistic rather than observational | V5 confirmation |

## Falsifier execution

The bootstrap simulation in the falsifier is **retrospective by construction**: it
operates only on archived completed projects whose TEST splits have already been
reported and locked in publications or archive (post-quarantine data). It does
NOT authorize any live or in-progress project to peek at its quarantined TEST
split, and the actionable rule above (threshold_source = VAL, TEST disallowed at
schema level) remains fully in force for all live pipelines. Implementation of
the simulation harness is deferred until promotion to `established` is attempted
(i.e., when >=3 cohort confirmations accrue); until then, the criterion is
retained as a theoretically-executable weakening path.

## Related

- [[equations/fixed-spec-threshold]] — the specific threshold rule this condensate governs (also applies to Youden and max F1)
- [[condensates/fixed-spec-for-screening]] — complements this condensate; addresses WHICH threshold objective to use
- ADR-009 (cel-risk) — decision record this condensate formalizes
- ADR-001 (cel-risk) — provides the VAL split this condensate requires
<!-- TODO: verify slug exists after batch merge --> - [[condensates/three-way-split-prevents-threshold-leakage]]
<!-- TODO: verify slug exists after batch merge --> - [[protocols/v5-confirmation]]
