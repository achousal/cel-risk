---
type: condensate
depends_on: ["[[equations/case-control-ratio-downsampling]]"]
applies_to:
  - "extremely imbalanced cohorts (population prevalence < 1%)"
  - "pipelines where tuning cost scales with training-set size (Optuna, nested CV)"
  - "models evaluated on discrimination metrics invariant to prevalence (AUROC, AUPRC on a fixed operating curve)"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "~60× speedup (43,662 controls → ~740 controls at 5:1) with no reported AUROC loss vs undersampled baselines"
    date: "2026-01-20"
    source: "ADR-003, docs/adr/ADR-003-control-downsampling.md"
falsifier: |
  Direction: if control-downsampling at 5:1 (case:control) shows a AUROC drop
  of |Δ| ≥ 0.02 relative to no-downsampling (1:N full) on the same test split,
  with 95% bootstrap CI excluding 0, the condensate is confirmed as inverted —
  downsampling DAMAGES discrimination and should be retired. If |Δ| < 0.01 with
  95% CI ⊂ [−0.02, 0.02], condensate is confirmed as Equivalence (current
  status). Decision rule: the condensate holds only if Equivalence is met; any
  Direction in either sign retires it.
---

# Random control downsampling to a 5:1 case:control ratio preserves discrimination while cutting training compute roughly 60-fold on cohorts with population prevalence below 1%

## Claim

For cohorts with extreme class imbalance (population prevalence below 1%), the majority class carries redundant information: most controls sit far from the decision boundary and contribute negligibly to the fitted model. Random downsampling of controls to a fixed case:control ratio (ADR-003 locks 5:1) reduces the training-set size proportionally, cutting tuning and fitting compute by roughly the downsampling factor, without measurable loss on discrimination metrics that are invariant to class prevalence (AUROC, ranking-based). The savings enable otherwise-infeasible hyperparameter searches (50k+ Optuna fits on the celiac cohort).

## Mechanism

AUROC depends only on score ordering across case/control pairs, not on the marginal class rate. Downsampling removes controls uniformly at random; by exchangeability, the retained controls are representative of the population control distribution in expectation. The decision boundary shifts (empirical prevalence becomes $1/(1+r)$, see [[equations/case-control-ratio-downsampling]]), but the ranking of cases vs controls in score-space is preserved up to sampling variance.

The $\sim 60\times$ speedup comes from $n_\text{control} / (r \cdot n_\text{case}) = 43{,}662 / (5 \cdot 148) \approx 59$. Calibration, however, is NOT preserved — probability outputs must be prevalence-adjusted before use. That requirement is a separate condensate: [[condensates/downsample-requires-prevalence-adjustment]].

## Actionable rule

- The V0 gate locks `train_control_per_case` from {1.0, 2.0, 5.0} based on measured AUROC equivalence. 5.0 is the ADR-003 default but the factorial should test all three.
- Downsampling MUST happen after the stratified split allocation, never before. Hook-enforced by `downsample_controls` operating within per-split dataframes.
- If a downstream calibration step is planned, the prevalence-adjustment equation must be applied before calibration fits on VAL. See [[condensates/downsample-requires-prevalence-adjustment]].
- Ratios below 2:1 are NOT permitted on the celiac cohort — the factorial grid explicitly bounds the axis at {1, 2, 5}.

## Boundary conditions

- Applies when population prevalence is below ~1% AND the primary metric is discrimination (AUROC/AUPRC on fixed operating points). For prevalent diseases or calibration-primary evaluations, the downsampling trade-off flips.
- Does NOT apply when the negative-class feature distribution is multimodal with rare but important modes — random downsampling may drop them entirely. Cluster-aware downsampling is the alternative in that regime.
- Does NOT apply to tree ensembles that internally subsample (some RF/XGBoost configurations already subsample controls per tree). Downsampling before the tree does not stack multiplicatively with internal subsampling.
- SMOTE-style oversampling alternatives were considered and rejected in ADR-003 because synthetic proteomics features may not preserve the joint biomarker structure. This boundary is worth re-testing if a future cohort has denser case counts.

## Evidence

| Dataset | n_case | n_control (raw) | n_control (1:5) | AUROC delta observed | Source gate |
|---|---|---|---|---|---|
| Celiac (UKBB) | 148 | 43,662 | ~740 | ADR-003 reports "preserves adequate negative signal" — no head-to-head AUROC measurement at 1:5 vs 1:N documented in the ADR | v0-strategy 2026-01-20 |

## Evidence gaps

ADR-003 asserts the compute savings arithmetic (60× speedup — verifiable) and claims discrimination is preserved, but does NOT report the AUROC delta from a matched run at `train_control_per_case=1.0` vs `5.0`. This is a candidate V0 gate measurement: three cells ($r \in \{1, 2, 5\}$) × same recipe × held-out TEST gives the empirical Equivalence check the falsifier requires. Flag as "compute claim verified, discrimination-preservation claim asserted but not measured on this cohort."

## Related

- [[equations/case-control-ratio-downsampling]] — math of the ratio and the prevalence shift
- [[condensates/downsample-requires-prevalence-adjustment]] — required follow-up for probability outputs
- [[condensates/three-way-split-prevents-threshold-leakage]] — downsampling operates within the split structure this rule defines
- [[condensates/prevalent-restricted-to-train]] — downsampling is applied after prevalent injection
<!-- TODO: verify slugs exist after batch 2/3 merge -->
- ADR-003 (cel-risk, 2026-01-20) — source decision record
