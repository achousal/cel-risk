---
type: condensate
depends_on:
  - "[[equations/nested-cv]]"
applies_to:
  - "any pipeline with data-dependent hyperparameter selection"
  - "small-n or rare-event settings where a single tuning split is too volatile"
  - "pipelines whose reported OOF metric feeds downstream decisions (model gate, panel selection, deployment threshold)"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "ADR-005 mandates 5×10 outer × 5 inner = 50k fits per model precisely to prevent optimism bias; single-loop CV with the same tuning budget would overstate OOF AUROC"
    date: "2026-01-20"
    source: "ADR-005 (nested CV), citing Varma & Simon 2006"
falsifier: |
  Equivalence claim: OOF AUROC from single-loop CV (tune and evaluate on same
  folds, same trial budget) vs. nested CV stays within |ΔAUROC| < 0.01 with
  95% bootstrap CI inside [−0.02, 0.02] across >=3 datasets with p > n and
  rare-event prevalence (π < 0.01). If this holds, the optimism-bias mandate
  is weakened. Retire the nested-CV requirement at 5 such datasets.
  Direction claim (expected): single-loop CV produces higher OOF AUROC
  (|Δ| >= 0.02, 95% CI excludes 0) — this is the predicted failure mode.
---

# Nested cross-validation is required to produce an unbiased OOF performance estimate when hyperparameters are tuned on the data

## Claim

When a pipeline uses the same cross-validation folds to both select hyperparameters and report a held-out metric, the reported metric is upward-biased by an amount that scales with tuning freedom (Varma & Simon 2006). Nested CV — outer folds for evaluation, inner folds for tuning, strict isolation between them — is the standard remedy. The celiac pipeline adopts $K_\text{out} = 5$, $R = 10$ repeats, $K_\text{in} = 5$ inner folds, $T = 200$ trials because (a) repeats stabilize the outer OOF estimate at rare-event prevalence, and (b) 5 inner folds preserve sufficient positives per inner validation partition for hyperparameter tuning to be stable rather than noisy.

## Mechanism

See [[equations/nested-cv]]. Tuning on the evaluation folds lets the hyperparameter search exploit noise in those folds: the "best" hyperparameters are those that overfit the specific held-out samples. Reporting AUROC on those same folds then conflates genuine generalization with that exploitation. The bias is bounded above by the tuning freedom (search space volume and trial budget) — a one-hyperparameter search with 10 trials has small bias; a 200-trial search over a 10-dimensional space has large bias.

Nested CV isolates the two roles. Hyperparameter choice for outer fold $i$ depends only on outer fold $i$'s training rows, never on its held-out rows. The outer held-out AUROC is then an unbiased estimate of deployment performance under the same data distribution.

Repeats ($R$) are a variance-reduction mechanism, not a bias-correction mechanism. A single 5-fold outer loop still produces unbiased point estimates but with high variance at rare-event prevalence; averaging across $R$ repeats tightens the CI around that unbiased point.

## Actionable rule

- Any gate that reports OOF AUROC, PR-AUC, or Brier MUST use nested CV if hyperparameters are tuned on the same cohort.
- Hyperparameters tuned inside outer fold $i$ MUST NOT be reused in outer fold $j$. Each outer fold independently re-tunes.
- The inner loop is the correct home for Optuna (see [[equations/optuna-tpe]]). Placing Optuna at the outer level violates isolation.
- The full nested structure — including inner tuning — re-runs under every permutation when computing [[equations/perm-test-pvalue]]. See [[condensates/perm-validity-full-pipeline]].
- Inner fold count $K_\text{in}$ may be reduced dynamically if positives per inner fold would fall below 2 (ADR-005 safeguard for calibration CV); this is a documented exception logged at gate entry, not a silent fallback.

## Boundary conditions

- **Does not apply to pre-registered fixed hyperparameters.** If hyperparameters are declared ex ante with no on-data search, single-loop CV produces unbiased estimates. ADR-005 applies when tuning happens.
- **Rare-event prevalence demands positive floor.** At $\pi < 0.01$, inner fold count cannot be arbitrary. Default $K_\text{in} = 5$ at celiac prevalence yields ~24 positives per inner validation fold; reducing inner folds to 3 or below risks zero-positive folds that crash or bias tuning.
- **Tuning budget must match search complexity.** $T = 200$ for 5-10 dimensional mixed spaces is the celiac default; simpler spaces (LR_EN: 2 hyperparameters) tolerate $T = 50$; more complex spaces (XGBoost: 10+ hyperparameters) may require $T = 400$.
- **Computational ceiling.** $F_\text{total} = K_\text{out} \cdot R \cdot K_\text{in} \cdot T = 50{,}000$ fits per model is the celiac baseline; multiplying any factor scales linearly. HPC runtime of 12 hours at the default is load-bearing for protocol design.

## Evidence

| Dataset | n | p | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | 2,920 | ADR-005 Rationale: "Prevents optimistic bias in hyperparameter selection"; 50k-fit budget adopted precisely because single-loop CV with tuning would overstate OOF AUROC | Migrated from ADR-005 2026-01-20 |

## Related

- [[equations/nested-cv]] — the decomposition and unbiased-estimator statement
- [[equations/optuna-tpe]] — what runs inside the inner loop
- [[equations/perm-test-pvalue]] — requires the full nested inner pipeline per permutation
- [[condensates/perm-validity-full-pipeline]] — failure mode when the inner pipeline is not re-run
- ADR-005 (nested CV) — canonical source
- Varma & Simon (2006) — original derivation
