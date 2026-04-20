---
type: condensate
depends_on: ["[[equations/fixed-spec-threshold]]"]
applies_to:
  - "clinical screening applications where false-positive burden is explicit"
  - "low-prevalence cohorts (prevalence < 0.01) where high PPV requires high specificity"
  - "decision rules whose downstream action is confirmatory testing with nontrivial cost or harm"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v5-confirmation
    delta: "ADR-010 locks default threshold objective = fixed-spec at 0.95; Youden and max_F1 are supported as comparators but not as defaults because they can yield specificity < 0.90 on low-prevalence cohorts"
    date: "2026-01-20"
    source: "ADR-010, ADR-009"
falsifier: |
  Dominance claim: fixed-spec-at-0.95 dominates Youden and max_F1 on both
  (a) PPV at deploy-time prevalence AND (b) specificity held at >= 0.90. If
  Youden or max_F1 achieves spec >= 0.90 AND PPV within Equivalence of
  fixed-spec-0.95's PPV (|Delta PPV| < 0.01) on a low-prevalence cohort
  (prevalence < 0.01), the specificity-first default is weakened for that
  cohort. At least 3 such cohorts -> condensate retires in favor of a
  data-driven selection among {fixed-spec, Youden, max_F1}.
---

# Fixed-specificity thresholds dominate Youden and max F1 for low-prevalence screening because specificity drives positive predictive value

## Claim

For clinical screening on low-prevalence cohorts, the default threshold objective MUST be fixed-specificity at a clinically-declared target (cel-risk default: 0.95), not Youden's $J$ or max $F_1$. The justification is that at prevalence $\pi \ll 1$, positive predictive value (PPV) is dominated by specificity:

$$\text{PPV} = \frac{\pi \cdot \text{sens}}{\pi \cdot \text{sens} + (1 - \pi)(1 - \text{spec})}$$

so a 1-point drop in specificity (from 0.95 to 0.85) can halve PPV even when sensitivity rises. Youden's $J = \text{sens} + \text{spec} - 1$ weights both equally, which is the wrong tradeoff when downstream false-positive cost is asymmetric with false-negative cost.

## Mechanism

At deploy-time prevalence $\pi = 0.003$ (celiac in UK Biobank), a threshold achieving $\text{sens} = 0.70$ and $\text{spec} = 0.95$ yields PPV $= 0.003 \cdot 0.70 / (0.003 \cdot 0.70 + 0.997 \cdot 0.05) = 0.0406$. The same sensitivity at $\text{spec} = 0.85$ yields PPV $= 0.003 \cdot 0.70 / (0.003 \cdot 0.70 + 0.997 \cdot 0.15) = 0.0138$ — a 3x reduction in PPV from a 10-point specificity loss.

Youden's $J$ is agnostic to prevalence. It returns the threshold that maximizes the distance above the chance-diagonal on the ROC, which corresponds to equal weighting of sensitivity and specificity. On low-prevalence cohorts this systematically selects thresholds with specificity in the 0.80–0.90 range, producing unacceptable false-positive rates when the test is deployed at scale.

Max $F_1$ weights precision and recall equally but does not cap specificity; on low-prevalence cohorts with few positive cases, max $F_1$ thresholds can be even lower than Youden's.

Fixed-spec at 0.95 is a **prior** encoding a clinical judgment: "we accept up to 5% false positives, no more." The judgment is domain-specific (e.g., newborn screening targets 0.99+; screening with benign follow-up accepts 0.90), so the target is a configuration choice, not a universal constant.

## Actionable rule

- `ThresholdConfig.objective` default MUST be `fixed_spec` with `fixed_spec_target` set to the clinically-declared value (0.95 on the celiac cohort).
- Youden and max_F1 MUST be computed and reported as comparators in the V4/V5 ledger. Their operating points are **not** the default deploy-time threshold but serve as sensitivity checks: if Youden's specificity is much lower than the fixed-spec target, flag the gap.
- Threshold selection follows [[equations/fixed-spec-threshold]] on the VAL split — see [[condensates/threshold-on-val-not-test]].
- Pre-registration: the target specificity is declared in the V4 ledger BEFORE observing test-set performance. Moving the target after peeking (e.g., relaxing from 0.95 to 0.90 because sensitivity disappointed) is a search-space violation.

## Boundary conditions

- Applies when prevalence is low ($\pi < 0.01$) AND downstream false-positive cost is nontrivial (extra testing, anxiety, workup). Does NOT apply to:
  - Balanced cohorts ($\pi \approx 0.5$), where Youden's equal-weighting is defensible.
  - Diagnostic settings where false negatives are catastrophic (e.g., ruling out acute myocardial infarction), where high sensitivity trumps specificity. In that regime, the dual rule — fixed-sensitivity at high target — is preferred and this condensate is inverted.
- Requires PPV to be the decision-relevant metric. If the downstream action is continuous risk stratification (no hard threshold) then no threshold is selected and this condensate does not apply.
- Assumes the target-specificity value is clinically justified. A target of 0.95 was chosen for celiac screening in ADR-010; on a different disease, the target is revisited.

## Evidence

| Dataset | prevalence | Phenomenon | Source gate |
|---|---|---|---|
| Celiac (UKBB) | 0.00338 | ADR-010 locks fixed-spec at 0.95 as the default because Youden returns spec ~0.85 on this cohort and max_F1 returns spec ~0.80; both are below the 0.90 clinical floor | V4 threshold selection, V5 confirmation |

## Related

- [[equations/fixed-spec-threshold]] — the threshold-selection math
- [[condensates/threshold-on-val-not-test]] — complements this condensate; addresses WHICH SPLIT the threshold is selected on
- ADR-010 (cel-risk) — decision record this condensate formalizes
- ADR-009 (cel-risk) — threshold-on-VAL dependency
<!-- TODO: verify slug exists after batch merge --> - [[protocols/v5-confirmation]]
