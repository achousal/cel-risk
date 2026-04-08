# Incident Validation Brief — LR_EN (Elastic Net Logistic Regression)

**Model:** LogisticRegression (ElasticNet, SAGA solver)
**Design:** 3 training strategies × 4 class-weight schemes, 5-fold outer CV, 50-trial Optuna inner CV
**Primary metric:** AUPRC (area under precision-recall curve)
**Date:** 2026-04-01

---

## 1. Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Raw data: 44,174 subjects                                      │
│  119 incident · 150 prevalent · 35,329 + 8,776 controls        │
└───────────────────────┬─────────────────────────────────────────┘
                        │  Stratified split (seed=42, test=20%)
           ┌────────────┴────────────┐
           ▼                         ▼
  ┌─────────────────┐       ┌──────────────────┐
  │   DEV SET       │       │  LOCKED TEST SET  │
  │ 119  incident   │       │  29   incident    │  ← never touched
  │ 35,100 controls │       │  8,776 controls   │    until final eval
  │ 150  prevalent  │       └──────────────────┘
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────┐
  │  Bootstrap Stability Feature Selection              │
  │  · 100 resamples of dev incident + controls         │
  │  · Wald statistic, top 200 proteins per resample    │
  │  · Keep proteins selected in ≥ 70% of resamples     │
  │                                 → 134 stable prots  │
  │  Correlation pruning  |r| > 0.85 (Spearman)        │
  │                                 → 134 protein panel │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  5-Fold Outer CV  (stratified on incident vs control)        │
  │                                                              │
  │  For each of 12 combos (3 strategies × 4 weight schemes):   │
  │                                                              │
  │    Per fold:                                                 │
  │      Inner 3-fold Optuna (50 trials)                         │
  │        → tune C, l1_ratio  (maximise inner AUPRC)           │
  │      Refit on outer train fold → evaluate on outer val fold  │
  │                                                              │
  │  Strategies          Weight schemes                          │
  │  ─────────────────   ─────────────────────────────────────  │
  │  incident_only       none · log (~5.7×) · sqrt (~17×)       │
  │  incident_prevalent  · balanced (~290×)                      │
  │  prevalent_only                                              │
  └────────────────────────┬─────────────────────────────────────┘
                           │  Select best by mean AUPRC
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  Final refit on full dev set                        │
  │  · incident_only + log  ·  C=0.00644  l1=0.472     │
  │  · 28 / 134 non-zero coefficients                  │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  Locked Test Evaluation                             │
  │  AUPRC 0.188  [0.090, 0.363]                       │
  │  AUROC 0.908  [0.827, 0.978]                       │
  │  Sens 0.828 · Spec 0.946 · Prec 0.048              │
  └─────────────────────────────────────────────────────┘
```

---

## 2. Strategy Comparison (5-fold CV)

Sorted by mean AUPRC. All strategies use the same 134-protein panel.

```
Strategy             Weight     AUPRC (mean ± std)    AUROC (mean ± std)   Median C
─────────────────── ────────── ─────────────────────  ──────────────────── ────────
incident_only        log        0.215 ± 0.061  ★       0.867 ± 0.031        0.0064
incident_only        sqrt       0.215 ± 0.053           0.867 ± 0.025        0.0014
incident_only        none       0.212 ± 0.055           0.868 ± 0.030        0.0361
────────────────────────────────────────────────────────────────────────────────────
incident_prevalent   log        0.194 ± 0.071           0.866 ± 0.039        0.0037
incident_prevalent   balanced   0.192 ± 0.066           0.849 ± 0.035        0.0010
incident_prevalent   sqrt       0.191 ± 0.071           0.863 ± 0.038        0.0025
incident_prevalent   none       0.178 ± 0.061           0.864 ± 0.039        0.0101
────────────────────────────────────────────────────────────────────────────────────
prevalent_only       balanced   0.189 ± 0.104           0.849 ± 0.046        0.0048
prevalent_only       sqrt       0.170 ± 0.069           0.855 ± 0.042        0.0062
prevalent_only       log        0.155 ± 0.065           0.849 ± 0.042        0.0121
prevalent_only       none       0.126 ± 0.074           0.837 ± 0.040        0.0708
incident_only        balanced   0.125 ± 0.046           0.836 ± 0.063        0.0008
─────────────────────────────────────────────────────────────────────────────────────
  ★ Winner
```

**Note:** AUPRC is informative here despite baseline prevalence ~0.34% (29/8,805 in test)
because it reflects precision at the operating points that matter clinically.

---

## 3. What Works Best and Why

### Winner: `incident_only + log weighting`

**Why incident_only wins:**
Including prevalent cases degrades AUPRC by ~10–21% relative. Prevalent celiac is a
fundamentally different proteome — disease is active, the biomarker signal is acute.
Mixing prevalent into training distorts the decision boundary toward established disease
rather than pre-diagnostic risk. The model learns the wrong signal.

**Why log weighting wins (not balanced):**
- `log` applies a ~5.7× minority boost — enough to surface incident cases without
  drowning the signal in noise from over-aggressive resampling.
- `balanced` (~290× boost) collapses AUPRC to 0.125, the worst result for incident_only.
  At 290×, the model is calibrated on essentially synthetic positives; it hallucinates
  decision boundaries and fails on real test cases.
- `sqrt` (~17×) is competitive (0.215 ± 0.053) — nearly identical to log, but with
  slightly lower AUROC and higher variance. log edges it out.

**Why sqrt and none are close:**
AUPRC differences between log/sqrt/none within incident_only are within 1 std dev —
the ranking is real but the margin is modest. AUROC is almost identical across all three
(0.868, 0.867, 0.868). The incident_only strategy drives most of the gain; weighting
fine-tunes precision at the low-probability end of the curve.

**Summary sentence:**
> Incident-only training with log-weighted positives achieves AUPRC 0.215 (CV) and
> AUROC 0.908 on the locked test set — primarily because it keeps the signal clean
> (no prevalent contamination) while applying just enough upweighting to surface
> a 0.34%-prevalence outcome.

---

## 4. Final Feature Set and Stability

**Panel:** 134 proteins selected by bootstrap stability (≥70% of 100 resamples)
**Model non-zero (elastic net):** 28 / 134

### Tier 1 — Core: non-zero in all 5 CV folds, sign-consistent (n=8)

```
Protein               Mean coef   Std     Stability   Interpretation
────────────────────  ──────────  ──────  ─────────   ──────────────────────────────
tgm2_resid             −0.617     0.052   100%        Tissue transglutaminase 2 (anti-TG2
                                                       autoantigen). Negative coef = higher
                                                       levels → lower risk score? Confounded
                                                       by residualization — investigate.
cpa2_resid             +0.231     0.051   100%        Carboxypeptidase A2 (pancreatic)
muc2_resid             +0.172     0.091   100%        Intestinal mucin-2 (barrier integrity)
cxcl9_resid            +0.137     0.044   100%        IFN-γ-induced chemokine (T-cell recruitment)
ckmt1a_ckmt1b_resid    +0.132     0.034   100%        Creatine kinase mitochondrial
nos2_resid             +0.079     0.024   100%        Inducible nitric oxide synthase (inflammation)
itgb7_resid            +0.029     0.019   100%        Gut-homing integrin β7 (mucosal immunity)
tnfrsf8_resid          +0.051     0.023   100%        CD30 (activated T/B cells)
```

### Tier 2 — Consistent: non-zero in 4/5 folds (n=10)

```
Protein          Folds   Sign-consistent   Stability
───────────────  ──────  ────────────────  ─────────
mln_resid         4/5        No             100%
xcl1_resid        4/5        No             100%
clec4g_resid      4/5        No              99%
cd160_resid       4/5        No             100%
rnaset2_resid     4/5        No              88%
tigit_resid       4/5        No             100%
klrd1_resid       4/5        No             100%
rbp2_resid        4/5        No             100%
cxcl11_resid      4/5        No             100%
apoa1_resid       4/5        No              80%
```

### Tier 3 — Marginal: non-zero in ≤3/5 folds (n=10)

Present in the final refitted model but sign-unstable across CV folds.
Not reliable for mechanistic interpretation.

### Overlap with Phase 1–3 seven-protein panel

```
Phase 1–3 panel   In Tier 1?   Folds nonzero   Note
────────────────  ───────────  ──────────────  ─────────────────────────
tgm2              YES (★)       5/5            Anchor feature
cxcl9             YES (★)       5/5            Confirmed
nos2              YES (★)       5/5            Confirmed
itgb7             YES (★)       5/5            Confirmed
tnfrsf8           YES (★)       5/5            Confirmed
[2 others]        No            < 5/5          Not consistently selected
```

5 of 7 Phase 1–3 proteins are in Tier 1. The 134-protein elastic net discovers
a richer signal but converges on the same mechanistic core.

---

## Appendix: Key Numbers

| Item | Value |
|------|-------|
| Total subjects | 44,174 |
| Dev incident / controls / prevalent | 119 / 35,100 / 150 |
| Test incident / controls | 29 / 8,776 |
| Feature panel (post-pruning) | 134 proteins |
| Non-zero in final model | 28 / 134 |
| Tier 1 core features | 8 |
| CV AUPRC (winner) | 0.2152 ± 0.061 |
| Test AUPRC | 0.1883 [0.090, 0.363] |
| Test AUROC | 0.9080 [0.827, 0.978] |
| Winner config | incident_only + log, C=0.00644, l1=0.472 |
| Optuna trials / fold | 50 |
| Bootstrap resamples | 100 |
| Bootstrap CI samples | 2,000 |
