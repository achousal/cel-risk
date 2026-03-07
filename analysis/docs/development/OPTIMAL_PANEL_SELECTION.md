# Optimal Panel Selection: Deterministic Stopping Rules

**Date**: 2026-03-07
**Status**: Code complete. HPC experiment ready to run.
**Goal**: Replace intuitive panel size selection with statistical decision rules

---

## Problem Statement

Current `find_recommended_panels` produces 4 candidates (95%/90%/85% of max AUROC + knee point).
The jump from these heuristics to "pick this panel for validation" requires human judgment.
We need a deterministic, pre-specified decision procedure that:

1. Selects the smallest panel statistically non-inferior to the full model
2. Validates stability across split seeds
3. Confirms every protein is essential (not a passenger)

---

## Pre-Specified Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **delta (primary)** | 0.02 AUROC | Triage before cheap confirmatory (anti-tTG) |
| **delta (sensitivity)** | 0.01 AUROC | Conservative bound for publication claim |
| **alpha** | 0.05 (one-sided) | Standard non-inferiority testing |
| **n_bootstrap** | 2000 | Adequate for percentile CI at n_pos=148 |
| **cross-seed CV threshold** | 0.05 | Max coefficient of variation across seeds |
| **essentiality gate** | Drop-column 95% CI excludes 0 | Every protein must contribute |
| **bootstrap seed** | 42 | Reproducibility |

---

## Three-Rule Decision Procedure

### Rule 1: Bootstrap Non-Inferiority Test

For each panel size k on the Pareto curve:

```
H0: AUROC(full) - AUROC(k) > delta   (inferior)
H1: AUROC(full) - AUROC(k) <= delta  (non-inferior)
```

Paired bootstrap resampling. Reject H0 if the upper (1-alpha) percentile
of the bootstrap difference distribution <= delta. The smallest k where
H0 is rejected is the statistical minimum.

Multi-seed extension pools bootstrap replicates across seeds
(n_bootstrap/n_seeds per seed) to capture both within-seed and cross-seed
variability.

### Rule 2: Cross-Seed Stability Gate

Panel size k is eligible only if:

```
CV(AUROC_k) = std(AUROC_k across seeds) / mean(AUROC_k across seeds) < 0.05
```

Filters out sizes that pass Rule 1 on average but are unstable across
data partitions.

### Rule 3: Marginal Contribution Test (Essentiality)

For each protein in the selected panel:
- Compute delta_AUROC via drop-column across seeds
- 95% CI lower bound must exclude 0

Proteins failing this gate are passengers. Remove them and re-evaluate.

### Decision Flowchart

```
For k in [smallest ... largest] on Pareto curve:
  1. Cross-seed CV < 0.05?       NO  -> skip, try next larger
  2. Non-inferior (delta)?        NO  -> skip, try next larger
  3. All proteins essential?      NO  -> remove passengers, re-check
                                  YES -> ACCEPT k
```

---

## HPC Experiment Plan

Discovery seeds: 100-104. Validation seeds: 105-109.
All jobs on Minerva LSF (`acc_Chipuk_Laboratory`, `premium` queue, 12 cores, 96 GB).

### Step 0: Generate Splits

```bash
ced save-splits
```

### Step 1: Full Training

```bash
ced run-pipeline \
  --config configs/pipeline_hpc.yaml \
  --split-seeds 100,101,102,103,104 \
  --hpc
```

4 models (LR_EN, LinSVM_cal, RF, XGBoost) x 5 seeds + ensemble + consensus.

### Step 2: Permutation Testing

```bash
ced permutation-test --run-id <RUN_ID> --model LR_EN     --n-perms 200 --n-jobs 12 --hpc
ced permutation-test --run-id <RUN_ID> --model LinSVM_cal --n-perms 200 --n-jobs 12 --hpc
ced permutation-test --run-id <RUN_ID> --model RF         --n-perms 200 --n-jobs 12 --hpc
ced permutation-test --run-id <RUN_ID> --model XGBoost    --n-perms 200 --n-jobs 12 --hpc
```

### Step 3: Panel Optimization

RFE + essentiality + optimal panel selection (all integrated).

```bash
ced optimize-panel \
  --run-id <RUN_ID> \
  --require-significance \
  --step-strategy fine \
  --min-size 3 \
  --hpc
```

Outputs per model in `<MODEL>/aggregated/optimize_panel/`:
- `optimal_panel.txt` -- selected panel
- `optimal_panel_selection.json` -- decision audit trail
- `panel_curve_annotated.png` -- Pareto curve with non-inferiority annotations

### Step 4: Consensus Panel

```bash
ced consensus-panel --run-id <RUN_ID>
```

### Step 5: Validation Run (Fresh Seeds)

```bash
# Per-model optimal panel
ced run-pipeline \
  --config configs/pipeline_hpc.yaml \
  --split-seeds 105,106,107,108,109 \
  --fixed-panel results/run_<RUN_ID>/<BEST_MODEL>/aggregated/optimize_panel/optimal_panel.txt \
  --hpc

# Or consensus panel
ced run-pipeline \
  --config configs/pipeline_hpc.yaml \
  --split-seeds 105,106,107,108,109 \
  --fixed-panel results/run_<RUN_ID>/consensus/final_panel.txt \
  --hpc
```

### Step 6: Confirmatory Non-Inferiority

```bash
ced aggregate-splits --run-id <VALIDATION_RUN_ID>
```

Compare validation AUROC against discovery full-model AUROC.

**Success criteria**:
- Validation AUROC within delta=0.02 of discovery full-model AUROC
- Cross-seed CV < 0.05 on validation seeds
- No new passenger proteins

---

## Monitoring (Minerva LSF)

```bash
bjobs -w                              # all jobs
bjobs -w | grep opt_panel             # panel optimization jobs
bhist -l <JOB_ID>                     # job history
tail -f logs/hpc/<JOB_NAME>.log       # live log
bkill <JOB_ID>                        # kill job
```

---

## References

- Wellek (2010). Testing Statistical Hypotheses of Equivalence and Noninferiority. Chapman & Hall.
- DeLong, DeLong, Clarke-Pearson (1988). Comparing AUCs of correlated ROC curves. Biometrics.

---

**Last Updated**: 2026-03-07
**Owner**: Andres Chousal
