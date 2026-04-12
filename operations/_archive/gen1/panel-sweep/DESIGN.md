# Panel Saturation Sweep: Experimental Design

**Question:** What is the optimal protein panel size for each model class, and can this be determined non-arbitrarily?

**Context:** Adding 3 proteins (cxcl9, cd160, muc2) to the BH-significant 4-protein panel (tgm2, cpa2, itgb7, gip) increases AUROC from 0.78-0.80 to 0.87-0.88 -- a massive gain. Tree models benefit more (+0.09 AUROC) than linear models (+0.07 AUROC), and the model rank ordering inverts: linear wins at p=4, trees win at p=7. We extend this analysis across the full RRA-ranked protein list (p=4..25) with multiple addition orders.

**Vault grounding:**
- [[a sparse 30-protein panel captures approximately 90% of the NRI improvement conferred by all 2923 proteins for IHD prediction]]: Signal concentrates in sparse subsets; diminishing returns beyond ~30 proteins in IHD.
- [[hypothesis-driven protein selection improves cardiovascular risk prediction beyond TRFs but random protein selection of equal panel size does not]]: Selection order matters -- biology-informed > random at equal panel size.
- [[learning curves support three supervised ml decisions - data acquisition, early stopping, and model selection]]: Learning curve theory directly applies to feature-count curves, not just sample-count curves.

---

## I. Mathematical Framework

### 1. AUROC as a Function of Panel Size: A(p)

For model class M ∈ {LR, SVM, RF, XGBoost} with panel size p:

```
A_M(p) = A_M(∞) - α_M · exp(-β_M · p) + ε(p)
```

Where:
- `A_M(∞)` = asymptotic AUROC (model-specific ceiling)
- `α_M` = total gain from 0 to ceiling
- `β_M` = saturation rate (higher = faster saturation)
- `ε(p)` = noise term (higher at small p due to variance)

**Prediction:** β_linear > β_tree (linear models saturate faster because they can only use additive signal). But A_tree(∞) ≥ A_linear(∞) (trees can additionally capture interactions).

### 2. Bias-Variance Decomposition vs. Panel Size

```
E[Error_M(p)] = Bias²_M(p) + Variance_M(p) + σ²
```

**Linear models (LR, SVM):**
- Bias²(p) = Σ_{interactions not captured} → decreases slowly with p (interactions still missed)
- Variance(p) ≈ p/n → increases linearly, but negligible when n >> p (n=43K)
- Net: A(p) driven almost entirely by bias reduction from main effects

**Tree ensembles (RF, XGBoost):**
- Bias²(p) ≈ 0 for sufficient depth (universal approximator)
- Variance(p) = ρ(p)·σ² + (1-ρ(p))·σ²/B, where ρ(p) = tree pairwise correlation
- ρ(p) decreases with p: more features → more candidate splits → more diverse trees
- Net: A(p) driven by variance reduction through tree decorrelation

**The crossover:** At small p, linear wins because trees can't decorrelate. As p grows, tree variance drops while linear bias remains (can't capture interactions). The crossover point p* satisfies:

```
Bias²_linear(p*) = Variance_tree(p*) - Variance_linear(p*)
```

### 3. Marginal Gain from Protein k+1

Define the marginal AUROC gain from adding the k+1-th protein:

```
Δ_M(k) = A_M(k+1) - A_M(k)
```

Under the exponential saturation model:
```
Δ_M(k) = α_M · β_M · exp(-β_M · k)
```

This decays exponentially. The "stopping point" p* for model M is where Δ_M(p*) falls below a significance threshold δ.

### 4. Interaction Order and Model Capacity

The total signal at panel size p decomposes into:

```
Signal(p) = Σ_{j=1..p} main_effect(j)        [order-1: captured by all models]
          + Σ_{j<k} interaction(j,k)          [order-2: captured by trees, not linear]
          + Σ_{j<k<l} interaction(j,k,l)      [order-3: captured by deep trees]
          + ...
```

- Linear models capture: order-1 only → Signal_linear(p) = Σ main_effects
- RF/XGBoost capture: order-1 + order-2 + ... → Signal_tree(p) ≥ Signal_linear(p)
- The gap widens with p because C(p,2) interaction terms grow quadratically

**Implication:** If the 3 extra proteins (cxcl9, cd160, muc2) participate in interactions with the core 4, trees benefit disproportionately. This is testable via SHAP interaction values.

### 5. Addition Order Effects

For proteins ranked r₁, r₂, ..., r_p, the order matters if:

```
A({r₁,...,r_k, r_{k+1}}) ≠ A({r₁,...,r_k, r_j}) for j ≠ k+1
```

Two conditions where order matters:
1. **Redundancy:** If r_{k+1} is correlated with {r₁,...,r_k}, adding it gives less gain than adding an orthogonal protein
2. **Interaction dependence:** If r_{k+1} participates in interactions with proteins already in the set, its value depends on what's already included

**Order is irrelevant** (approximately) when:
- Features are independent (no redundancy)
- The signal is purely additive (no interactions)

For our data, cxcl9 and cd160 show moderate correlation with the core proteins and strong interaction effects in trees → order likely matters.

### 6. Statistical Testing Framework

**Primary test:** Paired permutation test on AUROC difference across 10 holdout seeds:
```
H₀: E[A_M(k+1) - A_M(k)] ≤ 0
H₁: E[A_M(k+1) - A_M(k)] > 0
```

One-sided paired t-test or Wilcoxon signed-rank across 10 seeds. With 10 paired observations, we can detect effect sizes > ~0.01 AUROC at α=0.05 (based on observed σ ≈ 0.03 within-seed).

**Multiple testing correction:** 21 sequential tests (k=4..24). Options:
- **Hochberg step-up:** most powerful, valid under independence
- **Alpha spending (O'Brien-Fleming):** spends more alpha at larger panels where we expect smaller gains → better power where it matters
- **Closure principle:** test intersection hypotheses for formal strong FWER control

**Model selection criterion (non-testing alternative):**
```
BIC_panel(p) = -2 · log L(A_M(p)) + log(n) · df_M(p)
```
Where df_M(p) is the effective degrees of freedom of model M at panel size p.

---

## II. Hypotheses (Co-Scientist Perspectives)

### Perspective A: The Statistician
**H_stat:** The optimal panel size is 7 for all models. The 4→7 jump captures the three strongest sub-BH proteins (cxcl9, cd160, muc2, all with raw p < 0.002). Beyond 7, proteins have raw p > 0.005 and the marginal AUROC gain will be non-significant. The BH cutoff at N=126 was accidentally correct, not because of the correction itself, but because it happened to coincide with the natural signal boundary.

**Prediction:** Saturation curve shows a knee at p≈7 for all models. Delta(k) drops below significance for k≥8.

### Perspective B: The ML Engineer
**H_ml:** The optimal panel is model-specific. Linear models saturate at p=5-7 (all main effects captured). Trees keep gaining until p=12-15 because they exploit pairwise interactions among the immune/intestinal pathway proteins. The rank inversion (linear→tree) happens at p≈5-6, not p≈7.

**Prediction:** Two distinct saturation curves with different knees. The crossover point p* is between 5 and 6.

### Perspective C: The Biologist
**H_bio:** The signal is pathway-structured. The first 7 proteins (tgm2, cpa2, itgb7, gip, cxcl9, cd160, muc2) cover 3 biological axes: mucosal integrity (tgm2, muc2), immune surveillance (itgb7, cxcl9, cd160), and metabolic (cpa2, gip). Proteins 8-25 are downstream effectors or correlated bystanders -- they add noise, not new biological axes. Performance will plateau or degrade beyond p≈10.

**Prediction:** AUROC improvement from p=8-25 is flat or negative, with possible degradation from collinearity.

### Perspective D: The Information Theorist
**H_info:** The useful information is in the top ~10-12 proteins. Beyond that, the incremental mutual information I(Y; X_{k+1} | X_{1..k}) → 0. But tree models can extract a tiny bit more because they can exploit conditional dependencies. The "optimal" panel size depends on your loss function: if you're maximizing AUROC, ~10 proteins; if you're maximizing calibration (Brier), ~7 proteins (excess features degrade probability estimates more than discrimination).

**Prediction:** AUROC and Brier saturation curves have different knees. Brier saturates earlier than AUROC.

### Perspective E: The Experimentalist
**H_exp:** Addition order will matter more than expected. The "importance-ranked" order (where cxcl9 enters before gip) will produce a steeper saturation curve than the "RRA-ranked" order (where gip enters at position 4). This is because gip is partially redundant with cpa2 (both in GI metabolic pathway), while cxcl9 provides orthogonal immune signal. Swapping positions 4 and 5 (gip ↔ cxcl9) may shift AUROC at p=4 by 1-2 percentage points.

**Prediction:** Importance-order saturation curve dominates RRA-order at p=4-6, converges by p=7.

---

## III. Visualization Suite

### Plot 1: Saturation Curves (Primary)
- X-axis: panel size p (4..25)
- Y-axis: AUROC (pooled across 10 seeds)
- Lines: one per model class + ENSEMBLE
- Shading: 95% CI from 10 seeds
- Annotations: BH cutoffs (vertical lines at p=4, p=7), crossover point p*
- **Subpanels:** same layout for PR-AUC, Brier, Sensitivity@95%Spec

### Plot 2: Marginal Gain (Delta Curves)
- X-axis: protein added (labeled by name, in RRA order)
- Y-axis: Δ AUROC from adding that protein
- Bars: one cluster per model, colored by model class
- Horizontal line: significance threshold (based on paired test power)
- **Purpose:** Which proteins matter, and to which models?

### Plot 3: Model Rank at Each Panel Size
- X-axis: panel size p
- Y-axis: model rank (1=best, 4=worst)
- Lines: one per model, showing rank trajectory
- **Purpose:** Visualize the rank inversion directly

### Plot 4: Bias-Variance Decomposition
- X-axis: panel size p
- Y-axis: train AUROC - test AUROC (proxy for variance)
- Lines: one per model
- **Purpose:** Confirm that tree overfitting decreases with p while linear gap stays constant

### Plot 5: Addition Order Comparison
- X-axis: panel size p
- Y-axis: AUROC
- Lines: one per addition order (RRA rank, importance rank, biology-pathway rank)
- Fixed model: run for best linear (LinSVM) and best tree (XGBoost)
- **Purpose:** Quantify how much addition order matters

### Plot 6: Interaction Heatmap (SHAP-based)
- Matrix: protein × protein SHAP interaction values
- One panel per model at p=7 (or at saturation point)
- **Purpose:** Identify which protein pairs create the interaction signal that drives tree advantage

### Plot 7: Metric Divergence
- X-axis: panel size p
- Y-axes (dual): AUROC (left), Brier (right)
- Lines: one per model
- **Purpose:** Test H_info prediction that AUROC and Brier have different saturation points

---

## IV. Experimental Protocol

### Addition Orders to Test

**Order 1: RRA rank (statistical)**
tgm2 → cpa2 → itgb7 → gip → cxcl9 → cd160 → muc2 → nos2 → fabp6 → agr2 → reg3a → mln → ccl25 → pafah1b3 → tnfrsf8 → tigit → cxcl11 → ckmt1a_ckmt1b → acy3 → hla_a → xcl1 → nell2 → pof1b → ppp1r14d → ada2

**Order 2: Cross-model importance rank**
Rank proteins by mean importance across all 4 models (from 7p run), then extend by RRA rank for proteins 8-25.

tgm2 → cpa2 → itgb7 → cxcl9 → cd160 → muc2 → gip → [8-25 by RRA]

**Order 3: Pathway-informed**
Group by biological axis, add one axis at a time:
- Core mucosal: tgm2, muc2
- Immune surveillance: itgb7, cxcl9, cd160
- Metabolic: cpa2, gip
- Extended immune: nos2, cxcl11, tigit, tnfrsf8, ...
- Extended GI: fabp6, agr2, reg3a, mln, ccl25, ...

### Per-Order Protocol

For each order, for each panel size p ∈ {4, 5, ..., 25}:
1. Create fixed_panel CSV with top-p proteins in that order
2. Run full holdout pipeline (10 seeds × 4 models + ENSEMBLE)
3. Collect: AUROC, PR-AUC, Brier, Sensitivity@95%Spec, train-test gap

### Statistical Analysis

1. **Fit exponential saturation model** A_M(p) = A_M(∞) - α·exp(-β·p) per model
2. **Paired sequential tests** for each Δ_M(k), with Hochberg correction
3. **Compute crossover point** p* where tree AUROC overtakes linear AUROC
4. **Compare addition orders** via paired AUROC at each p

### Stopping Rules (Non-Arbitrary Panel Selection)

Apply three independent criteria; the panel size where 2/3 agree is the recommendation:

1. **Statistical:** Last p where Δ_M(p) is significant after Hochberg correction
2. **Saturation model:** p where A_M(p) reaches 95% of fitted A_M(∞)
3. **Parsimony (1-SE rule):** Smallest p within 1 SE of the maximum AUROC

### Computational Budget

- 22 panel sizes × 3 orders × 10 seeds × 4 models = 2,640 training runs
- Existing: 2 panel sizes × 1 order × 10 seeds × 4 models = 80 runs (reusable for Order 1)
- New: 2,560 training runs
- At ~50 Optuna trials × ~1 min/trial = ~50 min/run → ~2,133 CPU-hours
- On Minerva (96 cores, free CPU): ~22 wall-hours if fully parallelized
- Add ensemble: +660 runs (lightweight, ~5 min each → +55 CPU-hours)

---

## V. Decision Matrix

After the sweep, the panel recommendation depends on the deployment goal:

| Goal | Decision Rule | Expected Panel Size |
|------|--------------|-------------------|
| Minimum viable (cost, simplicity) | Smallest p passing all 3 stopping rules for LR | 4-7 |
| Maximum discrimination | p at peak AUROC for best ensemble | 10-15 |
| Robust (model-agnostic) | p where all models are within 1% of their peaks | 7-10 |
| Calibration-optimal | p at minimum Brier for best model | 5-8 |

---

## VI. What This Experiment Answers

1. **Is 7 special, or is it an artifact of BH at N=126?**
2. **Do trees and linear models have genuinely different optimal panel sizes?**
3. **Does addition order matter, and by how much?**
4. **Where does each model saturate?**
5. **Is there a single "best" panel size, or must it be model-specific?**
6. **Do excess proteins hurt (overfitting) or just stop helping (plateau)?**

## References

- Bourgon et al. 2010 PNAS 107:9546 (independent filtering)
- Zehetmayer & Posch 2012 BMC Bioinformatics 13:81 (two-stage FDR)
- Meinshausen & Bühlmann 2010 JRSS-B (stability selection)
- Barber & Candès 2015 Annals of Statistics (knockoff filter)
- Ignatiadis et al. 2016 Nature Methods (IHW)
- Mohr & van Rijn 2022 (learning curves for model selection)
