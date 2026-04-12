# Factorial Cell Tuning — Experiment Design

## Objective

Find the optimal recipe (panel composition + ordering) × model × calibration × weighting × downsampling combination before holdout confirmation. Every panel is a generated artifact of declared rules — no hand-picking. The design must generalize: on a different dataset, the same rules produce different (but principled) panels and sizes.

---

## Source Data

Two trunks feed the recipes:

| Trunk | Source | What it captures |
|-------|--------|-----------------|
| **T1** (consensus-operating) | RRA significance across models | Cross-model agreement on which proteins matter |
| **T2** (incident-sparse) | Bootstrap stability + CV sign consistency | Features reliably non-zero in incident validation |

---

## Size Rules

### 3-criterion rule

Operates on the sweep results (`compiled_results_aggregated.csv`, 30-seed averaged AUROC across panel sizes 4–25). For each size p:

1. **C1 — Non-inferiority**: one-sided z-test, H0: gap ≥ δ (0.02). Can we reject that p is meaningfully worse than the best? Strict — accounts for variance.
2. **C2 — Within 1 SE**: AUROC(p) ≥ AUROC(best) − SE(best). Heuristic parsimony rule (Breiman 1-SE tradition). More permissive than C1.
3. **C3 — Marginal gain insignificant**: two-sided z-test on Δ(p→p+1), Holm-corrected. Has adding one more protein stopped helping?

**Criterion** (`three_criterion`): smallest p where ≥2 pass. Catches where parsimony arguments start holding.
**Unanimous** (`three_criterion_unanimous`): smallest p where all 3 pass. Catches the unambiguous performance plateau.

The sweep data is pooled across all orderings (importance, pathway, rra) and optionally filtered to a single model. Pooling makes the size decision ordering-agnostic — the ordering choice is then a separate axis handled by the recipe.

### Per-model size derivation

Model-specific recipes filter the sweep to the pinned model before running the unanimous rule. This captures model architecture capacity:

| Model | Plateau | Behavior |
|-------|---------|----------|
| LR_EN | 4p | Linear — extracts signal from core proteins, more features are noise |
| LinSVM_cal | 4p | Same — linear capacity limit |
| RF | 8p | Trees need feature diversity for ensemble splits |
| XGBoost | 9p | Most feature-hungry — gradient boosting exploits more signal |

### Other size rules

- **`significance_count`**: size = count of proteins passing BH q ≤ 0.05 (=4 on this dataset).
- **`stability`**: size = count of bootstrap-stable + sign-consistent features (=19 on this dataset).

### Pre-registration principle

The 3-criterion rule uses old sweep data (50 Optuna trials) to pick sizes. The factorial re-runs everything under production conditions (200 trials). This is intentional — the rule is a pre-registration step, not the final arbiter. V1 in the validation tree adjudicates.

### Vanillamax discovery principle

Protein-disease associations are biological, not methodological. The discovery phase's job is **sensitivity** (don't miss real signals). Every restriction on data (prevalent handling, control ratio, class weighting, downsampling) is a hypothesis that the factorial should test, not a decision the discovery should bake in.

**Vanillamax discovery settings (ideal for any dataset):**

| Setting | Vanillamax | Rationale |
|---------|-----------|-----------|
| Cases | All available (incident + prevalent, frac=1.0) | Maximum power to detect associations |
| Controls | All available (no downsampling) | Natural class distribution |
| Class weighting | None | No artificial loss rebalancing |
| Proteins | All assayed | No pre-filtering |
| Models | All candidate architectures | No model bias in consensus |
| Significance | Full-universe BH correction | Most conservative |

The factorial then tests which **restrictions** improve the model: V0 (strategy, control ratio), V1 (panel composition), V3 (weighting, downsampling).

**Gen1 caveat (this dataset):** The discovery phase used `prevalent_train_frac: 0.5`, `train_control_per_case: 5`, and log class weights — all deviations from vanillamax. The BH core (4p) is robust to this (survives full-universe correction even with restrictions), but the 8p boundary and model-specific orderings may reflect the restricted settings. V0 and V1 compensate by testing across strategies and recipes, but on a new dataset the discovery sweep should run vanillamax from the start.

---

## Ordering Strategies

### Shared-panel orderings (model-agnostic)

- **`consensus_score_descending`**: rank by RRA consensus score. Optional q-filter (`q_threshold: null` = unfiltered).
- **`stream_balanced`**: correlation clustering (|ρ| ≥ 0.50) → within-stream rank → round-robin interleave. Requires training data (`--data-path`).
- **`abs_coefficient_descending`**: rank by |mean coefficient| from incident validation. T2 trunk only.

### Model-specific orderings (pinned to single model)

- **`oof_importance`**: out-of-fold permutation importance ranking. Top 4 proteins shared across all models; divergence at positions 5+. More stable.
- **`rfe_elimination`**: backward elimination ordering from discovery-phase RFE. Highly model-specific — driven by model's inductive bias (|coef| for linear, permutation importance for trees). Most divergent across models.

---

## Recipes

### Shared-panel recipes (8) — 108 cells each

All 4 models compete on the same panel. Tests ordering and size independently of model choice.

| Recipe | Trunk | Ordering | Size Rule | Derived p |
|--------|-------|----------|-----------|-----------|
| R1_sig | T1 | consensus desc, q≤0.05 | significance_count | 4 |
| R1_criterion | T1 | consensus desc, unfiltered | 3-criterion ≥2 | 5 |
| R1_plateau | T1 | consensus desc, unfiltered | 3-criterion unanimous | 8 |
| R2_sig | T1 | stream-balanced, q≤0.05 | significance_count | 4 |
| R2_criterion | T1 | stream-balanced, unfiltered | 3-criterion ≥2 | 5 |
| R2_plateau | T1 | stream-balanced, unfiltered | 3-criterion unanimous | 8 |
| R3_incident_sparse | T2 | |coef| descending | stability | 19 |
| R3_consensus | T2 | stability_freq desc | stability | 19 |

**R1 vs R2** tests ordering (consensus rank vs stream-balanced).
**sig vs criterion vs plateau** tests size (4p vs 5p vs 8p).
**R3** tests trunk independence (T1 vs T2).
**R3 vs R3_consensus** tests ordering on the T2 trunk (|coef| vs stability ranking), deconfounding trunk from ordering.

### Model-specific recipes (8 base) — 27 cells each

Each recipe is pinned to a single model. Factorial crosses only calibration × weighting × downsampling.

| Recipe | Model | Ordering | Plateau p |
|--------|-------|----------|-----------|
| MS_oof_LR_EN | LR_EN | OOF importance | 4 |
| MS_oof_LinSVM_cal | LinSVM_cal | OOF importance | 4 |
| MS_oof_RF | RF | OOF importance | 8 |
| MS_oof_XGBoost | XGBoost | OOF importance | 9 |
| MS_rfe_LR_EN | LR_EN | RFE elimination | 4 |
| MS_rfe_LinSVM_cal | LinSVM_cal | RFE elimination | 4 |
| MS_rfe_RF | RF | RFE elimination | 8 |
| MS_rfe_XGBoost | XGBoost | RFE elimination | 9 |

**OOF vs RFE** tests ordering source. **Cross-model comparison** tests whether model-specific tailoring beats the shared consensus panel.

### Nested expansion

Model-specific recipes with plateau > 4p auto-expand into sub-recipes at every step from plateau down to 4 (the significance core). Each sub-recipe uses the same ordering, truncated to top-N. This maps each model's performance floor without RFE retraining — single protein steps, clean comparison.

| Base recipe | Expansion | Sub-recipes |
|-------------|-----------|-------------|
| MS_oof_RF (8p) | 8→7→6→5→4 | MS_oof_RF_p7, _p6, _p5, _p4 |
| MS_oof_XGBoost (9p) | 9→8→7→6→5→4 | MS_oof_XGBoost_p8, _p7, _p6, _p5, _p4 |
| MS_rfe_RF (8p) | 8→7→6→5→4 | MS_rfe_RF_p7, _p6, _p5, _p4 |
| MS_rfe_XGBoost (9p) | 9→8→7→6→5→4 | MS_rfe_XGBoost_p8, _p7, _p6, _p5, _p4 |
| LR_EN, LinSVM_cal | plateau = 4 = core | No expansion needed |

Linear models plateau at the core — confirming that 4 proteins saturate their capacity.

---

## Factorial Factors

| Factor | Levels | What it tests |
|--------|--------|---------------|
| **Model** | LinSVM_cal, XGBoost, LR_EN, RF | Which classifier? (shared recipes only) |
| **Calibration** | logistic_intercept, beta, isotonic | How to map scores to probabilities? |
| **Weighting** | log, sqrt, none | Class imbalance via loss function? |
| **Downsampling** | 1.0, 2.0, 5.0 | Class imbalance via data? |

Shared recipes: 4 × 3 × 3 × 3 = **108 cells**.
Pinned recipes: 1 × 3 × 3 × 3 = **27 cells**.

Per cell: 200 Optuna trials, 30 shared splits (seeds 100–129).

### Parsimony Ordering

When statistical tests show no significant difference, prefer the simpler option:

| Factor | Complexity ordering (simplest → most complex) |
|--------|-----------------------------------------------|
| Panel size | Fewer proteins preferred |
| Model | LR_EN (1) < LinSVM_cal (2) < RF (3) < XGBoost (4) |
| Calibration | logistic_intercept (1) < beta (2) < isotonic (3) |
| Weighting | none (1) < sqrt (2) < log (3) |
| Downsampling | 1.0 (1) < 2.0 (2) < 5.0 (3) |

Rationale: LR_EN is a single linear model with L1/L2 penalty (fewest effective parameters). LinSVM_cal adds a kernel-like margin objective. RF is an ensemble of trees. XGBoost is a sequential ensemble with the most hyperparameters.

### Design Constraint: Single-Model Primary, Ensemble Comparison

The factorial's primary output is a single interpretable model suitable for clinical translation. Ensemble methods (stacking, blending) are not factorial factors — they cannot be crossed with recipes since they combine multiple models' predictions.

Instead, ensemble comparison is a **post-tree informational step (V6)**: after V1–V5 lock a single-model winner, V6 tests whether stacking the non-dominated models from V2 beats it. The decision rule applies a higher bar (gain > δ=0.02 with full CI above 0) to account for the interpretability cost. V6 **recommends** but does not auto-select — the human decides.

---

## Cell Count

| Group | Recipes | Cells each | Subtotal |
|-------|---------|------------|----------|
| Shared (R1/R2/R3) | 8 | 108 | 864 |
| Model-specific (base) | 8 | 27 | 216 |
| Nested expansion (RF + XGBoost, OOF + RFE) | 18 | 27 | 486 |
| **Total** | **34** | | **1,566** |

Note: R2 variants (3 recipes) require `--data-path` for stream-balanced ordering. Without data: 4 shared × 108 + 26 pinned × 27 = **1,134 cells**.

---

## Validation Decision Tree

```
V1: Recipe (stratified comparison)
│   Within-type: shared vs shared, MS vs MS (seed-level SE)
│   Cross-type: model-matched bridge comparison
│   Decision: CI includes 0 → prefer fewer proteins
│
├── V2: Model (within winning recipe group)
│   Pareto dominance: AUROC vs Brier reliability (orthogonal axes)
│   Bootstrap CIs on Pareto front (n_boot=1000)
│   Robustly dominated models eliminated (dominated in ≥95% of bootstraps)
│   Parsimony tiebreaker: LR_EN < LinSVM_cal < RF < XGBoost
│
├── V3: Imbalance (JOINT weighting × downsampling)
│   3×3 grid (9 combinations) per recipe × model group
│   Normalized utility: 0.5 × AUPRC_norm + 0.5 × (1 − reliability_norm)
│   Bootstrap CI on utility
│   CI overlaps (none, 1.0) → prefer (none, 1.0) (parsimony)
│
├── V4: Calibration (post-hoc, applied last)
│   Brier reliability component with bootstrap CI overlap test
│   Parsimony: logistic_intercept < beta < isotonic
│   CI overlap with simpler → prefer simpler
│
└── V5: Confirmation (seed-split validation)
    20 selection seeds (100–119) + 10 confirmation seeds (120–129)
    Run V1–V4 on selection seeds, evaluate winner on confirmation seeds
    Flag if confirmation AUROC drops > 1 SE

V6: Ensemble Comparison (post-tree, informational)
    Takes non-dominated models from V2 (already fully optimized)
    Combines calibrated OOF predictions via meta-learner
    Higher bar: gain > δ (0.02) AND full CI above 0
    Human decides: is the gain worth the interpretability tradeoff?
```

**V1** uses a stratified comparison to fix cell-count asymmetry: shared recipes (108 cells) and model-specific recipes (27 cells) are compared within-type first, then bridged via model-matched comparisons. This prevents the 4× cell-count difference from inflating shared-recipe variance estimates. V1 answers:
- **Size**: 4p vs 5p vs 8p vs 9p vs 19p (and every step between for tree models)
- **Ordering**: consensus vs stream-balanced vs model-specific (OOF vs RFE)
- **Trunk**: T1 (consensus) vs T2 (incident)
- **Tailoring**: shared panel vs model-specific panel

**V2** selects the model within the winning recipe group using Pareto dominance on two orthogonal axes (discrimination via AUROC, calibration via Brier reliability).

**V3** resolves weighting and downsampling jointly as a single imbalance decision. These were previously separate steps (V4 weighting, V5 downsampling) but their interaction is non-negligible — aggressive downsampling can mask the effect of class weighting, and vice versa. The 3×3 grid evaluates all 9 combinations under a single normalized utility metric.

**V4** (calibration) is positioned after training-time decisions because calibration is a post-hoc transformation applied to an already-trained model. Evaluating it before training-time choices (imbalance handling) would conflate training dynamics with post-processing.

**V5** is new — a seed-split confirmation protocol. The 30 seeds are partitioned into 20 selection seeds (100–119) used for V1–V4 decisions, and 10 held-out confirmation seeds (120–129) used to validate the winner. This guards against decision-tree overfitting to the selection seeds.

---

## Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| Manifest | `analysis/configs/manifest.yaml` | Declarative source of truth |
| Schema | `analysis/src/ced_ml/recipes/schema.py` | Pydantic models for manifest validation |
| Size rules | `analysis/src/ced_ml/recipes/size_rules.py` | 3-criterion, stability, significance_count |
| Ordering | `analysis/src/ced_ml/recipes/ordering_rules.py` | Consensus, stream, OOF, RFE dispatch |
| Streams | `analysis/src/ced_ml/recipes/streams.py` | Correlation clustering + round-robin |
| Derivation | `analysis/src/ced_ml/recipes/derive.py` | Manifest → panels + audit logs + nested expansion |
| Config gen | `analysis/src/ced_ml/recipes/config_gen.py` | Recipe × factorial → merged YAML pairs + storage/user_attrs |
| CLI | `analysis/src/ced_ml/cli/commands/recipes.py` | `ced derive-recipes` command |
| Submission | `experiments/optimal-setup/factorial/submit_factorial.sh` | SLURM array (two-phase scout/main) |
| Compilation | `experiments/optimal-setup/factorial/compile_factorial.py` | Cells → results (filesystem + Optuna storage modes) |
| Validation | `experiments/optimal-setup/factorial/validate_tree.R` | V1–V5 statistical tests |
| Scout extraction | `experiments/optimal-setup/factorial/extract_scout_params.py` | Top-K params per model for warm-start |
| Live monitoring | `experiments/optimal-setup/factorial/monitor_factorial.py` | Query JournalStorage for progress |
| Analysis program | `experiments/optimal-setup/factorial/analysis/factorial_analysis_program.md` | V1–V5 analysis instructions (dataset-agnostic) |
| Analysis runner | `experiments/optimal-setup/factorial/analysis/runner.py` | R script execution + artifact logging |
| Analysis theme | `experiments/optimal-setup/factorial/analysis/_theme_factorial.R` | Dynamic RECIPE_COLORS, factorial palettes |
| Sweep CLI | `analysis/src/ced_ml/cli/commands/sweep.py` | `ced sweep --spec <yaml> [--dry-run]` |
| Sweep orchestrator | `experiments/optimal-setup/sweeps/sweep_orchestrator.py` | PROPOSE→SUBMIT→POLL→EVALUATE→DECIDE state machine |

## Provenance

- Panels derived via: `ced derive-recipes --manifest configs/manifest.yaml --data-path <parquet>`
- Configs generated by: `ced_ml.recipes.config_gen` (fully-merged, no _base chains)
- Source: `analysis/configs/manifest.yaml` + trunk CSVs + model-specific importance files
- Audit trail: `size_derivation.json` + `ordering_derivation.json` per recipe

## Generalization

The entire design is dataset-adaptive:
- Swap trunk CSVs → different proteins pass significance
- 3-criterion rule → different plateau sizes per model
- Nested expansion → step-down ladders adjust to new plateaus
- No hardcoded panel sizes, protein names, or model counts
