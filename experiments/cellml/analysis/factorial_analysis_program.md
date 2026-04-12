# Factorial Analysis Program

**Purpose:** Guide an agent through systematic interpretation of factorial experiment results. This is the `program.md` — read it before generating any analysis scripts.

**Dataset-agnostic:** These instructions never reference specific recipe names, protein counts, or winning models. The agent discovers all structure from the data.

---

## Goal

Interpret the factorial results to:
1. Identify the optimal cell (recipe × model × weighting × downsampling × calibration)
2. Explain *why* it wins (which factors matter, which don't)
3. Surface unexpected patterns for follow-up investigation

---

## Data

### Primary input
- `factorial_compiled.csv` — one row per cell, columns include factorial factors and aggregated metrics

### Guaranteed columns
**Factorial factors** (always present):
- `recipe_id` — recipe identifier
- `factorial_model` — model name
- `factorial_calibration` — calibration method
- `factorial_weighting` — class weight strategy
- `factorial_downsampling` — majority:minority ratio

**Core metrics** (always present):
- `summary_auroc_mean`, `summary_auroc_std`
- `summary_prauc_mean`, `summary_prauc_std`
- `summary_brier_score_mean`
- `summary_brier_reliability_mean`

### Discovery pattern
Additional columns vary by dataset. On your first step:
1. Read `head(df)` and `names(df)` to discover the full schema
2. Report the column list in the narrative before proceeding

### Validation outputs (may exist)
- `factorial_validation_v1_summary.csv`, `_v1_comparisons.csv`
- `factorial_validation_v2_pareto.csv`
- `factorial_validation_v3_imbalance.csv`
- `factorial_validation_v4_calibration.csv`
- `factorial_validation_v5_confirmation.csv`

If validation CSVs exist, read them as supplementary evidence. If they don't, the agent computes the relevant statistics directly.

---

## Style

- Source `_theme_factorial.R` in every R script
- Use `theme_cel()` for all plots
- Use the named palettes: `MODEL_COLORS`, `CALIBRATION_COLORS`, `WEIGHTING_COLORS`, `DOWNSAMPLING_COLORS`, `RECIPE_COLORS` (auto-generated from data)
- Output via `factorial_save_fig(p, "name")` → PDF + PNG at 300 DPI in `figures/`
- Summary tables via `factorial_save_table(df, "name")` → CSV in `tables/`
- Each script is standalone: `source("_theme_factorial.R")` at the top

---

## Analysis Structure

Follow V1 → V2 → V3 → V4 → V5 → Exploration → Synthesis. You may deviate if a level reveals something that demands immediate investigation.

### V1: Recipe Comparison
**Question:** Which recipe wins within each type, and does the best shared-panel recipe beat the best model-specific recipe?

Suggested figures:
- Heatmap: mean AUROC by recipe × model (rows = recipes, columns = models, fill = AUROC)
- **Stratified forest plot:** within-shared pairwise CIs, within-MS pairwise CIs, cross-type bridge CIs (model-matched)
- Size-performance curve: AUROC vs panel size (derived from recipe metadata if available)
- Trunk comparison: T1 vs T2 recipe groups (if both exist)

Statistical tests:
- Pairwise z-tests using **seed-level SE** (mean of per-cell auroc_std / sqrt(n_seeds)), NOT cell-level SE
- Comparisons are stratified: shared recipes compared against shared only, MS against MS only
- Cross-type bridge: best shared vs best MS matched on model (e.g., R1_plateau RF cells vs MS_oof_RF cells)

Decision rule: if CI includes 0, prefer fewer proteins.

### V2: Model Selection (within winning recipe group)
**Question:** Which model dominates on AUROC vs calibration quality?

Suggested figures:
- Pareto front scatterplot: AUROC (x) vs **Brier reliability** (y), colored by model
- Mark dominated vs non-dominated points
- **Bootstrap Pareto membership frequency** (bar chart: % of 1000 bootstraps each model is non-dominated)

Decision rule: Robustly Pareto-dominated models eliminated (dominated in ≥95% of bootstraps). Ties broken by complexity: LR_EN < LinSVM_cal < RF < XGBoost.

### V3: Imbalance Handling (Joint Weighting × Downsampling)
**Question:** What's the best combined class-imbalance strategy?

Weighting (loss function) and downsampling (data) both address class imbalance and interact — choosing them sequentially can miss joint optima. They are resolved as a 3×3 grid.

Suggested figures:
- Heatmap: normalized utility by weighting × downsampling (3×3 grid, one panel per recipe×model group)
- Interaction plot: AUPRC by weighting, faceted by downsampling (and vice versa)
- **Utility bootstrap CI forest plot** for all 9 combinations

Decision rule: Highest normalized utility (0.5 × AUPRC_norm + 0.5 × reliability_norm). If CI of best overlaps with (none, 1.0) → prefer (none, 1.0) for parsimony.

### V4: Calibration
**Question:** Which calibration method produces lowest reliability component?

Suggested figures:
- Brier reliability by calibration method (boxplot or dotplot)
- Reliability diagram overlay (if per-fold calibration data available)
- **Bootstrap CI forest plot** for reliability by calibration method

Decision rule: Lowest `summary_brier_reliability_mean` preferred. Bootstrap CI overlap with simpler method → prefer simpler. Complexity: logistic_intercept < beta < isotonic.

### V5: Seed-Split Confirmation
**Question:** Does the winner generalize to held-out seeds?

Seeds 100–119 are used for V1–V4 selection. Seeds 120–129 are held out for confirmation. This catches overfitting to the selection seeds.

Suggested figures:
- Paired bar chart: selection AUROC vs confirmation AUROC for top-3 candidates
- Bootstrap CI comparison: selection seeds vs confirmation seeds

Decision rule: If confirmation AUROC drops > 1 SE below selection AUROC, flag the winner as unstable. If runner-up is stable and within the V1 CI, consider promoting it.

**Note:** If per-seed compiled data is not yet available, V5 falls back to a variance-ratio check: is the winner's summary_auroc_std comparable to runners-up? Unusually high variance is flagged.

### Exploration
**Question:** What unexpected patterns exist?

Look for:
- Interaction effects: does calibration matter more for some models than others?
- Outlier cells: any cell dramatically better/worse than its neighbors?
- Seed stability: are results consistent across seeds, or are some cells unstable?
- Recipe group patterns: do shared-panel recipes behave differently from model-specific ones?
- V3 interaction surface: is there a weighting × downsampling interaction, or are the factors approximately additive?
- Seed stability: compare selection-seed and confirmation-seed rankings

No prescribed figures — follow the data.

### V6: Ensemble Comparison (Post-Tree)
**Question:** Does stacking the non-dominated models beat the locked single-model winner?

Explanation: "This is an informational comparison, not an automatic selection. Ensembles trade interpretability for potential performance gains. The decision rule uses a higher bar (δ=0.02, full CI above 0) to reflect this tradeoff."

Suggested figures:
- Bar chart: single-model AUROC vs ensemble AUROC with bootstrap CIs
- If ensemble flagged: component model contributions (per-model OOF AUROC vs ensemble)

Decision rule: "Ensemble gain > δ (0.02) AND full CI above 0 → recommend human review. Otherwise, single model is sufficient. **The human decides** — this level never auto-selects."

Note: "The V6 estimate in validate_tree.R uses averaged base-model AUROCs as a conservative proxy. If V6 flags the ensemble, run a full stacking comparison using OOF prediction files before making the final decision."

### Synthesis
Compile findings into a final summary:
- Winning cell specification (all 5 factors: recipe, model, weighting, downsampling, calibration)
- Ensemble comparison result (V6 — flagged or not)
- Which factors mattered most → least
- Key surprises or concerns
- Recommended follow-up analyses (these become sweep specs for F1)

---

## Narrative Format

After each analysis step, write to `narrative.md` using this structure:

```markdown
## V{N}: {Level Name}

### {Figure Title}

**Observation:** [Factual statement about what the data shows. Numbers, directions, magnitudes. No interpretation.]

**Interpretation:** [Tentative explanation. Why might this pattern exist? What does it mean for the decision? Hedged language — "suggests", "consistent with", "may indicate".]

![{figure_name}](figures/{figure_name}.png)
```

Rules:
- Observations are facts. Interpretations are hypotheses.
- Never state a conclusion as fact — the human decides.
- If something surprises you, flag it explicitly: "**Unexpected:**"
- Cross-reference earlier findings when relevant

---

## New Dataset Guidance

If no prior exists (no Gen 1 reference values, no historical baselines):
- Establish baselines from the data: best-performing recipe = reference, others compared to it
- Do not compare to hardcoded thresholds (e.g., "AUROC > 0.85")
- Report absolute values and relative differences
- The narrative should note "This is the first analysis of this dataset — no historical comparison available"

---

## Constraints

- **Read-only on pipeline:** Never modify training configs, pipeline code, or results data
- **No retraining:** All analysis operates on existing results
- **No definitive conclusions:** The human reviews the narrative before anything enters a manuscript
- **Script-per-figure:** Each R script produces one figure (possibly with a companion table). Do not bundle multiple unrelated plots.
