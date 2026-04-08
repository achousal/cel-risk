# cviz: Cornerstone Figures for cel-risk Optimization

6 figures tracing the decision chain: `panel-selection` -> `panel-sweep` -> `holdout-confirmation`.

## Figures

| Fig | File | Question | Data source |
|-----|------|----------|-------------|
| 01 | `fig01_signal_discovery.R` | Which proteins are real signals? | `rra_significance_corrected.csv` (N=2920, BH-corrected) |
| 02 | `fig02_saturation_curve.R` | How many proteins? Where does gain plateau? | `compiled_results_aggregated.csv` (LinSVM_cal, 3 orders) |
| 03 | `fig03_decision_landscape.R` | What matters most — model, size, or order? | `compiled_results_aggregated.csv` (ANOVA + heatmap) |
| 04 | `fig04_marker_architecture.R` | Which proteins are core vs passengers? | `panel-sweep/panels/pathway_*.csv` |
| 05 | `fig05_pareto_frontier.R` | What is the optimal AUROC-Brier trade-off? | `compiled_results_aggregated.csv` (264 configs) |
| 06 | `fig06_holdout_confirmation.R` | Does the locked setup hold on unseen data? | `run_phase3_holdout/` + `run_phase3_holdout_4protein/` |

## Key results encoded

- **Truth set**: 4 proteins (TGM2, CPA2, ITGB7, GIP) survive BH correction at N=2920.
- **Operating point**: Pathway order, LinSVM_cal, 10 proteins — AUROC 0.861 (sweep), 0.874 (holdout).
- **Model choice is load-bearing** (32% of variance); ordering is not (5%).
- **10p vs 4p on holdout**: AUROC 0.874 [0.854, 0.894] vs 0.803 [0.775, 0.831].

## Usage

```bash
cd cel-risk/experiments/optimal-setup/claude_viz/cviz
Rscript fig01_signal_discovery.R
Rscript fig02_saturation_curve.R
Rscript fig03_decision_landscape.R
Rscript fig04_marker_architecture.R
Rscript fig05_pareto_frontier.R
Rscript fig06_holdout_confirmation.R
```

All scripts source `_theme.R` for shared palettes and paths. Output goes to `out/`.

## Dependencies

R packages: `dplyr`, `tidyr`, `ggplot2`, `stringr`, `scales`, `ggrepel`, `jsonlite`, `patchwork`.
