# CEL-Risk Optimal-Setup Consolidated Viz (`cviz`)

This directory consolidates the exploratory `codex_viz` route and the
decision-oriented `claude_viz` route into one figure package.

## Story

The optimization result has two layers:

1. A strict truth-level core panel of 4 proteins.
2. A broader operating panel region around 8-10 proteins, with pathway order
   plus `LinSVM_cal` as the preferred practical setup.

The figures in this package are organized to defend that story rather than
showing a pure leaderboard.

## Figure Set

- `01_decision_frontier.R`
  - Performance vs panel size with non-inferiority and stability context.
- `02_acceptance_landscape.R`
  - Gate-by-gate pass/fail plus the stability/performance tradeoff phase plot.
- `03_core_extension_persistence.R`
  - Pathway-order milestone heatmap showing the core-to-extension transition.
- `04_cross_model_agreement.R`
  - Cross-model agreement on the top proteins.
- `05_panel_growth_route.R`
  - Milestone panel-growth route from the 4-protein core to larger operating
    panels.
- `FIGURE_STRATEGY.md`
  - Cornerstone ideas and recommended figure order.

## Notes

- Scripts assume execution from the repository root.
- R scripts target the run at `results/run_20260317_131842/`.
- The route figure reads milestone panel files from
  `experiments/optimal-setup/panel-sweep/panels/`.
