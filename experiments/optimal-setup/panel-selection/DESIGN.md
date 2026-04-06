# Panel Selection

## Goal

Identify the core non-arbitrary protein panel from Phase 1 consensus ranking.

## Design

Two evidence streams were combined:

1. RRA-style cross-model significance from the stable-protein rankings.
2. RFE panel-size sufficiency from per-model optimization curves.

The original Phase 1 analysis yielded a 7-protein panel when BH correction was applied over the 126 consensus candidates. The later universe-sensitivity analysis applied the conservative BH correction over the full 2920-protein universe and retained 4 proteins:

- `tgm2`
- `cpa2`
- `itgb7`
- `gip`

This experiment treats that 4-protein set as the locked truth core. Downstream sweep and comparison work extends or challenges that core but does not replace it.

## Execution assets

- `configs/pipeline_hpc.yaml`
- `configs/training_config.yaml`
- `configs/splits_config.yaml`
- `scripts/compile_results.py`

## Result handoff

Use `results.md` for the decision summary and artifact pointers.
