# ADR-013: Five-Strategy Feature Selection

**Status:** Accepted | **Date:** 2026-01-26 | **Updated:** 2026-01-29

## Decision

**Five distinct feature selection strategies for different use cases:**

| Strategy | When | Pipeline | Speed |
|----------|------|----------|-------|
| **1. Hybrid Stability** | Production (default) | screen → k-best (tuned) → stability | 30 min |
| **2. Nested RFECV** | Scientific discovery | screen → RFECV (per fold) → consensus | 5-22 hrs |
| **3. Post-hoc RFE** | Single-model deployment | trained model → RFE → Pareto curve | 5 min |
| **4. Consensus Panel** | Cross-model deployment | multi-model → RRA → uncertainty | 15 min |
| **5. Fixed Panel** | Regulatory validation | train on predetermined panel | 30 min |

**Mutually exclusive (choose during training):** 1 vs 2
**Complementary (post-training):** 3, 4, 5

## Rationale

No single method satisfies all needs:
- Production: fast, reproducible, tunable k
- Discovery: feature stability metrics
- Deployment: panel size optimization
- Cross-model: robust biomarkers across algorithms
- Validation: unbiased estimates on fixed panels

## Typical Workflow

```bash
# Discovery (choose 1 or 2 per model)
ced train --model LR_EN,RF,XGBoost --split-seed 0  # hybrid_stability
ced aggregate-splits --run-id RUN_ID --model LR_EN

# Single-model optimization (optional)
ced optimize-panel --run-id RUN_ID --model LR_EN

# Cross-model consensus (optional)
ced consensus-panel --run-id RUN_ID

# Validation (new seed critical)
ced train --fixed-panel panel.csv --split-seed 10
```

## Strategy Details

### 1. Hybrid Stability (Default)
- **Output:** `feature_selection/stability/stable_panel_t{thresh}.csv`
- **Config:** `feature_selection_strategy: hybrid_stability`
- **Use:** Fast production models (~30 min)

### 2. Nested RFECV
- **Output:** `cv/rfecv/consensus_panel.csv`, `feature_stability.csv`
- **Config:** `feature_selection_strategy: rfecv`
- **Use:** Scientific papers, automatic panel sizing (5-10× slower)

### 3. Post-hoc RFE
- **Output:** `optimize_panel/panel_curve.png`, `recommendations.json`
- **CLI:** `ced optimize-panel --model-path ... --start-size 100`
- **Use:** Stakeholder sizing decisions (~5 min)

### 4. Consensus Panel
- **Output:** `final_panel.txt`, `consensus_ranking.csv`, `uncertainty_summary.csv`
- **CLI:** `ced consensus-panel --run-id RUN_ID`
- **Use:** Model-agnostic robust panels, uncertainty quantification
- **Metrics:** `n_models_present`, `agreement_strength`, `rank_std`, `rank_cv`

### 5. Fixed Panel
- **CLI:** `ced train --fixed-panel panel.csv --split-seed 10`
- **Use:** Validate discovery panels, benchmark literature, regulatory submission
- **Critical:** Use NEW split seed to prevent peeking

## Runtime

(LR_EN, 43k samples, 2920 proteins, 5×10 CV):

| Strategy | Time | Relative |
|----------|------|----------|
| Hybrid Stability | 30 min | 1.0× |
| Nested RFECV | 5-22 hrs | 10-45× |
| Post-hoc RFE | 5 min | 0.16× |
| Consensus Panel | 15 min | 0.5× |
| Fixed Panel | 30 min | 1.0× |

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| Single unified method | No method satisfies speed + stability + deployment trade-offs |
| Two methods (Hybrid + RFECV) | Missing post-training optimization, consensus, validation |
| RFECV only | Too slow for routine use (45× slower) |
| Post-hoc RFE only | Cannot be used during training |

## Consequences

| Positive | Negative |
|----------|----------|
| Clear use case separation | More complex than single method |
| Speed-rigor trade-offs | Users must understand which method |
| Post-hoc rapid iteration | Five output formats to track |
| Cross-model robust biomarkers | Documentation overhead |
| Regulatory-grade validation | |

## Evidence

**Code:**
- Hybrid: [screening.py](../../src/ced_ml/features/screening.py), [kbest.py](../../src/ced_ml/features/kbest.py), [stability.py](../../src/ced_ml/features/stability.py)
- RFECV: [nested_rfe.py](../../src/ced_ml/features/nested_rfe.py)
- Post-hoc: [rfe.py](../../src/ced_ml/features/rfe.py), [optimize_panel.py](../../src/ced_ml/cli/optimize_panel.py)
- Consensus: [consensus.py](../../src/ced_ml/features/consensus.py), [consensus_panel.py](../../src/ced_ml/cli/consensus_panel.py)
- Fixed: [train.py](../../src/ced_ml/cli/train.py) `--fixed-panel`

**Tests:** `test_features_screening.py`, `test_features_kbest.py`, `test_features_stability.py`, `test_features_nested_rfe.py`, `test_features_rfe.py`, `test_model_selector.py`, `test_consensus_panel.py`, `test_cli_train.py`

**Docs:** [FEATURE_SELECTION.md](../reference/FEATURE_SELECTION.md), [UNCERTAINTY_QUANTIFICATION.md](../reference/UNCERTAINTY_QUANTIFICATION.md), [CLI_REFERENCE.md](../reference/CLI_REFERENCE.md)

**Refs:** Guyon (2002) RFE. Meinshausen (2010) Stability. Kolde (2012) RRA

## Related

- Supersedes: ADR-004 (now Strategy 1), ADR-005 (used by 1,2,3)
- Depends: ADR-006 (nested CV provides folds)
- Complements: ADR-008 (Optuna tuning), ADR-009 (ensemble after feature selection)
