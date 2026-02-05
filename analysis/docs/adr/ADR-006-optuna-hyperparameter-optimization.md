# ADR-006: Optuna Hyperparameter Optimization

**Status:** Accepted | **Date:** 2026-01-20

## Decision

**Add Optuna as optional hyperparameter backend** (coexists with RandomizedSearchCV).

- **Config-driven:** `OptunaConfig.enabled` toggles Optuna vs RandomizedSearchCV
- **Optional dependency:** `pip install ced-ml[optuna]`
- **Sklearn-compatible:** `OptunaSearchCV` mimics `RandomizedSearchCV` API


**Default:** TPE sampler + MedianPruner

## Rationale

**RandomizedSearchCV limits:**
- Random sampling (no learning from trials)
- No pruning (wastes compute on poor hyperparameters)
- Fixed budget upfront

**Optuna advantages:**
- Bayesian optimization (TPE learns from history)
- Pruning stops unpromising trials early
- Adaptive sampling explores promising regions
- Study persistence and parallel support

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| Grid search only | Exponentially expensive |

## Consequences

| Positive | Negative |
|----------|----------|
| 2-5× faster tuning (pruning + TPE) | Adds Optuna dependency (~10 MB) |
| Better hyperparameters in limited trials | More config options |
| Backward compatible (opt-in) | Stochastic (requires seed) |
| Trial history + visualization | ~5-10ms overhead/trial |

## Evidence

**Code:** [optuna_search.py](../../src/ced_ml/models/optuna_search.py) - `OptunaSearchCV`
[training.py](../../src/ced_ml/models/training.py) - integration
[hyperparams.py](../../src/ced_ml/models/hyperparams.py) - Optuna search spaces
**Config:** [schema.py](../../src/ced_ml/config/schema.py) - `OptunaConfig`
**Tests:** `test_optuna_search.py`, `test_training.py`
**Benchmark:** XGBoost tuning: 45 min → 12 min (3× speedup)
**Refs:** Akiba (2019). Optuna. *KDD*. Bergstra (2011). TPE. *NIPS*

## Related

- Depends: ADR-005 (nested CV, where Optuna applies)
