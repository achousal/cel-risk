# Deferred Inconsistency Issues

**Date**: 2026-01-31
**Count**: 13 issues deferred from the 51-issue remediation

---

## Summary

These issues were intentionally deferred because fixing them would break existing artifacts, config files, or public APIs with insufficient benefit to justify the risk.

---

## Breaking-change risk (artifacts / APIs)

| # | Issue | Current state | Why deferred |
|---|-------|--------------|-------------|
| 23 | Metric key casing drift (`auroc` vs `AUROC`) | Producers emit uppercase, consumers normalize ad hoc | Changing keys breaks parsing of all existing metric JSON/CSV artifacts. Requires coordinated producer+consumer migration with artifact versioning. |
| 24 | Threshold key naming (`spec_target`/`spec95`/`alpha`) | Multiple aliases coexist | Aliases serve backward compatibility with saved threshold bundles. Removing them breaks reprocessing of historical runs. |
| 25 | Split artifact naming inconsistencies | Aggregation accepts both `split_seed*` and `split_*` dirs | Legacy fallbacks needed for existing run directories. Safe to remove only after all historical runs are reprocessed or archived. |
| 27 | Control label singular vs plural | Schema defines `Controls`, some outputs use `Control` | Changing labels breaks downstream text-based filtering and aggregation scripts. |
| 45 | Return types (dict/tuple vs dataclass) | Mixed across modules | Migrating to dataclasses changes public function signatures. Best done incrementally as modules are touched. |
| 47 | Random state param naming (`seed` vs `random_state` vs `sampler_seed`) | Each module uses its own convention | Renaming parameters breaks existing config YAML files and CLI invocations. Current naming follows sklearn conventions where applicable. |

## Scope too large / low ROI

| # | Issue | Current state | Why deferred |
|---|-------|--------------|-------------|
| 13 | Model name string literals vs enum | Raw strings like `"XGBoost"`, `"LR_EN"` used throughout | `VALID_MODELS` in `data/schema.py` already validates. Enum would touch 10+ files for marginal type-safety gain. |
| 20 | Mixed random state paradigms | `np.random.seed()` (global) vs `RandomState()` (isolated) | Full migration to isolated state requires threading `RandomState` through every function in the pipeline. |
| 44/48 | Config access patterns (schema vs raw dicts) | Schema objects dominant; `hpc/lsf.py` and `cli/aggregate_splits.py` use raw dicts | Remaining raw dict access is for HPC job params and aggregation overrides that have no schema definition. Fixing requires HPC schema additions. |
| 51 | Validation depth inconsistent | Validate at boundaries (dominant), no validation for internal calls | Formalizing boundary-only validation across all modules adds boilerplate with no practical benefit. Already the de facto pattern. |

## Contextually appropriate as-is

| # | Issue | Current state | Why deferred |
|---|-------|--------------|-------------|
| 17 | Metric display precision (`.3f` vs `.4f`) | Plots use `.3f`, CLI output uses `.4f` | Different contexts warrant different precision. Plots need readability; CLI output needs detail. |
| 18 | Bootstrap `n_boot` defaults vary (50/500/1000) | `rfe.py`: 50, `EvaluationConfig`: 500, final test: 1000 | Each default matches its performance/accuracy tradeoff context. Fast RFE needs speed; final metrics need precision. |

---

## Recommended approach for future remediation

1. **Issues 23-25, 27**: Address together in a single "artifact schema v2" migration with explicit versioning and a conversion utility for historical artifacts.
2. **Issues 44/48**: Add HPC config schema objects when HPC support is next modified.
3. **Issues 13, 45, 47**: Address incrementally -- when a module is touched for other reasons, migrate its types/naming.
4. **Issue 20**: Address if/when reproducibility auditing requires fully isolated random state.
5. **Issues 17, 18, 51**: No action needed unless a concrete problem arises.
