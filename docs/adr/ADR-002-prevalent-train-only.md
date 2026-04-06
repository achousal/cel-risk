# ADR-002: Prevalent Cases → TRAIN Only

**Status:** Accepted | **Date:** 2026-01-20

## Decision

**Add prevalent cases (n=150) to TRAIN only at 50% sampling.** VAL/TEST remain incident-only.

- **Incident cases** (n=148): Biomarkers collected *before* diagnosis (prospective)
- **Prevalent cases** (n=150): Biomarkers collected *after* diagnosis (retrospective)
- **TRAIN positives:** 148 incident + 75 prevalent (50% sampled) = 223 total
- **VAL/TEST:** Incident-only → prospective evaluation

## Rationale

- Prevalent cases provide training signal but represent different distribution
- VAL/TEST must reflect prospective screening (incident-only)
- 50% sampling balances signal enrichment vs distribution shift

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| Exclude prevalent entirely | Wastes 150 positive samples |
| Prevalent in all splits | VAL/TEST no longer prospective |
| Separate prevalent model | Unnecessary complexity |
| 100% prevalent sampling | Larger distribution shift |

## Consequences

| Positive | Negative |
|----------|----------|
| +75 TRAIN positives (signal boost) | TRAIN ≠ VAL/TEST distribution |
| VAL/TEST remain clinically relevant | Requires prevalence adjustment |
| Balanced signal vs shift (50%) | |

## Evidence

**Code:** [splits.py:326-366](../../src/ced_ml/data/splits.py#L326-L366) - `add_prevalent_to_train`
[schema.py:49](../../src/ced_ml/data/schema.py#L49) - `SCENARIO_DEFINITIONS`
**Tests:** `test_add_prevalent_to_train`, `test_prevalent_never_in_val_test`

## Related

- Depends: ADR-001 (split strategy)
