---
type: condensate
depends_on:
  - "[[equations/optuna-tpe]]"
  - "[[equations/nested-cv]]"
applies_to:
  - "inner-loop hyperparameter search with intermediate-scorable objectives"
  - "hyperparameter spaces of at least 5 dimensions"
  - "trial budgets large enough to clear Optuna's TPE startup ( T >= 50, ideally T >= 100)"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "XGBoost tuning 45 min -> 12 min at matched OOF AUROC under TPE+MedianPruner vs. RandomizedSearchCV; reported in ADR-006 Benchmark"
    date: "2026-01-20"
    source: "ADR-006 (Optuna hyperparameter optimization)"
falsifier: |
  Direction claim: Optuna (TPE sampler + MedianPruner) achieves OOF AUROC at
  least equivalent to RandomizedSearchCV at the same trial budget, and does so
  in less wall time. Criterion: wall-time ratio tpe_plus_pruner / random_search
  < 0.7 AND |ΔAUROC| within [−0.005, +inf] with 95% bootstrap CI excluding
  tpe_plus_pruner being worse by > 0.005. If at trial budget T >= 100 the
  wall-time ratio stays within [0.9, 1.1] across >=3 datasets (Equivalence),
  the claim is weakened. Retire at 5 such datasets. If Optuna is slower or
  achieves lower AUROC by > 0.005 at matched T, Direction is violated.
---

# TPE sampling with median pruning outperforms random search on wall-time at matched OOF metric in the nested-CV inner loop

## Claim

Replacing RandomizedSearchCV with Optuna's TPE sampler plus MedianPruner inside the [[equations/nested-cv]] inner loop reduces wall-clock tuning time without degrading OOF AUROC, under two conditions: (a) the trial budget is large enough for the TPE prior to stabilize ($T \ge 50$, ideally $\ge 100$), and (b) the scoring pipeline reports intermediate fold-wise scores so the pruner can act. ADR-006 reports a 3x speedup for XGBoost tuning (45 min -> 12 min) at matched OOF AUROC on the celiac pipeline. The speedup comes from two complementary mechanisms: pruning terminates clearly-bad trials early, and TPE biases new trials toward the observed good region instead of uniformly sampling the declared space.

## Mechanism

See [[equations/optuna-tpe]]. Two forces combine:

1. **Pruning effect.** MedianPruner terminates trials whose intermediate fold-wise score is worse than the median of completed trials at the same step. For a 5-fold inner CV, a pruned trial spends 1 fold instead of 5 — an 80% saving per pruned trial. Proportion pruned depends on the hyperparameter space, but even 30% pruned at 5 folds gives a ~25% total budget reduction.
2. **TPE effect.** After the startup phase ($S \approx 10$ trials), the sampler weights new trials toward the $\gamma$-quantile "good" region via the $\ell(x)/g(x)$ ratio. At matched trial count, TPE concentrates compute on configurations more likely to win. In low-dim, well-separated search spaces this is a small gain; in mixed-type 10-dim spaces typical of XGBoost, the gain is material.

The two effects compose: TPE picks a promising configuration, pruning lets a bad pick die cheaply. Neither alone is the 3x.

## Actionable rule

- In V1 (recipe) and later gates, when tuning budget $T \ge 50$: use Optuna (TPE + MedianPruner) as the default inner-loop search backend.
- Below $T = 50$: RandomizedSearchCV is acceptable (TPE startup dominates; speedup small).
- `sampler_seed` MUST be set explicitly; TPE sampling is stochastic and reproducibility requires a separate seed from the CV splitter.
- Intermediate scoring: the objective MUST call `trial.report(score, step)` at each inner fold to enable pruning. Without intermediate reports, MedianPruner is a no-op and TPE alone gives a smaller speedup.
- TPE + HyperbandPruner: not default. Use only when $T \ge 40$ and the user has declared resource budgets per trial. ADR-006 mentions Hyperband support but does not adopt it as default.
- Optuna is an optional dependency (`pip install ced-ml[optuna]`); gates that declare Optuna backend must also declare the dependency in their protocol.

## Boundary conditions

- **Trial budget floor.** TPE's advantage over random search requires enough completed trials to fit $\ell(x)$ and $g(x)$. Below $S$ trials (default 10), TPE is random. Studies with $T < 50$ operate largely in the random regime.
- **Intermediate scores required for pruning.** If the objective returns only a final score, MedianPruner cannot act. Pruning benefit is zero in that mode.
- **Search space must be larger than trivially enumerable.** At $< 3$ hyperparameters with small discrete domains, grid search or random search suffice; TPE adds overhead ($\sim 5$-$10$ ms per trial) with no benefit.
- **Noisy objectives stress pruning.** If fold-wise score variance is high, MedianPruner can incorrectly prune trials that would recover in later folds. Use PercentilePruner with a stricter $q$ (e.g. 25th percentile) for noisy objectives.
- **Warm-start is a separate decision.** Even when TPE+pruner is the search backend, warm-starting from a prior gate's top trials couples the gates and is governed by [[condensates/optuna-warm-start-couples-gates]].

## Evidence

| Dataset | n | p | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | 2,920 | XGBoost tuning: 45 min -> 12 min (3x) under TPE+MedianPruner vs. RandomizedSearchCV at matched OOF AUROC | Migrated from ADR-006 2026-01-20 |

## Related

- [[equations/optuna-tpe]] — TPE and pruning rules
- [[equations/nested-cv]] — the loop Optuna runs inside
- [[condensates/optuna-warm-start-couples-gates]] — orthogonal decision about coupling gates
- [[condensates/nested-cv-prevents-tuning-optimism]] — why the inner loop needs isolation regardless of backend
- ADR-006 (Optuna) — canonical source
