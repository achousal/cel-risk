---
type: equation
symbol: "EI(x)"
depends_on: ["[[equations/nested-cv]]"]
computational_cost: "O(T * (C_fit + C_acquisition))"
assumptions:
  - "trial scores are comparable across trials within a single inner-loop study (same CV splits, same objective)"
  - "gamma-quantile split of past trials yields non-empty l(x) and g(x) densities before acquisition is queried"
  - "pruner intermediate scores are monotone-informative — an under-performing intermediate step predicts an under-performing final score"
failure_modes:
  - "TPE startup phase (default ~10 trials) behaves like random search; premature budget truncation before startup completes wastes the prior"
  - "pruning a trial that would have converged (false prune) under-explores a region of hyperparameter space"
  - "warm-starting from a prior study's top trials couples this gate's tuning trajectory to that prior gate — see [[condensates/optuna-warm-start-couples-gates]]"
---

# TPE expected-improvement acquisition with median pruning rule

## Statement

Tree-structured Parzen Estimator (TPE) acquisition:

Past trials are split at the $\gamma$-quantile of observed scores into "good" (top $\gamma$ fraction) and "bad" (remaining $1 - \gamma$). TPE fits two densities:

- $\ell(x) = p(x \mid y \le y^*)$ — density over hyperparameters among good trials
- $g(x) = p(x \mid y > y^*)$ — density over hyperparameters among bad trials

where $y^*$ is the score at the $\gamma$-quantile. The next trial samples from $\ell(x)$ and selects the candidate maximizing:

$$\text{EI}(x) \propto \frac{\ell(x)}{g(x)}$$

This is monotone in expected improvement over $y^*$ under the Bergstra (2011) formulation.

Median pruning rule: after the $S$-th startup trial, at intermediate step $t$ of the current trial, prune if

$$y_t^{\text{current}} \,\text{worse than}\, \text{median}\{y_t^{(b)} : b = 1, \ldots, B\}$$

where $y_t^{(b)}$ is the intermediate score of completed trial $b$ at step $t$, and $B$ is the number of completed trials with recorded intermediate score at step $t$. "Worse than" is direction-aware (lower is worse for maximize, higher is worse for minimize).

Percentile pruner generalizes to the $q$-th percentile (median = 50th). Hyperband pruner replaces median-rule with successive-halving brackets: at pre-specified resource budgets $r_i$, the top $1/\eta$ fraction advances, the rest are pruned.

## Derivation

TPE: Bergstra et al. (2011). Standard expected-improvement Bayesian optimization requires a surrogate over the score $y = f(x)$. TPE instead models the conditional hyperparameter density $p(x \mid y)$, which is easier to fit in mixed discrete/continuous spaces. The ratio $\ell(x)/g(x)$ is proportional to expected improvement under mild assumptions. Relative to RandomizedSearchCV, TPE uses completed trials to bias future sampling toward the observed "good" region.

MedianPruner: Ja et al. / Akiba et al. (2019). At rare-event prevalence, many hyperparameter configurations give roughly equivalent cross-validation scores; the informative ones are the clearly bad configurations. Stopping them at a partial-CV intermediate score lets the study allocate more of its fixed trial budget to promising regions. Startup trials $S$ protect against early pruning before the median is established.

HyperbandPruner: Li et al. (2017). Trades off exploration vs. exploitation by geometric resource allocation. Recommended when the objective is noisy and the user wants to evaluate many configurations briefly rather than few configurations thoroughly.

## Boundary conditions

- **Startup trials matter.** Below $S$ trials TPE is random. Optuna defaults $S \approx 10$ for TPE. For TPE+Hyperband, documentation recommends $\ge 40$ trials before the TPE prior becomes useful. Studies with $T < 50$ operate largely in the random regime.
- **Intermediate scores required for pruning.** MedianPruner requires `trial.report(score, step)` calls. Absent intermediate reports, pruning is disabled and the study reduces to pure TPE + full-trial evaluation.
- **Seed stochasticity.** TPE draws from $\ell(x)$ are random. Reproducibility requires `sampler_seed`; `random_state` alone seeds the CV splitter, not the sampler.
- **Warm-start coupling.** When `warm_start_params_file` and `warm_start_top_k` enqueue a prior gate's top trials, the first $k$ trials are deterministic and the TPE prior is shaped by the prior gate's search space. See [[condensates/optuna-warm-start-couples-gates]].
- **TPE vs. RandomizedSearchCV equivalence threshold.** At low $T$ and with no pruning, TPE's advantage over random search is small. Benchmarks at ADR-006 show a useful gap only when pruning is active (observed 3x XGBoost speedup: 45 min -> 12 min).

## Worked reference

XGBoost inner loop with $T = 200$ trials, $S = 10$ startup, MedianPruner active, 5-fold inner CV reporting intermediate fold-wise AUROC.

Trial 15 fold-1 AUROC = 0.61. Completed trials at fold 1 report median AUROC = 0.69. Because 15 > $S$ = 10 and 0.61 < 0.69 for a maximize study, trial 15 is pruned after fold 1 without spending folds 2-5. The saved budget ($4 \cdot C_\text{fit}$ for this trial) is consumed by subsequent trials.

Under the TPE prior after trial 100, $\ell(x)$ for `learning_rate` concentrates around the log-uniform region $[0.05, 0.2]$ because top-$\gamma$ completed trials cluster there. Trial 101 is sampled preferentially from that region, not uniformly over the declared search space.

## Sources

- Bergstra et al. (2011). Algorithms for hyper-parameter optimization. NIPS.
- Akiba et al. (2019). Optuna: A next-generation hyperparameter optimization framework. KDD.
- Li et al. (2017). Hyperband: A novel bandit-based approach to hyperparameter optimization. JMLR 18.
- ADR-006 (Optuna hyperparameter optimization).
- `analysis/src/ced_ml/models/optuna_search.py` — OptunaSearchCV reference implementation.

## Used by

- [[condensates/optuna-beats-random-search]]
- [[condensates/optuna-warm-start-couples-gates]]
- [[equations/nested-cv]] — TPE/pruning operate inside the inner loop of nested CV
<!-- TODO: verify slug exists after batch merge — protocols/v1-recipe.md should cite this equation for tuning budget declarations -->
