---
type: condensate
depends_on:
  - "[[equations/optuna-tpe]]"
  - "[[equations/nested-cv]]"
applies_to:
  - "sequential factorial gates that share model families across gates"
  - "workflows using Optuna with warm_start_params_file / warm_start_top_k enqueued from prior gate storage"
  - "studies where the prior gate's top-K trials were selected on a different cohort, search space, or outer-fold seed than the current gate"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "ADR-006 / cellml workflow documents warm-start as an explicit optional feature (OptunaSearchCV.warm_start_params_file, top_k=5 default), and extract_scout_params.py is an explicit operations tool for this purpose — indicating the team treats this as a deliberate coupling to manage, not an invisible default"
    date: "2026-01-20"
    source: "ADR-006, operations/cellml/extract_scout_params.py, ced_ml.utils.optuna_warmstart"
falsifier: |
  Direction claim: gates that warm-start from a prior gate's top-K trials
  produce different winning hyperparameters than cold-start gates at matched
  trial budget. Criterion: cosine distance on hyperparameter vectors between
  warm-started and cold-started best trials > 0.2 with 95% bootstrap CI
  excluding 0.2. If warm-start and cold-start converge to within cosine
  distance < 0.05 and 95% CI inside [0, 0.1] (Equivalence) across >=3
  datasets at matched T, the coupling is weak enough that warm-start is
  safe as default. Retire the coupling concern at 5 such datasets. If
  warm-start converges faster to a clearly worse region (Direction holds
  AND OOF AUROC degrades by > 0.005), warm-start should be removed as default.
---

# Optuna warm-starting couples downstream gate decisions to the prior gate's search trajectory

## Claim

Enqueuing the prior gate's top-$k$ trials as starting points for the current gate's Optuna study makes the current gate's best-trial selection a function of the prior gate's cohort, search space, and outer-fold seed. This is sometimes desirable (a tested coordination contract between gates) and sometimes a hidden coupling hazard (the prior gate's conditions no longer match the current gate's conditions, but the warm-start prior still biases sampling). The celiac pipeline exposes warm-start as an explicit `warm_start_params_file` + `warm_start_top_k` switch, and operates `extract_scout_params.py` as a dedicated tool for emitting the warm-start payload — signalling that the team treats coupling as a decision to declare, not a convenience to enable silently.

## Mechanism

See [[equations/optuna-tpe]]. When $k$ trials are enqueued, they become the first $k$ completed trials in the current study. TPE's $\ell(x)$ and $g(x)$ densities are then seeded with those points. The $\gamma$-quantile threshold $y^*$ is computed over them. Subsequent sampling from $\ell(x)/g(x)$ concentrates probability near the warm-start region, regardless of whether that region remains appropriate under the current gate's data, features, or search space.

Three ways this silently misfires:

1. **Cohort drift.** The prior gate's top-$k$ may have been selected on a training strategy (e.g. incident-only, 1:N control ratio) different from the current gate's training data. Parameters that were optimal there may be suboptimal now, but TPE still biases sampling toward them.
2. **Search-space mismatch.** If the current gate narrows a hyperparameter range (e.g. `learning_rate` $\in [0.01, 0.3]$ -> $[0.05, 0.2]$), warm-start points outside the new range are clipped or ignored; the prior's density shape is preserved but evaluated on a truncated space, producing distorted $\ell(x)/g(x)$.
3. **Seed coupling.** The prior's top-$k$ were chosen under specific outer-fold splits. Re-using them in a new outer-fold configuration means the "good" label transfers without the splits that validated it.

## Actionable rule

- Warm-start is OFF by default. A gate MUST declare `warm_start_params_file` explicitly in its protocol ledger to enable it.
- When enabled, the protocol ledger MUST record: source gate slug, source storage hash, `warm_start_top_k`, and the cohort / search-space delta vs. the source gate.
- If the current gate's training strategy or outer-fold seed differs from the source gate, warm-start is a potential tension. The gate's tension ledger flags this for review even if the run completed.
- `extract_scout_params.py` outputs MUST include source study metadata (gate slug, cohort fingerprint). Consumers must verify metadata match before enqueuing.
- To break coupling without losing scout information: pass warm-start points through a buffer of cold-start trials (e.g. 20 cold trials between scouts and main study) so TPE's prior is not dominated by the warm-start seeds. The default `warm_start_top_k = 5` against $T = 200$ is a 2.5% prior seed fraction — tolerable; $k = 50$ against $T = 200$ would be a 25% seed and should be flagged as high coupling.

## Boundary conditions

- **Inside a single gate, warm-start across outer folds is fine.** ADR-005 isolation requires each outer fold's inner tune to be independent of held-out rows, but not of the other outer folds' tuning histories for the same data. Sharing scout trials across outer folds within one study is a variance-reduction optimization, not a leakage path.
- **Does not apply when search space is strictly equal.** If the current gate uses byte-identical search space, objective, and data fingerprint as the source gate, warm-start is equivalent to resuming the source study and has no coupling hazard. Verify via dataset fingerprint hash match.
- **Risk scales with $k / T$.** Small $k$ against large $T$ gives TPE room to explore past the warm-start region; large $k$ effectively locks the study near the warm-start region.
- **Independent of TPE vs. random.** Warm-start with random sampler enqueues $k$ points as the first $k$ trials but does not bias the sampler (random is memoryless). Coupling is specific to Bayesian samplers.

## Evidence

| Dataset | n | p | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | 2,920 | ADR-006 supports warm-start; `extract_scout_params.py` exists as explicit operations tooling with documented `--top-k` switch; default `warm_start_top_k = 5` indicates intentional small-prior seeding | Migrated from ADR-006 2026-01-20 |

## Related

- [[equations/optuna-tpe]] — TPE density is what warm-start shapes
- [[equations/nested-cv]] — the loop where warm-start is applied
- [[condensates/optuna-beats-random-search]] — orthogonal decision (sampler/pruner choice) that is independent of warm-start
- ADR-006 (Optuna) — canonical source
- `analysis/src/ced_ml/utils/optuna_warmstart.py` — reference implementation
- `operations/cellml/extract_scout_params.py` — operations tooling
