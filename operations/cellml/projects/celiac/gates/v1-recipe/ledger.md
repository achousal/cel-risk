---
gate: v1-recipe
project: celiac
rulebook_snapshot: "rb-v0.1.0"
dataset_fingerprint: "sha256:8c02e33cc693edfb361a4d901113cd3d5d79c8c43193919440305d84c278c0e9"
created: "2026-04-20"
author: llm-advisor
---

> **Live notice.** This ledger is written at V1 gate ENTRY, before any V1 cell
> has executed. Predictions are pre-run (falsifiable against a future
> `observation.md`). It inherits V0 locks from
> [[projects/celiac/gates/v0-strategy/decision.md]] (retrospective) and
> complies with [[protocols/v1-recipe]] under `rulebook_snapshot: rb-v0.1.0`.
> No TEST partition is consulted at any point in V1 (50/25/25 TRAIN/VAL/TEST
> from ADR-001 holds; TEST quarantine preserved).

---

## Hypothesis

V1 asks which **recipe** — a composition of (trunk × ordering × size) — is
the best discrimination vehicle on the V0-locked cell configuration
(IncidentOnly × control_ratio=5). The recipe-family hypothesis is:

> **Under the V0-locked strategy and the Stage-1/stability pre-filter chain
> mandated by [[condensates/feature-selection-needs-model-gate]] and
> [[condensates/stability-as-hard-filter]], the shared-panel R1 family
> (T1-consensus trunk × consensus-descending ordering) at the
> three_criterion_unanimous plateau (p=8) discrimination-equivalences the
> same family at the significance_count core (p=4). The tie resolves to
> p=4 via [[condensates/parsimony-tiebreaker-when-equivalence]]. No single
> model-specific recipe dominates the shared-panel family on AUROC once
> ΔAUROC and the paired-seed bootstrap CI from v1-recipe.md §3.3 are
> enforced.**

V1 locks four fields on success: `recipe_id`, `panel_composition`,
`panel_size`, `ordering_strategy`. Model, calibration, weighting,
downsampling remain open (deferred to V2/V3/V4 per v1-recipe.md
`axes_deferred`). The cross-family bridge (§3.4 of the protocol) is the
only test allowed to adjudicate shared vs. model-specific at V1; all four
per-model bridges must agree on claim type for V1 to lock a cross-family
direction.

The hypothesis is informed by V0's forward predictions
([[projects/celiac/gates/v0-strategy/decision.md]] §Predictions for
v1-recipe): R1 vs R2 Equivalence expected; MS_* not dominating R1/R2 at
matched sizes; 4-protein BH core robust across partitions. V1 re-runs
these under the corrected SE definition.

---

## Search-space restriction

The v1-recipe protocol declares 34 recipes in a within-family pairwise
tournament (§2.2). Running the full pair set naively is expensive: 8
shared recipes × 108 cells = 864 shared cells; 26 model-specific recipes
(8 base + 18 nested expansion) × 27 cells = 702 MS cells; 1,566 total
factorial cells. The LLM advisor's job at this gate is to propose a
**staged restriction** that routes scarce compute toward the highest-
information pairs first, without prejudging any axis the protocol
requires V1 to test.

Restriction is **aggressive on compute order**, **non-aggressive on recipe
scope**: every recipe in the 34-set is still eligible; the advisor only
reorders when they run.

### Stage A (first pass — 14 recipes, 6 of 8 shared + all 8 MS base; defer nested expansion)

Rationale: the cleanest parsimony signals and the cross-family bridge
anchor on the smallest distinct sizes. Nested expansion sub-recipes (18
recipes at p = 4..8 for RF and p = 4..9 for XGBoost) are derived from the
MS plateau winners; running them before the plateau cell has converged
spends compute on a step-down ladder anchored at a point the factorial
has not yet validated. Defer the ladders until Stage A reveals which
ordering (OOF vs RFE) survives on which model.

Stage A cells: 6 shared × 108 + 8 MS base × 27 = 648 + 216 = **864 cells**.

| Recipe pair tested in Stage A | Which axis it isolates | Condensate authority |
|---|---|---|
| R1_sig (p=4) vs R1_plateau (p=8) | Size, within T1 × consensus ordering | [[condensates/three-criterion-size-rule]] predicts the plateau at p=8 with parsimony pulling to p=4 |
| R1_sig (p=4) vs R1_criterion (p=5) | Size, within T1 × consensus ordering | [[condensates/three-criterion-size-rule]] — the ≥2-criterion variant vs the significance core |
| R2_sig (p=4) vs R2_plateau (p=8) | Size, within T1 × stream-balanced ordering | Size axis under alt ordering; mirror of R1_sig vs R1_plateau |
| R1_sig vs R2_sig (both p=4) | Ordering at matched minimum size | [[condensates/size-ordering-pooling]] — size was fixed ordering-agnostic, so any residual ordering effect is clean here |
| R1_plateau vs R2_plateau (both p=8) | Ordering at matched plateau size | Same as above, larger panel |
| R3_incident_sparse vs R3_consensus (both p=19) | Ordering on T2 trunk | Deconfounds trunk from ordering per DESIGN.md §Recipes |
| MS_oof_* vs MS_rfe_* per model | Ordering source within model | Isolates OOF vs RFE at each model's plateau |
| Cross-family bridge: R1_plateau-restricted-to-model-m vs MS_oof_m and MS_rfe_m | Shared vs model-specific, per-model | v1-recipe.md §3.4 — four per-model deltas must agree |

Recipes **excluded from Stage A**: R1_criterion (p=5) vs anything other
than R1_sig (already in table), R2_criterion, and all 18 nested expansion
sub-recipes (MS_oof_RF_p{4..7}, MS_oof_XGBoost_p{4..8}, MS_rfe_RF_p{4..7},
MS_rfe_XGBoost_p{4..8}).

### Stage B (conditional — run iff Stage A returns Inconclusive or triggers a ladder)

Run the 18 nested expansion sub-recipes **only if**:

1. Stage A's MS_oof_RF vs MS_rfe_RF returns Inconclusive or Equivalence
   at RF plateau (p=8). If RF plateau shows Direction toward one
   ordering, that ordering's ladder (RF_p4..p7, 4 sub-recipes × 27 cells
   = 108 cells) runs; the loser's ladder is dropped.
2. Same rule for XGBoost at its plateau (p=9). Winner's ladder is
   p4..p8, 5 sub-recipes × 27 cells = 135 cells; loser dropped.
3. R1_criterion and R2_criterion run **only** if Stage A returns
   Equivalence between the two endpoints of their ordering (R1_sig ≈
   R1_plateau or R2_sig ≈ R2_plateau). Their role is to resolve the p=5
   middle point when the endpoints are tied; if one endpoint dominates,
   the middle point is redundant.

Stage B cell budget (worst case, both ladders and both criteria run):
108 + 135 + 2 × 108 = 459 cells.

### What is **NOT** restricted

1. **Model axis stays full in Stage A.** All four {LR_EN, LinSVM_cal, RF,
   XGBoost} are evaluated in every shared-panel cell (108-cell expansion
   preserved) and the four MS base recipes cover each model at its
   plateau. V1 does not lock the model; that is v2-model's job
   (v1-recipe.md `axes_deferred: model`). Pruning the model axis at V1
   would violate the protocol.
2. **Calibration × weighting × downsampling axes stay full in every
   cell.** Each shared recipe remains 4 × 3 × 3 × 3 = 108 cells; each MS
   recipe remains 1 × 3 × 3 × 3 = 27 cells. These are deferred axes at
   V1 but are not dropped — they are **aggregated over** when computing
   recipe-mean AUROC per seed per v1-recipe.md §3.3. Dropping them would
   collapse the recipe-mean into a point estimate with no variance
   structure for the paired bootstrap.
3. **30 outer-fold seeds (100–129) stay full.** §3.3 mandates the
   seed-level paired bootstrap; reducing seeds below the full 30 would
   invalidate the CI construction.
4. **Stage-1 model gate + stability hard-filter chain stays mandatory.**
   Per [[condensates/feature-selection-needs-model-gate]] and
   [[condensates/stability-as-hard-filter]], any recipe consuming T1
   consensus must run through the gate + filter chain before the
   consensus-rank-aggregation ordering is computed. Skipping these
   filters would make the R1 family's panels undefined, not just
   sub-optimal. Pre-condition check from v1-recipe.md §1.4 and §1.5
   blocks entry if this chain is not wired.

### Restriction summary

| Stage | Recipes run | Cells | Cumulative cells |
|---|---|---|---|
| A (first pass) | 14 of 34 (6 shared + 8 MS base) | 864 | 864 |
| B (conditional) | Up to 20 of 34 (2 criterion + 18 nested expansion) | Up to 459 | Up to 1,323 |
| **Deferred entirely** | 0 of 34 | 0 | — |

Total if Stage B fully triggers: 1,323 cells vs 1,566 naive = **15% compute
saving at worst, up to 58% saving if Stage A resolves cleanly** (no
ladders, no criterion middle-points). The LLM advisor's bet is that
parsimony signals live on the size endpoints (p=4 vs p=8/9), not on the
middle point (p=5), and that the MS winners on OOF vs RFE will often
settle the ladder before the sub-sizes run.

---

## Cited rulebook entries

All slugs verified present at `rulebook/condensates/` and
`rulebook/equations/` on disk at gate entry (ls-check against the
`rb-v0.1.0` snapshot).

### Condensates

- **[[condensates/feature-selection-needs-model-gate]]** — V1 R1/R2 recipes
  depend on T1 consensus, which is only valid over Stage-1-gated models
  (per-model permutation p < 0.05). Without this gate the shared-panel
  family's panels are noise aggregations.
- **[[condensates/stability-as-hard-filter]]** — Per-model ranked lists
  feeding consensus-rank-aggregation must pass the selection-frequency
  threshold (default 0.90) before aggregation. Composite weighting of
  stability into ranking is rejected.
- **[[condensates/nested-cv-prevents-tuning-optimism]]** — Each V1 cell
  uses 5×10×5×200 nested CV (50k fits); outer-fold isolation is how the
  paired bootstrap over seeds in §3.3 remains unbiased.
- **[[condensates/optuna-warm-start-couples-gates]]** — V1 cells warm-start
  from V0 scout trials. Ledger records source gate slug
  (`v0-strategy`), `warm_start_top_k = 5` against T = 200 (2.5% prior
  seed fraction — below the coupling-hazard threshold), source storage
  hash = V0 scout manifest, and the dataset fingerprint match (V0 and
  V1 share
  `sha256:8c02e33cc693edfb361a4d901113cd3d5d79c8c43193919440305d84c278c0e9`).
- **[[condensates/three-criterion-size-rule]]** — The size axis is derived
  from three_criterion_unanimous (p=8 plateau) and significance_count
  (p=4 core) per DESIGN.md §Size rules. Prediction P1 is a direct test
  of this rule.
- **[[condensates/size-ordering-pooling]]** — Size was derived on sweep
  data pooled across {importance, pathway, rra} orderings, so V1 can
  treat size and ordering as independent axes. Without pooling, any
  ordering-at-matched-size test is apples-to-oranges (prediction P3
  depends on this).
- **[[condensates/nested-expansion-vs-refit]]** — Stage B ladders use
  step-down truncation of the parent ordering, not refit-per-size. This
  condensate flags RF's flexibility risk (step-down may under-estimate
  refit by > 0.01 AUROC at p=4 from an 8p parent). Stage B is gated on
  this risk.
- **[[condensates/perm-validity-full-pipeline]]** — The Stage-1 model gate
  p-values inherited from V0 must have been computed under full-pipeline
  permutation (re-run feature selection per permutation). Partial
  permutation invalidates the gate per ADR-011.
- **[[condensates/parsimony-tiebreaker-when-equivalence]]** — The
  meta-condensate that binds V1's size tiebreaker. When the rubric
  returns Equivalence between p=4 and p=8, V1 locks p=4 (smaller). It
  does NOT invoke parsimony silently; `decision.md` must name the claim
  as Equivalence and cite the axis-specific order in v1-recipe.md §3.2.

### Equations

- **[[equations/consensus-rank-aggregation]]** — Geometric-mean normalized
  reciprocal-rank aggregator. Governs R1/R2 ordering when T1 consensus
  is the trunk. Valid only post-gate and post-stability-filter (its own
  boundary conditions).
- **[[equations/nested-cv]]** — 5×10×5×200 fold decomposition. Defines
  the OOF estimator on which every V1 AUROC and the seed-level paired
  bootstrap are computed.
- **[[equations/optuna-tpe]]** — TPE with MedianPruner at T = 200. The
  inner-loop search inside nested-cv. Warm-start from V0 shapes the TPE
  prior (coupling condensate above bounds the hazard).

---

## Falsifier criteria

All V1 claims use the fixed rubric from SCHEMA.md §Fixed falsifier
rubric, as carried into v1-recipe.md §3.1. AUROC is the primary
discrimination metric; Brier REL is reported but not adjudicated
(v1-recipe.md §3.5). No metric overrides are declared at V1 (SCHEMA.md
§Per-protocol metric overrides — no `metric_overrides` block in
v1-recipe frontmatter).

| Claim type | Criterion on AUROC | V1 decision |
|---|---|---|
| **Direction** (A > B) | \|ΔAUROC\| ≥ 0.02 AND 95% paired-seed bootstrap CI (1000 resamples over seeds 100–129) excludes 0 | Winner locked; loser retired from the family |
| **Equivalence** (A ≈ B) | \|ΔAUROC\| < 0.01 AND 95% paired-seed bootstrap CI ⊂ [−0.02, 0.02] | Parsimony tiebreaker applies (v1-recipe.md §3.2) |
| **Dominance** (A ≻ B, multi-axis) | Direction criterion holds independently on each of the four per-model bridges (v1-recipe.md §3.4) | Cross-family lock |
| **Inconclusive** | Neither Direction nor Equivalence met | Both recipes forwarded to V2; no lock on that pair |

**Counts** (panel size `p`, cell count): exact comparison, no CI, per
SCHEMA.md metric-specific rules.

### SE definition — mandatory, fixes T-V1-01

Per v1-recipe.md §3.3 (which fixes the V1 SE bug documented in
DECISION_TREE_AUDIT.md §1.3), every CI above is computed as:

    auroc_ci = BOOTSTRAP(
        unit       = seed,                   # outer-fold / seed-level PAIRED resample
        n_boot     = 1000,
        aggregation = "recipe_mean",          # mean AUROC across cells in the recipe, per seed
        stat       = "delta_AUROC",           # ΔAUROC between the paired recipes
    )

The pre-fix cell-averaged SE (`sd(summary_auroc_mean) / sqrt(n_cells)`)
is **explicitly forbidden** at V1 and any V1 observation.md that emits
it must be rejected by the tension detector. This ledger's predictions
are all stated against the paired-seed bootstrap.

### Cell-count asymmetry — mandatory, fixes T-V1-02

Cross-family (shared vs MS) comparisons use the model-matched bridge per
v1-recipe.md §3.4: for each model m ∈ {LR_EN, LinSVM_cal, RF, XGBoost},
restrict the shared recipe to the 27 cells with that model and compare
against the model-specific recipe pinned to m. The four deltas must
all return the same claim type for V1 to lock cross-family. Otherwise
the cross-family lock defers to V2 (v1-recipe.md §4 fallback 6).

### TEST-quarantine preservation

Every metric computed at V1 is on VAL partitions within the 30-seed
outer-fold CV. The 25% TEST partition (per ADR-001, locked at V0) is
**not consulted** at V1. No prediction in this ledger references TEST
values.

---

## Predictions with criteria

All predictions are **pre-run**. None cites a specific post-run point
estimate; each cites the claim type and the threshold that would
adjudicate it.

- **P1: R1_sig (p=4, consensus-desc) ≈ R1_plateau (p=8,
  consensus-desc) on AUROC.** Rubric claim type: **Equivalence**.
  Criterion: \|ΔAUROC\| < 0.01 AND 95% paired-seed bootstrap CI ⊂
  [−0.02, 0.02]. Decision: parsimony tiebreaker locks p=4.
  Rationale: [[condensates/three-criterion-size-rule]] predicts a
  parsimony plateau at p=4 (significance_count) when the >=2-criterion
  rule identifies 5p and the unanimous rule identifies 8p (DESIGN.md
  §Size rules). [[condensates/parsimony-tiebreaker-when-equivalence]]
  binds the tiebreaker to the v1-recipe.md §3.2 axis order (smaller
  `p` wins).

- **P2: Shared-panel family (R1_plateau and R1_sig) does not
  Dominate the model-specific base family (MS_oof_m, MS_rfe_m) on
  the four-per-model bridge.** Rubric claim type: **Inconclusive or
  Equivalence** on at least one bridge model. Criterion: at least
  one of {LR_EN, LinSVM_cal, RF, XGBoost} bridge returns Equivalence
  or Inconclusive under the rubric; therefore the cross-family
  Dominance criterion (all four agree on Direction) fails. Decision:
  per v1-recipe.md §4 fallback 6, the shared vs MS axis is deferred
  to V2. Rationale: [[equations/consensus-rank-aggregation]] produces
  a single ordering across all gated models; this should absorb
  idiosyncratic per-model noise, making shared ≈ MS on the linear
  models (LR_EN, LinSVM_cal) where OOF and RFE both plateau at the
  significance core. On the tree models (RF, XGBoost) the MS ordering
  may have a small edge by step-down construction. Heterogeneous
  bridge outcomes are therefore the expected pattern, which is
  exactly what fails the Dominance criterion.

- **P3: R1_sig (p=4, consensus-desc) ≈ R2_sig (p=4, stream-balanced)
  on AUROC at the matched p=4 size.** Rubric claim type:
  **Equivalence**. Criterion: \|ΔAUROC\| < 0.01 AND 95% paired-seed
  bootstrap CI ⊂ [−0.02, 0.02]. Decision: parsimony by ordering
  (v1-recipe.md §3.2 secondary tiebreaker) — consensus-desc ≺
  stream-balanced ≺ model-specific; R1_sig wins on ordering
  simplicity. Rationale: [[condensates/size-ordering-pooling]]
  guarantees the size decision was ordering-agnostic, so any residual
  ordering effect at a matched small size should be within
  measurement noise on the core 4 proteins.

- **P4: MS_oof_RF (p=8) vs MS_rfe_RF (p=8) is Inconclusive on
  AUROC; same for MS_oof_XGBoost (p=9) vs MS_rfe_XGBoost (p=9).**
  Rubric claim type: **Inconclusive**. Criterion: \|ΔAUROC\| falls in
  the zone [0.01, 0.02] OR the 95% paired-seed bootstrap CI
  straddles both ±0.01 and ±0.02 boundaries, preventing Direction
  and Equivalence from closing. Decision: ordering-within-MS defers
  to V2's model-conditional view. Rationale:
  [[condensates/nested-expansion-vs-refit]] notes MS_oof orderings
  are more step-down-stable than MS_rfe, but both are designed for
  the same model with the same plateau size — intra-model ordering
  differences on 8 or 9 proteins tend to be small and noisy.
  Inconclusive is the expected outcome; this ledger flags that a
  non-inconclusive result would be a surprise.

- **P5: R3_incident_sparse (p=19, |coef|-desc, T2 trunk) does not
  Direction-dominate R1_plateau (p=8, consensus-desc, T1 trunk) on
  AUROC.** Rubric claim type: **Equivalence** or **Inconclusive**.
  Criterion: \|ΔAUROC\| < 0.02 OR 95% paired-seed bootstrap CI
  includes 0; either outcome defeats Direction. Decision: if
  Equivalence, parsimony by size (v1-recipe.md §3.2) locks R1_plateau
  (p=8) over R3_incident_sparse (p=19) — the smaller panel wins. If
  Inconclusive, both forward to V2. Rationale: T2 trunk's stability
  size (p=19) is far above both T1 plateaus (p=8, p=4); if T2's
  extra 11 proteins carried discrimination signal above T1's top
  proteins, it would already be visible in the V0 discovery sweep.
  The [[equations/consensus-rank-aggregation]] output on T1 with the
  full Stage-1 + stability chain is the design's preferred panel
  construction; R3 is the trunk-deconfounder, not a competitor.

### Prediction-to-axis map

| Prediction | Axis | Claim type | Tiebreaker invoked on Equivalence |
|---|---|---|---|
| P1 | Size, within T1-consensus ordering | Equivalence | Size parsimony → p=4 |
| P2 | Cross-family (shared vs MS) via 4-model bridge | Inconclusive | None (defer to V2) |
| P3 | Ordering, within size p=4 | Equivalence | Ordering parsimony → consensus-desc |
| P4 | Ordering within MS at plateau (OOF vs RFE) | Inconclusive | None (defer to V2) |
| P5 | Trunk (T1 vs T2) at p=8 vs p=19 | Equivalence / Inconclusive | Size parsimony → T1 p=8 |

---

## Risks & fallbacks

Each risk maps to a v1-recipe.md §4 fallback and records how the LLM
advisor would re-direct the run.

- **R1 — Stage-1 gate fails on any model.** Per v1-recipe.md §4
  fallback 5, V1 does not lower alpha. If zero of {LR_EN,
  LinSVM_cal, RF, XGBoost} passes the permutation gate at α = 0.05,
  the R1/R2 family is undefined and V1 halts; the escalation is
  cohort/feature-engineering, not a protocol patch. If one or two
  models fail, the consensus aggregation runs only on the
  gate-passing models (per
  [[condensates/feature-selection-needs-model-gate]]) and V1 records
  the restricted model set in `decision.md`.

- **R2 — All pairs in a family return Inconclusive (including Stage
  A).** Per v1-recipe.md §4 fallback 2, the whole family forwards
  to V2 as a set. Nested expansion in Stage B would not be
  triggered because there is no per-model winner to anchor the
  ladder. Stage B becomes a no-op and V1's cell count caps at
  Stage A's 864 cells.

- **R3 — Cross-family bridge disagreement (3-of-4 on Direction, 1
  disagrees).** Per v1-recipe.md §4 fallback 6 / Known tension
  T-V1-04, V1 locks nothing on the cross-family axis and records
  the dissenting model as a V2 input. The sensitivity run (T-V1-02
  residual risk) reports the four per-model deltas un-aggregated
  so V2 can condition its model-dominance search on the dissent
  pattern.

- **R4 — Missing artefacts block a size rule.** Per v1-recipe.md §4
  fallback 4, missing
  `results/compiled_results_aggregated.csv` or the RRA consensus
  ranking drops the affected recipes from the tournament and emits
  `tensions.md`; V1 does NOT fall back to a default size. A V0-era
  default would silently re-enable the cell-averaged SE bug (T-V1-01
  risk surface) by collapsing the variance structure.

- **R5 — Warm-start coupling mis-registers cohort drift.** Per
  [[condensates/optuna-warm-start-couples-gates]] and the
  `warm_start_top_k / T = 5 / 200 = 2.5%` seed fraction, coupling
  hazard is bounded. V1's ledger records the source gate slug and
  dataset-fingerprint match; if a mismatch is detected at cell
  start, the cell falls back to cold-start and logs a tension
  pointing at the coupling condensate.

- **R6 — Nested expansion (Stage B) step-down under-estimates refit
  on RF.** Per [[condensates/nested-expansion-vs-refit]] boundary
  conditions, RF with `max_depth=None` is the flexible case where
  step-down bias can exceed 0.01 AUROC at p=4 from an 8p parent.
  If Stage B's RF ladder winner's p is ≤ 6, V1 MUST run a single
  refit-per-size cell at the winning p as a confirmation (condensate
  actionable rule); if the refit AUROC exceeds the step-down by ≥
  0.01 with CI excluding 0, the lock defers to the refit value and
  a tension is filed.

- **R7 — Prediction P1 fails with Direction (plateau beats sig).**
  This would be a genuine falsifier of the
  [[condensates/three-criterion-size-rule]] claim that the
  significance core equivalences the unanimous plateau on the celiac
  dataset. `decision.md` would record Direction (not Equivalence),
  parsimony would not be invoked, and the condensate's evidence row
  for celiac at v1-recipe would flip from "rule confirmed" to
  "rule challenged." The tension detector should route this to
  `projects/celiac/tensions/rule-vs-observation/three-criterion-size-rule.md`.

- **R8 — Prediction P2 fails with Dominance (all 4 bridges agree on
  shared > MS).** Unexpected but not prohibited. If all four
  per-model bridges return the same Direction on AUROC, V1 locks
  cross-family toward shared-panel and retires the MS family for
  V2's purposes. The failure would challenge
  [[condensates/nested-expansion-vs-refit]]'s prediction that tree
  models benefit from model-specific ordering; the tension routes
  there.

---

## Open questions surfaced by the advisor

These are not falsifiers; they are methodology gaps the advisor noticed
while writing this ledger. They do not block V1 entry but should be
addressed before V1 decision.

1. **Stage B triggering rule is stated informally above.** The advisor
   defers RF/XGBoost ladders until Stage A reveals which ordering
   (OOF vs RFE) survives; v1-recipe.md §2.2 permits all 18 ladder
   recipes in the naive tournament. If the protocol is authoritative
   and the advisor's staging is descriptive-only, Stage B must run
   in full regardless of Stage A. This ledger treats the staging as
   a compute-ordering heuristic, not a protocol divergence — all 34
   recipes remain in scope; only the execution ORDER is proposed.
2. **Cross-family bridge dataframe structure is not mirrored in
   `observation.md` schema.** The tension detector needs a known
   key to attach a cross-family tension to
   [[condensates/feature-selection-needs-model-gate]] vs
   [[condensates/nested-expansion-vs-refit]] depending on which
   prediction failed; v1-recipe.md §3.4 does not specify the
   `observation.md` key.
3. **Warm-start source-gate metadata verification is a pre-run
   check, not a ledger field.** This ledger records the source
   (`v0-strategy`, fingerprint `sha256:8c02e33cc...`) in the
   rulebook entries list above, but the protocol does not say
   whether `ledger.md` must echo the source's storage hash. Added
   here conservatively so the tension detector can match.
