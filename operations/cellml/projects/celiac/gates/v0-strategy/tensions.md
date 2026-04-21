---
gate: v0-strategy
project: celiac
created: "2026-04-20"
---

# V0 Tensions

Auto-populated tensions (delta between ledger predictions and observation
metrics) do not apply here — this is a retrospective reconstruction with no
formal pre-registered predictions. Instead, this file lists V0-relevant
architectural tensions logged in `operations/cellml/DECISION_TREE_AUDIT.md`
that bear on V0 decisions or on how V0 outputs propagate downstream, plus
the rb-v0.2.0 axis-redefinition tension introduced by the Strategy C
rulebook rewrite.

Each bullet is a summary + reference. Do not duplicate full tension text;
see `DECISION_TREE_AUDIT.md` for the authoritative record.

## Rulebook-version tensions

- **V0 axis redefinition (rb-v0.1.0 → rb-v0.2.0).** The V0 imbalance-handling
  axis was redefined from `control_ratio ∈ {1, 2, 5}` (numeric level within
  the downsample family) to `imbalance_probe ∈ {none, downsample_5, cw_log}`
  (categorical probe across three families: none / downsample / weight).
  V0's output lock is now `imbalance_family` (categorical), not a numeric
  ratio. This retrospective ledger, decision, and observation were
  originally written against `rb-v0.1.0` axis semantics; they have been
  updated in place to reflect rb-v0.2.0.

  **Claim-level consequence.** Level-specific claims from Gen 1 (e.g.,
  "`control_ratio = 5` preserves AUROC while cutting compute") are now
  re-interpreted as FAMILY-level claims (e.g., "the `downsample` family
  preserves AUROC at the `downsample_5` probe point"). The numeric level
  within the downsample family (1 vs 2 vs 5) is NO LONGER a V0 output; it
  is deferred to V3. Downstream gates (V1-V6) that previously inherited a
  numeric `train_control_per_case = 5` lock now inherit a categorical
  `imbalance_family = downsample` lock and operate at the Gen 1 probe
  level (`downsample_5`) as a placeholder until V3 refines it.

  **Source.** `rulebook/CHANGELOG.md` rb-v0.2.0 entry. Route candidate:
  `tensions/rulebook-version/` (new category for rulebook-version-driven
  axis redefinitions; if the category does not yet exist, promote to
  `tensions/rule-vs-observation/` pending a schema revision).

## V0-relevant tensions from DECISION_TREE_AUDIT.md

- **§3.5 V0 prevalent fraction coverage** — original V0 spec tested only
  `prevalent_frac` in {0.5, 1.0}, missing lower values (0.2, 0.3). Resolved
  2026-04-08 by expanding to {0.25, 0.5, 1.0}. This V0 retrospective
  reflects the expanded (post-resolution) axis. See `DECISION_TREE_AUDIT.md`
  §3.5 and §5 row "P3: V0 prevalent_frac". Route candidate:
  `tensions/gate-decisions/`.

- **§1.1 Parsimony principle applied unevenly** — does not directly affect
  V0 (V0 uses a Dominance criterion, not a parsimony tiebreaker), but V0
  locks are consumed by V2/V3 which depend on parsimony orderings. Relevant
  because the V0 decision rule ("if strategy winner is the same across all
  4 models, lock both") is implicitly a Direction-or-Equivalence test with
  no explicit parsimony fallback. See `DECISION_TREE_AUDIT.md` §1.1. Route
  candidate: `tensions/rule-vs-observation/`.

- **§1.2 Statistical rigor inconsistent** — V0 as specified in
  `MASTER_PLAN.md` does not explicitly require bootstrap CIs on the
  strategy × imbalance-probe comparison. If V0 is to be re-run under the
  current SCHEMA rubric, it inherits the requirement for 1000-resample
  bootstrap CIs on AUROC/PR-AUC/Brier per `rulebook/SCHEMA.md` "Fixed
  falsifier rubric" section. See `DECISION_TREE_AUDIT.md` §1.2. Route
  candidate: `tensions/rule-vs-observation/`.

- **§2.2 V0 -> V1 transition is correct** — audit validates the V0-first
  ordering (training strategy defines data distribution; must be locked
  before recipe selection). No tension; recorded here as an anchor that V0
  is structurally in the right place. See `DECISION_TREE_AUDIT.md` §2.2
  row "V0 -> V1 Correct".

- **§3.6 HPO seed variance** — V0 uses 50 Optuna trials per cell (reduced
  from the main factorial's 200). The DECISION_TREE_AUDIT disposition
  records this as "documented as acceptable" because the sampler seed is
  derived from the split seed, giving fair within-seed comparison. Note
  for retrospective use: V0's narrower HPO budget means some cells may not
  have converged to the same quality as main factorial cells will — the
  Dominance claim is therefore on relative performance under a fixed
  (narrower) HPO budget, not on fully-tuned performance. See
  `DECISION_TREE_AUDIT.md` §3.6. Route candidate: `tensions/gate-decisions/`.

- **§3.7 No tree cross-validation** (historical) — resolved 2026-04-08
  via 20 selection / 10 confirmation seed split. V0 is in the selection
  bucket (seeds 100-119). The V5 confirmation protocol was retrofitted
  after V0 submitted, so V0 predictions were not held out from selection.
  See `DECISION_TREE_AUDIT.md` §3.7 and §5 row "P3: Tree cross-validation".
  Relevant because any claim that the V0 lock generalizes must be confirmed
  on seeds 120-129 at V5; the V0 retrospective cannot claim confirmation.
  Route candidate: `tensions/gate-decisions/`.

## Not from DECISION_TREE_AUDIT.md, but worth noting

- **Dataset-total mismatch** — README.md reports 43,960 subjects (148
  incident + 150 prevalent + controls). SCHEMA.md expects
  `n_cases + n_controls`. With incident-only (n_cases=148) and
  n_controls=43662 from ADR-003, the sum is 43,810 — short of 43,960 by
  150, consistent with the 150 prevalent cases being excluded from the
  fingerprint count. Route candidate: `tensions/data-quality/`. No action
  needed at V0 lock time; documented for fingerprint rebuild.
