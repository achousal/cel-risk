---
type: tension
severity: high
category: data-quality
discovered: 2026-04-20
discovered_by: celiac-scaffold-agent
status: open
affects:
  - "operations/cellml/projects/celiac/dataset/fingerprint.yaml"
  - "operations/cellml/projects/celiac/gates/v0-strategy/ledger.md"
  - "operations/cellml/projects/celiac/gates/v3-imbalance/ledger.md"
  - "operations/cellml/projects/celiac/gates/v3-imbalance/observation.md"
references:
  - "docs/adr/ADR-003-control-downsampling.md"
  - "docs/adr/ADR-002-prevalent-train-only.md"
  - "operations/cellml/MASTER_PLAN.md"
  - "[[condensates/nested-downsampling-composition]]"  # TODO: condensate not yet written
---

# train_control_per_case is used as both a V0 split-generation lock and a V3 training-dynamics axis without disambiguation

## Contradiction

The parameter name `train_control_per_case` (equivalently: "control ratio", "downsampling") appears in two distinct gate contexts, at two distinct abstraction levels, with no rulebook entry or ADR that explicitly disambiguates them:

1. **V0 (split-generation, locked):** per `MASTER_PLAN.md` V0 spec, V0 sweeps training strategy × control ratio, and the V0 decision rule locks `(strategy, control_ratio)` forward. This `control_ratio` governs *which control subjects appear in the TRAIN fold at all* — it is a dataset-level partition parameter applied once per seed before any model sees data.
2. **V3 (training-dynamics, explored):** per MASTER_PLAN Decision Architecture, V3 is "Imbalance (JOINT weighting × downsampling)" with a 3×3 grid and the phrase "prefer (none, 1.0)" — here `downsampling` is a training-time resampling ratio applied within already-partitioned TRAIN data.

`ADR-003-control-downsampling.md` describes only the V0-flavor (split-level, 5:1 stratified by split). It does not mention the V3 usage at all. `MASTER_PLAN.md` Open Questions §1 even asks "Should V0 also test control ratio (train_control_per_case: 1, 5, 10), or keep that as supplementary sweep #9?" — the existence of this open question is evidence that the two usages are being conflated in the operator's head.

If the two knobs compose multiplicatively (and nothing in the docs says they don't), then a V3 cell running `downsampling=0.5` on top of a V0-locked `train_control_per_case=5` produces an effective ratio of `5 × 0.5 = 2.5:1`, not 5:1 and not 0.5:1. This is a high-severity foot-gun.

## Evidence

- `operations/cellml/MASTER_PLAN.md` §"Factorial Scope" V0 cell count (line 193):
  "5 strategies × **3 control ratios** × 4 models × 2 recipes = 120 cells"
- `operations/cellml/MASTER_PLAN.md` §"Decision Architecture" V3 line (lines 110-112):
  "V3: Imbalance (JOINT weighting × downsampling)
   3×3 grid, normalized utility (AUPRC + calibration)
   CI overlaps parsimony default → prefer (none, 1.0)"
- `operations/cellml/MASTER_PLAN.md` §"Discovery Methodology" Separation of Concerns (line 145-147):
  "- V0: training strategy + control ratio
   - V1: panel composition + ordering
   - V3: class weighting + downsampling"
  (V0 uses "control ratio"; V3 uses "downsampling" — different names, plausibly the same knob.)
- `operations/cellml/MASTER_PLAN.md` §"Open Questions" (line 432):
  "V0 scope: Should V0 also test control ratio (train_control_per_case: 1, 5, 10), or keep that as supplementary sweep #9?"
- `docs/adr/ADR-003-control-downsampling.md` §"Decision" + §"Evidence" only describes one application, at split-generation time via `SplitsConfig.train_control_per_case`. No V3 context.

## Why it matters

This is high severity because:

1. **Silent multiplicative composition.** If `splits_config.yaml` writes out a TRAIN fold with `train_control_per_case=5` and then V3's `downsampling=0.3` is applied inside the pipeline, the model trains on `5 × 0.3 = 1.5` controls per case — not 5, not 0.3. The V3 observation would then be conditional on the V0 lock in a way that V3's ledger cannot state unless this composition is explicit.
2. **Gate output mis-lock.** The V0 decision.md will write something like "lock `train_control_per_case=5`". V3 then re-explores a knob that is documented as locked. Either V3 is over-riding the V0 lock (in which case V0 did not actually lock anything and ADR-003 is misleading), or V3 is composing on top of the lock (in which case the V3 axis values need to be interpreted as *multipliers*, not *absolute ratios*).
3. **Rulebook impact.** Any condensate that emerges from V3 data (e.g., "downsampling to 1.0 beats 5.0 at low prevalence") will have an evidence table with a `downsampling` column whose numeric value is ambiguous — is 1.0 the effective ratio or the V3 multiplier?

The falsifier rubric in `rulebook/SCHEMA.md` requires a `|Δ|` threshold with bootstrap CIs; if the underlying axis semantics are ambiguous, two analysts can compute `Δ` against different baselines and reach opposite claim types.

## Proposed resolution

Human should:

1. **Rename in at least one place.** Keep `train_control_per_case` for the V0 split-level parameter (matches the existing `SplitsConfig` field). Rename V3's axis to something unambiguous — `train_downsample_factor` or `minority_class_upsample_ratio` — in `MASTER_PLAN.md` §"Decision Architecture" and §"Discovery Methodology".
2. **Write the composition condensate.** Author `rulebook/condensates/nested-downsampling-composition.md` (referenced in the frontmatter above as a forward-ref TODO) that states explicitly: "effective_ratio = V0_control_ratio × V3_downsample_factor", with a worked numeric example, a boundary condition (applies when V0 and V3 axes are both active), and a falsifier ("If two cells with identical effective_ratio but different V0/V3 decompositions produce ΔAUROC > 0.02, this composition rule is weakened"). Until that condensate exists, keep the forward-reference marker in this tension's frontmatter.
3. **Amend ADR-003** to add a "Scope" paragraph: "This ADR governs split-generation-time control subsetting only (applied in `splits.py` before model training). Training-time resampling is governed by V3 imbalance decisions and may compose with this ratio."

Do not attempt to modify the rulebook from this tension — that route is `rule-vs-observation/`, not `data-quality/`. The condensate work above is project-level specification work, not a rulebook promotion.

## Status

Open — awaiting human adjudication. Condensate `[[condensates/nested-downsampling-composition]]` not yet written; forward-ref marker retained.
