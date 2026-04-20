# Rulebook Schema

Normative spec for rulebook entries and project gate artifacts. All tooling that
reads or writes these files must comply. Paths are relative to
`operations/cellml/`.

## Versioning

Rulebook releases are git tags `rb-v{MAJOR}.{MINOR}.{PATCH}`.

- MAJOR: breaks falsifier rubric or file schema
- MINOR: new equations, condensates, or protocols; retirements
- PATCH: wording, citations, evidence additions

Every gate ledger MUST cite `rulebook_snapshot: "rb-v{x.y.z}"` in frontmatter.
Snapshot is bound at gate ENTRY, never at decision time.

---

## File types

### Equations — `rulebook/equations/{slug}.md`

Propositional title. Slug is short; title carries the statement.

- Title: "Permutation test p-value with +1 correction" (statement)
- Slug: `perm-test-pvalue.md` (short, searchable)
- Bad title: "Permutation tests" (topic label, not a statement)

Frontmatter:

```yaml
---
type: equation
symbol: "p_perm"
depends_on: []                        # [[equation]] wiki-links
computational_cost: "O(B*C_inner)"
assumptions: ["..."]
failure_modes: ["..."]
---
```

Body sections (required, in order):

1. **Statement** — LaTeX or code
2. **Derivation** — cite sources
3. **Boundary conditions** — when it does and doesn't apply
4. **Worked reference** — small numeric example

### Condensates — `rulebook/condensates/{slug}.md`

Propositional title: a claim about model, data, or feature behavior.

- Title: "Per-fold calibration introduces optimism bias when calibrator fits on hyperparameter-selected folds"
- Slug: `calib-per-fold-leakage.md`

Frontmatter:

```yaml
---
type: condensate
depends_on: ["[[equations/...]]"]
applies_to:
  - "models: [RF, XGBoost]"
  - "n < 1000"
  - "prevalence < 0.1"
status: provisional | established | retired
confirmations: 1                      # promote to established at >=3
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "+0.018 AUROC"
    date: "2026-04-12"
falsifier: |
  If ΔAUROC between per_fold and oof_posthoc stays within ±0.005 across
  all four models at n>=500, this condensate is weakened.
---
```

Body:

1. **Claim** — restate title, 1-3 sentence elaboration
2. **Mechanism** — link to `[[equations/...]]`
3. **Actionable rule** — what the factorial or search space must or must not do
4. **Boundary conditions** — where the claim stops holding
5. **Evidence** — table: dataset | n | observed-delta | source-gate

Promotion `provisional -> established`: at least 3 dataset confirmations on
non-overlapping cohorts, falsifier never triggered.

### Protocols — `rulebook/protocols/{slug}.md`

Playbook at a single factorial gate. One protocol per gate version.

Slug convention: `v{k}-{name}.md` (e.g. `v0-strategy.md`, `v1-recipe.md`).

Frontmatter:

```yaml
---
type: protocol
gate: v0-strategy
inputs: ["dataset/fingerprint.yaml", "prior_gate: null"]
outputs: ["locks: [training_strategy, control_ratio]"]
axes_explored: ["incident_only", "train_control_per_case"]
axes_deferred: ["model", "calibration", "weighting"]
depends_on: ["[[condensates/...]]", "[[equations/...]]"]
metric_overrides:                     # OPTIONAL — see "Per-protocol metric overrides"
  REL:
    direction_margin: 0.005
    equivalence_band: [-0.01, 0.01]
    justification: "REL on rare-event cohorts (<0.01 prevalence) lives at ~1e-5 scale; the default 0.02 margin exceeds all expected signal"
---
```

Body:

1. **Pre-conditions** — what must be true before the gate runs
2. **Search space** — axes and allowed values, cite condensates
3. **Success criteria** — uses fixed falsifier rubric below
4. **Fallbacks** — when no axis value satisfies the rubric
5. **Post-conditions** — what locks, what advances to next gate

---

## Fixed falsifier rubric

These thresholds are defaults. Protocols may declare per-metric overrides —
see "Per-protocol metric overrides" below.

All LLM predictions and all gate decisions use exactly these claim types.

| Claim type | Criterion |
|---|---|
| **Direction** (X > Y) | \|Δ\| ≥ 0.02 AND 95% bootstrap CI excludes 0 |
| **Equivalence** (X ≈ Y) | \|Δ\| < 0.01 AND 95% bootstrap CI ⊂ [−0.02, 0.02] |
| **Dominance** (X ≻ Y multi-axis) | Direction criterion holds independently on each axis |
| **Inconclusive** | Neither Direction nor Equivalence met |

Metric-specific rules (non-negotiable):

- AUROC, PR-AUC, Brier: 95% bootstrap CI over 1000 resamples of outer folds
- Counts (panel size, cell count): exact comparison, no CI

**"Inconclusive" is a valid gate outcome** and triggers the protocol fallback
path. Gate decisions MUST log the claim type actually made.

---

## Per-protocol metric overrides

Protocols MAY override the default rubric thresholds on a per-metric basis when
the default is inappropriate for the metric's scale. Overrides are NOT license
to loosen criteria — they are a calibration of the rubric to the metric.

### Override requirements (all mandatory)

1. Declared in protocol frontmatter under `metric_overrides` with per-metric
   fields: `direction_margin`, `equivalence_band`, and a `justification` string.
2. The justification must cite either (a) a condensate or equation with a
   scale-of-signal argument, or (b) measured distribution of the metric on
   historical data.
3. Overrides affect ONLY the declared metric. All other metrics in the gate
   use rubric defaults.
4. Overrides do NOT propagate to downstream gates. Each gate declares its own.
5. `ledger.md` at gate entry MUST repeat the active override values in its
   frontmatter for audit (prevents silent drift if SCHEMA or protocol changes
   between ledger write and decision write).

### What overrides cannot do

- Eliminate the need for a CI (only margins and bands change; CI exclusion/
  inclusion logic is invariant)
- Move a metric from "measured with CI" to "point estimate" (this is a rubric
  violation regardless of the override)
- Set direction_margin lower than the measurement error floor for the metric
  on the cohort (implementers must compute this; no universal minimum)

### Precedent and review

New overrides require a rulebook version bump at MINOR (not PATCH) to signal
the rubric calibration change. The override's justification becomes part of
the rulebook-update audit trail.

---

## Gate lifecycle — `projects/<name>/gates/v{k}-{slug}/`

Four files, written in order. Once `observation.md` is written, `ledger.md` is
immutable by convention; the git hash provides audit.

### `ledger.md` — PRE-run reasoning (LLM or human)

```yaml
---
gate: v1-recipe
project: celiac
rulebook_snapshot: "rb-v0.3.2"
dataset_fingerprint: "sha256:9f2b..."
created: "2026-04-20T14:32"
author: llm-advisor | human
---

## Hypothesis
## Search-space restriction
## Cited rulebook entries
## Falsifier criteria (must use rubric claim types)
## Predictions with criteria
## Risks & fallbacks
```

### `observation.md` — POST-run measurements

Compiled metrics with CIs. Facts only, no reasoning. Generated by
`cellml-reduce` CLI, not the LLM.

### `decision.md` — POST-run reasoning

```yaml
---
gate: v1-recipe
observed_at: "2026-04-20T18:45"
---

## Predictions that held
## Predictions that failed
## Actual claim type (per rubric)
## Locks passed forward
## Predictions for v_{k+1}
```

### `tensions.md` — delta log

Auto-populated by the tension-detector when `ledger` predictions disagree with
`observation` metrics. Each tension is a candidate rulebook update. Routes to
`projects/<name>/tensions/rule-vs-observation/`.

---

## Dataset fingerprint — `projects/<name>/dataset/fingerprint.yaml`

```yaml
project: celiac
created: "2026-04-20"
hash: "sha256:9f2b..."                # over concat(splits/seed_*.csv)
n_cases: 148
n_controls: 43662
prevalence: 0.00337
n_features: 2920
missingness_profile:
  pct_features_gt_10pct_missing: 0.12
cohort: "UK Biobank incident celiac"
platform: "Olink Explore 3072"
```

Write-once. If inputs change, a new project slug is required.

---

## Rulebook updates

1. Tension accumulates in `projects/<name>/tensions/rule-vs-observation/`
2. At least 3 dataset confirmations on non-overlapping cohorts -> LLM drafts a
   rulebook PR
3. Human reviews; merge bumps `rb-v{x.y.z}` (MINOR for add, PATCH for wording)
4. `CHANGELOG.md` entry auto-links the condensate to its triggering tensions

---

## Slug conventions

- Equations: short noun-phrase of the computation (`perm-test-pvalue`, `brier-decomp`)
- Condensates: short phrase naming the phenomenon (`perm-validity-full-pipeline`, `calib-per-fold-leakage`)
- Protocols: `v{k}-{name}` for sort order
- Projects: lowercase cohort name (`celiac`, `ibd`, `t1d`)
