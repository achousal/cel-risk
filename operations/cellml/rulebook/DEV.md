# Cellml Rulebook — Developer Guide

Companion to [SCHEMA.md](SCHEMA.md). SCHEMA.md is normative (what must be true);
this doc is practical (how to work with the system). Read this if you are
picking up rulebook or advisor work in a later session.

**Last substantive update**: 2026-04-20 (rb-v0.1.0 cut).

---

## 1. What this is

An LLM-steerable grammar for factorial gate sequencing. The cel-risk factorial
used to be brute-force across 1,566 cells; the rulebook lets an LLM advisor
read prior gate evidence + methodological claims and narrow the search space
before each gate runs.

Three decoupled layers:

| Layer | Purpose | Files |
|---|---|---|
| **Rulebook** | Dataset-agnostic ML methodology (equations, behavioral claims, gate playbooks) | `operations/cellml/rulebook/` |
| **Projects** | Dataset-specific findings + gate ledgers | `operations/cellml/projects/<name>/` |
| **Advisor** | Sibling CLI reading rulebook + project state, emitting templates | `operations/cellml/advisor/` |

Runtime executor is `cellml` itself (unchanged). Advisor **reads cellml
outputs**; it never writes into cellml state.

---

## 2. The loop (per gate)

```
┌──────────┐   reads    ┌─────────┐   emits    ┌───────────┐
│ rulebook │ ─────────▶ │ advisor │ ─────────▶ │ ledger.md │  (pre-run)
└──────────┘            └─────────┘            └───────────┘
                             ▲                      │
                             │                      ▼
                             │               ┌─────────────┐
                             │               │    cellml   │  (executes)
                             │               └─────────────┘
                             │                      │
                             │                      ▼
                             │               ┌──────────────────┐
                             │               │ observation.md   │  (facts)
                             │               └──────────────────┘
                             │                      │
                             │                      ▼
                             │               ┌───────────────┐
                             └──reads────────│ decision.md   │  (post-run)
                                             └───────────────┘
                                                    │
                                                    ▼
                                             ┌────────────┐
                                             │ tensions.md│  (deltas)
                                             └────────────┘
```

**Per-gate sequence**:
1. Advisor reads `protocols/v{k}-{name}.md` + all prior `decision.md` +
   `dataset/fingerprint.yaml`
2. Advisor writes `ledger.md` with predictions (fixed rubric claim types) and
   axis restrictions
3. cellml runs the reduced factorial
4. Reduction tool emits `observation.md` with metrics + CIs
5. Advisor writes `decision.md` comparing predictions to observations
6. Tension detector writes `tensions.md` for predicted-vs-observed deltas that
   suggest rulebook updates

**Falsifier rubric** (governs every claim): Direction / Equivalence / Dominance
/ Inconclusive. See [SCHEMA.md § Fixed falsifier rubric](SCHEMA.md).

---

## 3. Where things are

```
operations/cellml/
├── CELLML_CLI.md                # cellml CLI reference (existing)
├── DESIGN.md                    # factorial design (existing)
├── DECISION_TREE_AUDIT.md       # known V0-V5 tensions (existing)
├── MASTER_PLAN.md               # experiment index (existing)
├── rulebook/
│   ├── SCHEMA.md                # normative spec
│   ├── DEV.md                   # this file
│   ├── equations/               # 12 files — math/stats definitions
│   ├── condensates/             # 24 files — behavioral claims about models
│   └── protocols/               # 7 files — gate playbooks v0 through v6
├── projects/
│   ├── README.md
│   ├── registry.csv             # project → active rulebook version
│   └── celiac/
│       ├── dataset/fingerprint.yaml
│       ├── gates/v0-strategy/   # retrospective ledger/observation/decision/tensions
│       ├── gates/v1-recipe/     # live advisor dry-run ledger only (no obs yet)
│       ├── tensions/            # data-quality, gate-decisions, rule-vs-observation
│       └── rulebook-snapshot/
└── advisor/                     # Python package (scaffold, v0.1.0)
    ├── README.md
    ├── pyproject.toml
    ├── cellml_advisor/
    │   ├── cli.py               # cellml-advisor entry point
    │   ├── models.py            # Pydantic models
    │   ├── rulebook_loader.py   # STUB
    │   ├── ledger_writer.py     # STUB
    │   ├── decision_writer.py   # STUB
    │   ├── tension_detector.py  # STUB
    │   └── rubric.py            # STUB
    └── tests/test_smoke.py      # imports test only
```

---

## 4. Current state (as of rb-v0.1.0)

| Area | Status |
|---|---|
| SCHEMA.md | Stable. Covers equations, condensates, protocols, gate lifecycle, falsifier rubric, per-protocol metric overrides, informational gates |
| Equations (12) | Migrated from ADRs 001-011 |
| Condensates (24) | All at `status: provisional, confirmations: 1`. Promotion to `established` requires ≥3 non-overlapping cohorts |
| Protocols (7) | v0-strategy through v5-confirmation (locking) + v6-ensemble-comparison (informational) |
| Celiac V0 | Retrospective ledger/observation/decision/tensions. Rulebook-snapshot bound to `rb-v0.1.0` |
| Celiac V1 | **Live advisor dry-run ledger only**. No observation.md yet — awaiting V1 execution via cellml |
| Advisor CLI | Scaffold — all function bodies raise `NotImplementedError("scaffold: implement in v0.2")` |
| Tag `rb-v0.1.0` | Pushed to origin, anchors at commit `fd8796e` |
| Merge to main | `cellml-cli` merged via `--no-ff` commit; pushed |

---

## 5. Action items (priority order)

### Tier 1 — Unblock V1 execution

- [ ] **Run V1 gate via cellml** on the celiac (recipe × calibration × weighting × downsampling) axes per `protocols/v1-recipe.md` §2. Produces `projects/celiac/gates/v1-recipe/observation.md`
- [ ] **Compute missingness profile** for `fingerprint.yaml` (currently `TODO: compute from source parquet`). One parquet query

### Tier 2 — User review of data-quality tensions

Five files under `projects/celiac/tensions/data-quality/`. Each has a
`## Proposed resolution` section. Sorted by severity:

| Severity | File | Action |
|---|---|---|
| HIGH | `control-ratio-semantics-double-use.md` | Sharpen ADR-003 to disambiguate V0 split-gen vs V3 training-dynamics ratios |
| MED | `adr002-vs-v0-prevalent-frac-spec.md` | Update ADR-002 to list {0.25, 0.5, 1.0} |
| MED | `adr007-vs-v6-restructure-stacking-status.md` | Add Superseded/Revisited note to ADR-007 |
| MED | `master-plan-header-vs-body-submission-status.md` | Reconcile MASTER_PLAN header vs body |
| LOW | `readme-vs-adr003-total.md` | Accept 43,810 or clarify 43,960 |

### Tier 3 — Rulebook PATCH revisions

From the V1 dry-run, `protocols/v1-recipe.md` has three protocol gaps flagged
in its "Open questions surfaced by the advisor" section:

- [ ] **Staging authority** — does the advisor have license to propose
      execution order (e.g., two-stage: deferred nested expansions)?
- [ ] **Cross-family bridge observation schema** — specify key structure
      `observation.md` should emit so cross-family tensions route to the right
      condensate
- [ ] **Warm-start source-gate echo** — mandate `ledger.md` echoes the V0
      Optuna storage hash for downstream tension matching

### Tier 4 — Advisor v0.2 implementation (work package)

Every stub in `operations/cellml/advisor/cellml_advisor/*.py` raises
`NotImplementedError`. Build order (each is independent enough to TDD):

1. `rulebook_loader.load_rulebook()` — parse markdown frontmatter in
   equations/condensates/protocols, return `Rulebook` container
2. `rubric.classify_claim()` — CI-based Direction/Equivalence/Dominance/
   Inconclusive classifier; honor per-protocol `metric_overrides`
3. `ledger_writer.draft_ledger()` — template generator from protocol +
   prior decisions + fingerprint
4. `ledger_writer.validate_ledger()` — schema conformance + wiki-link
   resolution + falsifier-rubric claim-type usage
5. `decision_writer.draft_decision()` — reads ledger + observation,
   emits decision template with predictions-that-held/failed sections
6. `tension_detector.detect_tensions()` — diff predicted vs observed,
   classify by rubric, write `tensions.md`

Replace `tests/test_smoke.py` with real coverage at each step.

### Tier 5 — Multi-cohort promotion (long horizon)

- [ ] 24 condensates at `provisional`. Need ≥3 non-overlapping cohorts to
      promote to `established`. Run rulebook-driven pipeline on a second
      dataset (IBD, T1D, MI, or similar)
- [ ] Each promotion wave bumps `rb-v0.x.y` at MINOR level
- [ ] Bootstrap simulation falsifier for `threshold-on-val-not-test`
      (archived post-quarantine cohorts) — deferred until promotion attempted

---

## 6. How-to guides

### Add a new condensate

1. Pick a propositional title (must parse as "This claim argues that
   [title]"). Example ok: "Per-fold calibration introduces optimism bias".
   Not ok: "Calibration leakage"
2. Pick a short slug (hyphenated, no version numbers). Example:
   `calib-per-fold-leakage`
3. Create `rulebook/condensates/{slug}.md` with SCHEMA frontmatter:
   - `status: provisional`, `confirmations: 1`
   - Falsifier using rubric claim types only
   - `depends_on` links verified to exist on disk
   - `evidence` table with dataset / n / delta / source-gate
4. Update topic map or protocol that will cite it
5. Bump rulebook at MINOR: `rb-v0.{n+1}.0`

### Add a new protocol

1. File at `rulebook/protocols/v{k}-{name}.md`
2. Frontmatter: `gate`, `inputs`, `outputs` (locks or report), `axes_explored`,
   `axes_deferred`, `depends_on`, optional `informational: true`, optional
   `metric_overrides`
3. Five required body sections: Pre-conditions / Search space / Success
   criteria / Fallbacks / Post-conditions
4. Success criteria MUST use the fixed falsifier rubric exclusively
5. Include a `## Known tensions` addendum for anything in
   DECISION_TREE_AUDIT.md the protocol acknowledges
6. Every wiki-link must resolve — `ls` the condensate/equation dirs before
   writing

### Add a new project (e.g., a second cohort)

1. Create `projects/<slug>/` tree matching the celiac layout (dataset/,
   gates/, tensions/, rulebook-snapshot/, archive/)
2. Write-once `dataset/fingerprint.yaml` (hash of raw data, n_cases,
   prevalence, feature count)
3. Append row to `projects/registry.csv`
4. Gate ledgers bind to the rulebook version at ENTRY (`rulebook_snapshot: "rb-vX.Y.Z"`)
5. Each condensate falsifier that fires on the new cohort bumps
   `confirmations` in the rulebook on the next patch

### Promote a condensate (provisional → established)

Requires:
- ≥3 dataset confirmations, non-overlapping cohorts
- Falsifier has never triggered
- Evidence table populated for each cohort

Workflow:
1. Verify confirmations count across `evidence` rows
2. Edit frontmatter: `status: established`
3. Rulebook bump at MINOR
4. Tag the new version

### Patch SCHEMA.md

- Adding new frontmatter field → MINOR bump if the field is optional,
  MAJOR if it changes required fields
- Adding new falsifier rubric claim type → MAJOR
- Tightening metric-override rules → MINOR
- Wording/clarity → PATCH

---

## 7. Conventions

### Slugs
- Short, hyphenated, lowercase, no version numbers
- Equations name the math: `perm-test-pvalue`, `brier-decomp`
- Condensates name the phenomenon: `calib-per-fold-leakage`,
  `downsample-preserves-discrimination-cuts-compute`
- Protocols: `v{k}-{name}` for sort order

### Wiki-links
- Format: `[[condensates/some-slug]]` (no `.md` suffix)
- Every link must resolve to a file on disk — run
  `ls operations/cellml/rulebook/{condensates,equations,protocols}/` before writing
- Protocol forward-references (e.g., v1 citing v3) are OK if the target
  exists

### Titles
- Propositional (statement, not topic)
- Composability test: "This claim argues that [title]" must parse as a
  sentence

### Frontmatter
- All string values double-quoted
- Dates in ISO: `"2026-04-20"` or `"2026-04-20T14:32"`
- Lists either block-style (YAML preferred by the linter) or inline
  — be consistent per file

### Status lifecycle
- `provisional` (new) → `established` (≥3 cohorts) → `retired` (weakened
  by ≥5 cohorts or superseded)
- Never skip `provisional` unless the condensate is a restatement of an
  existing established claim

### Rulebook versioning
- Git tags `rb-v{MAJOR}.{MINOR}.{PATCH}`
- MAJOR: breaks falsifier rubric or file schema
- MINOR: new equations/condensates/protocols; retirements; SCHEMA optional
  field additions
- PATCH: wording, citations, evidence additions

---

## 8. Gotchas

- **Advisor is read-only on cellml.** It never writes to `analysis/`,
  `operations/cellml/sweeps/`, or `results/`. Strict separation.
- **TEST quarantine is sacred.** Any protocol move that looks at
  held-out TEST data before V5 is a rubric violation. `threshold-on-val-not-test`
  forbids this at the claim level.
- **V1 SE bug is mandated-fixed.** `protocols/v1-recipe.md` §3.3 forbids
  cell-averaged SE; must use paired bootstrap over outer-fold seeds.
- **V0 control_ratio ≠ V3 downsampling.** V0 locks split-generation ratio;
  V3 explores within-training downsampling. They compose multiplicatively.
  See `condensates/nested-downsampling-composition.md` and the HIGH-severity
  data-quality tension
- **V6 is informational only.** Any protocol change that adds locking to V6
  is a schema violation (`informational: true` constraint)
- **All condensates are provisional until promotion.** Never treat a
  `provisional` condensate as authoritative in a PATCH-level decision
- **Celiac is incident-only at V0.** `n_cases = 148` (incident), not
  including the 150 prevalent cases that are used only in
  IncidentPlusPrevalent training strategies. `n_cases + n_controls = 43,810`;
  README reports 43,960 total subjects
- **Parsimony is a tiebreaker, not a preference.** Only apply parsimony
  ordering when Equivalence has been established per rubric. See
  `condensates/parsimony-tiebreaker-when-equivalence.md`
- **Rulebook snapshot binds at gate ENTRY.** `ledger.md`'s
  `rulebook_snapshot` is frozen when the ledger is written. Subsequent
  rulebook changes do not retroactively apply

---

## 9. Glossary

- **Gate** — a single decision node (V0-V6). Locks one or more axes or
  emits a report (informational)
- **Recipe** — panel composition + ordering + size. 34 recipes derived
  from T1/T2 trunks per `DESIGN.md`
- **Cell** — one full-factorial combination at a gate. V0 has 120 cells;
  V1 has up to 1,566
- **Substrate** — the prediction vector a utility metric is computed on.
  V3 requires prevalence-adjusted OOF-posthoc substrate; see
  `v3-utility-provenance-chain`
- **Dominance** — rubric claim type: Direction holds independently on
  every compared axis
- **Inconclusive** — rubric claim type: neither Direction nor Equivalence
  met. **A valid gate outcome**, not a failure
- **Warm-start** — V0 Optuna scout finds hyperparameters; V1+ inherit via
  shared Optuna storage. Couples downstream gates to V0; see
  `optuna-warm-start-couples-gates`
- **Informational gate** — protocol with `informational: true`; reports
  verdict, does not lock. Canonical: v6-ensemble-comparison
- **Fingerprint** — write-once dataset hash + metadata in
  `projects/<name>/dataset/fingerprint.yaml`. Binds a project to a
  specific data state

---

## 10. Resuming work — checklist for a new session

1. Read this file (DEV.md) and [SCHEMA.md](SCHEMA.md)
2. Check current rulebook tag: `cd /path/to/cel-risk && git tag | grep rb-`
3. Check git status; `cellml-cli` is the active dev branch
4. Pick from § 5 Action items. Tier 1 first unless blocked
5. Before writing rulebook content, `ls` the three subdirs to know what
   slugs exist
6. Every new condensate must have a falsifier in rubric claim types
7. Every new protocol must cite only slugs verified on disk
8. Test the advisor package with `pytest operations/cellml/advisor/tests/`
   before and after your changes
9. Commit convention: `feat(cellml): ...` for rulebook additions,
   `fix(cellml): ...` for wiki-link repairs and TODO rebinds,
   `feat(cellml/advisor): ...` for advisor code

---

## 11. Questions this doc doesn't answer

- How to wire the advisor into cellml's HPC orchestration (not yet
  designed)
- How tension-promotion actually decides between weaken/retire (§3.5 of
  the "rulebook updates" section of SCHEMA.md is minimal)
- Whether `informational: true` gates should count against
  `confirmations` in the condensates they test (open — `v6` doesn't at
  present)
- How to evolve the rubric itself as evidence accumulates (requires MAJOR
  version bump)

If you resolve any of these in a session, update this doc.
