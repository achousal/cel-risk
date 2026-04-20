---
type: tension
severity: medium
category: data-quality
discovered: 2026-04-20
discovered_by: celiac-scaffold-agent
status: open
affects:
  - "operations/cellml/projects/celiac/gates/v0-strategy/ledger.md"
  - "operations/cellml/projects/celiac/gates/v0-strategy/observation.md"
references:
  - "docs/adr/ADR-002-prevalent-train-only.md"
  - "operations/cellml/MASTER_PLAN.md"
  - "operations/cellml/DECISION_TREE_AUDIT.md"
---

# ADR-002 records prevalent_train_frac=0.5 as decided but V0 factorial tests three values

## Contradiction

`ADR-002-prevalent-train-only.md` is authored with status `Accepted` and documents `prevalent_train_frac: 0.5` as *the* chosen value — the ADR is framed as a locked architectural decision, not a hypothesis under test. The V0 Training Strategy Gate in `MASTER_PLAN.md` explicitly reopens this parameter and sweeps over three levels `{0.25, 0.5, 1.0}`. ADR-002 has not been amended to mark itself as "Revisited by V0" or to list the expanded axis, despite MASTER_PLAN's own ADR Map table explicitly flagging ADR-002 as revisited by V0.

## Evidence

- `docs/adr/ADR-002-prevalent-train-only.md` §"Decision" (lines 7-10):
  "Add prevalent cases (n=150) to TRAIN only at **50% sampling**... TRAIN positives: 148 incident + 75 prevalent (50% sampled) = 223 total."
  Status field (line 3): "**Status:** Accepted | **Date:** 2026-01-20". No "Revisited" or "Superseded" note.
- `operations/cellml/MASTER_PLAN.md` §"Factorial Scope" → "V0: Training Strategy Gate" (lines 188-189):
  "Training strategy | IncidentOnly, IncidentPlusPrevalent, PrevalentOnly ; Prevalent fraction | **0.25, 0.5, 1.0** (only for IncidentPlusPrevalent)"
- `operations/cellml/MASTER_PLAN.md` §"ADR Map" (line 255):
  "ADR-002 | Prevalent train-only, frac=0.5 | **Yes** — V0 gate"
- `operations/cellml/MASTER_PLAN.md` §"Decision Architecture" (line 98):
  "Prevalent fraction levels: 0.25, 0.5, 1.0 (expanded from 0.5, 1.0)"
- `operations/cellml/projects/celiac/gates/v0-strategy/tensions.md` §"V0-relevant tensions" first bullet confirms the expansion was resolved 2026-04-08.

## Why it matters

Documentation is authoritative for external reviewers (and for the LLM advisor at gate entry). An advisor reading ADR-002 in isolation will predict V0's search space is `{0.5}` (a singleton, which cannot produce a Direction or Equivalence claim by the SCHEMA rubric) rather than the actual three-level factor. This can:

1. Corrupt the V0 ledger's cited rulebook entries — the ledger would cite ADR-002 as "locked at 0.5" when V0 is in fact testing whether 0.5 is Pareto-dominant.
2. Cause the LLM to flag the V0 observation as contradicting ADR-002 (triggering a spurious rulebook-update tension in `rule-vs-observation/`) when in reality ADR-002 is simply stale.
3. Confuse downstream V1-V5 gates that consume the V0 lock: the lock might be any of `{0.25, 0.5, 1.0}`, not just 0.5.

## Proposed resolution

Human should update `ADR-002-prevalent-train-only.md` to add a "Revisited by" line citing V0 and listing the three tested values. One sentence is enough:

> "**Revisited by V0 Gate (2026-04-08):** V0 factorial tests `prevalent_train_frac ∈ {0.25, 0.5, 1.0}`. The value locked at V0 supersedes the 0.5 recorded here."

Leave the original decision block intact (it documents the initial 2026-01-20 reasoning — historical record). Do not remove the 50% number; append the expansion as an addendum with its own date.

## Status

Open — awaiting human adjudication.
