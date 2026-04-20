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
  - "operations/cellml/MASTER_PLAN.md"
---

# MASTER_PLAN header claims V0 submitted but body still lists V0 as ready-to-submit with blocker

## Contradiction

The header of `MASTER_PLAN.md` (line 4) claims V0 has already been submitted to the scheduler with specific job IDs 237012328–237012447. The body of the *same file*, in the "Current Phase" block (lines 77-83) and the "Execution Roadmap" §"Phase 1: V0 Gate" (lines 327-335), still describes V0 as "**Ready to submit**" with an explicit blocker ("splits overlay still needed"). Both claims cannot be simultaneously true — either V0 has been submitted (in which case the splits overlay was resolved and Phase 1 has advanced) or V0 is still pending (in which case the header is aspirational or stale).

## Evidence

- `operations/cellml/MASTER_PLAN.md` header (line 4):
  "**Status:** V0 gate submitted (job IDs 237012328–237012447). Vanillamax discovery (10 seeds) running."
- `operations/cellml/MASTER_PLAN.md` §"Current Phase" (lines 77-79):
  "**Phase:** Pre-execution — all infrastructure built and validated, V0 gate ready to submit
   **Blocker:** `config_gen.py` splits overlay for V0 (still needed), data path for `ced derive-recipes`
   **Next action:** Implement V0 splits overlay, generate V0 configs, submit scout batch"
- `operations/cellml/MASTER_PLAN.md` §"Current Phase" table (line 83):
  "V0 Gate | **Ready to submit** | Optuna storage + warm-start infra built; splits overlay still needed"
- `operations/cellml/MASTER_PLAN.md` §"Execution Roadmap" Phase 1 (lines 327-331):
  "1. ~~Extend `config_gen.py` with splits overlay support~~ (still needed for V0 splits)
   2. Generate V0 configs: 120 cells ...
   3. Submit V0 batch: `bash submit_experiment.sh --experiment v0_gate ...`"

  Only step 1 has a strikethrough (and even that is annotated "still needed"); steps 2 and 3 have no completion mark.

## Why it matters

The V0 gate's `ledger.md` (if/when written) must cite a `rulebook_snapshot` that was frozen at V0 entry, not at decision time (per `rulebook/SCHEMA.md` §Versioning). If the header is authoritative, V0 entry has already happened and the snapshot is locked in the submission commit. If the body is authoritative, V0 has not entered and the ledger can still cite a newer rulebook snapshot. The LLM advisor routing to V0 needs to know which state applies before choosing what to surface at the gate — and which `observation.md` TODOs are live vs still-in-ledger-phase.

Concretely:

- If submitted: the live TODO is "wait for 237012328–237012447, then compile and write observation.md".
- If not submitted: the live TODO is still "implement splits overlay and generate configs".

These trigger entirely different advisor behaviors.

## Proposed resolution

Human should reconcile by editing `MASTER_PLAN.md` so the header and body agree. Two canonical paths:

1. **If V0 was submitted:** update lines 77-83 to "Phase: V0 running (seeds 100-119)" and "Blocker: none; awaiting job completion". Strike through the Roadmap Phase 1 steps that are now complete. Cite job IDs in the Current Phase block as well, not just the header.
2. **If V0 was not yet submitted:** revert the header status line to match the body, e.g. "V0 gate ready to submit; splits overlay pending." The job IDs 237012328–237012447 then refer to some other submission (vanillamax discovery?) and should be relocated to the sentence that actually owns them.

Do not edit either ADR files — the contradiction is entirely within MASTER_PLAN.md.

## Status

Open — awaiting human adjudication.
