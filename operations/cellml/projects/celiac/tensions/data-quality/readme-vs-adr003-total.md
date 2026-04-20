---
type: tension
severity: low
category: data-quality
discovered: 2026-04-20
discovered_by: celiac-scaffold-agent
status: open
affects:
  - "operations/cellml/projects/celiac/dataset/fingerprint.yaml"
  - "operations/cellml/projects/celiac/gates/v0-strategy/ledger.md"
references:
  - "README.md"
  - "docs/adr/ADR-003-control-downsampling.md"
  - "docs/adr/ADR-002-prevalent-train-only.md"
---

# README dataset total and ADR-003 incident-only counts disagree by exactly the 150 prevalent cases

## Contradiction

The top-level `README.md` advertises the celiac dataset as **43,960 subjects (148 incident + 150 prevalent cases + controls)**. `ADR-003-control-downsampling.md` states the pre-downsampling inputs are **148 incident cases and 43,662 controls** — summing to 43,810. The residual of 150 equals exactly the prevalent-case count from `ADR-002`. The two numbers refer to *different population bases* (full cohort vs incident-only training pool) but the README phrasing reads as if 43,960 is the fingerprint-relevant `n_cases + n_controls`.

## Evidence

- `README.md` §"Results: Celiac Disease" (line 127):
  "Dataset: 43,960 subjects (148 incident cases, 150 prevalent cases), 2,920 proteins, plus demographics..."
- `docs/adr/ADR-003-control-downsampling.md` §"Decision" (lines 9-10):
  "Original: 148 incident cases, 43,662 controls (~1:300)"
- `docs/adr/ADR-002-prevalent-train-only.md` §"Decision" (lines 7-10):
  "Add prevalent cases (n=150) to TRAIN only at 50% sampling... Incident cases (n=148), Prevalent cases (n=150)"
- Arithmetic: 43,960 − (148 + 43,662) = 150, matching `n_prevalent`.

## Why it matters

`rulebook/SCHEMA.md` §"Dataset fingerprint" expects `n_cases` and `n_controls` to sum to the cohort total that gets hashed into the sha256 fingerprint. The README's headline number (43,960) is a superset of whatever the V0 fingerprint will commit to, because incident-only V0 splits exclude the 150 prevalent cases entirely (per ADR-002, prevalent goes to TRAIN only, which is orthogonal to the fingerprint cohort definition). A future gate ledger citing "the README cohort" vs "the fingerprint cohort" will diverge by 150 subjects, and any downstream comparison against a cohort-total figure will be off by the same 150.

Additionally: a reader who takes 43,960 as authoritative and then reads ADR-002 might incorrectly conclude 43,960 includes both incident *and* prevalent splits, when in fact prevalent cases only ever appear in TRAIN.

## Proposed resolution

Human should decide the canonical `n_cases` / `n_controls` convention for `fingerprint.yaml` and add a one-sentence note reconciling the two figures. Two low-cost options:

1. **Fingerprint is incident-only.** Write `n_cases: 148, n_controls: 43662` into `fingerprint.yaml`. Add a README footnote: "43,960 = full cohort (incident + prevalent + controls); V0 training-pool fingerprint uses incident-only and is 43,810."
2. **Fingerprint includes prevalent.** Write `n_cases: 298, n_controls: 43662` and update ADR-002 references to clarify that the prevalent-TRAIN subsetting is a downstream split-generation step, not a fingerprint-level exclusion.

Do not modify `ADR-003` — its counts are internally consistent with its scope ("incident only, pre-downsampling").

## Status

Open — awaiting human adjudication.
