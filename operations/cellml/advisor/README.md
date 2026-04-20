# cellml-advisor

LLM advisor CLI for the CellML factorial gate workflow. Scaffold v0.1.0.

## Purpose

`cellml-advisor` is a sibling package to `cellml`. It reads the rulebook
(`operations/cellml/rulebook/`) and prior gate state from a project
(`operations/cellml/projects/<name>/gates/...`) and emits templates that scaffold
LLM reasoning at each factorial gate. It drafts `ledger.md` at gate entry,
drafts `decision.md` once `observation.md` exists, and detects tensions when
predicted and observed deltas disagree under the fixed falsifier rubric.

The advisor is **strictly read-only** on cellml state. It never writes into
`analysis/`, `operations/cellml/sweeps/`, or any cellml run artifact. Its only
writes are the ledger/decision/tension template files it is asked to emit, and
only under `operations/cellml/projects/<name>/gates/<gate>/`.

## Installation (future)

```bash
pip install -e operations/cellml/advisor
```

The scaffold currently uses a flat layout: package code lives directly under
`advisor/cellml_advisor/`. A future restructure may move to
`advisor/src/cellml_advisor/` as the package grows; the entry point and import
path (`cellml_advisor`) remain stable.

## CLI command reference

| Command | Purpose |
| --- | --- |
| `cellml-advisor rulebook show [--filter TYPE]` | List rulebook entries, optionally filtered by type (`equation`, `condensate`, `protocol`) |
| `cellml-advisor rulebook validate` | Walk wiki-links, report dangling/cycle errors |
| `cellml-advisor ledger draft <project> <gate> [--out PATH]` | Draft `ledger.md` for a gate from the protocol + prior locks |
| `cellml-advisor ledger validate <ledger_path>` | Check required sections, frontmatter, links, falsifier rubric types |
| `cellml-advisor decision draft <project> <gate>` | Draft `decision.md` from the ledger + observation (facts only; reasoning is LLM-filled) |
| `cellml-advisor tension detect <project> <gate>` | Diff predicted vs observed, classify under the rubric, emit `tensions.md` |

## Data flow

```
rulebook/  ---reads--->  advisor  ---emits--->  projects/<name>/gates/<gk>/ledger.md
                            |
                            v
                       (cellml runs)
                            |
                            v
                  observation.md exists
                            |
                            v
advisor  ---emits--->  decision.md
                       tensions.md
```

## Boundary statement

The advisor is read-only on cellml: it does not modify sweeps, configs, or
results, and it never imports `ced_ml`. It reads the rulebook and prior gate
artifacts; it writes only to `projects/<name>/gates/<gate>/` files that the
rulebook schema explicitly assigns to the advisor (`ledger.md` drafts,
`decision.md` drafts, `tensions.md`). Observations are produced by
`cellml-reduce`, not by the advisor.

## Version

`0.1.0 (scaffold)` — every function raises `NotImplementedError`. CLI
subcommands echo a scaffold message. Real behavior lands in v0.2.
