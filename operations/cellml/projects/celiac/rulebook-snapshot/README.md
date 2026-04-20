# rulebook-snapshot/

Binds this project's currently executing gate run to a specific rulebook
tag. Per `rulebook/SCHEMA.md`, every gate ledger cites
`rulebook_snapshot: "rb-v{x.y.z}"` in frontmatter, and snapshots are bound
at gate ENTRY (never at decision time).

## Files (expected)

- `hash.txt` — two lines:
  1. `rb-v{MAJOR}.{MINOR}.{PATCH}` — the git tag
  2. `<commit sha>` — the commit that tag points to

Currently absent. Will be written when the first rulebook tag
(`rb-v0.1.0`) is cut after ADR -> condensate migration completes.

## Current state

- V0 gate ledger cites `rulebook_snapshot: "rb-v0.0.0-unfinalized"` as a
  placeholder. This value intentionally does NOT match any existing tag —
  it flags that V0 ran before the rulebook was formalized.
- `projects/registry.csv` records `active_rulebook_version =
  rb-v0.0.0-unfinalized` for the celiac project.
- Once `rb-v0.1.0` is tagged:
  1. Write `hash.txt` with the tag and commit sha.
  2. Update `projects/registry.csv` row for celiac.
  3. Update the V0 ledger frontmatter TODO to cite `rb-v0.1.0` (the
     retrospective notice at the top of the ledger preserves the audit
     that V0 ran before the rulebook existed).
  4. From V1 onward, bind at gate entry, never at decision time.

## Binding policy

A rulebook-snapshot binding is immutable once a gate has written its
`observation.md`. If the rulebook advances during gate execution, the
next gate binds to the new tag; the in-flight gate stays on its entry
tag. This preserves the audit chain: every gate decision is reproducible
against the exact rulebook text that governed it.
