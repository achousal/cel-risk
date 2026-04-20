# tensions/

Project-scoped tension log. Three subfolders separate tension types so
rulebook-facing tensions (which can trigger PR-style updates) do not mix
with dataset-specific or decision-specific frictions.

## Subfolders

### `data-quality/`

Dataset-level frictions. Missing covariates, QC surprises, Olink LOD
behavior, split-generation asymmetries, demographic imbalance discovered
after fingerprint freeze. These do NOT trigger rulebook updates — they
usually trigger a new project slug (e.g. `celiac-v2`) or a corrective
residualization.

### `gate-decisions/`

Per-gate decision frictions. Example: "V2 parsimony ordering chose LR_EN
but XGBoost had higher AUROC by Δ=0.015 with overlapping CIs." Records
the human/LLM judgment call and the evidence. These inform but do not
drive rulebook updates — they are project history.

### `rule-vs-observation/`

**Rulebook-facing tensions.** Each tension is a candidate rulebook update.
When a ledger prediction cites a condensate and the observation violates
the falsifier criterion, the tension routes here. Per `rulebook/SCHEMA.md`
§Rulebook updates, >=3 non-overlapping-cohort confirmations trigger an
LLM-drafted rulebook PR.

## Writing a tension

One file per tension, named `YYYY-MM-DD-short-slug.md`. Frontmatter:

```yaml
---
tension_type: data-quality | gate-decisions | rule-vs-observation
gate: v0-strategy | v1-recipe | ...
condensate: "[[condensates/...]]"    # rule-vs-observation only
direction: supports | weakens | dissolves | triggers
observed_delta: "Δ_metric = value, 95% CI [lo, hi]"
---
```

Body (at minimum): what was predicted, what was observed, how the falsifier
rubric classifies the result (Direction / Equivalence / Inconclusive /
Dominance), what the tension implies for the rulebook entry cited.
