---
type: equation
symbol: "s_i"
depends_on: []
computational_cost: "O(P * M)"
assumptions:
  - "per-model rankings are comparable after normalization by list length"
  - "missing from a model means absent from that model's ranked survivor set (post stability filter)"
  - "higher OOF-importance rank is better (rank 1 is the top protein)"
failure_modes:
  - "aggregating across models that have not passed the Stage 1 model gate dilutes signal with noise rankings"
  - "using raw ranks without list-length normalization biases against models with shorter ranked lists"
  - "not all ADR-004 ingredients feed this score: stability is a hard pre-filter, RFE sizes the panel, drop-column is post-hoc interpretation"
---

# Geometric mean of normalized reciprocal ranks with missing penalty

## Statement

For protein $i$ across models $m \in \mathcal{M}$:

$$s_i = \left( \prod_{m \in \mathcal{M}} \frac{N_m}{r_{i,m}} \right)^{1/|\mathcal{M}|}$$

where:

- $r_{i,m}$ is the OOF-importance rank of protein $i$ in model $m$ (rank 1 = top)
- $N_m = |\text{ranked list from model } m|$ after the stability hard filter
- If protein $i$ is absent from model $m$'s ranked list, set $r_{i,m} = r_\text{missing} = \max_m N_m + 1$
- The final panel is the top-$N$ proteins sorted by $s_i$ descending

The $N_m / r_{i,m}$ factor is the normalized reciprocal rank: ranks near the top of a longer list earn a larger contribution. Geometric mean (rather than arithmetic mean) penalizes disagreement multiplicatively — one model ranking a protein near the bottom pulls $s_i$ down sharply.

## Derivation

Normalization by $N_m$ is required because models contribute ranked lists of different lengths after the stability filter. Without normalization, a protein ranked 10th in a 50-item list would be treated the same as 10th in a 2920-item list. Dividing by list length converts rank position to a relative quantile-like quantity.

This is a **geometric-mean consensus aggregation** (commonly called RRA, though not formal Kolde RRA) — the rank-aggregation form used in gene-list robustness literature. It is NOT the formal beta-model Robust Rank Aggregation of Kolde et al. (2012); the celiac pipeline uses the geometric-mean approximation because it avoids the RRA beta null-distribution assumption and is sufficient for panel construction without distributional p-values. Throughout this rulebook and the cel-risk codebase any bare reference to "RRA" means this geometric-mean consensus aggregator defined here, not the Kolde et al. formal aggregator; the slug `consensus-rank-aggregation` encodes this disambiguation explicitly. See ADR-004 alternatives section.

The missing-protein penalty of $r_\text{missing} = \max_m N_m + 1$ ensures that absence from a model's list always contributes a worse normalized rank than the lowest-ranked item in any list. A protein seen in 1 of 4 models receives three penalty terms in the product, which drives $s_i$ toward zero — the desired behavior for cross-model consensus.

## Boundary conditions

- **Stage 1 gate required.** Valid only when $\mathcal{M}$ is restricted to models that pass the permutation test model gate (ADR-004 Stage 1). Aggregating across noise models is unsafe. See [[condensates/feature-selection-needs-model-gate]].
- **Post stability filter.** The ranked lists fed in must already be hard-filtered by selection stability (e.g. selection frequency $\ge 0.90$). See [[condensates/stability-as-hard-filter]].
- **Panel sizing is independent.** RFE+Pareto determines $N$; this equation produces the ordering only. Size is not derived from $s_i$.
- **Correlation clustering applies downstream.** Top candidates by $s_i$ are correlation-clustered; one representative per cluster enters the final panel. Raw $s_i$ ordering is not the final panel.
- **Drop-column is post-hoc.** Essentiality $\Delta$AUROC per cluster is interpretation on the already-selected panel, not an input to $s_i$.

## Worked reference

Three models on 5 proteins (rank 1 = top). Stability filter already applied — P5 missing from Model C.

| Protein | Model A rank (N=5) | Model B rank (N=5) | Model C rank (N=4) |
|---|---|---|---|
| P1 | 1 | 2 | 1 |
| P2 | 2 | 1 | 3 |
| P3 | 3 | 4 | 2 |
| P4 | 4 | 3 | 4 |
| P5 | 5 | 5 | missing |

$r_\text{missing} = \max(5, 5, 4) + 1 = 6$.

$s_{P1} = (5/1 \cdot 5/2 \cdot 4/1)^{1/3} = (50)^{1/3} \approx 3.68$
$s_{P2} = (5/2 \cdot 5/1 \cdot 4/3)^{1/3} = (16.67)^{1/3} \approx 2.56$
$s_{P3} = (5/3 \cdot 5/4 \cdot 4/2)^{1/3} = (4.17)^{1/3} \approx 1.61$
$s_{P4} = (5/4 \cdot 5/3 \cdot 4/4)^{1/3} = (2.08)^{1/3} \approx 1.28$
$s_{P5} = (5/5 \cdot 5/5 \cdot 4/6)^{1/3} = (0.67)^{1/3} \approx 0.87$

Final order: P1, P2, P3, P4, P5. The P5 missing penalty pushes it below P4 despite P4 also being bottom-ranked in two models.

## Sources

- ADR-004 (three-stage feature selection; Stage 3 consensus).
- `analysis/src/ced_ml/features/consensus/aggregation.py::_compute_geometric_mean_score` — reference implementation.
- Kolde et al. (2012). Robust rank aggregation for gene list integration. Bioinformatics 28(4):573-580. (Formal RRA; this equation is the simpler geometric-mean approximation.)

## Used by

- [[condensates/feature-selection-needs-model-gate]]
- [[condensates/stability-as-hard-filter]]
<!-- TODO: verify slug exists after batch merge — protocols/v1-recipe.md should cite this equation once batch 3 lands -->
