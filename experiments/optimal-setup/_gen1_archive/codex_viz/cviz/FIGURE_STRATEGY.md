# Figure Strategy

## Cornerstone Ideas

1. Separate the truth panel from the operating panel.
2. Show optimization as a constrained tradeoff, not a leaderboard.
3. Distinguish invariant core markers from optional extensions.
4. Show where models agree and where operating choices diverge.
5. Make panel growth interpretable at milestone sizes.

## Essential Figures

### 1. Decision Frontier

Primary figure. Show AUROC vs panel size with:

- full-model reference
- non-inferiority margin
- seed variability
- accepted vs rejected points

Question answered:
Where does added complexity stop paying off?

### 2. Acceptance Landscape

Explain why candidate sizes passed or failed:

- stability gate
- non-inferiority gate
- overall acceptance

Question answered:
Why was a panel accepted or rejected?

### 3. Core-to-Extension Persistence

Focus on milestone panel sizes rather than every size:

- 4 proteins: truth-level core
- 7 proteins: consensus panel
- 8 and 10 proteins: operating range
- larger accepted panel(s): extension zone

Question answered:
Which proteins are anchors, and which are added for operational lift?

Implementation note:
Use the pathway-order milestone panels as the shared operating trajectory.

### 4. Cross-Model Agreement

Show agreement across LR, SVM, RF, and XGBoost for the top proteins.

Question answered:
Is the biological signal stable across model families?

### 5. Panel Growth Route

Show how the panel grows from core to operating region.

Question answered:
What is retained, and what gets added, as the panel expands?

## Recommended Story Order

1. Conservative core exists.
2. Performance saturates in a broader operating region.
3. Acceptance depends on both performance and stability.
4. Larger panels extend the core rather than replacing it.
5. The core signal is shared across model families.
