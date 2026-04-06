# Optimize-Panel Visualization Brainstorm

This folder captures candidate visualization routes for the `cel-risk` optimize-panel experiment.

## Core Questions

1. How does performance change as panel size increases?
2. Is the preferred panel specific to `SVM` or stable across models?
3. Which marker-selection routes recur among top-performing panels?
4. Where do gains plateau, and where does complexity stop paying off?

## Recommended Story Order

1. Start with a simple `SVM`-only performance view.
2. Add a heatmap to show repeated marker patterns.
3. Expand to all-model comparison.
4. Add a route view to explain how top panels are assembled.

## Visualization Routes

### Route A: SVM-first main story

Best if `SVM` is the main scientific or operational model.

- Figure 1: line plot of performance vs panel size for `SVM`
- Figure 2: heatmap of marker inclusion frequency across top `SVM` panels
- Figure 3: ranked dot plot of the top `SVM` candidate panels

Why use it:
- Cleanest read
- Makes plateau and diminishing returns obvious
- Avoids clutter from weak comparator models

### Route B: all-model comparison

Best if the question is robustness across algorithms.

- Figure 1: faceted line plots by model, x = panel size, y = performance
- Figure 2: heatmap with rows = models and columns = panel size
- Figure 3: delta-to-best plot showing how far each model is from the best score at each panel size

Why use it:
- Shows whether `SVM` is genuinely preferred or just one acceptable option
- Makes model-specific sweet spots visible

### Route C: marker-pattern emphasis

Best if the marker composition matters more than the algorithm.

- Figure 1: binary heatmap of top panels, rows = markers, columns = candidate panels
- Figure 2: co-occurrence heatmap of marker pairs in top panels
- Figure 3: cluster map of markers or panels by similarity

Why use it:
- Exposes reusable marker blocks
- Helps identify redundant or mutually reinforcing markers

### Route D: route exploration / optimization path

Best if the experiment has a sequential search or nested panel construction logic.

- Figure 1: alluvial/Sankey from panel size 3 -> 5 -> 7 -> 9
- Figure 2: marker co-occurrence network among top panels
- Figure 3: trajectory plot of search iterations vs score
- Figure 4: bump chart of rank changes as panel size increases

Why use it:
- Explains how candidate panels evolve
- Highlights stable routes rather than only end-state winners

## Chart Types By Question

### Heatmaps for patterns

Use when looking for repeated structure.

Good encodings:
- rows = markers, columns = top panels, fill = selected or not
- rows = markers, columns = panel sizes, fill = selection frequency
- rows = models, columns = panel sizes, fill = AUROC or AUPRC

Best for:
- recurring marker blocks
- stable markers
- model-size performance patterns

### Line graphs for tendencies

Use when the x-axis has order.

Good encodings:
- x = panel size, y = AUROC
- one line per model
- uncertainty ribbon = CV fold variation or bootstrap CI

Best for:
- plateau detection
- diminishing returns
- comparing slope and stability across models

### SVM-only view

Use if `SVM` is the primary decision-making model.

Recommended components:
- mean performance line with uncertainty ribbon
- point labels for best 1-3 panel sizes
- companion heatmap for selected markers in top panels

### Showing all models

Use if method comparison is part of the story.

Recommended formats:
- faceted small multiples rather than one crowded plot
- shared y-axis if scales are comparable
- highlight `SVM` consistently with color or annotation

## Practical Recommendation

For the clearest first pass:

1. Make `SVM` the main figure set.
2. Put all-model comparison in a second figure or supplement.
3. Use one route-exploration figure only if the optimization process itself matters.

## Minimal Figure Set

If time is limited, produce these three:

1. `svm_panel_size_tendency`
2. `svm_marker_pattern_heatmap`
3. `all_models_panel_size_comparison`

## Nice-to-Have Figure Set

If exploration is the goal rather than a final manuscript:

1. `svm_panel_size_tendency`
2. `top_panel_binary_heatmap`
3. `marker_cooccurrence_network`
4. `model_by_size_heatmap`
5. `search_route_alluvial`
