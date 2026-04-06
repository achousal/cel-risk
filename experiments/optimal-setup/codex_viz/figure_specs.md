# Figure Specs

These specs turn the brainstorm into concrete plot designs.

## 1. `svm_panel_size_tendency`

Purpose: show whether larger panels improve performance enough to justify complexity.

- Plot type: line plot with points and uncertainty ribbon
- Data:
  - x = panel size
  - y = mean validation performance
  - group = `SVM`
  - ymin/ymax = cross-validation interval
- Optional:
  - vertical marker at chosen operating panel size
  - annotation for plateau onset

Questions answered:
- Where does `SVM` peak?
- Where do gains flatten?
- Is the optimum narrow or broad?

## 2. `svm_marker_pattern_heatmap`

Purpose: show which markers recur in the best `SVM` panels.

- Plot type: binary or frequency heatmap
- Data:
  - rows = markers
  - columns = top-ranked `SVM` panels or panel sizes
  - fill = selected/not selected or inclusion frequency
- Sort:
  - markers by frequency or hierarchical clustering
  - panels by score

Questions answered:
- Which markers are persistent anchors?
- Which markers only appear in larger panels?
- Are there obvious redundant blocks?

## 3. `top_svm_panels_ranked`

Purpose: make the best candidate panels easy to compare.

- Plot type: ranked dot plot or lollipop chart
- Data:
  - y = panel identifier
  - x = validation score
  - color = panel size
  - label = marker composition or shorthand ID

Questions answered:
- How close are the top solutions?
- Are several panels effectively tied?
- Does a smaller panel achieve near-best performance?

## 4. `all_models_panel_size_comparison`

Purpose: compare whether the observed trend is specific to `SVM`.

- Plot type: faceted line plots
- Data:
  - x = panel size
  - y = mean validation score
  - facet = model
  - ribbon = uncertainty
- Design:
  - keep consistent y-axis if possible
  - highlight `SVM` in title or with stronger color

Questions answered:
- Do models agree on the best panel size?
- Does `SVM` dominate or merely compete?
- Are some models unstable at small or large panels?

## 5. `model_by_size_heatmap`

Purpose: compressed comparison across algorithms and panel sizes.

- Plot type: heatmap
- Data:
  - rows = model
  - columns = panel size
  - fill = validation performance

Questions answered:
- Which model-size combinations are strongest?
- Are there clear bands of stable performance?
- Does performance depend more on model choice or panel size?

## 6. `marker_cooccurrence_network`

Purpose: explore reusable marker combinations among strong panels.

- Plot type: network graph
- Data:
  - nodes = markers
  - edges = co-occurrence count or normalized overlap in top panels
  - node size = selection frequency
  - edge width = co-occurrence strength

Questions answered:
- Which markers travel together?
- Are there modular marker communities?
- Is the best panel built from a stable core plus optional extensions?

## 7. `search_route_alluvial`

Purpose: show how top panels evolve as panel size grows.

- Plot type: alluvial / Sankey
- Data:
  - stage = panel size
  - strata = marker sets or panel families
  - flow = retained/expanded panel routes

Questions answered:
- Which small panels persist into larger winners?
- Are there multiple viable construction routes?
- Does optimization converge or branch?

## Decision Rule For What To Show

Use this simple filter:

- If the main decision is panel size: prioritize line plots.
- If the main decision is marker composition: prioritize heatmaps.
- If the main decision is algorithm choice: prioritize faceted model comparisons.
- If the main decision is optimization logic: prioritize route/network figures.

## Suggested First Pass

Build these first:

1. `svm_panel_size_tendency`
2. `svm_marker_pattern_heatmap`
3. `all_models_panel_size_comparison`

Then decide whether the route figure should be:

- `marker_cooccurrence_network` for composition insight
- `search_route_alluvial` for sequential optimization insight
