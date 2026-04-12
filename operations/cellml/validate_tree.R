#!/usr/bin/env Rscript
# validate_tree.R — V1–V5 statistical validation of factorial cell results (v2)
#
# Rewritten to address audit findings:
#   - V1: Stratified recipe comparison with seed-level SE (fixes cell-level SE bias)
#   - V2: AUROC vs Reliability Pareto with bootstrap CIs (fixes collinear AUROC-Brier axes)
#   - V3: Joint weighting x downsampling (fixes sequential V4-V5 interaction)
#   - V4: Calibration moved to end, reliability CI overlap + parsimony (fixes no-uncertainty ranking)
#   - V5: Seed-split confirmation (new level)
#
# Usage:
#   Rscript validate_tree.R \
#     --input results/factorial_compiled.csv \
#     --output results/factorial_validation.csv \
#     --n-seeds 30

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
input_path <- NULL
output_path <- NULL
n_seeds <- 30

i <- 1
while (i <= length(args)) {
  if (args[i] == "--input") {
    input_path <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--output") {
    output_path <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--n-seeds") {
    n_seeds <- as.integer(args[i + 1])
    i <- i + 2
  } else {
    i <- i + 1
  }
}

if (is.null(input_path)) {
  stop("Usage: Rscript validate_tree.R --input <compiled.csv> [--output <report.csv>] [--n-seeds 30]")
}

if (is.null(output_path)) {
  output_path <- sub("\\.csv$", "_validation.csv", input_path)
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df <- read.csv(input_path, stringsAsFactors = FALSE)
message(sprintf("Loaded %d rows from %s", nrow(df), input_path))

required_cols <- c(
  "recipe_id", "factorial_model", "factorial_calibration",
  "factorial_weighting", "factorial_downsampling",
  "summary_auroc_mean", "summary_auroc_std",
  "summary_prauc_mean", "summary_prauc_std",
  "summary_brier_score_mean",
  "summary_brier_reliability_mean"
)
missing <- setdiff(required_cols, names(df))
if (length(missing) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing, collapse = ", ")))
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_COMPLEXITY <- c(LR_EN = 1, LinSVM_cal = 2, RF = 3, XGBoost = 4)
CALIBRATION_COMPLEXITY <- c(logistic_intercept = 1, beta = 2, isotonic = 3)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#' Compute seed-level SE from per-cell summary_auroc_std values
#' SE = mean(std_across_seeds) / sqrt(n_seeds)
seed_se <- function(std_values, n_seeds) {
  mean(std_values, na.rm = TRUE) / sqrt(n_seeds)
}

#' Two-sided z-test for difference in means given SEs
z_test <- function(mean_a, se_a, mean_b, se_b) {
  diff_val <- mean_a - mean_b
  se_diff <- sqrt(se_a^2 + se_b^2)
  z <- if (se_diff > 0) diff_val / se_diff else 0
  p <- 2 * (1 - pnorm(abs(z)))
  ci_includes_zero <- (abs(diff_val) < 1.96 * se_diff)
  data.frame(diff = diff_val, se_diff = se_diff, z = z, p = p,
             ci_includes_zero = ci_includes_zero)
}

#' Extract panel size from recipe_id where possible
#' e.g., MS_oof_RF_p7 -> 7, R1_sig -> NA (needs metadata)
extract_panel_size <- function(recipe_id) {
  m <- regmatches(recipe_id, regexpr("_p(\\d+)", recipe_id))
  ifelse(length(m) > 0 & m != "",
         as.integer(sub("_p", "", m)),
         NA_integer_)
}

#' Check Pareto dominance: returns TRUE if point i is dominated by any other
#' Objectives: maximize obj1, minimize obj2
is_dominated <- function(obj1, obj2, idx) {
  others <- setdiff(seq_along(obj1), idx)
  if (length(others) == 0) return(FALSE)
  any(obj1[others] >= obj1[idx] &
      obj2[others] <= obj2[idx] &
      (obj1[others] > obj1[idx] | obj2[others] < obj2[idx]))
}

# ---------------------------------------------------------------------------
# V1: Recipe comparison — STRATIFIED
# Audit fix: seed-level SE instead of cell-level SE; stratified comparison
# to handle asymmetric cell counts between shared (108) and MS (27) recipes.
# ---------------------------------------------------------------------------
v1_recipe_comparison <- function(df, n_seeds) {
  # 1. Classify recipes
  df <- df %>%
    mutate(recipe_type = ifelse(grepl("^R[0-9]", recipe_id), "shared", "model_specific"))

  # 2. Within-type summary: recipe x model level
  #    For shared recipes: each recipe appears across 4 models, 27 downstream cells each

  #    For MS recipes: pinned to 1 model, 27 downstream cells
  recipe_model <- df %>%
    group_by(recipe_type, recipe_id, factorial_model) %>%
    summarise(
      auroc_mean = mean(summary_auroc_mean, na.rm = TRUE),
      auroc_seed_se = seed_se(summary_auroc_std, n_seeds),
      n_cells = n(),
      .groups = "drop"
    )

  # Recipe-level: average across models (for shared, this averages 4 model strata)
  recipe_summary <- recipe_model %>%
    group_by(recipe_type, recipe_id) %>%
    summarise(
      auroc_mean = mean(auroc_mean, na.rm = TRUE),
      auroc_se = mean(auroc_seed_se, na.rm = TRUE),
      n_models = n_distinct(factorial_model),
      n_cells_total = sum(n_cells),
      .groups = "drop"
    ) %>%
    mutate(panel_size = sapply(recipe_id, extract_panel_size))

  # 3. Pairwise z-tests WITHIN type
  run_pairwise <- function(sub) {
    sub <- sub %>% arrange(desc(auroc_mean))
    recipes <- sub$recipe_id
    if (length(recipes) < 2) return(data.frame())
    comparisons <- list()
    for (a in seq_along(recipes)) {
      for (b in seq_along(recipes)) {
        if (a >= b) next
        ra <- sub[sub$recipe_id == recipes[a], ]
        rb <- sub[sub$recipe_id == recipes[b], ]
        zt <- z_test(ra$auroc_mean, ra$auroc_se, rb$auroc_mean, rb$auroc_se)
        # If CI includes 0, prefer fewer proteins (parsimony)
        ps_a <- ra$panel_size
        ps_b <- rb$panel_size
        parsimony_winner <- NA_character_
        if (zt$ci_includes_zero && !is.na(ps_a) && !is.na(ps_b)) {
          parsimony_winner <- ifelse(ps_a <= ps_b, recipes[a], recipes[b])
        }
        comparisons[[length(comparisons) + 1]] <- data.frame(
          recipe_a = recipes[a],
          recipe_b = recipes[b],
          auroc_diff = zt$diff,
          se_diff = zt$se_diff,
          z_stat = zt$z,
          p_value = zt$p,
          ci_includes_zero = zt$ci_includes_zero,
          parsimony_winner = parsimony_winner,
          stringsAsFactors = FALSE
        )
      }
    }
    bind_rows(comparisons)
  }

  within_shared <- recipe_summary %>% filter(recipe_type == "shared")
  within_ms <- recipe_summary %>% filter(recipe_type == "model_specific")
  comp_shared <- run_pairwise(within_shared)
  comp_ms <- run_pairwise(within_ms)

  # Select best within each type: highest mean, or parsimony winner if tied
  pick_best <- function(sub, comps) {
    if (nrow(sub) == 0) return(character(0))
    sub <- sub %>% arrange(desc(auroc_mean))
    best <- sub$recipe_id[1]
    # Check if top is statistically tied with a smaller-panel recipe
    if (nrow(comps) > 0) {
      ties <- comps %>%
        filter((recipe_a == best | recipe_b == best) & ci_includes_zero)
      if (nrow(ties) > 0 && any(!is.na(ties$parsimony_winner))) {
        pw <- ties$parsimony_winner[!is.na(ties$parsimony_winner)]
        # If any parsimony winner differs from best, switch
        smaller <- pw[pw != best]
        if (length(smaller) > 0) {
          # Pick the one with smallest panel
          cand <- sub %>% filter(recipe_id %in% c(best, smaller))
          cand <- cand %>% arrange(panel_size, desc(auroc_mean))
          if (!is.na(cand$panel_size[1])) best <- cand$recipe_id[1]
        }
      }
    }
    best
  }

  best_shared <- pick_best(within_shared, comp_shared)
  best_ms <- pick_best(within_ms, comp_ms)

  # 4. Cross-type bridge: match on model
  #    Compare best shared recipe's per-model strata vs best MS recipe
  cross_type <- data.frame()
  if (length(best_shared) > 0 && length(best_ms) > 0) {
    # Determine which model the MS recipe is pinned to
    ms_models <- recipe_model %>%
      filter(recipe_id == best_ms) %>%
      pull(factorial_model)

    shared_strata <- recipe_model %>%
      filter(recipe_id == best_shared, factorial_model %in% ms_models)

    if (nrow(shared_strata) > 0) {
      ms_row <- recipe_model %>% filter(recipe_id == best_ms)
      # Fair comparison: same model, same 27 downstream cells
      zt <- z_test(
        shared_strata$auroc_mean[1], shared_strata$auroc_seed_se[1],
        ms_row$auroc_mean[1], ms_row$auroc_seed_se[1]
      )
      cross_type <- data.frame(
        shared_recipe = best_shared,
        ms_recipe = best_ms,
        bridge_model = ms_models[1],
        shared_auroc = shared_strata$auroc_mean[1],
        ms_auroc = ms_row$auroc_mean[1],
        auroc_diff = zt$diff,
        se_diff = zt$se_diff,
        z_stat = zt$z,
        p_value = zt$p,
        ci_includes_zero = zt$ci_includes_zero,
        stringsAsFactors = FALSE
      )
    }
  }

  # Combined recommendation
  recommended <- c(best_shared, best_ms)
  if (nrow(cross_type) > 0 && cross_type$ci_includes_zero[1]) {
    # Statistically tied: prefer smaller panel
    ps_s <- extract_panel_size(best_shared)
    ps_m <- extract_panel_size(best_ms)
    if (!is.na(ps_s) && !is.na(ps_m)) {
      recommended <- ifelse(ps_s <= ps_m, best_shared, best_ms)
    }
  } else if (nrow(cross_type) > 0) {
    # Significant difference: pick higher mean
    recommended <- ifelse(cross_type$auroc_diff[1] > 0, best_shared, best_ms)
  }

  # Build combined summary with recommendation flag
  combined_summary <- recipe_summary %>%
    mutate(recommended = recipe_id %in% recommended)

  list(
    within_shared = within_shared %>% mutate(best = recipe_id == best_shared),
    within_ms = within_ms %>% mutate(best = recipe_id == best_ms),
    comp_shared = comp_shared,
    comp_ms = comp_ms,
    cross_type = cross_type,
    summary = combined_summary,
    recommended_recipes = recommended
  )
}

# ---------------------------------------------------------------------------
# V2: Model — AUROC vs Reliability Pareto + Bootstrap CIs
# Audit fix: replaced collinear AUROC-Brier with AUROC-Reliability axes.
# Added bootstrap CI on Pareto dominance and defined parsimony tiebreaker.
# ---------------------------------------------------------------------------
v2_model_pareto <- function(df, recommended_recipes, n_boot = 1000, n_seeds) {
  set.seed(42)

  # Filter to recommended recipes from V1
  df_sub <- df %>% filter(recipe_id %in% recommended_recipes)

  # 1. Summarize per recipe x model
  model_summary <- df_sub %>%
    group_by(recipe_id, factorial_model) %>%
    summarise(
      auroc_mean = mean(summary_auroc_mean, na.rm = TRUE),
      reliability_mean = mean(summary_brier_reliability_mean, na.rm = TRUE),
      auroc_se = seed_se(summary_auroc_std, n_seeds),
      reliability_se = sd(summary_brier_reliability_mean, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  # 2. Bootstrap Pareto dominance
  models_list <- unique(model_summary$factorial_model)
  n_models <- length(models_list)

  # Per-recipe bootstrap
  results_list <- list()
  for (rec in unique(model_summary$recipe_id)) {
    ms <- model_summary %>% filter(recipe_id == rec)

    # Bootstrap: count how often each model is dominated / non-dominated
    dom_count <- setNames(rep(0, nrow(ms)), ms$factorial_model)
    nd_count <- setNames(rep(0, nrow(ms)), ms$factorial_model)

    for (b in seq_len(n_boot)) {
      # Perturb means by N(0, SE)
      auroc_pert <- ms$auroc_mean + rnorm(nrow(ms), 0, ms$auroc_se)
      rel_pert <- ms$reliability_mean + rnorm(nrow(ms), 0, ms$reliability_se)

      for (k in seq_len(nrow(ms))) {
        if (is_dominated(auroc_pert, rel_pert, k)) {
          dom_count[ms$factorial_model[k]] <- dom_count[ms$factorial_model[k]] + 1
        } else {
          nd_count[ms$factorial_model[k]] <- nd_count[ms$factorial_model[k]] + 1
        }
      }
    }

    ms <- ms %>%
      mutate(
        frac_dominated = dom_count[factorial_model] / n_boot,
        frac_nondominated = nd_count[factorial_model] / n_boot,
        robust_dominated = frac_dominated >= 0.95,
        robust_nondominated = frac_nondominated >= 0.50,
        complexity = MODEL_COMPLEXITY[factorial_model]
      )

    results_list[[rec]] <- ms
  }

  results <- bind_rows(results_list)

  # 3. Parsimony tiebreaker among non-dominated models
  results <- results %>%
    group_by(recipe_id) %>%
    mutate(
      # Among robust non-dominated models, prefer lower complexity
      selected = if (any(robust_nondominated)) {
        nd_set <- which(robust_nondominated)
        # Check CI overlap among non-dominated: if auroc CIs overlap, prefer simpler
        # Simple approach: pick lowest complexity among non-dominated
        factorial_model == factorial_model[nd_set[which.min(complexity[nd_set])]]
      } else {
        # Fallback: pick highest AUROC
        factorial_model == factorial_model[which.max(auroc_mean)]
      }
    ) %>%
    ungroup()

  results
}

# ---------------------------------------------------------------------------
# V3: Imbalance — JOINT weighting x downsampling
# Audit fix: resolves weighting and downsampling jointly instead of
# sequentially (old V4 then V5), avoiding missed interaction optima.
# ---------------------------------------------------------------------------
v3_imbalance_joint <- function(df, v2_selections, n_seeds) {
  set.seed(42)

  # Filter to selected recipe x model combos from V2
  selected_combos <- v2_selections %>%
    filter(selected) %>%
    select(recipe_id, factorial_model)

  df_sub <- df %>%
    semi_join(selected_combos, by = c("recipe_id", "factorial_model"))

  # 1. Summarize per recipe x model x calibration x (weighting, downsampling)
  #    9 combinations per calibration group
  imb_summary <- df_sub %>%
    group_by(recipe_id, factorial_model, factorial_calibration,
             factorial_weighting, factorial_downsampling) %>%
    summarise(
      auprc_mean = mean(summary_prauc_mean, na.rm = TRUE),
      reliability_mean = mean(summary_brier_reliability_mean, na.rm = TRUE),
      auprc_se = seed_se(summary_prauc_std, n_seeds),
      reliability_se = sd(summary_brier_reliability_mean, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  # 2. Normalize metrics within each recipe x model x calibration group
  # 3. Compute utility = 0.5 * auprc_norm + 0.5 * reliability_norm
  # 4. Bootstrap CI on utility
  n_boot <- 1000

  result_list <- list()
  group_keys <- imb_summary %>%
    distinct(recipe_id, factorial_model, factorial_calibration)

  for (g in seq_len(nrow(group_keys))) {
    gk <- group_keys[g, ]
    grp <- imb_summary %>%
      semi_join(gk, by = c("recipe_id", "factorial_model", "factorial_calibration"))

    if (nrow(grp) == 0) next

    # Normalize
    auprc_range <- diff(range(grp$auprc_mean))
    rel_range <- diff(range(grp$reliability_mean))

    grp <- grp %>%
      mutate(
        auprc_norm = if (auprc_range > 0) {
          (auprc_mean - min(auprc_mean)) / auprc_range
        } else { 0.5 },
        reliability_norm = if (rel_range > 0) {
          (max(reliability_mean) - reliability_mean) / rel_range
        } else { 0.5 },
        utility = 0.5 * auprc_norm + 0.5 * reliability_norm
      )

    # Bootstrap utility CIs
    boot_utilities <- matrix(NA, nrow = n_boot, ncol = nrow(grp))
    for (b in seq_len(n_boot)) {
      auprc_pert <- grp$auprc_mean + rnorm(nrow(grp), 0, grp$auprc_se)
      rel_pert <- grp$reliability_mean + rnorm(nrow(grp), 0, grp$reliability_se)

      # Re-normalize
      ap_range <- diff(range(auprc_pert))
      rp_range <- diff(range(rel_pert))
      ap_norm <- if (ap_range > 0) (auprc_pert - min(auprc_pert)) / ap_range else rep(0.5, nrow(grp))
      rp_norm <- if (rp_range > 0) (max(rel_pert) - rel_pert) / rp_range else rep(0.5, nrow(grp))
      boot_utilities[b, ] <- 0.5 * ap_norm + 0.5 * rp_norm
    }

    grp <- grp %>%
      mutate(
        utility_ci_lo = apply(boot_utilities, 2, quantile, probs = 0.025),
        utility_ci_hi = apply(boot_utilities, 2, quantile, probs = 0.975)
      )

    # 4. Decision: best utility. If CI overlaps with (none, 1.0) -> prefer parsimony
    best_idx <- which.max(grp$utility)

    # Check if "none" weighting + 1.0 downsampling exists (parsimony baseline)
    parsimony_idx <- which(
      grp$factorial_weighting == "none" &
      grp$factorial_downsampling == 1.0
    )

    selected_idx <- best_idx
    if (length(parsimony_idx) == 1 && parsimony_idx != best_idx) {
      # CI overlap test: does best's CI overlap with parsimony's CI?
      overlap <- grp$utility_ci_hi[parsimony_idx] >= grp$utility_ci_lo[best_idx] &
                 grp$utility_ci_hi[best_idx] >= grp$utility_ci_lo[parsimony_idx]
      if (overlap) {
        selected_idx <- parsimony_idx
      }
    }

    grp$selected <- seq_len(nrow(grp)) == selected_idx
    result_list[[g]] <- grp
  }

  bind_rows(result_list)
}

# ---------------------------------------------------------------------------
# V4: Calibration — Reliability with CI overlap + parsimony
# Audit fix: moved calibration to end (post-hoc). Added bootstrap CI
# overlap test and complexity-based parsimony tiebreaker.
# ---------------------------------------------------------------------------
v4_calibration_reliability <- function(df, v3_selections, n_seeds) {
  set.seed(42)

  # Filter to winning recipe x model x (weighting, downsampling) from V3
  winners <- v3_selections %>%
    filter(selected) %>%
    select(recipe_id, factorial_model, factorial_weighting, factorial_downsampling)

  df_sub <- df %>%
    semi_join(winners, by = c("recipe_id", "factorial_model",
                              "factorial_weighting", "factorial_downsampling"))

  # 1. Summarize per group x calibration
  cal_summary <- df_sub %>%
    group_by(recipe_id, factorial_model, factorial_weighting, factorial_downsampling,
             factorial_calibration) %>%
    summarise(
      reliability_mean = mean(summary_brier_reliability_mean, na.rm = TRUE),
      reliability_se = seed_se(
        # Use std of reliability across cells as proxy; actual SE from seeds
        rep(sd(summary_brier_reliability_mean, na.rm = TRUE), n()),
        n_seeds
      ),
      n_cells = n(),
      .groups = "drop"
    )

  # 2. Bootstrap CI on reliability
  n_boot <- 1000
  result_list <- list()
  group_keys <- cal_summary %>%
    distinct(recipe_id, factorial_model, factorial_weighting, factorial_downsampling)

  for (g in seq_len(nrow(group_keys))) {
    gk <- group_keys[g, ]
    grp <- cal_summary %>%
      semi_join(gk, by = c("recipe_id", "factorial_model",
                            "factorial_weighting", "factorial_downsampling"))

    if (nrow(grp) == 0) next

    boot_rel <- matrix(NA, nrow = n_boot, ncol = nrow(grp))
    for (b in seq_len(n_boot)) {
      boot_rel[b, ] <- grp$reliability_mean + rnorm(nrow(grp), 0, grp$reliability_se)
    }

    grp <- grp %>%
      mutate(
        ci_lo = apply(boot_rel, 2, quantile, probs = 0.025),
        ci_hi = apply(boot_rel, 2, quantile, probs = 0.975),
        rank = rank(reliability_mean, ties.method = "min"),
        complexity = CALIBRATION_COMPLEXITY[factorial_calibration]
      )

    # 3. Rank by mean reliability (lower is better)
    #    If CI of rank-1 overlaps with simpler method -> prefer simpler
    best_rank_idx <- which.min(grp$reliability_mean)
    best_ci_lo <- grp$ci_lo[best_rank_idx]
    best_ci_hi <- grp$ci_hi[best_rank_idx]

    # Find simpler methods whose CIs overlap with the best
    overlapping <- which(
      grp$ci_lo <= best_ci_hi &
      grp$ci_hi >= best_ci_lo
    )

    if (length(overlapping) > 1) {
      # Among overlapping, pick simplest (lowest complexity)
      simplest <- overlapping[which.min(grp$complexity[overlapping])]
      grp$selected <- seq_len(nrow(grp)) == simplest
      grp$overlaps_simpler <- seq_len(nrow(grp)) %in% overlapping
    } else {
      grp$selected <- seq_len(nrow(grp)) == best_rank_idx
      grp$overlaps_simpler <- FALSE
    }

    result_list[[g]] <- grp
  }

  bind_rows(result_list)
}

# ---------------------------------------------------------------------------
# V5: Confirmation — Seed-split validation (NEW)
# This level requires per-seed data. Falls back to variance-ratio check
# when only cell-level aggregates are available.
# ---------------------------------------------------------------------------
v5_seed_confirmation <- function(df, v4_selections,
                                  selection_seeds = 100:119,
                                  confirmation_seeds = 120:129) {
  # Identify the final winner from V4
  winner <- v4_selections %>%
    filter(selected) %>%
    head(1)

  if (nrow(winner) == 0) {
    return(data.frame(
      status = "no_winner",
      message = "No configuration selected by V4",
      stringsAsFactors = FALSE
    ))
  }

  # Check if per-seed data is available (column "seed" in df)
  has_per_seed <- "seed" %in% names(df)

  if (!has_per_seed) {
    # Fallback: variance-ratio check using cell-level aggregates
    # Compare winner's std vs runners-up
    winner_key <- winner %>%
      select(recipe_id, factorial_model, factorial_weighting,
             factorial_downsampling, factorial_calibration)

    winner_cells <- df %>%
      semi_join(winner_key, by = c("recipe_id", "factorial_model",
                                    "factorial_weighting", "factorial_downsampling",
                                    "factorial_calibration"))

    # Runners-up: same recipe+model but different downstream settings
    runner_cells <- df %>%
      filter(recipe_id == winner_key$recipe_id[1],
             factorial_model == winner_key$factorial_model[1]) %>%
      anti_join(winner_key, by = c("factorial_weighting", "factorial_downsampling",
                                    "factorial_calibration"))

    winner_std <- mean(winner_cells$summary_auroc_std, na.rm = TRUE)
    runner_std <- mean(runner_cells$summary_auroc_std, na.rm = TRUE)
    ratio <- if (runner_std > 0) winner_std / runner_std else NA_real_

    # Flag if winner has unusually high variance (>1.5x runners-up)
    unstable <- !is.na(ratio) && ratio > 1.5

    result <- data.frame(
      recipe_id = winner_key$recipe_id,
      factorial_model = winner_key$factorial_model,
      factorial_calibration = winner_key$factorial_calibration,
      factorial_weighting = winner_key$factorial_weighting,
      factorial_downsampling = winner_key$factorial_downsampling,
      status = "confirmation_pending",
      message = "Per-seed data not available; variance-ratio fallback used",
      winner_auroc_std = winner_std,
      runner_auroc_std = runner_std,
      variance_ratio = ratio,
      stability_flag = ifelse(unstable, "UNSTABLE", "OK"),
      stringsAsFactors = FALSE
    )

    return(result)
  }

  # Per-seed data IS available
  winner_key <- winner %>%
    select(recipe_id, factorial_model, factorial_weighting,
           factorial_downsampling, factorial_calibration)

  # Split seeds
  df_select <- df %>%
    filter(seed %in% selection_seeds) %>%
    semi_join(winner_key, by = c("recipe_id", "factorial_model",
                                  "factorial_weighting", "factorial_downsampling",
                                  "factorial_calibration"))

  df_confirm <- df %>%
    filter(seed %in% confirmation_seeds) %>%
    semi_join(winner_key, by = c("recipe_id", "factorial_model",
                                  "factorial_weighting", "factorial_downsampling",
                                  "factorial_calibration"))

  select_auroc <- mean(df_select$summary_auroc_mean, na.rm = TRUE)
  confirm_auroc <- mean(df_confirm$summary_auroc_mean, na.rm = TRUE)
  select_se <- sd(df_select$summary_auroc_mean, na.rm = TRUE) / sqrt(nrow(df_select))

  drop <- select_auroc - confirm_auroc
  drop_flag <- drop > select_se

  data.frame(
    recipe_id = winner_key$recipe_id,
    factorial_model = winner_key$factorial_model,
    factorial_calibration = winner_key$factorial_calibration,
    factorial_weighting = winner_key$factorial_weighting,
    factorial_downsampling = winner_key$factorial_downsampling,
    status = ifelse(drop_flag, "confirmation_drop", "confirmed"),
    message = ifelse(drop_flag,
                     sprintf("Confirmation drop %.4f > 1 SE (%.4f)", drop, select_se),
                     "Winner confirmed on held-out seeds"),
    selection_auroc = select_auroc,
    confirmation_auroc = confirm_auroc,
    auroc_drop = drop,
    selection_se = select_se,
    drop_exceeds_1se = drop_flag,
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# V6: Ensemble Comparison (post-tree, informational)
# Not a gate — surfaces whether stacking non-dominated models from V2
# beats the locked single-model winner. Human decides the tradeoff.
# Uses calibrated OOF predictions from fully-optimized base models.
# ---------------------------------------------------------------------------
v6_ensemble_comparison <- function(df, v2_results, v4_selections,
                                   delta = 0.02, n_seeds) {
  set.seed(42)

  # Get non-dominated models from V2
  non_dominated <- v2_results %>%
    filter(robust_nondominated) %>%
    distinct(recipe_id, factorial_model)

  if (nrow(non_dominated) < 2) {
    return(data.frame(
      status = "skipped",
      message = "Fewer than 2 non-dominated models; ensemble not applicable",
      stringsAsFactors = FALSE
    ))
  }

  # Get the locked single-model winner from V4
  winner <- v4_selections %>% filter(selected) %>% head(1)
  if (nrow(winner) == 0) {
    return(data.frame(
      status = "skipped",
      message = "No single-model winner from V4",
      stringsAsFactors = FALSE
    ))
  }

  # Ensemble approximation: average AUROC across non-dominated models
  # (True stacking requires OOF prediction files; this estimates the ceiling
  # using per-model cell-level metrics as a proxy.)
  #
  # For each non-dominated model, get its best cell (matching V3/V4 winners
  # where possible, otherwise best within that model's cells).

  ensemble_cells <- list()
  for (i in seq_len(nrow(non_dominated))) {
    nd <- non_dominated[i, ]
    # Try to match the V3/V4 winning downstream config for this model
    matched <- df %>%
      filter(recipe_id == nd$recipe_id,
             factorial_model == nd$factorial_model,
             factorial_weighting == winner$factorial_weighting,
             factorial_downsampling == winner$factorial_downsampling,
             factorial_calibration == winner$factorial_calibration)

    if (nrow(matched) == 0) {
      # Fallback: best cell for this model
      matched <- df %>%
        filter(recipe_id == nd$recipe_id,
               factorial_model == nd$factorial_model) %>%
        arrange(desc(summary_auroc_mean)) %>%
        head(1)
    }
    if (nrow(matched) > 0) {
      ensemble_cells[[i]] <- matched[1, ]
    }
  }

  if (length(ensemble_cells) < 2) {
    return(data.frame(
      status = "skipped",
      message = "Could not match enough non-dominated models to cells",
      stringsAsFactors = FALSE
    ))
  }

  ens_df <- bind_rows(ensemble_cells)

  # Ensemble estimate: mean of base model AUROCs
  # (conservative lower bound — true stacking typically does better than
  # averaging, but we don't have the OOF prediction files here)
  ensemble_auroc <- mean(ens_df$summary_auroc_mean)
  ensemble_se <- seed_se(ens_df$summary_auroc_std, n_seeds)

  # Single-model winner metrics
  winner_cell <- df %>%
    filter(recipe_id == winner$recipe_id,
           factorial_model == winner$factorial_model,
           factorial_weighting == winner$factorial_weighting,
           factorial_downsampling == winner$factorial_downsampling,
           factorial_calibration == winner$factorial_calibration)

  if (nrow(winner_cell) == 0) {
    return(data.frame(
      status = "error",
      message = "Cannot find winner cell in data",
      stringsAsFactors = FALSE
    ))
  }

  single_auroc <- winner_cell$summary_auroc_mean[1]
  single_se <- seed_se(c(winner_cell$summary_auroc_std[1]), n_seeds)

  # Comparison: ensemble vs single
  gain <- ensemble_auroc - single_auroc
  se_diff <- sqrt(ensemble_se^2 + single_se^2)
  ci_lo <- gain - 1.96 * se_diff
  ci_hi <- gain + 1.96 * se_diff

  # Higher bar: gain > delta AND full CI above 0
  significant_gain <- (gain > delta) && (ci_lo > 0)

  data.frame(
    status = ifelse(significant_gain, "ensemble_recommended", "single_model_preferred"),
    message = ifelse(significant_gain,
                     sprintf("Ensemble gain %.4f [%.4f, %.4f] exceeds delta=%.3f — recommend human review",
                             gain, ci_lo, ci_hi, delta),
                     sprintf("Ensemble gain %.4f [%.4f, %.4f] does not exceed delta=%.3f — single model sufficient",
                             gain, ci_lo, ci_hi, delta)),
    ensemble_models = paste(ens_df$factorial_model, collapse = " + "),
    ensemble_auroc = ensemble_auroc,
    single_model = winner$factorial_model,
    single_auroc = single_auroc,
    auroc_gain = gain,
    gain_ci_lo = ci_lo,
    gain_ci_hi = ci_hi,
    delta_threshold = delta,
    exceeds_delta = significant_gain,
    note = "Ensemble estimate uses averaged base-model AUROCs (conservative). True stacking with OOF predictions may perform better. If flagged, run full stacking comparison before final decision.",
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# Run all levels
# ---------------------------------------------------------------------------
message("\n=== V1: Recipe comparison (stratified) ===")
v1 <- v1_recipe_comparison(df, n_seeds)
message(sprintf("  Shared recipes: %d | MS recipes: %d",
                nrow(v1$within_shared), nrow(v1$within_ms)))
message(sprintf("  Recommended: %s", paste(v1$recommended_recipes, collapse = ", ")))

message("\n=== V2: Model Pareto (AUROC vs Reliability) ===")
v2 <- v2_model_pareto(df, v1$recommended_recipes, n_boot = 1000, n_seeds = n_seeds)
v2_selected <- v2 %>% filter(selected)
message(sprintf("  %d model-recipe combos evaluated, %d robustly dominated",
                nrow(v2), sum(v2$robust_dominated)))
for (i in seq_len(nrow(v2_selected))) {
  r <- v2_selected[i, ]
  message(sprintf("  Selected: %s / %s (AUROC=%.4f, Reliability=%.4f, Complexity=%d)",
                  r$recipe_id, r$factorial_model,
                  r$auroc_mean, r$reliability_mean, r$complexity))
}

message("\n=== V3: Imbalance (joint weighting x downsampling) ===")
v3 <- v3_imbalance_joint(df, v2, n_seeds)
v3_selected <- v3 %>% filter(selected)
for (i in seq_len(nrow(v3_selected))) {
  r <- v3_selected[i, ]
  message(sprintf("  Selected: %s / %s / wt=%s ds=%.1f (utility=%.4f [%.4f, %.4f])",
                  r$recipe_id, r$factorial_model,
                  r$factorial_weighting, r$factorial_downsampling,
                  r$utility, r$utility_ci_lo, r$utility_ci_hi))
}

message("\n=== V4: Calibration (reliability + parsimony) ===")
v4 <- v4_calibration_reliability(df, v3, n_seeds)
v4_selected <- v4 %>% filter(selected)
for (i in seq_len(nrow(v4_selected))) {
  r <- v4_selected[i, ]
  message(sprintf("  Selected: %s / %s / cal=%s (reliability=%.6f [%.6f, %.6f])",
                  r$recipe_id, r$factorial_model,
                  r$factorial_calibration,
                  r$reliability_mean, r$ci_lo, r$ci_hi))
}

message("\n=== V5: Seed-split confirmation ===")
v5 <- v5_seed_confirmation(df, v4)
message(sprintf("  Status: %s", v5$status[1]))
if ("message" %in% names(v5)) {
  message(sprintf("  %s", v5$message[1]))
}
if ("variance_ratio" %in% names(v5) && !is.na(v5$variance_ratio[1])) {
  message(sprintf("  Variance ratio (winner/runners): %.3f  Flag: %s",
                  v5$variance_ratio[1], v5$stability_flag[1]))
}

# ---------------------------------------------------------------------------
# V6: Ensemble comparison (post-tree)
# ---------------------------------------------------------------------------
message("\n=== V6: Ensemble comparison (post-tree, informational) ===")
v6 <- v6_ensemble_comparison(df, v2, v4, delta = 0.02, n_seeds = n_seeds)
message(sprintf("  Status: %s", v6$status[1]))
message(sprintf("  %s", v6$message[1]))
if ("ensemble_models" %in% names(v6) && !is.na(v6$ensemble_models[1])) {
  message(sprintf("  Ensemble: %s (AUROC=%.4f) vs Single: %s (AUROC=%.4f)",
                  v6$ensemble_models[1], v6$ensemble_auroc[1],
                  v6$single_model[1], v6$single_auroc[1]))
  message(sprintf("  Gain: %.4f [%.4f, %.4f]  Threshold: %.3f",
                  v6$auroc_gain[1], v6$gain_ci_lo[1], v6$gain_ci_hi[1],
                  v6$delta_threshold[1]))
}

# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------
base <- tools::file_path_sans_ext(output_path)

write.csv(v1$within_shared, paste0(base, "_v1_within_shared.csv"), row.names = FALSE)
write.csv(v1$within_ms, paste0(base, "_v1_within_ms.csv"), row.names = FALSE)
write.csv(v1$cross_type, paste0(base, "_v1_cross_type.csv"), row.names = FALSE)
write.csv(v1$summary, paste0(base, "_v1_summary.csv"), row.names = FALSE)
write.csv(v2, paste0(base, "_v2_pareto.csv"), row.names = FALSE)
write.csv(v3, paste0(base, "_v3_imbalance.csv"), row.names = FALSE)
write.csv(v4, paste0(base, "_v4_calibration.csv"), row.names = FALSE)
write.csv(v5, paste0(base, "_v5_confirmation.csv"), row.names = FALSE)
write.csv(v6, paste0(base, "_v6_ensemble.csv"), row.names = FALSE)

message(sprintf("\nOutput written to %s_v[1-6]_*.csv", base))

# ---------------------------------------------------------------------------
# Final cascade summary
# ---------------------------------------------------------------------------
message("\n========================================")
message("         VALIDATION CASCADE             ")
message("========================================")

# V1 winner
message(sprintf("V1 Recipe:       %s", paste(v1$recommended_recipes, collapse = ", ")))

# V2 winner
if (nrow(v2_selected) > 0) {
  message(sprintf("V2 Model:        %s",
                  paste(v2_selected$factorial_model, collapse = ", ")))
}

# V3 winner
if (nrow(v3_selected) > 0) {
  message(sprintf("V3 Imbalance:    wt=%s, ds=%s",
                  paste(unique(v3_selected$factorial_weighting), collapse = "/"),
                  paste(unique(v3_selected$factorial_downsampling), collapse = "/")))
}

# V4 winner
if (nrow(v4_selected) > 0) {
  message(sprintf("V4 Calibration:  %s",
                  paste(unique(v4_selected$factorial_calibration), collapse = ", ")))
}

# V5 status
message(sprintf("V5 Confirmation: %s", v5$status[1]))

# V6 ensemble
message(sprintf("V6 Ensemble:     %s", v6$status[1]))

# Final locked config
if (nrow(v4_selected) > 0) {
  final <- v4_selected[1, ]
  message("\n--- LOCKED CONFIGURATION ---")
  message(sprintf("  Recipe:       %s", final$recipe_id))
  message(sprintf("  Model:        %s", final$factorial_model))
  message(sprintf("  Weighting:    %s", final$factorial_weighting))
  message(sprintf("  Downsampling: %s", final$factorial_downsampling))
  message(sprintf("  Calibration:  %s", final$factorial_calibration))
}

message("\nDone.")
