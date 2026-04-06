#!/usr/bin/env Rscript
# plot_optuna_trials.R
#
# Visualize Optuna hyperparameter search trials across model types.
# Generates 4 figures exploring convergence, tradeoffs, and optimal stopping.
#
# Usage:
#   Rscript analysis/scripts/plot_optuna_trials.R
#
# Outputs: analysis/figures/optuna/fig{1-4}_*.pdf

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(scales)
  library(stringr)
})

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR <- "results"
OUT_DIR     <- "analysis/figures/optuna"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Source: 300-trial run with 30 splits — richest convergence data
RUNS <- list(
  "300t" = list(
    dir        = file.path(RESULTS_DIR, "run_20260217_194153"),
    n_trials   = 300L,
    label      = "300 trials (full)"
  ),
  "100t" = list(
    dir        = file.path(RESULTS_DIR, "run_phase2_val_consensus_100t"),
    n_trials   = 100L,
    label      = "100 trials"
  ),
  "50t"  = list(
    dir        = file.path(RESULTS_DIR, "run_phase2_val_consensus"),
    n_trials   = 50L,
    label      = "50 trials"
  )
)

MODELS <- c("LR_EN", "LinSVM_cal", "RF", "XGBoost")

MODEL_LABELS <- c(
  LR_EN      = "Logistic (EN)",
  LinSVM_cal = "Linear SVM",
  RF         = "Random Forest",
  XGBoost    = "XGBoost"
)

MODEL_COLORS <- c(
  LR_EN      = "#4878CF",
  LinSVM_cal = "#6ACC65",
  RF         = "#D65F5F",
  XGBoost    = "#B47CC7"
)

CUTOFFS <- c(25L, 50L, 75L, 100L, 150L, 200L, 300L)

# ── Helpers ───────────────────────────────────────────────────────────────────
load_trials <- function(run_dir, model, n_trials_per_split) {
  path <- file.path(run_dir, model, "aggregated", "cv", "optuna_trials.csv")
  if (!file.exists(path)) return(NULL)

  df <- read.csv(path, stringsAsFactors = FALSE)

  # Assign split_id: rows are stacked blocks of n_trials_per_split
  df <- df %>%
    mutate(
      split_id  = (seq_len(n()) - 1L) %/% n_trials_per_split,
      trial_num = number,      # 0-indexed within split
      auroc     = values_0,
      neg_brier = values_1,
      brier     = -values_1,
      model     = model
    ) %>%
    select(model, split_id, trial_num, auroc, brier, neg_brier)

  df
}

# Cumulative best AUROC within each split
add_best_so_far <- function(df) {
  df %>%
    arrange(model, split_id, trial_num) %>%
    group_by(model, split_id) %>%
    mutate(best_auroc = cummax(auroc)) %>%
    ungroup()
}

theme_cel <- function() {
  theme_bw(base_size = 11) +
    theme(
      panel.grid.minor  = element_blank(),
      strip.background  = element_rect(fill = "grey92", color = NA),
      legend.position   = "bottom",
      plot.title        = element_text(face = "bold", size = 12),
      plot.subtitle     = element_text(color = "grey40", size = 10)
    )
}

# ── Load 300-trial data (main convergence source) ─────────────────────────────
message("Loading trial data …")

trials_300 <- map_dfr(MODELS, function(m) {
  load_trials(RUNS[["300t"]]$dir, m, RUNS[["300t"]]$n_trials)
}) %>% add_best_so_far()

stopifnot(nrow(trials_300) > 0)
message(sprintf("  Loaded %d rows (%d splits × %d trials × %d models)",
  nrow(trials_300),
  n_distinct(trials_300$split_id),
  RUNS[["300t"]]$n_trials,
  n_distinct(trials_300$model)))


# ══ FIG 1: Convergence curves ═════════════════════════════════════════════════
# Best-so-far AUROC vs trial number; mean ± ribbon across splits; 4 models.
message("Fig 1: Convergence curves …")

conv_summary <- trials_300 %>%
  group_by(model, trial_num) %>%
  summarise(
    mean_best = mean(best_auroc),
    se_best   = sd(best_auroc) / sqrt(n()),
    lo95      = mean_best - 1.96 * se_best,
    hi95      = mean_best + 1.96 * se_best,
    .groups   = "drop"
  ) %>%
  mutate(
    model_label = MODEL_LABELS[model],
    model_label = factor(model_label, levels = MODEL_LABELS)
  )

cutoff_lines <- tibble(
  xintercept = c(50, 100, 150, 200),
  label      = paste0(c(50, 100, 150, 200), "t")
)

fig1 <- ggplot(conv_summary, aes(x = trial_num + 1, color = model_label, fill = model_label)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), alpha = 0.12, color = NA) +
  geom_line(aes(y = mean_best), linewidth = 0.9) +
  geom_vline(
    data = cutoff_lines,
    aes(xintercept = xintercept),
    linetype = "dashed", color = "grey50", linewidth = 0.5
  ) +
  geom_text(
    data = cutoff_lines,
    aes(x = xintercept + 2, y = -Inf, label = label),
    inherit.aes = FALSE,
    vjust = -0.5, hjust = 0, size = 2.8, color = "grey40"
  ) +
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS, name = NULL) +
  scale_fill_manual(values  = MODEL_COLORS, labels = MODEL_LABELS, name = NULL) +
  scale_x_continuous(breaks = c(1, 50, 100, 150, 200, 300)) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  labs(
    title    = "Hyperparameter search convergence by model",
    subtitle = "Best AUROC found up to trial t (mean ± 95% CI across CV splits)",
    x        = "Trial number",
    y        = "Best AUROC (cumulative max)"
  ) +
  theme_cel()

ggsave(file.path(OUT_DIR, "fig1_convergence.pdf"),
       fig1, width = 8, height = 5)
message("  Saved fig1_convergence.pdf")


# ══ FIG 2: Performance at N-trial cutoffs ════════════════════════════════════
# For each cutoff T, the best AUROC achieved within T trials, per model.
# Shows: marginal value of running more trials.
message("Fig 2: Performance at N-trial cutoffs …")

cutoff_perf <- map_dfr(CUTOFFS, function(t) {
  trials_300 %>%
    filter(trial_num < t) %>%
    group_by(model, split_id) %>%
    summarise(best = max(auroc), .groups = "drop") %>%
    mutate(cutoff = t)
}) %>%
  group_by(model, cutoff) %>%
  summarise(
    mean_best = mean(best),
    se_best   = sd(best) / sqrt(n()),
    lo95      = mean_best - 1.96 * se_best,
    hi95      = mean_best + 1.96 * se_best,
    .groups   = "drop"
  ) %>%
  mutate(
    model_label = factor(MODEL_LABELS[model], levels = MODEL_LABELS)
  )

# Gain relative to 50-trial performance
baseline_50 <- cutoff_perf %>%
  filter(cutoff == 50) %>%
  select(model, baseline = mean_best)

cutoff_perf <- cutoff_perf %>%
  left_join(baseline_50, by = "model") %>%
  mutate(gain_vs_50 = (mean_best - baseline) * 1000)  # in mAUROC units

fig2 <- ggplot(cutoff_perf, aes(x = cutoff, color = model_label, group = model_label)) +
  geom_errorbar(aes(ymin = lo95, ymax = hi95),
                width = 6, linewidth = 0.5, alpha = 0.6) +
  geom_line(aes(y = mean_best), linewidth = 0.9) +
  geom_point(aes(y = mean_best), size = 2.5, fill = "white", shape = 21, stroke = 1.2) +
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS, name = NULL) +
  scale_x_continuous(breaks = CUTOFFS) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  labs(
    title    = "Optimal performance vs. Optuna trial budget",
    subtitle = "Best AUROC within T trials, mean ± 95% CI across CV splits",
    x        = "Trial budget (T)",
    y        = "Best AUROC within T trials"
  ) +
  theme_cel()

ggsave(file.path(OUT_DIR, "fig2_trial_budget.pdf"),
       fig2, width = 8, height = 5)
message("  Saved fig2_trial_budget.pdf")


# ══ FIG 3: Pareto front (AUROC vs Brier score) ═══════════════════════════════
# Multi-objective landscape: all trials as scatter + Pareto-optimal frontier.
message("Fig 3: Pareto front …")

# Sample trials for readability (up to 1500 per model)
set.seed(42)
trials_sample <- trials_300 %>%
  group_by(model) %>%
  slice_sample(n = 1500) %>%
  ungroup()

# Pareto-optimal trials (maximize AUROC, minimize Brier)
is_pareto <- function(auroc, brier) {
  n <- length(auroc)
  dominated <- logical(n)
  for (i in seq_len(n)) {
    dominated[i] <- any(auroc[-i] >= auroc[i] & brier[-i] <= brier[i] &
                        (auroc[-i] > auroc[i] | brier[-i] < brier[i]))
  }
  !dominated
}

pareto_pts <- trials_300 %>%
  group_by(model) %>%
  slice_sample(n = 3000) %>%   # keep manageable for Pareto check
  group_modify(~ {
    idx <- is_pareto(.x$auroc, .x$brier)
    .x[idx, ]
  }) %>%
  ungroup()

fig3 <- ggplot() +
  # All trials (background)
  geom_point(
    data = trials_sample %>% mutate(model_label = factor(MODEL_LABELS[model], levels = MODEL_LABELS)),
    aes(x = auroc, y = brier, color = model_label),
    size = 0.6, alpha = 0.15
  ) +
  # Pareto front
  geom_point(
    data = pareto_pts %>% mutate(model_label = factor(MODEL_LABELS[model], levels = MODEL_LABELS)),
    aes(x = auroc, y = brier, color = model_label),
    size = 2.0, alpha = 0.9, shape = 18
  ) +
  facet_wrap(~ model_label, nrow = 1) +
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS, name = NULL) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  scale_x_continuous(labels = label_number(accuracy = 0.001)) +
  labs(
    title    = "Pareto front: AUROC vs. Brier score per model",
    subtitle = "Diamonds = Pareto-optimal trials (maximize AUROC, minimize Brier); dots = all sampled trials",
    x        = "AUROC (↑ better)",
    y        = "Brier score (↓ better)"
  ) +
  theme_cel() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(file.path(OUT_DIR, "fig3_pareto.pdf"),
       fig3, width = 11, height = 4)
message("  Saved fig3_pareto.pdf")


# ══ FIG 4: Best-model ranking across trial budgets ═══════════════════════════
# Heatmap of mean best AUROC × model × trial budget.
# Diverging from LR_EN baseline to show which model leads at each budget.
message("Fig 4: Model ranking heatmap …")

heatmap_dat <- cutoff_perf %>%
  select(model, model_label, cutoff, mean_best) %>%
  group_by(cutoff) %>%
  mutate(
    rank   = rank(-mean_best, ties.method = "first"),
    best   = mean_best == max(mean_best)
  ) %>%
  ungroup()

# Pivot to relative gain vs LR_EN at each cutoff (mAUROC)
lren_vals <- heatmap_dat %>%
  filter(model == "LR_EN") %>%
  select(cutoff, lren_auroc = mean_best)

heatmap_dat <- heatmap_dat %>%
  left_join(lren_vals, by = "cutoff") %>%
  mutate(delta_vs_lren = (mean_best - lren_auroc) * 1000)

fig4 <- ggplot(heatmap_dat,
               aes(x = factor(cutoff), y = model_label,
                   fill = delta_vs_lren)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(
    label = sprintf("%.3f\n(Δ%+.1f)", mean_best, delta_vs_lren)
  ), size = 2.8, lineheight = 0.9) +
  # Highlight best in each column
  geom_tile(
    data = filter(heatmap_dat, best),
    color = "black", fill = NA, linewidth = 1.2
  ) +
  scale_fill_gradient2(
    low      = "#d73027",
    mid      = "white",
    high     = "#1a9641",
    midpoint = 0,
    name     = "Δ vs LR_EN\n(mAUROC)"
  ) +
  scale_x_discrete(labels = function(x) paste0(x, "t")) +
  labs(
    title    = "Model AUROC across trial budgets",
    subtitle = "Mean best AUROC; Δ relative to LR_EN baseline; black border = best model at that budget",
    x        = "Trial budget",
    y        = NULL
  ) +
  theme_cel() +
  theme(legend.position = "right")

ggsave(file.path(OUT_DIR, "fig4_model_ranking.pdf"),
       fig4, width = 9, height = 4)
message("  Saved fig4_model_ranking.pdf")


# ── Summary table ─────────────────────────────────────────────────────────────
message("\n── Optimal stopping summary (mean best AUROC) ──")
cutoff_perf %>%
  select(model, cutoff, mean_best, gain_vs_50) %>%
  filter(cutoff %in% c(50, 100, 200, 300)) %>%
  mutate(across(c(mean_best, gain_vs_50), ~ round(.x, 4))) %>%
  arrange(cutoff, desc(mean_best)) %>%
  print(n = 40)

message("\nAll figures saved to: ", OUT_DIR)
