#!/usr/bin/env Rscript
# fig02_model_pareto.R -- Model Pareto front (AUROC vs Reliability) + bootstrap membership
#
# Panel A: Pareto scatter with dominated points faded
# Panel B: Bootstrap Pareto membership (from validation CSV) or placeholder

source(file.path("..", "_theme_factorial.R"))
library(patchwork)

# ── Load data ────────────────────────────────────────────────────────────────
df <- load_factorial()

# ── Parsimony ranking ────────────────────────────────────────────────────────
MODEL_COMPLEXITY <- c(LR_EN = 1, LinSVM_cal = 2, RF = 3, XGBoost = 4)

# ── Try to load validation CSV ───────────────────────────────────────────────
v2_files <- list.files(path = ".", pattern = "v2_pareto", recursive = TRUE,
                       full.names = TRUE)
has_validation <- length(v2_files) > 0

if (has_validation) {
  validation_df <- read.csv(v2_files[1], stringsAsFactors = FALSE)
  message(sprintf("Loaded validation CSV: %s (%d rows)", v2_files[1], nrow(validation_df)))
}

# ── Compute model summary from raw data ──────────────────────────────────────
model_summary <- df %>%
  group_by(recipe_id, factorial_model) %>%
  summarise(
    auroc_mean       = mean(summary_auroc_mean, na.rm = TRUE),
    reliability_mean = mean(summary_brier_reliability_mean, na.rm = TRUE),
    .groups = "drop"
  )

# ── Pareto dominance (AUROC higher = better, reliability lower = better) ─────
model_summary$dominated <- FALSE
for (i in seq_len(nrow(model_summary))) {
  for (j in seq_len(nrow(model_summary))) {
    if (i != j &&
        model_summary$auroc_mean[j] >= model_summary$auroc_mean[i] &&
        model_summary$reliability_mean[j] <= model_summary$reliability_mean[i] &&
        (model_summary$auroc_mean[j] > model_summary$auroc_mean[i] ||
         model_summary$reliability_mean[j] < model_summary$reliability_mean[i])) {
      model_summary$dominated[i] <- TRUE
      break
    }
  }
}

# ── Pareto front (non-dominated, sorted for step line) ───────────────────────
pareto_front <- model_summary %>%
  filter(!dominated) %>%
  arrange(reliability_mean)

# ── Crosshair at medians ────────────────────────────────────────────────────
med_auroc <- median(model_summary$auroc_mean)
med_reliability <- median(model_summary$reliability_mean)

# ── Panel A: Pareto scatter ──────────────────────────────────────────────────
pa <- ggplot(model_summary, aes(x = reliability_mean, y = auroc_mean)) +
  # Crosshair
  geom_hline(yintercept = med_auroc, linetype = "dashed", color = "grey60") +
  geom_vline(xintercept = med_reliability, linetype = "dashed", color = "grey60") +
  # Pareto step line
  geom_step(data = pareto_front, direction = "vh",
            linetype = "dashed", color = "grey40", linewidth = 0.6) +
  # Points
  geom_point(aes(color = factorial_model,
                 alpha = ifelse(dominated, 0.3, 1.0)),
             size = 3) +
  scale_color_manual(values = MODEL_COLORS, name = "Model") +
  scale_alpha_identity() +
  labs(
    title = "A. AUROC vs Reliability (Pareto Front)",
    x = expression("Brier Reliability (lower is better)" %->% ""),
    y = "AUROC (higher is better)"
  ) +
  theme_cel() +
  theme(legend.position = "bottom")

# ── Panel B: Bootstrap membership or placeholder ────────────────────────────
if (has_validation && "frac_nondominated" %in% names(validation_df)) {
  # Aggregate to model level if recipe-level
  if ("factorial_model" %in% names(validation_df)) {
    model_boot <- validation_df %>%
      group_by(factorial_model) %>%
      summarise(frac_nondominated = mean(frac_nondominated, na.rm = TRUE),
                .groups = "drop")
  } else {
    model_boot <- validation_df
  }

  # Add complexity annotation
  model_boot <- model_boot %>%
    mutate(complexity = MODEL_COMPLEXITY[as.character(factorial_model)])

  pb <- ggplot(model_boot,
               aes(x = frac_nondominated,
                   y = reorder(factorial_model, frac_nondominated))) +
    geom_pointrange(aes(xmin = 0, xmax = frac_nondominated,
                        color = factorial_model), size = 1) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey50") +
    # Parsimony annotation on right side
    geom_text(aes(label = sprintf("complexity: %d", complexity)),
              x = 0.97, hjust = 1, size = 2.8, color = "grey40") +
    scale_color_manual(values = MODEL_COLORS, name = "Model") +
    scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
    labs(
      title = "B. Bootstrap Pareto Membership",
      x = "Fraction non-dominated (1000 bootstraps)",
      y = NULL
    ) +
    theme_cel() +
    theme(legend.position = "bottom")

} else {
  # Placeholder panel
  placeholder_df <- data.frame(x = 0.5, y = 0.5)
  pb <- ggplot(placeholder_df, aes(x = x, y = y)) +
    annotate("text", x = 0.5, y = 0.5,
             label = "Run validate_tree.R for bootstrap results",
             size = 4.5, color = "grey40", fontface = "italic") +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "B. Bootstrap Pareto Membership", x = NULL, y = NULL) +
    theme_cel() +
    theme(
      axis.text  = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    )

  message("No validation CSV found -- Panel B shows placeholder.")
}

# ── Compose ──────────────────────────────────────────────────────────────────
p <- (pa | pb) +
  plot_layout(guides = "collect", widths = c(1.2, 1)) +
  plot_annotation(
    title = "V2: Model Selection -- Pareto Dominance",
    subtitle = "Non-dominated models survive; ties broken by parsimony (fewer parameters preferred)",
    caption = "Pareto front: AUROC (higher) vs Reliability (lower). Dominated points faded."
  )

factorial_save_fig(p, "fig02_model_pareto", width = 14, height = 7)

# ── Agent-friendly summary ───────────────────────────────────────────────────
n_total <- nrow(model_summary)
n_pareto <- sum(!model_summary$dominated)
pareto_models <- unique(pareto_front$factorial_model)

message(sprintf(
  "fig02_model_pareto: %d recipe-model points, %d non-dominated (Pareto front). Pareto models: %s. Validation CSV: %s.",
  n_total, n_pareto,
  paste(pareto_models, collapse = ", "),
  if (has_validation) "loaded" else "not found (placeholder Panel B)"
))
