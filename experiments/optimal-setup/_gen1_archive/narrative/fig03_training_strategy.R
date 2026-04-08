#!/usr/bin/env Rscript
# fig03_training_strategy.R
#
# Two-panel figure:
#   A) Strategy comparison: dot plot of mean AUPRC (primary) by strategy x weight
#   B) Feature importance: top 20 coefficients from the final incident_only+log model
#
# Data: results/incident_validation/

source("_theme.R")
library(patchwork)

# ── Paths ────────────────────────────────────────────────────────────────────
IV_DIR <- file.path(RESULTS_DIR, "incident_validation")

# ── Panel A: Strategy comparison ─────────────────────────────────────────────
strat <- read.csv(file.path(IV_DIR, "strategy_comparison.csv"),
                  stringsAsFactors = FALSE)

strat <- strat %>%
  mutate(
    label = paste0(
      str_replace_all(strategy, "_", " "),
      "\n(",
      str_replace_all(weight_scheme, "_", " "),
      ")"
    ),
    is_best = strategy == "incident_only" & weight_scheme == "log"
  ) %>%
  arrange(desc(mean_auprc)) %>%
  mutate(label = factor(label, levels = rev(label)))

pa <- ggplot(strat, aes(x = mean_auprc, y = label)) +
  geom_errorbarh(
    aes(xmin = mean_auprc - std_auprc,
        xmax = mean_auprc + std_auprc),
    height = 0.3, color = "grey60"
  ) +
  geom_point(aes(color = is_best), size = 3) +
  scale_color_manual(
    values = c(`TRUE` = "#1B9E77", `FALSE` = "grey50"),
    guide = "none"
  ) +
  labs(
    title    = "A. Training Strategy Comparison",
    subtitle = "5-fold CV mean AUPRC (+/- SD). Green = best.",
    x = "Mean AUPRC", y = ""
  ) +
  theme_cel(base_size = 10) +
  theme(
    panel.grid.major.y = element_blank(),
    axis.text.y        = element_text(size = 8)
  )

# ── Panel B: Feature coefficients ───────────────────────────────────────────
coefs <- read.csv(file.path(IV_DIR, "feature_coefficients.csv"),
                  stringsAsFactors = FALSE)

top_coefs <- coefs %>%
  head(20) %>%
  mutate(
    protein_clean = clean_protein(protein),
    direction     = ifelse(coefficient > 0, "Risk", "Protective"),
    protein_clean = factor(protein_clean, levels = rev(protein_clean))
  )

pb <- ggplot(top_coefs, aes(x = abs_coef, y = protein_clean, fill = direction)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = sprintf("%.0f%%", stability_freq * 100)),
    hjust = -0.15, size = 2.5, color = "grey40"
  ) +
  scale_fill_manual(
    values = c(Risk = "#D95F02", Protective = "#4C78A8"),
    name = "Direction"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(
    title    = "B. Top 20 Feature Coefficients",
    subtitle = "Final model (incident-only, log weights). % = bootstrap stability.",
    x = "|Coefficient|", y = ""
  ) +
  theme_cel(base_size = 10) +
  theme(
    panel.grid.major.y = element_blank(),
    legend.position    = c(0.75, 0.2)
  )

# ── Compose ──────────────────────────────────────────────────────────────────
message("Fig 3: Training strategy ...")

p3 <- pa | pb
p3 <- p3 + plot_annotation(
  title = "Training Strategy Validation: Incident-Only with Log Weights",
  subtitle = "Locked test AUROC = 0.908 [0.827, 0.978]",
  caption = "12 strategy-weight combinations. Elastic-net logistic regression, 134 stable proteins, Optuna-tuned.",
  theme = theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(color = "grey40", size = 10),
    plot.caption  = element_text(color = "grey50", size = 8, hjust = 0)
  )
)

save_fig(p3, "fig03_training_strategy", width = 14, height = 8)
message("Done: fig03")
