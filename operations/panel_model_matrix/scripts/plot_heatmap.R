#!/usr/bin/env Rscript
# Render panel x model PRAUC heatmap from panel_model_prauc.csv.
# Usage: Rscript operations/panel_model_matrix/scripts/plot_heatmap.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
})

here <- function(...) file.path(
  normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", "..", "..")),
  ...
)
# Fallback when sourced without ofile: use cwd
root <- tryCatch(here(), error = function(e) normalizePath("."))

in_csv  <- file.path(root, "operations/panel_model_matrix/analysis/panel_model_prauc.csv")
out_png <- file.path(root, "operations/panel_model_matrix/analysis/panel_model_prauc_heatmap.png")

df <- read_csv(in_csv, show_col_types = FALSE) %>%
  filter(status == "ok") %>%
  mutate(
    panel = factor(panel, levels = c("LinSVM_cal", "LR_EN", "RF", "XGBoost")),
    model = factor(model, levels = c("LinSVM_cal", "LR_EN", "RF", "XGBoost")),
    label = sprintf("%.2f\n±%.2f", prauc_mean, prauc_std)
  )

p <- ggplot(df, aes(x = model, y = panel, fill = prauc_mean)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = label), size = 3.2) +
  scale_fill_gradient(low = "#f7f7f7", high = "#1b6ca8",
                      limits = c(0, 1), name = "PRAUC") +
  labs(
    title    = "Panel x Model PRAUC (holdout ds5, IncidentOnly, top-20 panels)",
    subtitle = "Rows = panel source; columns = classifier; cell = mean ± sd across 11 splits",
    x = "Classifier", y = "Panel source"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid = element_blank(),
    plot.title.position = "plot"
  )

ggsave(out_png, p, width = 7.5, height = 6, dpi = 200)
cat("wrote", out_png, "\n")
