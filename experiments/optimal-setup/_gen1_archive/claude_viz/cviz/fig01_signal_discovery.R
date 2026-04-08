#!/usr/bin/env Rscript
# fig01_signal_discovery.R
#
# Waterfall of RRA significance across 2920 proteins.
# BH threshold separates the 4 truth-set proteins from everything else.
# Shows the gap between signal and noise in the full proteomic universe.
#
# Data: results/experiments/rra_universe_sensitivity/rra_significance_corrected.csv

source("_theme.R")

# ── Load ─────────────────────────────────────────────────────────────────────
rra <- read.csv(
  file.path(RESULTS_DIR, "experiments", "rra_universe_sensitivity",
            "rra_significance_corrected.csv"),
  stringsAsFactors = FALSE
)

rra <- rra %>%
  mutate(
    protein_clean = clean_protein(protein),
    neg_log10_bh  = -log10(bh_adjusted_p),
    rank          = rank(-observed_rra, ties.method = "first"),
    significant   = as.character(significant)
  ) %>%
  arrange(rank)

# BH threshold at alpha = 0.05
bh_threshold <- -log10(0.05)

# Top N to label (truth set + next few for context)
n_label <- 10

# ── Fig 1: Waterfall ────────────────────────────────────────────────────────
message("Fig 1: Signal discovery waterfall ...")

top <- rra %>% slice_head(n = n_label)
truth_set <- c("TGM2", "CPA2", "ITGB7", "GIP")

p1 <- ggplot(rra, aes(x = rank, y = neg_log10_bh)) +
  # Significance threshold
  geom_hline(yintercept = bh_threshold, linetype = "dashed",
             color = "grey50", linewidth = 0.4) +
  annotate("text", x = nrow(rra) * 0.85, y = bh_threshold + 0.15,
           label = "BH-corrected alpha = 0.05", color = "grey40",
           size = 3, hjust = 1) +
  # All proteins as points

  geom_point(
    aes(color = significant),
    size = 0.8, alpha = 0.6
  ) +
  # Truth-set proteins highlighted
  geom_point(
    data = top %>% filter(protein_clean %in% truth_set),
    color = "#1B9E77", size = 3, shape = 16
  ) +
  # Labels for top proteins
  ggrepel::geom_text_repel(
    data = top,
    aes(label = protein_clean,
        fontface = ifelse(protein_clean %in% truth_set, "bold", "plain")),
    size = 3, color = "grey20",
    nudge_y = 0.3, segment.color = "grey60", segment.size = 0.3,
    max.overlaps = 15
  ) +
  # Bracket annotation for truth set
  annotate("segment",
           x = 0.5, xend = 4.5,
           y = min(top$neg_log10_bh[top$protein_clean %in% truth_set]) - 0.3,
           yend = min(top$neg_log10_bh[top$protein_clean %in% truth_set]) - 0.3,
           color = "#1B9E77", linewidth = 0.6) +
  annotate("text",
           x = 2.5,
           y = min(top$neg_log10_bh[top$protein_clean %in% truth_set]) - 0.6,
           label = "4-protein truth set",
           color = "#1B9E77", size = 3, fontface = "bold") +
  scale_color_manual(
    values = c(`True` = "#1B9E77", `False` = "grey70"),
    labels = c(`True` = "Significant", `False` = "Not significant"),
    name = ""
  ) +
  scale_x_continuous(
    breaks = c(1, 10, 25, 50, 100),
    trans = "log10",
    labels = comma_format()
  ) +
  labs(
    title    = "Signal Discovery: RRA Significance Across 2,920 Proteins",
    subtitle = "BH-corrected permutation p-values. Universe = full Olink panel.",
    x = "Protein rank (log scale)",
    y = expression(-log[10](p[BH])),
    caption  = "Source: Phase 1 consensus ranking with universe-corrected RRA"
  ) +
  theme_cel() +
  theme(legend.position = c(0.85, 0.85))

save_fig(p1, "fig01_signal_discovery", width = 9, height = 5.5)
message("Done: fig01")
