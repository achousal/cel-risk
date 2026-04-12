#!/usr/bin/env Rscript
# fig04_marker_architecture.R
#
# Pathway-order marker inclusion heatmap: rows = proteins (entry order),
# columns = panel sizes 4-25. Binary fill = in/out.
# Annotates the 4 truth-set proteins and biological axis groupings.
#
# Data: experiments/optimal-setup/panel-sweep/panels/pathway_*.csv

source("_theme.R")

# ── Load panel files ─────────────────────────────────────────────────────────
sizes <- 4:25

load_panel <- function(size) {
  path <- file.path(PANELS_DIR, sprintf("pathway_%dp.csv", size))
  proteins <- readLines(path) %>% str_trim() %>% .[. != ""]
  data.frame(size = size, protein = proteins, stringsAsFactors = FALSE)
}

panels <- bind_rows(lapply(sizes, load_panel))

# Canonical pathway position from the largest panel file (preserves addition order)
canonical <- readLines(file.path(PANELS_DIR, "pathway_25p.csv")) %>%
  str_trim() %>% .[. != ""]
canonical_pos <- setNames(seq_along(canonical), canonical)

# Entry order: order by first appearance, break ties by canonical pathway position
entry_order <- panels %>%
  group_by(protein) %>%
  summarize(first_entry = min(size), .groups = "drop") %>%
  mutate(
    pathway_pos   = canonical_pos[protein],
    protein_clean = clean_protein(protein)
  ) %>%
  arrange(first_entry, pathway_pos)

# ── Build presence matrix ────────────────────────────────────────────────────
presence <- panels %>%
  mutate(
    protein_clean = clean_protein(protein),
    present = TRUE
  )

full_grid <- expand.grid(
  protein_clean = entry_order$protein_clean,
  size = sizes,
  stringsAsFactors = FALSE
)

presence_full <- full_grid %>%
  left_join(presence %>% select(protein_clean, size, present),
            by = c("protein_clean", "size")) %>%
  mutate(
    present = ifelse(is.na(present), FALSE, TRUE),
    protein_clean = factor(protein_clean, levels = rev(entry_order$protein_clean))
  )

# ── Biological axis annotations ─────────────────────────────────────────────
truth_set <- c("TGM2", "CPA2", "ITGB7", "GIP")

# Axis assignments from sweep-analysis.md
axis_map <- c(
  TGM2 = "Mucosal", MUC2 = "Mucosal",
  ITGB7 = "Immune", CXCL9 = "Immune", CD160 = "Immune",
  CPA2 = "Metabolic", GIP = "Metabolic",
  NOS2 = "Ext. immune", CXCL11 = "Ext. immune", TIGIT = "Ext. immune"
)

axis_colors <- c(
  "Mucosal"     = "#4C78A8",
  "Immune"      = "#1B9E77",
  "Metabolic"   = "#E69F00",
  "Ext. immune" = "#7570B3"
)

entry_annotated <- entry_order %>%
  mutate(
    axis = axis_map[protein_clean],
    is_truth = protein_clean %in% truth_set,
    label_text = ifelse(is_truth,
                        paste0(protein_clean, " *"),
                        protein_clean)
  )

# ── Fig 4: Marker architecture heatmap ──────────────────────────────────────
message("Fig 4: Marker architecture ...")

# Y-axis labels with truth-set markers bolded
y_labels <- setNames(entry_annotated$label_text,
                     entry_annotated$protein_clean)

p4 <- ggplot(presence_full, aes(x = factor(size), y = protein_clean,
                                 fill = present)) +
  geom_tile(color = "white", linewidth = 0.3) +
  # Sweet-spot bracket at top
  annotate("rect", xmin = 4.5, xmax = 7.5, ymin = 0.4, ymax = Inf,
           fill = "#d9ead3", alpha = 0.15) +
  scale_fill_manual(
    values = c(`TRUE` = "#1B9E77", `FALSE` = "#F7F7F7"),
    labels = c(`TRUE` = "In panel", `FALSE` = "Excluded"),
    name = ""
  ) +
  scale_y_discrete(labels = y_labels) +
  labs(
    title    = "Marker Architecture: Pathway Addition Order",
    subtitle = "Proteins ordered by entry position. * = BH-significant truth set.",
    x = "Panel size",
    y = "",
    caption  = "First 10 proteins span 3 biological axes: mucosal integrity, immune surveillance, metabolic"
  ) +
  theme_cel() +
  theme(
    axis.text.y  = element_text(size = 7, family = "mono"),
    axis.text.x  = element_text(size = 8),
    panel.grid   = element_blank()
  )

save_fig(p4, "fig04_marker_architecture", width = 11, height = 9)
message("Done: fig04")
