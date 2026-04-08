#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(stringr)
})

OUT_DIR <- "experiments/optimal-setup/codex_viz/cviz/out"
PANEL_DIR <- "experiments/optimal-setup/panel-sweep/panels"
MILESTONES <- c(4, 7, 8, 10)

theme_cel <- function() {
  theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(color = "grey40", size = 10)
    )
}

load_panel <- function(size) {
  path <- file.path(PANEL_DIR, sprintf("pathway_%sp.csv", size))
  data.frame(
    size = size,
    marker = readLines(path, warn = FALSE),
    stringsAsFactors = FALSE
  )
}

panels <- bind_rows(lapply(MILESTONES, load_panel)) %>%
  mutate(marker_clean = str_to_upper(str_remove(marker, "_resid$")))

marker_levels <- panels %>%
  distinct(marker_clean) %>%
  pull(marker_clean)

route <- expand.grid(
  marker_clean = marker_levels,
  size = MILESTONES,
  stringsAsFactors = FALSE
) %>%
  left_join(
    panels %>% transmute(size, marker_clean, present = TRUE),
    by = c("marker_clean", "size")
  ) %>%
  mutate(
    present = ifelse(is.na(present), FALSE, TRUE),
    marker_clean = factor(marker_clean, levels = rev(marker_levels))
  )

p <- ggplot(route, aes(x = factor(size), y = marker_clean, fill = present)) +
  geom_tile(color = "white", linewidth = 0.4) +
  scale_fill_manual(
    values = c(`TRUE` = "#1B9E77", `FALSE` = "#F7F7F7"),
    labels = c(`TRUE` = "Included", `FALSE` = "Absent"),
    name = ""
  ) +
  labs(
    title = "Pathway Panel Growth Route at Milestones",
    subtitle = "Milestones show retention and new additions as the pathway-ordered panel expands.",
    x = "Panel size milestone",
    y = "Marker"
  ) +
  theme_cel() +
  theme(axis.text.y = element_text(size = 8))

ggsave(file.path(OUT_DIR, "fig05_panel_growth_route.pdf"), p, width = 7.5, height = 7, create.dir = TRUE)
