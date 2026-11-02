## ============================================================================
## Figures and model comparison for one scenario.
##
## Set `scenario` below and run. This file replaces poisson_figures_hotfix.R
## and gaussian_figures_hotfix.R: keeping two near-identical copies is how the
## AME_true bug got into the Gaussian version, so the two differ only in the
## config block at the top.
##
## Outputs
##   images/<tag>_accuracy_differences.png   main accuracy figure
##   images/<tag>_effects_bias.png           main effects figure
##   images/<tag>_accuracy_bars.png          appendix
##   images/<tag>_effects_bars.png           appendix
##   logs/<tag>_accuracy_table.csv           table for the report (Mean (SE))
##   logs/<tag>_{accuracy,effects}_{anova,contrasts}.csv
## ============================================================================

library(tidyr)
library(ggplot2)
library(dplyr)

## ---------------------------------------------------------------------------
## Config
## ---------------------------------------------------------------------------

scenario <- "poisson"          # "poisson" or "gaussian"

cfg <- switch(scenario,
              poisson  = list(file = "code/poisson_data.csv",
                              tag  = "poisson",
                              r2_label = "McFadden R2"),
              gaussian = list(file = "code/gaussian_data.csv",
                              tag  = "gaussian",
                              r2_label = "R2"),
              stop("scenario must be 'poisson' or 'gaussian'")
)

metrics <- read.csv(cfg$file)

loss_levels <- c("MSE", "MAE", "POISSON", "GAUSSIAN", "NBINOM", "GLM")

loss_colors <- c(MSE      = "#440154",
                 MAE      = "#3B528B",
                 POISSON  = "#21908C",
                 GAUSSIAN = "#5DC863",
                 NBINOM   = "#C8C544",
                 GLM      = "grey60")

## RMSE against the observed response is carried through so that the table and
## the contrasts can show how much the irreducible sampling variance compresses
## the differences; the figure shows only the three primary metrics.
metric_names  <- c(R2        = cfg$r2_label,
                   RMSE_true = "RMSE (vs. true mean)",
                   RMSE      = "RMSE (vs. observed)",
                   Spearman  = "Spearman")
metric_levels   <- names(metric_names)
figure_metrics  <- c("R2", "RMSE_true", "Spearman")

## GLM first, as the reference, in the table and on the figure axis
table_order <- c("GLM", "MSE", "MAE", "POISSON", "GAUSSIAN", "NBINOM")
diff_order  <- c("GLM", "MAE", "POISSON", "GAUSSIAN", "NBINOM")

## Reference for all paired contrasts.
## MSE is the default loss in practice, so "does switching away from the
## default help?" is the question the study actually asks. Referencing the
## GLM instead would make it impossible to show the GLM in the figure.
reference_model <- "MSE"

dodge_w <- 0.8

dir.create("logs",   showWarnings = FALSE)
dir.create("images", showWarnings = FALSE)

## If a metric column is named differently in the data, drop it rather than
## aborting (e.g. observed RMSE stored as RMSE_obs instead of RMSE).
.missing <- setdiff(metric_levels, names(metrics))
if (length(.missing)) {
  warning("metric column(s) not found and skipped: ",
          paste(.missing, collapse = ", "), call. = FALSE)
  metric_names   <- metric_names[setdiff(metric_levels, .missing)]
  metric_levels  <- names(metric_names)
  figure_metrics <- intersect(figure_metrics, metric_levels)
}

## ---------------------------------------------------------------------------
## Shared helpers
## ---------------------------------------------------------------------------

## Render a data frame as a PNG table, sized to fit its own content.
## Same gridExtra default theme as the coverage table, so the two match.
save_table_png <- function(df, path, base_size = 11, dpi = 300, pad = 0.3) {
  if (!requireNamespace("gridExtra", quietly = TRUE))
    stop("install.packages('gridExtra')")
  
  tg <- gridExtra::tableGrob(
    df, rows = NULL,
    theme = gridExtra::ttheme_default(base_size = base_size))
  
  w <- grid::convertWidth( sum(tg$widths),  "in", valueOnly = TRUE) + pad
  h <- grid::convertHeight(sum(tg$heights), "in", valueOnly = TRUE) + pad
  
  if (capabilities("cairo")) {
    png(path, width = w, height = h, units = "in", res = dpi,
        bg = "white", type = "cairo")
  } else {
    png(path, width = w, height = h, units = "in", res = dpi, bg = "white")
  }
  grid::grid.newpage()
  grid::grid.draw(tg)
  dev.off()
  invisible(path)
}

## keep only iterations in which every model produced a value, so the
## blocked design stays balanced
complete_iterations <- function(d) {
  d |>
    select(Iteration, Loss_function, Value) |>
    pivot_wider(names_from = Loss_function, values_from = Value) |>
    tidyr::drop_na() |>
    pull(Iteration)
}

## Repeated-measures ANOVA. All models are fitted to the SAME simulated data
## within an iteration, so Error(Iteration) blocks out the between-iteration
## variance, which is far larger than the differences of interest.
rm_anova <- function(d) {
  d    <- droplevels(d)          # needed when a loss has been filtered out
  keep <- complete_iterations(d)
  dd   <- d |> filter(Iteration %in% keep) |> droplevels()
  tab  <- summary(aov(Value ~ Loss_function + Error(Iteration),
                      data = dd))[["Error: Within"]][[1]]
  data.frame(n_iter       = length(keep),
             df_num       = tab$Df[1],
             df_den       = tab$Df[2],
             F_value      = tab$`F value`[1],
             p_value      = tab$`Pr(>F)`[1],
             eta2_partial = tab$`Sum Sq`[1] / (tab$`Sum Sq`[1] + tab$`Sum Sq`[2]))
}

## Paired differences against the reference, within iteration. Pairing removes
## the iteration-level variance, so the CI is far narrower than the raw spread.
## Report Mean_diff and the CI; at n = 1000 the p-values are not informative.
paired_contrasts <- function(d, reference = reference_model) {
  wide <- d |>
    select(Iteration, Loss_function, Value) |>
    pivot_wider(names_from = Loss_function, values_from = Value) |>
    tidyr::drop_na()
  
  others <- setdiff(levels(droplevels(d$Loss_function)), reference)
  
  out <- lapply(others, function(lv) {
    dif <- wide[[lv]] - wide[[reference]]
    tt  <- t.test(dif)
    data.frame(Loss_function = lv,
               n_iter    = length(dif),
               Mean_diff = mean(dif),
               CI_low    = tt$conf.int[1],
               CI_high   = tt$conf.int[2],
               dz        = mean(dif) / sd(dif),   # Cohen's dz, paired
               p_t       = tt$p.value)
  }) |> bind_rows()
  
  out$p_holm <- p.adjust(out$p_t, method = "holm")
  out
}

## ===========================================================================
## ACCURACY
## ===========================================================================

## metrics has three rows per iteration x model (one per predictor), so the
## accuracy columns are duplicated; distinct() collapses them back to one row.
acc_iter <- metrics |>
  distinct(Iteration, Loss_function, .keep_all = TRUE) |>
  select(Iteration, Loss_function, all_of(metric_levels)) |>
  pivot_longer(cols = all_of(metric_levels),
               names_to = "Accuracy", values_to = "Value") |>
  mutate(Accuracy      = factor(Accuracy, levels = metric_levels),
         Loss_function = factor(Loss_function, levels = loss_levels),
         Iteration     = factor(Iteration))

## Means and Monte Carlo SEs. SE = SD/sqrt(n) is the uncertainty of the MEAN,
## not the spread between iterations (which is roughly 30x larger).
Accuracy_summary <- acc_iter |>
  group_by(Accuracy, Loss_function) |>
  summarise(n_iter = sum(!is.na(Value)),
            Mean   = mean(Value, na.rm = TRUE),
            SD     = sd(Value,   na.rm = TRUE),
            .groups = "drop") |>
  mutate(SE   = SD / sqrt(n_iter),
         ymin = Mean - 1.96 * SE,
         ymax = Mean + 1.96 * SE)

## Table 1 for the report: one row per model, "Mean (SE)" per metric.
## pivot_wider orders the new columns by the factor levels of Accuracy, so the
## column order follows metric_levels and the renaming below lines up.
accuracy_table <- Accuracy_summary |>
  mutate(cell = sprintf("%.3f (%.3f)", Mean, SE)) |>
  select(Accuracy, Loss_function, cell) |>
  pivot_wider(names_from = Accuracy, values_from = cell) |>
  mutate(Loss_function = factor(Loss_function, levels = table_order)) |>
  arrange(Loss_function) |>
  as.data.frame()

names(accuracy_table) <- c("Loss function", unname(metric_names[metric_levels]))

print(accuracy_table)
write.csv(accuracy_table, sprintf("logs/%s_accuracy_table.csv", cfg$tag),
          row.names = FALSE)
save_table_png(accuracy_table,
               sprintf("images/%s_accuracy_table.png", cfg$tag))
write.csv(Accuracy_summary, sprintf("logs/%s_accuracy_summary.csv", cfg$tag),
          row.names = FALSE)

## ANOVA, with and without MAE.
## Without MAE tests whether the remaining losses still differ systematically,
## or whether the omnibus result was driven by the one broken model.
acc_anova <- bind_rows(
  lapply(metric_levels, function(m)
    cbind(Accuracy = m, Models = "all six",
          rm_anova(filter(acc_iter, Accuracy == m)))),
  lapply(metric_levels, function(m)
    cbind(Accuracy = m, Models = "excl. MAE",
          rm_anova(filter(acc_iter, Accuracy == m, Loss_function != "MAE"))))
) |> arrange(Accuracy, Models)

cat("\n--- Repeated-measures ANOVA: accuracy ---\n"); print(acc_anova)

acc_contrasts <- lapply(metric_levels, function(m)
  cbind(Accuracy = m, paired_contrasts(filter(acc_iter, Accuracy == m)))
) |> bind_rows()

cat(sprintf("\n--- Paired contrasts vs %s: accuracy ---\n", reference_model))
print(acc_contrasts, digits = 3)

write.csv(acc_anova,     sprintf("logs/%s_accuracy_anova.csv", cfg$tag),     row.names = FALSE)
write.csv(acc_contrasts, sprintf("logs/%s_accuracy_contrasts.csv", cfg$tag), row.names = FALSE)

## --- main accuracy figure: paired differences vs MSE -----------------------
## A difference of 0.02 is invisible on an axis running 0 to 0.9, but perfectly
## readable as a point-and-interval against a zero line.
diff_plot <- acc_contrasts |>
  filter(Accuracy %in% figure_metrics) |>
  mutate(Loss_function = factor(Loss_function, levels = diff_order),
         Accuracy      = factor(Accuracy, levels = figure_metrics)) |>
  ggplot(aes(x = Loss_function, y = Mean_diff, color = Loss_function)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_pointrange(aes(ymin = CI_low, ymax = CI_high), linewidth = 0.6, size = 0.5) +
  facet_wrap(~ Accuracy, scales = "free_y",
             labeller = labeller(Accuracy = metric_names)) +
  scale_color_manual(values = loss_colors, guide = "none") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = NULL,
       y = sprintf("Difference from %s", reference_model),
       caption = sprintf(
         "Points: mean paired difference from %s over 1000 iterations. Bars: 95%% CI.",
         reference_model))

ggsave(sprintf("images/%s_accuracy_differences.png", cfg$tag), diff_plot,
       width = 9, height = 4, dpi = 600, bg = "white")
ggsave(sprintf("images/%s_accuracy_differences.pdf", cfg$tag), diff_plot,
       width = 9, height = 4)

## --- appendix: absolute values as bars -------------------------------------
accuracy_plot <- ggplot(Accuracy_summary,
                        aes(x = Accuracy, y = Mean, fill = Loss_function)) +
  geom_col(position = position_dodge(width = dodge_w), width = 0.7,
           color = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = ymin, ymax = ymax),
                position = position_dodge(width = dodge_w),
                width = 0.2, linewidth = 0.3) +
  scale_x_discrete(labels = metric_names) +
  scale_fill_manual(values = loss_colors) +
  theme_minimal(base_size = 11) +
  labs(x = "Accuracy metric", y = "Value", fill = "Loss function",
       caption = "Error bars: mean +/- 1.96 Monte Carlo SE over 1000 iterations")

ggsave(sprintf("images/%s_accuracy_bars.png", cfg$tag), accuracy_plot,
       width = 9, height = 5, dpi = 600, bg = "white")

## ===========================================================================
## EFFECTS
## ===========================================================================

## Bias per iteration. AME_true is the true average effect for that iteration;
## under the log link it depends on the realised mean and so varies between
## iterations, which is why it is differenced row-wise rather than against a
## single constant.
bias_iter <- metrics |>
  select(Iteration, Loss_function, Predictor, Effect_size, AME_true) |>
  mutate(Value         = Effect_size - AME_true,
         Loss_function = factor(Loss_function, levels = loss_levels),
         Iteration     = factor(Iteration))

bias_summary <- bias_iter |>
  group_by(Predictor, Loss_function) |>
  summarise(n_iter = sum(!is.na(Value)),
            Bias   = mean(Value, na.rm = TRUE),
            SD     = sd(Value,   na.rm = TRUE),
            .groups = "drop") |>
  mutate(SE      = SD / sqrt(n_iter),
         CI_low  = Bias - 1.96 * SE,
         CI_high = Bias + 1.96 * SE)

cat("\n--- Bias against the true effect ---\n"); print(bias_summary, n = Inf)
write.csv(bias_summary, sprintf("logs/%s_effects_bias.csv", cfg$tag), row.names = FALSE)

eff_iter <- metrics |>
  select(Iteration, Loss_function, Predictor, Value = Effect_size) |>
  mutate(Loss_function = factor(Loss_function, levels = loss_levels),
         Iteration     = factor(Iteration))

predictors <- unique(eff_iter$Predictor)

eff_anova <- bind_rows(
  lapply(predictors, function(p)
    cbind(Predictor = p, Models = "all six",
          rm_anova(filter(eff_iter, Predictor == p)))),
  lapply(predictors, function(p)
    cbind(Predictor = p, Models = "excl. MAE",
          rm_anova(filter(eff_iter, Predictor == p, Loss_function != "MAE"))))
) |> arrange(Predictor, Models)

eff_contrasts <- lapply(predictors, function(p)
  cbind(Predictor = p, paired_contrasts(filter(eff_iter, Predictor == p)))
) |> bind_rows()

cat("\n--- Repeated-measures ANOVA: effects ---\n"); print(eff_anova)
cat(sprintf("\n--- Paired contrasts vs %s: effects ---\n", reference_model))
print(eff_contrasts, digits = 3)

write.csv(eff_anova,     sprintf("logs/%s_effects_anova.csv", cfg$tag),     row.names = FALSE)
write.csv(eff_contrasts, sprintf("logs/%s_effects_contrasts.csv", cfg$tag), row.names = FALSE)

## --- main effects figure: bias against the truth ---------------------------
## Zero means unbiased. This shows size and significance in one panel, which
## the bar chart of absolute estimates cannot.
bias_plot <- bias_summary |>
  mutate(Loss_function = factor(Loss_function, levels = table_order)) |>
  ggplot(aes(x = Loss_function, y = Bias, color = Loss_function)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_pointrange(aes(ymin = CI_low, ymax = CI_high), linewidth = 0.6, size = 0.5) +
  facet_wrap(~ Predictor, scales = "free_y") +
  scale_color_manual(values = loss_colors, guide = "none") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = NULL, y = "Bias (estimate - true effect)",
       caption = "Points: mean bias over 1000 iterations. Bars: 95% CI. Zero = unbiased.")

ggsave(sprintf("images/%s_effects_bias.png", cfg$tag), bias_plot,
       width = 9, height = 4, dpi = 600, bg = "white")
ggsave(sprintf("images/%s_effects_bias.pdf", cfg$tag), bias_plot,
       width = 9, height = 4)

## --- appendix: absolute estimates as bars ----------------------------------
Effects <- metrics |>
  group_by(Loss_function, Predictor) |>
  summarise(Effect = mean(Effect_size, na.rm = TRUE),
            SE     = mean(SE, na.rm = TRUE),
            .groups = "drop") |>
  mutate(Loss_function = factor(Loss_function, levels = loss_levels))

ame_labels <- metrics |>
  group_by(Predictor) |>
  summarise(AME_true = mean(AME_true, na.rm = TRUE), .groups = "drop")

effects_plot <- ggplot(Effects, aes(x = Predictor, y = Effect, fill = Loss_function)) +
  geom_col(position = position_dodge(width = dodge_w), width = 0.7,
           color = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = Effect - SE, ymax = Effect + SE),
                position = position_dodge(width = dodge_w), width = 0.2) +
  geom_hline(data = ame_labels, aes(yintercept = AME_true),
             linetype = "dashed", color = "firebrick", linewidth = 0.4,
             inherit.aes = FALSE) +
  facet_wrap(~ Predictor, scales = "free", nrow = 1) +
  scale_fill_manual(values = loss_colors) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_blank()) +
  labs(x = "Predictor", y = "Effect size", fill = "Loss function",
       caption = paste("Error bars: mean bootstrap SE of a single estimate.",
                       "Red line: true average effect."))

ggsave(sprintf("images/%s_effects_bars.png", cfg$tag), effects_plot,
       width = 9, height = 5, dpi = 600, bg = "white")




coverage_table <- metrics |>
  mutate(error   = Effect_size - AME_true,
         covered = abs(error) <= qnorm(0.975) * SE) |>
  group_by(Loss_function, Predictor) |>
  summarise(n             = sum(!is.na(covered)),
            coverage      = mean(covered, na.rm = TRUE),
            coverage_MCSE = sqrt(coverage * (1 - coverage) / n),
            bias          = mean(error, na.rm = TRUE),
            bias_MCSE     = sd(error, na.rm = TRUE) / sqrt(n),
            .groups       = "drop")

if (interactive()) View(coverage_table)
write.csv(coverage_table, file = "code/poisson_noTuning_coverage.csv", row.names = FALSE)