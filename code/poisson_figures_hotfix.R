library(tidyr)
library(ggplot2)
library(dplyr)

metrics <- read.csv(file = "code/poisson_data.csv")

## ---------------------------------------------------------------------------
## Settings
## ---------------------------------------------------------------------------

loss_levels <- c("MSE", "MAE", "POISSON", "GAUSSIAN", "NBINOM", "GLM")

## [CLAUDE FIX 22] viridis palette as used in the other figures, GLM in grey
loss_colors <- c(MSE      = "#440154",
                 MAE      = "#3B528B",
                 POISSON  = "#21908C",
                 GAUSSIAN = "#5DC863",
                 NBINOM   = "#C8C544",
                 GLM      = "grey60")

## internal column name -> label shown in the figure
metric_names  <- c(R2        = "R2",
                   RMSE_true = "RMSE (vs. true mean)",
                   Spearman  = "Spearman")
metric_levels <- names(metric_names)

dodge_w    <- 0.8
glm_offset <- -dodge_w / 2 + (length(loss_levels) - 0.5) * (dodge_w / length(loss_levels))

## reference model for the paired contrasts.
## "GLM"  -> does any DNN match the correctly specified baseline?
## "MSE"  -> does switching away from the default loss help?
reference_model <- "GLM"

## ---------------------------------------------------------------------------
## [NEW 1] Per-iteration accuracy values
##
## metrics has three rows per iteration x model (one per predictor), so the
## accuracy columns are duplicated. distinct() collapses them back to one row.
## Keeping the per-iteration values (rather than summarising immediately) is
## what makes SEs and significance tests possible.
## ---------------------------------------------------------------------------

acc_iter <- metrics |>
  distinct(Iteration, Loss_function, .keep_all = TRUE) |>
  select(Iteration, Loss_function, all_of(metric_levels)) |>
  pivot_longer(cols      = all_of(metric_levels),
               names_to  = "Accuracy",
               values_to = "Value") |>
  mutate(Accuracy      = factor(Accuracy, levels = metric_levels),
         Loss_function = factor(Loss_function, levels = loss_levels),
         Iteration     = factor(Iteration))

## ---------------------------------------------------------------------------
## Means and Monte Carlo standard errors
##
## SE = SD across iterations / sqrt(n). This is the uncertainty of the MEAN,
## i.e. how precisely 1000 iterations pin down the expected performance.
## It is NOT the spread between iterations, which is much larger (see the
## train-test boxplots in the appendix).
## ---------------------------------------------------------------------------

Accuracy_summary <- acc_iter |>
  group_by(Accuracy, Loss_function) |>
  summarise(n_iter = sum(!is.na(Value)),
            Mean   = mean(Value, na.rm = TRUE),
            SD     = sd(Value,   na.rm = TRUE),
            .groups = "drop") |>
  mutate(SE   = SD / sqrt(n_iter),
         ymin = Mean - 1.96 * SE,
         ymax = Mean + 1.96 * SE)

print(Accuracy_summary, n = Inf)

## ---------------------------------------------------------------------------
## [NEW 2] Repeated-measures ANOVA, one per metric
##
## All six models are fitted to the SAME simulated dataset in each iteration,
## so the six values within an iteration are not independent. A plain one-way
## aov(Value ~ Loss_function) would ignore that and throw away most of the
## power, because between-iteration variance is far larger than the
## differences between losses. Error(Iteration) blocks it out.
## ---------------------------------------------------------------------------

complete_iterations <- function(d) {
  ## keep only iterations in which every model produced a value,
  ## otherwise the blocked design is unbalanced
  d |>
    select(Iteration, Loss_function, Value) |>
    pivot_wider(names_from = Loss_function, values_from = Value) |>
    tidyr::drop_na() |>
    pull(Iteration)
}

rm_anova <- function(d) {
  keep <- complete_iterations(d)
  dd   <- d |> filter(Iteration %in% keep) |> droplevels()
  
  fit <- aov(Value ~ Loss_function + Error(Iteration), data = dd)
  tab <- summary(fit)[["Error: Within"]][[1]]
  
  data.frame(n_iter       = length(keep),
             df_num       = tab$Df[1],
             df_den       = tab$Df[2],
             F_value      = tab$`F value`[1],
             p_value      = tab$`Pr(>F)`[1],
             ## partial eta squared: share of within-iteration variance
             ## attributable to the loss function
             eta2_partial = tab$`Sum Sq`[1] / (tab$`Sum Sq`[1] + tab$`Sum Sq`[2]))
}

anova_results <- lapply(metric_levels, function(m) {
  cbind(Accuracy = m, rm_anova(filter(acc_iter, Accuracy == m)))
}) |> bind_rows()

cat("\n--- Repeated-measures ANOVA (accuracy) ---\n")
print(anova_results)

## ---------------------------------------------------------------------------
## [NEW 3] Paired contrasts against the reference model
##
## The omnibus ANOVA only says "something differs". These contrasts say what
## differs and by how much. Because the comparison is paired within iteration,
## the iteration-level variance cancels and the CI is narrow.
##
## Report the mean difference and its CI, not the p-value: at n = 1000 even a
## difference of 0.002 will come out significant, which is not the question.
## ---------------------------------------------------------------------------

paired_contrasts <- function(d, reference = reference_model) {
  wide <- d |>
    select(Iteration, Loss_function, Value) |>
    pivot_wider(names_from = Loss_function, values_from = Value) |>
    tidyr::drop_na()
  
  others <- setdiff(levels(d$Loss_function), reference)
  
  out <- lapply(others, function(lv) {
    dif <- wide[[lv]] - wide[[reference]]
    tt  <- t.test(dif)
    wt  <- suppressWarnings(wilcox.test(dif))
    data.frame(Loss_function = lv,
               n_iter        = length(dif),
               Mean_diff     = mean(dif),
               CI_low        = tt$conf.int[1],
               CI_high       = tt$conf.int[2],
               ## Cohen's dz for paired data: mean difference / SD of differences
               dz            = mean(dif) / sd(dif),
               p_t           = tt$p.value,
               p_wilcoxon    = wt$p.value)
  }) |> bind_rows()
  
  out$p_holm <- p.adjust(out$p_t, method = "holm")
  out
}

contrast_results <- lapply(metric_levels, function(m) {
  cbind(Accuracy = m, paired_contrasts(filter(acc_iter, Accuracy == m)))
}) |> bind_rows()

cat(sprintf("\n--- Paired contrasts vs %s (accuracy) ---\n", reference_model))
print(contrast_results, digits = 3)

write.csv(Accuracy_summary, "logs/accuracy_summary_poisson.csv",   row.names = FALSE)
write.csv(anova_results,    "logs/accuracy_anova_poisson.csv",     row.names = FALSE)
write.csv(contrast_results, "logs/accuracy_contrasts_poisson.csv", row.names = FALSE)

## ---------------------------------------------------------------------------
## Accuracy figure
## ---------------------------------------------------------------------------

accuracy_plot <- ggplot(Accuracy_summary,
                        aes(x = Accuracy, y = Mean, fill = Loss_function)) +
  geom_col(position = position_dodge(width = dodge_w), width = 0.7,
           color = "black", linewidth = 0.3) +
  ## [CLAUDE FIX 21] correct variable names; [CLAUDE FIX 20] trailing "+"
  geom_errorbar(aes(ymin = ymin, ymax = ymax),
                position = position_dodge(width = dodge_w),
                width = 0.2, linewidth = 0.3) +
  geom_text(aes(label = round(Mean, 3)),
            position = position_dodge(width = dodge_w),
            vjust = -0.4, size = 2.8, color = "gray20") +
  scale_x_discrete(labels = metric_names) +
  scale_fill_manual(values = loss_colors) +
  theme_minimal() +
  labs(title = "Model accuracy",
       x = "Accuracy metric", y = "Value", fill = "Loss function",
       caption = "Error bars: mean +/- 1.96 Monte Carlo SE over 1000 iterations")

ggsave("images/poisson_accuracy.pdf", plot = accuracy_plot,
       device = "pdf", width = 9, height = 5)
ggsave("images/poisson_accuracy.png", plot = accuracy_plot,
       device = "png", width = 9, height = 5, dpi = 600, bg = "white")

## ---------------------------------------------------------------------------
## Effects: same machinery, one test per predictor
##
## Note the error bars here mean something different from the accuracy figure:
## SE is the mean BOOTSTRAP SE, i.e. the typical uncertainty of a single
## estimate, not the uncertainty of the mean across iterations. Both are
## legitimate, but the caption has to say which one is shown.
## ---------------------------------------------------------------------------

Effects <- metrics |>
  group_by(Loss_function, Predictor) |>
  summarise(Effect = mean(Effect_size, na.rm = TRUE),
            SE     = mean(SE, na.rm = TRUE),
            .groups = "drop") |>
  mutate(Loss_function = factor(Loss_function, levels = loss_levels))

eff_iter <- metrics |>
  select(Iteration, Loss_function, Predictor, Value = Effect_size) |>
  mutate(Loss_function = factor(Loss_function, levels = loss_levels),
         Iteration     = factor(Iteration))

effects_anova <- lapply(unique(eff_iter$Predictor), function(p) {
  cbind(Predictor = p, rm_anova(filter(eff_iter, Predictor == p)))
}) |> bind_rows()

effects_contrasts <- lapply(unique(eff_iter$Predictor), function(p) {
  cbind(Predictor = p, paired_contrasts(filter(eff_iter, Predictor == p)))
}) |> bind_rows()

cat("\n--- Repeated-measures ANOVA (effects) ---\n")
print(effects_anova)
cat(sprintf("\n--- Paired contrasts vs %s (effects) ---\n", reference_model))
print(effects_contrasts, digits = 3)

write.csv(effects_anova,     "logs/effects_anova_poisson.csv",     row.names = FALSE)
write.csv(effects_contrasts, "logs/effects_contrasts_poisson.csv", row.names = FALSE)

predictor_xpos <- c(Environment1 = 1, Environment2 = 2, Environment3 = 3)
ame_labels <- metrics |>
  group_by(Predictor) |>
  summarise(AME_true = mean(AME_true, na.rm = TRUE), .groups = "drop") |>
  mutate(x = predictor_xpos[Predictor] + glm_offset + 0.09)

effects_plot <- ggplot(Effects, aes(x = Predictor, y = Effect, fill = Loss_function)) +
  geom_col(position = position_dodge(width = dodge_w), width = 0.7,
           color = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = Effect - SE, ymax = Effect + SE),
                position = position_dodge(width = dodge_w), width = 0.2) +
  geom_text(aes(label = round(Effect, 3)),
            position = position_dodge(width = dodge_w),
            vjust = 1.5, size = 2.8, color = "gray20") +
  geom_text(data = ame_labels,
            aes(x = x, y = AME_true, label = paste("True:", round(AME_true, 4))),
            hjust = 0, vjust = -0.6, size = 3, color = "firebrick",
            inherit.aes = FALSE) +
  scale_fill_manual(values = loss_colors) +
  theme_minimal() +
  labs(title = "Effect of environment", x = "Predictor", y = "Effect size",
       fill = "Loss function",
       caption = "Error bars: mean bootstrap SE of a single estimate")

ggsave("images/poisson_effects.pdf", plot = effects_plot,
       device = "pdf", width = 9, height = 5)
## [CLAUDE FIX 23] was "poisson_efffects.png"
ggsave("images/poisson_effects.png", plot = effects_plot,
       device = "png", width = 9, height = 5, dpi = 600, bg = "white")

## ---------------------------------------------------------------------------
## [NEW 4] Paired-difference figure
##
## This is what the AI feedback (B4) asked for: differences per iteration as a
## point with an interval, zero line marked. It shows size AND significance in
## one panel, which the bar chart cannot, because a difference of 0.01 is
## invisible on an axis running from 0 to 0.9.
## Use it instead of, or alongside, the bar chart.
## ---------------------------------------------------------------------------

diff_plot <- contrast_results |>
  mutate(Loss_function = factor(Loss_function, levels = loss_levels),
         Accuracy      = factor(Accuracy, levels = metric_levels)) |>
  ggplot(aes(x = Loss_function, y = Mean_diff, color = Loss_function)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_pointrange(aes(ymin = CI_low, ymax = CI_high), linewidth = 0.6) +
  facet_wrap(~ Accuracy, scales = "free_y",
             labeller = labeller(Accuracy = metric_names)) +
  scale_color_manual(values = loss_colors, guide = "none") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = paste("Paired difference vs", reference_model),
       x = NULL, y = paste("Difference from", reference_model),
       caption = "Point: mean paired difference over 1000 iterations. Bars: 95% CI.")

ggsave("images/poisson_accuracy_differences.png", plot = diff_plot,
       device = "png", width = 9, height = 4, dpi = 600, bg = "white")
