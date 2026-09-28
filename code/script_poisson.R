###############################################################################
# Legend for review comments:
#   [CLAUDE FIX n]  = changed by Claude, see chat for reasoning
#   [CLAUDE NOTE]   = NOT changed, but worth a decision from you
#   FIX 1-14  = first review round
#   FIX 15-18 = second round (validation, sample size, bootstrapped GLM, coverage table)
###############################################################################

library(DHARMa)
library(cito)
library(torch)
torch_set_num_threads(1)
torch_set_num_interop_threads(1)
library(tidyr)
library(ggplot2)
library(dplyr)
library(parallel)
library(marginaleffects)
library(stringr)
library(pbmcapply)   # [CLAUDE FIX 1] replaces progressr (progressr does not relay progress from mclapply forks). install.packages("pbmcapply")

# [CLAUDE FIX 2] reproducibility: parallel-safe RNG + seed (mc.set.seed = TRUE needs L'Ecuyer to give reproducible streams)
RNGkind("L'Ecuyer-CMRG")
set.seed(42)

iter    <- 100
n_cores <- 20
n_boot  <- 20   # [CLAUDE FIX 17] one bootstrap count for GLM AND DNN, so both always use the same number
# [CLAUDE NOTE] 20 bootstrap replicates give a fairly noisy SE per fit (roughly +-16% relative error of the SE itself).
# Fine for a 6-week project, but if compute allows, 50 would stabilise coverage estimates.

rmse <- function(preds, obs) sqrt(mean((preds - obs)^2))

# [CLAUDE FIX 3] one McFadden helper used everywhere (GLM + DNN, train + test), so every model is measured with the same ruler.
# pmax() clamps non-positive predictions (MSE/MAE/GAUSSIAN losses can predict < 0). Before, dpois() returned NaN for these and
# na.rm = TRUE silently dropped them, which made the log-likelihood look BETTER than it was. eps is arbitrary -> see Neg_preds column.
mcfadden <- function(obs, preds, null_pred, eps = 1e-8) {
  ll_full <- sum(dpois(obs, lambda = pmax(preds, eps), log = TRUE))
  ll_null <- sum(dpois(obs, lambda = null_pred,        log = TRUE))
  1 - ll_full / ll_null
}

losses     <- c("mse", "mae", "poisson", "gaussian", "nbinom")
predictors <- c("Environment1", "Environment2", "Environment3")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------#
run_one_iteration <- function(i) {
  
  torch_manual_seed(i)   # [CLAUDE FIX 2b] torch has its own RNG (weight init), R's set.seed does not reach it
  
  #### Data Creation
  # [CLAUDE FIX 15] sampleSize 200 -> 250: after the 80/20 train-test split (200 train) and validation = 0.2 inside the DNN,
  # the DNN still fits on 160 rows, same as before validation was switched on.
  sim <- createData(sampleSize    = 250,
                    intercept    = -1,
                    fixedEffects = c(2, 0.4, 0.1),
                    overdispersion       = 0,
                    family               = poisson(),
                    randomEffectVariance = 0)
  sim$true_mu <- exp(-1 + 2 * sim$Environment1 + 0.4 * sim$Environment2 + 0.1 * sim$Environment3)
  true_effects <- c(Environment1 = 2, Environment2 = 0.4, Environment3 = 0.1)
  trainID <- sample(x = sim$ID, size = 0.8 * length(sim$ID))
  train <- sim[trainID, ]
  test  <- sim[-trainID, ]
  ame_reference <- data.frame(Predictor = names(true_effects),AME_true  = true_effects * mean(train$true_mu)) 
  train_clean <- train[, setdiff(names(train), "group")] #Somehow needed for avg_slopes AME calculation, as sim includes group
  null_pred <- mean(train$observedResponse)   # [CLAUDE FIX 3] shared null model for all McFadden calculations
  
  
  #### GLM
  # [CLAUDE FIX 16] bootstrapped GLM, mirroring what cito does with bootstrap = n_boot:
  #   - each replicate refits the GLM on a resample (rows drawn WITH replacement) of the training data
  #   - predictions = mean over replicates (bagged, like the DNN ensemble)
  #   - effect      = mean of the replicate AMEs, SE = SD of the replicate AMEs, p from a z-test (estimate / SE)
  # AMEs are always evaluated on the ORIGINAL training data, because that is where AME_true is defined.
  # vcov = FALSE skips the (unused) delta-method SEs in avg_slopes -> faster.
  # [CLAUDE NOTE] mean/SD/z-test is, to my knowledge, how cito's summary() reports bootstrap ACEs. If you want to be 100% sure
  # both sides are identical, check the cito source/docs for summary.citodnnBootstrap.
  boot_preds_test  <- matrix(NA_real_, nrow = nrow(test),  ncol = n_boot)
  boot_preds_train <- matrix(NA_real_, nrow = nrow(train), ncol = n_boot)
  boot_ame         <- matrix(NA_real_, nrow = n_boot, ncol = length(predictors), dimnames = list(NULL, predictors))
  
  for (b in seq_len(n_boot)) {
    boot_data <- train_clean[sample(nrow(train_clean), replace = TRUE), ]
    glm_b <- glm(formula = observedResponse ~ Environment1 + Environment2 + Environment3,
                 data = boot_data, family = poisson)
    boot_preds_test[, b]  <- predict(glm_b, newdata = test,  type = "response")
    boot_preds_train[, b] <- predict(glm_b, newdata = train, type = "response")
    ame_b <- avg_slopes(glm_b, newdata = train_clean, variables = predictors, vcov = FALSE)
    boot_ame[b, ] <- ame_b$estimate[match(predictors, ame_b$term)]   # match() guarantees correct predictor order
  }
  
  preds_glm       <- rowMeans(boot_preds_test)
  preds_glm_train <- rowMeans(boot_preds_train)
  glm_effect      <- colMeans(boot_ame)
  glm_SE          <- apply(boot_ame, 2, sd)
  glm_p           <- 2 * pnorm(-abs(glm_effect / glm_SE))
  
  # [CLAUDE FIX 4] GLM test R2 is now out-of-sample McFadden, exactly like the DNNs (was in-sample before, and R2_train was a copy)
  accuracy_glm <- c(
    RMSE     = rmse(preds_glm, test$observedResponse),
    RMSE_true = rmse(preds_glm, test$true_mu),
    Spearman = cor(preds_glm, test$observedResponse, method = "spearman"),
    R2       = mcfadden(test$observedResponse, preds_glm, null_pred)
  )
  accuracy_glm_train <- c(
    RMSE      = rmse(preds_glm_train, train$observedResponse),
    RMSE_true = rmse(preds_glm_train, train$true_mu),
    Spearman  = cor(preds_glm_train, train$observedResponse, method = "spearman"),
    R2        = mcfadden(train$observedResponse, preds_glm_train, null_pred)
  )
  
  glm_row <- data.frame(
    Iteration            = i,
    Loss_function         = "GLM",
    RMSE                  = accuracy_glm["RMSE"],
    RMSE_true             = accuracy_glm["RMSE_true"],
    Spearman              = accuracy_glm["Spearman"],
    R2                    = accuracy_glm["R2"],
    RMSE_train            = accuracy_glm_train["RMSE"],
    RMSE_true_train       = accuracy_glm_train["RMSE_true"],
    Spearman_train        = accuracy_glm_train["Spearman"],
    R2_train              = accuracy_glm_train["R2"],       # [CLAUDE FIX 4]
    Neg_preds             = sum(preds_glm <= 0),            # [CLAUDE FIX 5] always 0 for a Poisson GLM, needed so rbind() columns match
    Predictor             = predictors,                     # [CLAUDE FIX 16]
    Effect_size           = glm_effect,                     # [CLAUDE FIX 16]
    SE                    = glm_SE,                         # [CLAUDE FIX 16]
    p_value               = glm_p,                          # [CLAUDE FIX 16]
    row.names = NULL
  )
  
  
  #### DNN
  iter_rows <- vector("list", length(losses))
  for (l in seq_along(losses)) {
    dnn_fit <- dnn(formula    = observedResponse ~ Environment1 + Environment2 + Environment3,
                   data       = train,
                   hidden     = c(50L, 50L),
                   activation = "relu",
                   loss       = losses[l],
                   optimizer  = config_optimizer("ignite_adam", weight_decay = 0.01), # l2
                   epochs     = 200,
                   lr         = 0.003,
                   validation = 0.2,          # [CLAUDE FIX 15] early stopping now monitors held-out loss instead of training loss
                   early_stopping = 20L,
                   bootstrap  = n_boot,       # [CLAUDE FIX 17]
                   # predict() returns the MEAN over the bootstrap nets (bagged ensemble); the GLM now does the same (FIX 16)
                   bootstrap_parallel = 1L,
                   verbose    = FALSE,
                   plot       = FALSE)
    
    
    # Predictions
    # [CLAUDE FIX 6] as.vector(): bootstrap predict() returns an n x 1 matrix; cor() on a matrix returns a 1x1 matrix.
    preds       <- as.vector(predict(dnn_fit, newdata = test,  type = "response"))
    preds_train <- as.vector(predict(dnn_fit, newdata = train, type = "response"))
    
    # Accuracy metrics test
    accuracy_temp <- c(
      RMSE     = rmse(preds, test$observedResponse),
      RMSE_true = rmse(preds, test$true_mu),   # against the true, noise-free DGP mean
      Spearman = cor(preds, test$observedResponse, method = "spearman"),
      R2       = mcfadden(test$observedResponse, preds, null_pred)          # [CLAUDE FIX 3]
    )
    # Accuracy metrics train
    accuracy_train <- c(
      RMSE     = rmse(preds_train, train$observedResponse),
      RMSE_true = rmse(preds_train, train$true_mu),   # against the true, noise-free DGP mean
      Spearman = cor(preds_train, train$observedResponse, method = "spearman"),
      R2       = mcfadden(train$observedResponse, preds_train, null_pred)   # [CLAUDE FIX 7] was loglik_f / loglik_0 (missing "1 -")
    )
    
    # Effects and summary metrics
    sm <- summary(dnn_fit, type = "response", n_permute = 1)    
    effects_temp <- sm$ACE[, 1] |> as.numeric()
    SE_temp      <- sm$ACE[, 2] |> as.numeric()
    p_value_temp <- sm$ACE[, 4] |> as.numeric()
    
    iter_rows[[l]] <- data.frame(
      Iteration              = i,
      Loss_function           = toupper(losses[l]),
      Predictor               = predictors,
      RMSE                    = accuracy_temp["RMSE"],
      RMSE_true               = accuracy_temp["RMSE_true"],
      R2                      = accuracy_temp["R2"],
      Spearman                = accuracy_temp["Spearman"],
      RMSE_train              = accuracy_train["RMSE"],
      RMSE_true_train         = accuracy_train["RMSE_true"],
      Spearman_train          = accuracy_train["Spearman"],
      R2_train                = accuracy_train["R2"],
      Neg_preds               = sum(preds <= 0),   # [CLAUDE FIX 5] how many test predictions were impossible for count data
      Effect_size             = effects_temp,
      SE                      = SE_temp,
      p_value                 = p_value_temp,
      row.names = NULL
    )
  }
  
  results_i <- merge(rbind(glm_row, do.call(rbind, iter_rows)), ame_reference[, c("Predictor","AME_true")], by = "Predictor")
  return(results_i)   # [CLAUDE FIX 8] explicit return
}

# -----------------------------------------------------------------------------------------------------------------------------------------------------------#
# [CLAUDE FIX 1] pbmclapply = drop-in replacement for mclapply with progress bar + ETA
results_list <- pbmclapply(1:iter, run_one_iteration, mc.cores = n_cores, mc.set.seed = TRUE, mc.style = "ETA")

# Check for worker failures before combining - a failed fork returns a try-error object
# [CLAUDE FIX 9] also catch NULL (a worker killed e.g. by running out of memory returns NULL, which rbind would drop silently)
failed <- vapply(results_list, function(x) is.null(x) || inherits(x, "try-error"), logical(1))
if (any(failed)) {
  warning(sprintf("%d of %d iterations failed - inspect results_list[failed] for error messages",
                  sum(failed), iter))
}

metrics <- do.call(rbind, results_list[!failed])
#------------------------------------------------------------------------------------------------------------------------------------------------------------#
if (interactive()) View(metrics)   # [CLAUDE FIX 10] View() errors when the script runs non-interactively (e.g. Rscript)

write.csv(metrics, file = "code/poisson_data.csv", row.names = FALSE)

glm_ref <- metrics |> filter(Loss_function == "GLM")
dnn_metrics <- metrics |> filter(Loss_function != "GLM")


#### Coverage / bias table
# [CLAUDE FIX 18] performance measures following Morris, White & Crowther (2019), Statistics in Medicine 38(11).
# Each estimate is compared to ITS OWN iteration's AME_true (AME_true changes with mean(train$true_mu) every iteration).
#   coverage      = share of iterations where estimate +- 1.96*SE contains AME_true (should be ~0.95 if SEs are honest)
#   coverage_MCSE = Monte Carlo SE of coverage -> with 100 iterations ~0.02, so e.g. 0.93 vs 0.95 is NOT a meaningful difference
#   bias          = mean(estimate - truth); bias_MCSE = its Monte Carlo SE
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
write.csv(coverage_table, file = "code/poisson_coverage.csv", row.names = FALSE)


####Accuracy metrics
glm_accuracy_lines <- glm_ref |>
  summarise(
    RMSE     = mean(RMSE, na.rm = TRUE),
    Spearman = mean(Spearman, na.rm = TRUE),
    R2       = mean(R2, na.rm = TRUE)
  ) |>
  pivot_longer(everything(), names_to = "Accuracy", values_to = "glm_value")

Accuracy <- dnn_metrics |>
  group_by(Loss_function) |>
  summarise(RMSE     = mean(RMSE, na.rm = TRUE),
            Spearman = mean(Spearman, na.rm = TRUE),
            R2       = mean(R2, na.rm = TRUE))

metric_xpos <- c(R2 = 1, RMSE = 2, Spearman = 3)
bar_halfwidth <- 0.39  # adjust

Accuracy_long <- Accuracy |>
  pivot_longer(cols = c(RMSE, Spearman, R2),
               names_to = "Accuracy",
               values_to = "Value") |>
  mutate(Accuracy = factor(Accuracy, levels = names(metric_xpos)))   # [CLAUDE FIX 11] lock x order to metric_xpos

glm_accuracy_segments <- glm_accuracy_lines |>
  mutate(
    x    = metric_xpos[Accuracy] - bar_halfwidth,
    xend = metric_xpos[Accuracy] + bar_halfwidth
  )

accuracy_plot <- ggplot(Accuracy_long, aes(x = Accuracy, y = Value, fill = Loss_function)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7,
           color = "black", linewidth = 0.3) +
  geom_text(aes(label = round(Value, 3)),
            position = position_dodge(width = 0.8),
            vjust = -0.4, size = 2.8, color = "gray20") +
  geom_segment(data = glm_accuracy_segments,
               aes(x = x, xend = xend, y = glm_value, yend = glm_value),
               color = "gray30", linewidth = 0.8, linetype = "solid",
               inherit.aes = FALSE) +
  geom_text(data = glm_accuracy_segments,
            aes(x = xend, y = glm_value, label = paste("GLM:", round(glm_value, 3))),
            hjust = -0.1, vjust = 0, size = 3, color = "gray30",
            inherit.aes = FALSE) +
  scale_fill_manual(values = c(MAE      = "#56B4E9",
                               MSE      = "#0072B2",
                               GAUSSIAN = "#009E73",
                               POISSON  = "#6B969F",
                               NBINOM   = "#8EBEC7")) +
  theme_minimal() +
  labs(title = "Model accuracy", x = "Accuracy metric", y = "Value", fill = "Loss function")
# [CLAUDE FIX 12] plot = accuracy_plot (was effects_plot)   [CLAUDE FIX 13] fixed width/height
ggsave("images/poisson_accuracy.pdf", plot = accuracy_plot, device = "pdf", width = 9, height = 5)

####Effects:

glm_effects_lines <- glm_ref |>
  group_by(Predictor) |>
  summarise(glm_effect  = mean(Effect_size, na.rm = TRUE),
            AME_true    = mean(AME_true, na.rm = TRUE))

Effects <- dnn_metrics |>
  group_by(Loss_function, Predictor) |>
  summarise(Effect   = mean(Effect_size, na.rm = TRUE),
            SE       = mean(SE, na.rm = TRUE),   # average bootstrap SE per fit (calibration is in coverage_table, FIX 18)
            .groups  = "drop")

predictor_xpos <- c(Environment1 = 1, Environment2 = 2, Environment3 = 3)
bar_halfwidth  <- 0.39

glm_effects_segments <- glm_effects_lines |>
  mutate(
    x    = predictor_xpos[Predictor] - bar_halfwidth,
    xend = predictor_xpos[Predictor] + bar_halfwidth
  )

effects_plot <- ggplot(Effects, aes(x = Predictor, y = Effect, fill = Loss_function)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7,
           color = "black", linewidth = 0.3) +
  geom_text(aes(label = round(Effect, 3)),
            position = position_dodge(width = 0.8),
            vjust = 1.5, size = 2.8, color = "gray20") +
  geom_errorbar(aes(ymin = Effect - SE, ymax = Effect + SE),
                position = position_dodge(width = 0.8), width = 0.2) +
  # GLM reference segment
  geom_segment(data = glm_effects_segments,
               aes(x = x, xend = xend, y = glm_effect, yend = glm_effect),
               color = "gray30", linewidth = 0.8, linetype = "solid",
               inherit.aes = FALSE) +
  geom_text(data = glm_effects_segments,
            aes(x = xend, y = glm_effect, label = paste("GLM:", round(glm_effect, 3))),
            hjust = -0.1, vjust = -0, size = 3, color = "gray30",
            inherit.aes = FALSE) +
  # True AME reference segment
  geom_segment(data = glm_effects_segments,
               aes(x = x, xend = xend, y = AME_true, yend = AME_true),
               color = "firebrick", linewidth = 0.8, linetype = "solid",
               inherit.aes = FALSE) +
  geom_text(data = glm_effects_segments,
            aes(x = xend, y = AME_true, label = paste("True:", round(AME_true, 3))),
            hjust = -0.1, vjust = -2, size = 3, color = "firebrick",
            inherit.aes = FALSE) +
  scale_fill_manual(values = c(MAE      = "#E69F00",
                               MSE      = "#D55E00",
                               GAUSSIAN = "#F0E442",
                               POISSON  = "#9B5A21",
                               NBINOM   = "#EBC711")) +
  theme_minimal() +
  labs(title = "Effect of environment", x = "Predictor", y = "Effect size", fill = "Loss function")
ggsave("images/poisson_effects.pdf", plot = effects_plot, device = "pdf", width = 9, height = 5)   # [CLAUDE FIX 13]

#Training Accuracy (did the models converge?) -> appendix
metrics_long <- metrics |>
  distinct(Iteration, Loss_function, .keep_all = TRUE) |>   # [CLAUDE FIX 14] one accuracy row per model, not 3 duplicates
  pivot_longer(
    cols = c(RMSE, RMSE_train, Spearman, Spearman_train, R2, R2_train),
    names_to = "metric_raw",
    values_to = "value"
  ) |>
  mutate(
    split = if_else(str_detect(metric_raw, "_train"), "train", "test"),
    metric = str_remove(metric_raw, "_train")
  )

overfit_check <- ggplot(metrics_long, aes(x = Loss_function, y = value, fill = split)) +
  geom_boxplot() +
  facet_wrap(~metric, scales = "free_y") +
  theme_minimal()

ggsave("images/overfit_poisson.pdf", plot = overfit_check, device = "pdf", width = 9, height = 5)   # [CLAUDE FIX 13]
