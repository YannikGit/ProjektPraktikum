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
library(pbmcapply)   
library(gridExtra)
library(grid)

RNGkind("L'Ecuyer-CMRG")
set.seed(42)

iter    <- 100
n_cores <- 12
n_boot  <- 20 

rmse <- function(preds, obs) sqrt(mean((preds - obs)^2))
mcfadden <- function(obs, preds, null_pred, eps = 1e-8) {
  ll_full <- sum(dpois(obs, lambda = pmax(preds, eps), log = TRUE))
  ll_null <- sum(dpois(obs, lambda = null_pred,        log = TRUE))
  1 - ll_full / ll_null
}

losses     <- c("mse", "mae", "poisson", "gaussian", "nbinom")
predictors <- c("Environment1", "Environment2", "Environment3")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------#
run_one_iteration <- function(i) {
  
  cat(sprintf("%s  iter %d\n", format(Sys.time(), "%H:%M:%S"), i),
      file = "logs/progress.txt", append = TRUE)  
  torch_manual_seed(i)  
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
  train_clean <- train[, setdiff(names(train), "group")] 
  null_pred <- mean(train$observedResponse)  
  
  
  #### GLM
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
    boot_ame[b, ] <- ame_b$estimate[match(predictors, ame_b$term)]
  }
  
  preds_glm       <- rowMeans(boot_preds_test)
  preds_glm_train <- rowMeans(boot_preds_train)
  glm_effect      <- colMeans(boot_ame)
  glm_SE          <- apply(boot_ame, 2, sd)
  glm_p           <- 2 * pnorm(-abs(glm_effect / glm_SE))
  
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
    R2_train              = accuracy_glm_train["R2"],     
    Neg_preds             = sum(preds_glm <= 0),        
    Predictor             = predictors,                     
    Effect_size           = glm_effect,                    
    SE                    = glm_SE,                         
    p_value               = glm_p,                         
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
                   optimizer  = config_optimizer("ignite_adam"), weight_decay = 0.01)# l2
                   epochs     = 200,
                   lr         = 0.003,
                   validation = 0.2,        
                   early_stopping = 20L,
                   bootstrap  = n_boot,    
                   bootstrap_parallel = 1L,
                   verbose    = FALSE,
                   plot       = FALSE)
    
    
    # Predictions
    preds       <- as.vector(predict(dnn_fit, newdata = test,  type = "response"))
    preds_train <- as.vector(predict(dnn_fit, newdata = train, type = "response"))
    
    # Accuracy metrics test
    accuracy_temp <- c(
      RMSE     = rmse(preds, test$observedResponse),
      RMSE_true = rmse(preds, test$true_mu),  
      Spearman = cor(preds, test$observedResponse, method = "spearman"),
      R2       = mcfadden(test$observedResponse, preds, null_pred)  
    )
    # Accuracy metrics train
    accuracy_train <- c(
      RMSE     = rmse(preds_train, train$observedResponse),
      RMSE_true = rmse(preds_train, train$true_mu),   
      Spearman = cor(preds_train, train$observedResponse, method = "spearman"),
      R2       = mcfadden(train$observedResponse, preds_train, null_pred) 
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
      Neg_preds               = sum(preds <= 0), 
      Effect_size             = effects_temp,
      SE                      = SE_temp,
      p_value                 = p_value_temp,
      row.names = NULL
    )
  }
  
  results_i <- merge(rbind(glm_row, do.call(rbind, iter_rows)), ame_reference[, c("Predictor","AME_true")], by = "Predictor")
  return(results_i)   
}

# -----------------------------------------------------------------------------------------------------------------------------------------------------------#
results_list <- pbmclapply(1:iter, run_one_iteration, mc.cores = n_cores, mc.set.seed = TRUE, mc.style = "ETA")


failed <- vapply(results_list, function(x) is.null(x) || inherits(x, "try-error"), logical(1))
if (any(failed)) {
  warning(sprintf("%d of %d iterations failed - inspect results_list[failed] for error messages",
                  sum(failed), iter))
}
 
metrics <- do.call(rbind, results_list[!failed])
#------------------------------------------------------------------------------------------------------------------------------------------------------------#
if (interactive()) View(metrics)  

write.csv(metrics, file = "code/poisson_data.csv", row.names = FALSE)

glm_ref <- metrics |> filter(Loss_function == "GLM")
dnn_metrics <- metrics |> filter(Loss_function != "GLM")
loss_colors <- c(
  MSE      = "#440154",
  MAE      = "#3B528B",
  POISSON  = "#21908C",
  GAUSSIAN = "#5DC863",
  NBINOM   = "#C8C544", 
  GLM      = "grey60"     # reference model
)

#### Coverage / bias table
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

cov_fmt <- coverage_table |>
  mutate(
    Loss_function = factor(Loss_function, levels = loss_levels),
    Coverage = sprintf("%.3f (%.3f)", coverage, coverage_MCSE),
    Bias     = sprintf("%+.3f (%.3f)", bias, bias_MCSE)
  ) |>
  arrange(Predictor, Loss_function) |>
  select(`Loss function` = Loss_function, Predictor, Coverage, Bias)

png("images/poisson_coverage_table.png", width = 1000, height = 900, res = 150)
grid.table(cov_fmt, rows = NULL)
dev.off()

####Accuracy metrics.
loss_levels <- c("MSE", "MAE", "POISSON", "GAUSSIAN", "NBINOM", "GLM")
metric_levels <- c("R2", "RMSE", "Spearman")
dodge_w <- 0.8

glm_offset <- -dodge_w / 2 + (length(loss_levels) - 0.5) * (dodge_w / length(loss_levels))

Accuracy_long <- metrics |>
  distinct(Iteration, Loss_function, .keep_all = TRUE) |>  
  group_by(Loss_function) |>
  summarise(RMSE     = mean(RMSE, na.rm = TRUE),
            Spearman = mean(Spearman, na.rm = TRUE),
            R2       = mean(R2, na.rm = TRUE),
            .groups  = "drop") |>
  pivot_longer(cols = c(RMSE, Spearman, R2),
               names_to = "Accuracy",
               values_to = "Value") |>
  mutate(Accuracy      = factor(Accuracy, levels = metric_levels),
         Loss_function = factor(Loss_function, levels = loss_levels))

accuracy_plot <- ggplot(Accuracy_long, aes(x = Accuracy, y = Value, fill = Loss_function)) +
  geom_col(position = position_dodge(width = dodge_w), width = 0.7,
           color = "black", linewidth = 0.3) +
  geom_text(aes(label = round(Value, 3)),
            position = position_dodge(width = dodge_w),
            vjust = -0.4, size = 2.8, color = "gray20") +
  scale_fill_manual(values = loss_colors) +
  theme_minimal() +
  labs(title = "Model accuracy", x = "Accuracy metric", y = "Value", fill = "Loss function")
ggsave("images/poisson_accuracy.pdf", plot = accuracy_plot, device = "pdf", width = 9, height = 5)
ggsave("images/poisson_accuracy.png", plot = accuracy_plot, device = "png", width = 9, height = 5, dpi = 600, bg = "white")

#### Effects metrics
Effects <- metrics |>
  group_by(Loss_function, Predictor) |>
  summarise(Effect   = mean(Effect_size, na.rm = TRUE),
            SE       = mean(SE, na.rm = TRUE),  
            .groups  = "drop") |>
  mutate(Loss_function = factor(Loss_function, levels = loss_levels))

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
  labs(title = "Effect of environment", x = "Predictor", y = "Effect size", fill = "Loss function")
ggsave("images/poisson_effects.pdf", plot = effects_plot, device = "pdf", width = 9, height = 5)
ggsave("images/poisson_efffects.png", plot = effects_plot, device = "png", width = 9, height = 5, dpi = 600, bg = "white")

#Training Accuracy (did the models converge?) -> appendix
metrics_long <- metrics |>
  distinct(Iteration, Loss_function, .keep_all = TRUE) |> 
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
  facet_wrap(~metric, scales = "frsee_y") +
  theme_minimal()

ggsave("images/poisson_overfit.pdf", plot = overfit_check, device = "pdf", width = 9, height = 5)
ggsave("images/poisson_overfit.png", plot = overfit_check, device = "png", width = 9, height = 5, dpi = 600, bg = "white")

