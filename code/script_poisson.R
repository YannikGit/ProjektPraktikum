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
library(progressr)

iter <- 100
n_cores <- 20
rmse <- function(preds, obs) sqrt(mean((preds - obs)^2))

losses     <- c("mse", "mae", "poisson", "gaussian", "nbinom")
predictors <- c("Environment1", "Environment2", "Environment3")

handlers(handler_progress(
  format = "[:bar] :percent | Iteration :current/:total | Elapsed: :elapsed | ETA: :eta",
  width  = 80
))

# -----------------------------------------------------------------------------------------------------------------------------------------------------------#
run_one_iteration <- function(i) {
  
  #### Data Creation
  sim <- createData(sampleSize    = 200,
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
  
  
  #### GLM
  glm_fit <- glm(formula = observedResponse ~ Environment1 + Environment2 + Environment3,
                 data = train_clean, family = poisson)
  preds_glm <- predict(glm_fit, newdata = test, type = "response")
  preds_glm_train <- predict(glm_fit, newdata =  train, type = "response")

  logLik_full_glm <- as.numeric(logLik(glm_fit))
  logLik_null_glm <- as.numeric(logLik(update(glm_fit, . ~ 1))) #intercept only, for MCFadden 
  
  accuracy_glm <- c(
    RMSE     = rmse(preds_glm, test$observedResponse),
    RMSE_true = rmse(preds_glm, test$true_mu),
    Spearman = cor(preds_glm, test$observedResponse, method = "spearman"),
    R2       = 1 - logLik_full_glm / logLik_null_glm  # McFadden because poisson
  )

  ame_glm <- avg_slopes(glm_fit, newdata   = train_clean, variables = c("Environment1", "Environment2", "Environment3"))  
  
  glm_row <- data.frame(
    Iteration            = i,
    Loss_function         = "GLM",
    RMSE                  = accuracy_glm["RMSE"],
    RMSE_true             = accuracy_glm["RMSE_true"],
    Spearman              = accuracy_glm["Spearman"],
    R2                    = accuracy_glm["R2"],
    RMSE_train            = rmse(preds_glm_train, train$observedResponse),
    RMSE_true_train       = rmse(preds_glm_train, train$true_mu),
    Spearman_train        = cor(preds_glm_train, train$observedResponse, method = "spearman"),
    R2_train              = accuracy_glm["R2"],   # same in-sample fit, no separate calc needed
    Predictor             = ame_glm$term,
    Effect_size           = ame_glm$estimate,
    SE                    = ame_glm$std.error,
    p_value               = ame_glm$p.value,
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
                   #validation = 0.2,
                   early_stopping = 20L,
                   bootstrap  = 20L,
                   bootstrap_parallel = 1L,
                   verbose    = FALSE,
                   plot       = FALSE)
    
    
    # Predictions
    preds       <- predict(dnn_fit, newdata = test, type = "response")
    preds_train <- predict(dnn_fit, newdata = train, type = "response")
    
    # Accuracy metrics test
    loglik_test_f <- sum(dpois(test$observedResponse, lambda = preds, log = TRUE), na.rm = TRUE)
    loglik_test_0 <- sum(dpois(test$observedResponse, lambda = mean(train$observedResponse), log = TRUE))

    accuracy_temp <- c(
      RMSE     = rmse(preds, test$observedResponse),
      RMSE_true = rmse(preds, test$true_mu),   # against the true, noise-free DGP mean
      Spearman = cor(preds, test$observedResponse, method = "spearman"),
      R2       = 1 - loglik_test_f / loglik_test_0
    )
    # Accuracy metrics train
    loglik_f  <- sum(dpois(train$observedResponse, lambda = preds_train, log = TRUE), na.rm = TRUE)
    null_pred <- mean(train$observedResponse)
    loglik_0  <- sum(dpois(train$observedResponse, lambda = null_pred, log = TRUE))
    
    accuracy_train <- c(
      RMSE     = rmse(preds_train, train$observedResponse),
      RMSE_true = rmse(preds_train, train$true_mu),   # against the true, noise-free DGP mean
      Spearman = cor(preds_train, train$observedResponse, method = "spearman"),
      R2       = loglik_f / loglik_0
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
      Effect_size             = effects_temp,
      SE                      = SE_temp,
      p_value                 = p_value_temp,
      row.names = NULL
    )
  }
  
  results_i <- merge(rbind(glm_row, do.call(rbind, iter_rows)), ame_reference[, c("Predictor","AME_true")], by = "Predictor")}

# -----------------------------------------------------------------------------------------------------------------------------------------------------------#
results_list <- mclapply(1:iter, run_one_iteration, mc.cores = n_cores, mc.set.seed = TRUE)

# Check for worker failures before combining - a failed fork returns a
# try-error object here instead of stopping the whole mclapply() call
failed <- vapply(results_list, function(x) inherits(x, "try-error"), logical(1))
if (any(failed)) {
  warning(sprintf("%d of %d iterations failed - inspect results_list[failed] for error messages",
                  sum(failed), iter))
}

metrics <- do.call(rbind, results_list[!failed])
#------------------------------------------------------------------------------------------------------------------------------------------------------------#
View(metrics)

write.csv(metrics, file = "code/poisson_data.csv", row.names = FALSE)

glm_ref <- metrics |> filter(Loss_function == "GLM")
dnn_metrics <- metrics |> filter(Loss_function != "GLM")


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

Accuracy_long <- Accuracy |>
  pivot_longer(cols = c(RMSE, Spearman, R2),
               names_to = "Accuracy",
               values_to = "Value")

metric_xpos <- c(R2 = 1, RMSE = 2, Spearman = 3)
bar_halfwidth <- 0.39  # adjust

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
ggsave("images/poisson_accuracy.pdf", plot = effects_plot, device = "pdf",  dpi = 600)

####Effects:

glm_effects_lines <- glm_ref |>
  group_by(Predictor) |>
  summarise(glm_effect  = mean(Effect_size, na.rm = TRUE),
            AME_true    = mean(AME_true, na.rm = TRUE))

Effects <- dnn_metrics |>
  group_by(Loss_function, Predictor) |>
  summarise(Effect   = mean(Effect_size, na.rm = TRUE),
            SE       = mean(SE, na.rm = TRUE),
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
ggsave("images/poisson_effects.pdf", plot = effects_plot, device = "pdf",  dpi = 600)

#Training Accuracy (did the models converge?)
metrics_long <- metrics |>
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

ggsave("images/overfit_poisson.pdf", plot = overfit_check, device = "pdf", dpi =  600)
