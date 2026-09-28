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

iter <- 100
n_cores <- 20
rmse <- function(preds, obs) sqrt(mean((preds - obs)^2))

losses     <- c("mse", "mae", "poisson", "gaussian")
predictors <- c("Environment1", "Environment2", "Environment3")



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
      R2       = 1 - sum((preds_train - train$observedResponse)^2) /
        sum((train$observedResponse - mean(train$observedResponse))^2),
      MCFadden = 1 - loglik_test_f / loglik_test_0
    )
    r2_value <- if (losses[l] == "poisson") accuracy_temp["MCFadden"] else accuracy_temp["R2"]
    
    # Accuracy metrics train
    loglik_f  <- sum(dpois(train$observedResponse, lambda = preds_train, log = TRUE), na.rm = TRUE)
    null_pred <- mean(train$observedResponse)
    loglik_0  <- sum(dpois(train$observedResponse, lambda = null_pred, log = TRUE))
    
    accuracy_train <- c(
      RMSE     = rmse(preds_train, train$observedResponse),
      RMSE_true = rmse(preds_train, train$true_mu),   # against the true, noise-free DGP mean
      Spearman = cor(preds_train, train$observedResponse, method = "spearman"),
      R2       = 1 - sum((preds_train - train$observedResponse)^2) /sum((train$observedResponse - mean(train$observedResponse))^2),
      MCFadden = 1 - loglik_f / loglik_0
    )
    r2_train <- if (losses[l] == "poisson") accuracy_train["MCFadden"] else accuracy_train["R2"]
    
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
      R2                      = r2_value,
      Spearman                = accuracy_temp["Spearman"],
      RMSE_train              = accuracy_train["RMSE"],
      RMSE_true_train         = accuracy_train["RMSE_true"],
      Spearman_train          = accuracy_train["Spearman"],
      R2_train                = r2_train,
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

#write.csv(metrics, file = "lossFunction_data.csv", row.names = FALSE)

#Accuracy metrics
Accuracy <- metrics |>
  group_by(Loss_function) |>
  summarise(RMSE = mean(RMSE, na.rm = TRUE),
            RMSE_true = mean(RMSE_true, na.rm = TRUE),
            Spearman = mean(Spearman, na.rm = TRUE),
            R2 = mean(R2, na.rm = TRUE))

Accuracy_long <- tidyr::pivot_longer(
  Accuracy,
  cols = c(RMSE, Spearman, R2),
  names_to = "Accuracy",
  values_to = "Value"
)

accuracy_plot <- ggplot(Accuracy_long, aes(x = Accuracy, y = Value, fill = Loss_function)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "black", linewidth = 0.3) +
  scale_fill_manual(values = c(MAE = "darkslateblue", MSE = "deepskyblue4", GAUSSIAN = "cyan3", POISSON = "cyan", GLM = "gray")) +
  geom_text(aes(label = round(Value, 3)), position = position_dodge(width = 0.8), vjust = -0.3, size = 3.5) +
  theme_minimal() +
  labs(
    title = "Model Accuracy",
    x = "Accuracy metric",
    y = "Value",
    fill = "Loss function "
  )
#ggsave("accuracy_plot.pdf", plot = accuracy_plot, device = "pdf",  dpi = 600)


#Effects:
Effects <- metrics |>
  group_by(Loss_function, Predictor) |>
  summarise(Effect = mean(Effect_size, na.rm = TRUE),
            SE = mean(SE, na.rm = TRUE),
            p_value = mean(p_value, na.rm = TRUE),
            AME_true = mean(AME_true, na.rm = TRUE),
            .groups = "drop")


effects_plot <- ggplot(Effects, aes(x = Predictor, y = Effect, fill = Loss_function)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = Effect - SE, ymax = Effect + SE),position = position_dodge(width = 0.8),width = 0.2) +
  scale_fill_manual(values = c(MAE = "bisque", MSE = "chocolate", POISSON = "coral", GAUSSIAN = "orange", GLM = "gray")) +
  geom_text(aes(label = round(Effect, 3)), position = position_dodge(width = 0.8), vjust = -0.3, size = 3) +
  theme_minimal() +
  labs(title = "Effect of Environment",x = "Predictor",y = "Effect size",fill = "Loss function")

#ggsave("effects_plot.pdf", plot = effects_plot, device = "pdf",  dpi = 600)

