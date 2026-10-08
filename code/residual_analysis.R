library(DHARMa)
library(cito)

sim <- createData(sampleSize    = 250,
                  intercept    = -1,
                  fixedEffects = c(2, 0.4, 0.1),
                  overdispersion       = 0,
                  family               = poisson(),
                  randomEffectVariance = 0)
trainID <- sample(x = sim$ID, size = 0.8 * length(sim$ID))
train <- sim[trainID, ]
test  <- sim[-trainID, ]
train_clean <- train[, setdiff(names(train), "group")]
boot_data <- train_clean[sample(nrow(train_clean), replace = TRUE), ]


glm_b <- glm(formula = observedResponse ~ Environment1 + Environment2 + Environment3,
             data = boot_data, family = poisson)

dnn_fit <- dnn(formula    = observedResponse ~ Environment1 + Environment2 + Environment3,
               data       = train,
               hidden     = c(50L, 50L),
               activation = "relu",
               loss       = "gaussian",
               optimizer  = config_optimizer("ignite_adam", weight_decay = 0.01),
               epochs     = 200,
               lr         = 0.003,
               validation = 0.2,
               early_stopping = 20L,
               verbose    = TRUE,
               plot       = TRUE)

simulateResiduals(glm_b, plot = T)

preds <- as.vector(predict(dnn_fit, newdata = test, type = "response"))
summary(preds)
sims <- replicate(250, rpois(length(preds), lambda = pmax(preds, 1e-8)))

res <- createDHARMa(
  simulatedResponse = sims,
  observedResponse  = test$observedResponse,
  fittedPredictedResponse = preds,
  integerResponse   = TRUE
)
plot(res)




library(dplyr)
library(gt)

n_test <- 50   # 20% of 250; adjust if you changed sample size

neg_preds_table <- metrics |>
  distinct(Iteration, Loss_function, .keep_all = TRUE) |>
  group_by(Loss_function) |>
  summarise(
    mean_neg   = mean(Neg_preds, na.rm = TRUE),
    pct_neg    = 100 * mean(Neg_preds, na.rm = TRUE) / n_test,
    iter_any   = 100 * mean(Neg_preds > 0, na.rm = TRUE),
    max_neg    = max(Neg_preds, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(Loss_function = factor(Loss_function, levels = loss_levels)) |>
  arrange(Loss_function)

write.csv(neg_preds_table, "code/poisson_neg_preds.csv", row.names = FALSE)


library(gridExtra)
library(grid)

tbl_fmt <- neg_preds_table |>
  mutate(across(c(mean_neg, pct_neg, iter_any), ~round(.x, 1)))

png("images/poisson_neg_preds_table.png", width = 900, height = 300, res = 150)
grid.table(tbl_fmt, rows = NULL)
dev.off()


library(gridExtra)
library(grid)

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
