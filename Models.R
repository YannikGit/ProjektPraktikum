library(cito)
library(DHARMa)

fit_model <- function(loss) {
  dnn(
    observedResponse ~ Environment1 + Environment2 + Environment3,
    data       = train,
    hidden     = c(20L, 20L),
    activation = "selu",
    loss       = loss,          # <- the only thing that changes
    epochs     = 300,
    lr         = 0.03,
    validation = 0.2,
    early_stopping = 10,
    plot       = FALSE,
    verbose    = FALSE
  )
}

losses <- c("mse", "poisson", "nbinom")   # add "gaussian" if you want it
# distinguished from "mse"
models <- lapply(losses, fit_model)
names(models) <- losses
summary(models["mse"]) 
