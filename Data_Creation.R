library(DHARMa)
library(cito)

set.seed(187)

sim <- createData(sampleSize = 2000,
                  intercept = 1,
                  fixedEffects = c(1, -0.5, 0.3), #trueEffects of the three environments
                  overdispersion = 0.7, 
                  family = poisson(),
                  randomEffectVariance = 0,
                  )
head(sim)
summary(sim)

true_effects <- c(Environment1 = 1, Environment2 = -0.5, Environment3 = 0.3)

train_index <- sample(seq_len(nrow(sim)), size = 0.8*nrow(sim))
train <- sim[train_index, ]
test <- sim[-train_index, ]
