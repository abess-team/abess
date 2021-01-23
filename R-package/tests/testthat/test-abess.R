library(abess)
library(testthat)

test_that("abess (gaussian) works", {
  n <- 500
  p <- 1500
  support.size <- 3
  
  dataset <- generate.data(n, p, support.size, seed = 1)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]])
  best_model <- abess_fit[["best.model"]]
  
  ## support size
  expect_equal(best_model[["support.size"]], support.size)
  
  ## subset
  est_index <- best_model[["support.index"]]
  names(est_index) <- NULL
  true_index <- which(dataset[["beta"]] != 0)
  expect_equal(est_index, true_index)
  
  ## estimation
  # oracle estimation by lm function:
  dat <- cbind.data.frame("y" = dataset[["y"]], 
                          dataset[["x"]][, true_index])
  oracle_est <- lm(y ~ ., data = dat)
  oracle_beta <- coef(oracle_est)[-1]
  oracle_coef0 <- coef(oracle_est)[1]
  names(oracle_beta) <- NULL
  names(oracle_coef0) <- NULL
  # estimation by abess:
  est_beta <- best_model[["beta"]][est_index]
  est_coef0 <- best_model[["coef0"]]
  names(est_beta) <- NULL
  names(est_coef0) <- NULL
  
  expect_equal(oracle_beta, est_beta, tolerance = 1e-5)
  expect_equal(oracle_coef0, est_coef0, tolerance = 1e-5)
  
  ## deviance
  oracle_dev <- mean((oracle_est[["residuals"]])^2)
  expect_equal(oracle_dev, best_model[["dev"]])
})

test_that("abess (binomial) works", {
  require(survival)
  n <- 500
  p <- 1500
  support.size <- 3
  
  dataset <- generate.data(n, p, support.size, 
                           family = "binomial", seed = 1)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                     family = "binomial", tune.type = "cv", 
                     newton = "approx", max.newton.iter = 80)
  best_model <- abess_fit[["best.model"]]
  
  ## support size
  expect_equal(best_model[["support.size"]], support.size)
  
  ## subset
  est_index <- best_model[["support.index"]]
  names(est_index) <- NULL
  true_index <- which(dataset[["beta"]] != 0)
  expect_equal(est_index, true_index)
  
  ## estimation
  # oracle estimation by glm function:
  dat <- cbind.data.frame("y" = dataset[["y"]], dataset[["x"]][, true_index])
  oracle_est <- glm(y ~ ., data = dat, family = binomial())
  oracle_beta <- coef(oracle_est)[-1]
  oracle_coef0 <- coef(oracle_est)[1]
  names(oracle_beta) <- NULL
  names(oracle_coef0) <- NULL
  # estimation by abess:
  est_beta <- best_model[["beta"]][est_index]
  est_coef0 <- best_model[["coef0"]]
  names(est_beta) <- NULL
  names(est_coef0) <- NULL
  
  expect_equal(oracle_beta, est_beta, tolerance = 1e-5)
  expect_equal(oracle_coef0, est_coef0, tolerance = 1e-5)
  
  ## deviance
  oracle_dev <- oracle_est[["deviance"]] / 2
  expect_equal(oracle_dev, best_model[["dev"]])
})

test_that("abess (cox) works", {
  require(survival)
  n <- 500
  p <- 1500
  support.size <- 3
  
  dataset <- generate.data(n, p, support.size, 
                           family = "cox", seed = 1)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                     family = "cox", tune.type = "cv", 
                     newton = "newton", max.newton.iter = 80)
  best_model <- abess_fit[["best.model"]]
  
  ## support size
  expect_equal(best_model[["support.size"]], support.size)
  
  ## subset
  est_index <- best_model[["support.index"]]
  names(est_index) <- NULL
  true_index <- which(dataset[["beta"]] != 0)
  expect_equal(est_index, true_index)
  
  ## estimation
  # oracle estimation by coxph function:
  dat <- cbind.data.frame(dataset[["y"]], dataset[["x"]][, true_index])
  oracle_est <- coxph(Surv(time, status) ~ ., data = dat)
  oracle_beta <- coef(oracle_est)
  # oracle_coef0 <- coef(oracle_est)
  names(oracle_beta) <- NULL
  # names(oracle_coef0) <- NULL
  # estimation by abess:
  est_beta <- best_model[["beta"]][est_index]
  # est_coef0 <- best_model[["coef0"]]
  names(est_beta) <- NULL
  # names(est_coef0) <- NULL
  
  expect_equal(oracle_beta, est_beta, tolerance = 1e-5)
  expect_equal(oracle_beta, est_beta, tolerance = 1e-5)
  
  ## deviance
  oracle_dev <- mean((oracle_est[["residuals"]])^2)
  expect_equal(oracle_dev, best_model[["dev"]])
})
