library(abess)
library(testthat)

test_batch <- function(abess_fit, dataset, family) {
  support_size <- sum(dataset[["beta"]] != 0)
  
  ## support size
  fit_s_size <- abess_fit[["best.size"]]
  expect_equal(fit_s_size, support_size)
  
  ## subset
  coef_value <- coef(abess_fit, support.size = 3)
  est_index <- coef_value@i[-1]
  true_index <- which(dataset[["beta"]] != 0)
  expect_equal(est_index, true_index)
  
  ## estimation
  # oracle estimation by glm function:
  dat <- cbind.data.frame("y" = dataset[["y"]], 
                          dataset[["x"]][, true_index])
  
  oracle_est <- glm(y ~ ., data = dat, family = family)
  oracle_beta <- coef(oracle_est)[-1]
  oracle_coef0 <- coef(oracle_est)[1]
  names(oracle_beta) <- NULL
  names(oracle_coef0) <- NULL
  # estimation by abess:
  est_beta <- coef_value@x[-1]
  est_coef0 <- coef_value@x[1]
  names(est_beta) <- NULL
  names(est_coef0) <- NULL
  
  expect_equal(oracle_beta, est_beta, tolerance = 1e-5)
  expect_equal(oracle_coef0, est_coef0, tolerance = 1e-5)
  
  ## deviance
  f <- family()
  if (f[["family"]] == "gaussian") {
    oracle_dev <- mean((oracle_est[["residuals"]])^2)
  } else {
    oracle_dev <- deviance(oracle_est) / 2
  }
  expect_equal(oracle_dev, abess_fit[["dev"]][fit_s_size + 1])
}

test_that("abess (gaussian) works", {
  
  n <- 500
  p <- 1500
  support_size <- 3
  
  ## default interface
  dataset <- generate.data(n, p, support_size, seed = 1)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]])
  test_batch(abess_fit, dataset, gaussian)
  
  ## formula interface
  dat <- cbind.data.frame(dataset[["x"]][, 1:30], "y" = dataset[["y"]], 
                          dataset[["x"]][, 31:p])
  abess_fit <- abess(y ~ ., data = dat)
  test_batch(abess_fit, dataset, gaussian)
})

test_that("abess (binomial) works", {
  n <- 500
  p <- 1500
  support.size <- 3
  
  dataset <- generate.data(n, p, support.size, 
                           family = "binomial", seed = 1)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                     family = "binomial", tune.type = "cv", 
                     newton = "exact", max.newton.iter = 50, 
                     newton.thresh = 1e-8)
  test_batch(abess_fit, dataset, binomial)
})

test_that("abess (cox) works", {
  require(survival)
  n <- 500
  p <- 1500
  support.size <- 3
  
  dataset <- generate.data(n, p, support.size, 
                           family = "cox", seed = 1)
  t <- system.time(abess_fit <- abess(
    dataset[["x"]], dataset[["y"]], 
    family = "cox", tune.type = "cv", 
    newton = "exact", max.newton.iter = 60)
  )
  
  ## support size
  fit_s_size <- abess_fit[["best.size"]]
  expect_equal(fit_s_size, support.size)
  
  ## subset
  coef_value <- coef(abess_fit, support.size = 3)
  est_index <- coef_value@i
  true_index <- which(dataset[["beta"]] != 0)
  expect_equal(est_index, true_index)
  
  ## estimation
  # true value:
  true_beta <- dataset[["beta"]][true_index]
  # estimated by coxph:
  dat <- cbind.data.frame(dataset[["y"]], dataset[["x"]][, true_index])
  oracle_est <- coxph(Surv(time, status) ~ ., data = dat)
  oracle_beta <- coef(oracle_est)
  # estimated by abess:
  est_beta <- coef_value@x
  names(est_beta) <- NULL
  
  for (i in 1:support.size) {
    abs_abess_diff <- abs(est_beta[i] - true_beta[i])
    abs_coxph_diff <- abs(oracle_beta[i] - true_beta[i])
    expect_lt(abs_abess_diff / abs_coxph_diff, 1.05)
  }
})

test_that("abess (poisson) works", {
  skip("Skip poisson now!")
  n <- 500
  p <- 1500
  support.size <- 3
  
  dataset <- generate.data(n, p, support.size, 
                           family = "poisson", seed = 1)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]], support.size = 1:5, 
                     family = "poisson", tune.type = "cv", 
                     newton.thresh = 1e-8)
  test_batch(abess_fit, dataset, poisson)
})
