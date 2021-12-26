library(abess)
library(testthat)

test_batch <- function(abess_fit, dataset, family) {
  support_size <- length(dataset[["true.group"]])
  k <-
    ncol(dataset[["x"]]) / length(unique(dataset[["group.index"]]))
  true_index <- as.vector(sapply(dataset$true.group, function(i) {
    return(((i - 1) * k + 1):(i * k))
  }))

  ## support size
  fit_s_size <- abess_fit[["best.size"]]

  ## subset
  coef_value <- coef(abess_fit, support.size = fit_s_size)
  est_index <- coef_value@i[-1]
  expect_equal(extract(abess_fit)[["support.vars"]], paste0("x", true_index))

  ## estimation
  # oracle estimation by glm function:
  dat <- cbind.data.frame(
    "y" = dataset[["y"]],
    dataset[["x"]][, true_index]
  )

  f <- family()
  if (f[["family"]] != "gaussian") {
    oracle_est <- glm(y ~ ., data = dat, family = family)
  } else {
    oracle_est <- lm(y ~ ., data = dat)
  }
  oracle_beta <- coef(oracle_est)[-1]
  oracle_coef0 <- coef(oracle_est)[1]
  names(oracle_beta) <- NULL
  names(oracle_coef0) <- NULL
  # estimation by abess:
  est_beta <- coef_value@x[-1]
  est_coef0 <- coef_value@x[1]
  names(est_beta) <- NULL
  names(est_coef0) <- NULL

  expect_equal(oracle_beta, est_beta, tolerance = 1e-2)
  expect_equal(oracle_coef0, est_coef0, tolerance = 1e-2)

  ## deviance
  if (f[["family"]] == "gaussian") {
    oracle_dev <- mean((oracle_est[["residuals"]])^2) / 2
    expect_equal(oracle_dev, abess_fit[["dev"]][fit_s_size + 1])
  } else if (f[["family"]] != "poisson") {
    oracle_dev <- deviance(oracle_est) / 2
    expect_equal(oracle_dev, abess_fit[["dev"]][fit_s_size + 1])
  }
}

## Generate simulated data
generate.group <- function(n,
                           J,
                           k,
                           support.size = NULL,
                           rho = 0.5,
                           family = c("gaussian", "binomial", "poisson", "cox"),
                           cortype = 1,
                           weibull.shape = 1,
                           uniform.max = 1,
                           sigma1 = 1,
                           sigma2 = 1,
                           seed = 1) {
  family <- match.arg(family)
  set.seed(seed)
  group.index <- rep(1:J, each = k)

  if (cortype == 1) {
    Sigma <- diag(J)
  } else if (cortype == 2) {
    Sigma <- matrix(0, J, J)
    Sigma <- rho^(abs(row(Sigma) - col(Sigma)))
  } else if (cortype == 3) {
    Sigma <- matrix(rho, J, J)
    diag(Sigma) <- 1
  }
  if (cortype == 1) {
    x <- matrix(rnorm(n * J), nrow = n, ncol = J)
  } else {
    x <- MASS::mvrnorm(n, rep(0, J), Sigma)
  }
  z <- matrix(rnorm(n * J * k), nrow = n)
  z <- sapply(1:(J * k), function(i) {
    g <- floor((i - 1) / k) + 1
    return((x[, g] + z[, i]) / sqrt(2))
  })
  x <- matrix(unlist(z), n)
  beta <- rep(0, J * k)
  true.group <- sort(sample(1:J, support.size))
  nonzero <- as.vector(sapply(true.group, function(i) {
    return(((i - 1) * k + 1):(i * k))
  }))
  coef <- rep(0, support.size * k)
  for (i in 1:support.size) {
    temp <- stats::rnorm(k + 1, 0, sigma2)
    coef[((i - 1) * k + 1):(i * k)] <- (temp - mean(temp))[-1]
  }
  beta[nonzero] <- coef
  if (family == "gaussian") {
    y <- x %*% beta + rnorm(n, 0, sigma1)
  }
  if (family == "binomial") {
    eta <- x %*% beta + rnorm(n, 0, sigma1)
    PB <- apply(eta, 1, function(eta) {
      a <- exp(eta) / (1 + exp(eta))
      if (is.infinite(exp(eta))) {
        a <- 1
      }
      return(a)
    })
    y <- stats::rbinom(n, 1, PB)
  }
  if (family == "cox") {
    eta <- x %*% beta + rnorm(n, 0, sigma1)
    time <-
      (-log(stats::runif(n)) / drop(exp(eta)))^(1 / weibull.shape)
    ctime <- stats::runif(n, max = uniform.max)
    status <- (time < ctime) * 1
    censoringrate <- 1 - mean(status)
    time <- pmin(time, ctime)
    y <- cbind(time = time, status = status)
  }
  if (family == "poisson") {
    eta <- x %*% beta + stats::rnorm(n, 0, sigma1)
    eta <- ifelse(eta > 50, 50, eta)
    eta <- ifelse(eta < -50, -50, eta)
    eta <- exp(eta)
    y <- sapply(eta, stats::rpois, n = 1)
  }
  set.seed(NULL)

  colnames(x) <- paste0("x", 1:(J * k))
  return(list(
    x = x,
    y = y,
    beta = beta,
    group.index = group.index,
    true.group = true.group
  ))
}

test_that("Group selection: abess (gaussian) works", {
  n <- 200
  J <- 100
  k <- 4
  support_size <- 3

  dataset <- generate.group(n, J, k, support_size, seed = 1)

  ## default interface
  abess_fit <-
    abess(dataset[["x"]], dataset[["y"]], group.index = dataset$group.index)
  test_batch(abess_fit, dataset, gaussian)

  ## formula interface
  dat <- cbind.data.frame(dataset[["x"]][, 1:30],
    "y" = dataset[["y"]],
    dataset[["x"]][, 31:(J * k)]
  )
  abess_fit <-
    abess(y ~ ., data = dat, group.index = dataset$group.index)
  test_batch(abess_fit, dataset, gaussian)
})

test_that("Group selection: abess (logistic) works", {
  n <- 300
  J <- 50
  k <- 4
  support_size <- 3

  dataset <-
    generate.group(n, J, k, support_size, family = "binomial", seed = 1)

  ## default interface
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    group.index = dataset$group.index,
    family = "binomial",
    newton = "exact"
  )
  test_batch(abess_fit, dataset, binomial)

  ## formula interface
  dat <- cbind.data.frame(dataset[["x"]][, 1:30],
    "y" = dataset[["y"]],
    dataset[["x"]][, 31:(J * k)]
  )
  abess_fit <- abess(
    y ~ .,
    data = dat,
    group.index = dataset$group.index,
    family = "binomial",
    newton = "exact"
  )
  test_batch(abess_fit, dataset, binomial)
})

test_that("Group selection: abess (poisson) works", {
  n <- 200
  J <- 100
  k <- 3
  support_size <- 3

  dataset <-
    generate.group(
      n,
      J,
      k,
      support_size,
      family = "poisson",
      seed = 1,
      sigma1 = 0.1
    )

  ## default interface
  abess_fit <-
    abess::abess(dataset[["x"]],
      dataset[["y"]],
      group.index = dataset$group.index,
      family = "poisson"
    )
  test_batch(abess_fit, dataset, poisson)

  ## formula interface
  dat <- cbind.data.frame(dataset[["x"]][, 1:30],
    "y" = dataset[["y"]],
    dataset[["x"]][, 31:(J * k)]
  )
  abess_fit <-
    abess(
      y ~ .,
      data = dat,
      group.index = dataset$group.index,
      family = "poisson"
    )
  test_batch(abess_fit, dataset, poisson)
})

test_that("Group selection: abess (cox) works", {
  if (!require("survival")) {
    install.packages("survival")
  }
  n <- 150
  J <- 50
  k <- 4
  support_size <- 3

  dataset <-
    generate.group(n, J, k, support_size, family = "cox", seed = 1)

  ## default interface
  abess_fit <- abess::abess(dataset[["x"]],
    dataset[["y"]],
    group.index = dataset$group.index,
    family = "cox"
  )

  ## support size
  fit_s_size <- abess_fit[["best.size"]]
  expect_equal(fit_s_size, support_size)

  ## subset
  coef_value <- coef(abess_fit, support.size = fit_s_size)
  est_index <- coef_value@i
  true_index <- as.vector(sapply(dataset$true.group, function(i) {
    return(((i - 1) * k + 1):(i * k))
  }))
  expect_equal(est_index, true_index)

  ## estimation
  # true value:
  true_beta <- dataset[["beta"]][true_index]
  # estimated by coxph:
  dat <-
    cbind.data.frame(dataset[["y"]], dataset[["x"]][, true_index])
  oracle_est <-
    survival::coxph(survival::Surv(time, status) ~ ., data = dat)
  oracle_beta <- coef(oracle_est)
  # estimated by abess:
  est_beta <- coef_value@x
  names(est_beta) <- NULL

  for (i in 1:support_size) {
    abs_abess_diff <- abs(est_beta[i] - true_beta[i])
    abs_coxph_diff <- abs(oracle_beta[i] - true_beta[i])
    expect_lt(abs_abess_diff / abs_coxph_diff, 1.05)
  }
})
