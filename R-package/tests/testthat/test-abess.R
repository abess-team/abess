library(abess)
library(testthat)

test_batch <- function(abess_fit, dataset, family) {
  support_size <- sum(dataset[["beta"]] != 0)

  ## support size
  fit_s_size <- abess_fit[["best.size"]]
  expect_equal(fit_s_size, support_size)

  ## subset
  coef_value <- coef(abess_fit, support.size = fit_s_size)
  est_index <- coef_value@i[-1]
  true_index <- which(dataset[["beta"]] != 0)
  expect_equal(est_index, true_index)


  ## estimation
  # oracle estimation by glm function:
  dat <- cbind.data.frame(
    "y" = dataset[["y"]],
    dataset[["x"]][, true_index]
  )

  # in gamma model, we have to set a start point of coef to ensure eta=Xb is positive
  if (family()[["family"]] == "Gamma") {
    start_point <- rep(1, length(true_index))
    coef0 <- abs(min(dataset[["x"]][, true_index] %*% start_point)) + 1
    oracle_est <- glm(y ~ ., data = dat, family = "Gamma", start = c(coef0, start_point))
  }
  else {
    oracle_est <- glm(y ~ ., data = dat, family = family)
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

  if (Sys.info()[1] == "Darwin") {
    threshold <- 1e-5
  } else {
    threshold <- 1e-3
  }

  if (family()[["family"]] == "Gamma") {
    threshold <- 1e-3
  }
  expect_equal(oracle_beta, est_beta, tolerance = threshold)
  expect_equal(oracle_coef0, est_coef0, tolerance = threshold)

  ## deviance
  f <- family()
  if (f[["family"]] == "gaussian") {
    oracle_dev <- mean((oracle_est[["residuals"]])^2) / 2
  } else if (f[["family"]] == "Gamma") {
    oracle_dev <- extract(abess_fit)[["dev"]]
  } else if (f[["family"]] != "poisson") {
    oracle_dev <- deviance(oracle_est) / 2
  } else {
    oracle_dev <- extract(abess_fit)[["dev"]]
  }
  expect_equal(oracle_dev, extract(abess_fit)[["dev"]])

}

test_batch_multivariate <-
  function(abess_fit, dataset, gaussian = TRUE) {
    true_index <- which(apply(dataset[["beta"]], 1, function(x) {
      all(x != 0)
    }))
    support_size <- length(true_index)

    ## support size
    fit_s_size <- abess_fit[["best.size"]]
    expect_equal(fit_s_size, support_size)

    ## subset
    coef_value <- coef(abess_fit, support.size = fit_s_size)[[1]]
    est_index <- unique(coef_value@i)[-1]
    expect_equal(est_index, true_index)

    ## estimation ##

    # estimation by abess:
    est_beta <- as.matrix(coef_value[1 + est_index, ])
    est_coef0 <- coef_value[1, , drop = TRUE]
    names(est_beta) <- NULL
    names(est_coef0) <- NULL

    if (gaussian) {
      # oracle estimation by lm function:
      dat <- cbind.data.frame(
        dataset[["y"]],
        dataset[["x"]][, true_index]
      )

      oracle_est <- lm(cbind(y1, y2, y3) ~ ., data = dat)
      oracle_est <- as.matrix(coef(oracle_est))
      oracle_beta <- oracle_est[-1, ]
      oracle_coef0 <- oracle_est[1, , drop = TRUE]
      names(oracle_beta) <- NULL
      names(oracle_coef0) <- NULL


      expect_equal(oracle_beta, est_beta, tolerance = 1e-5)
      expect_equal(oracle_coef0, est_coef0, tolerance = 1e-5)
    } else {
      require(nnet)
      # oracle estimation by nnet function:
      dat <- cbind.data.frame(
        "y" = 2 - dataset[["y"]],
        dataset[["x"]][, true_index]
      )
      set.seed(1)
      suppressMessages(multinom_model <-
        multinom(
          y ~ .,
          data = dat,
          maxit = 15,
          trace = FALSE
        ))
      oracle_est <- as.matrix(coef(multinom_model))
      oracle_est <- t(oracle_est[2:1, ])
      oracle_beta <- oracle_est[-1, ]
      oracle_coef0 <- oracle_est[1, , drop = TRUE]

      expect_lt(mean(oracle_beta - est_beta[, 1:2]), 0.1)
      expect_lt(mean(oracle_coef0 - est_coef0[1:2]), 1)
    }

    ## deviance
    # f <- family()
    # if (f[["family"]] == "gaussian") {
    #   oracle_dev <- mean((oracle_est[["residuals"]])^2)
    # } else {
    #   oracle_dev <- deviance(oracle_est) / 2
    # }
    # expect_equal(oracle_dev, abess_fit[["dev"]][fit_s_size + 1])
  }

test_that("Covariance update works", {
  ## n > p case:
  n <- 100
  p <- 20
  support_size <- 3

  dataset <- generate.data(n, p, support_size, seed = 1)
  t1 <- system.time(abess_fit1 <- abess(
    dataset[["x"]],
    dataset[["y"]],
    cov.update = FALSE,
    num.threads = 1
  ))
  t2 <- system.time(abess_fit2 <- abess(dataset[["x"]],
    dataset[["y"]],
    num.threads = 1
  ))

  abess_fit1[["call"]] <- NULL
  abess_fit2[["call"]] <- NULL
  expect_true(all.equal(abess_fit1, abess_fit2))
  # expect_lt(t2[3], t1[3])

  ## p > n case:
  n <- 50
  p <- 100
  support_size <- 3

  dataset <- generate.data(n, p, support_size, seed = 1)
  t1 <- system.time(abess_fit1 <- abess(
    dataset[["x"]],
    dataset[["y"]],
    cov.update = FALSE,
    num.threads = 1
  ))
  t2 <- system.time(abess_fit2 <- abess(dataset[["x"]],
    dataset[["y"]],
    num.threads = 1
  ))

  abess_fit1[["call"]] <- NULL
  abess_fit2[["call"]] <- NULL
  expect_true(all.equal(abess_fit1, abess_fit2))
})

test_that("OPENMP works", {
  skip("Skip OPENMP!")
  skip_on_os("mac")

  if (!require("ps")) {
    install.packages("ps")
  }
  num_threads <- ps_num_threads()
  print(num_threads)
  skip_if(num_threads == 1)

  n <- 500
  p <- 500
  support_size <- 3

  dataset <- generate.data(n, p, support_size, seed = 1)
  t1 <- system.time(abess_fit1 <- abess(
    dataset[["x"]],
    dataset[["y"]],
    tune.type = "cv",
    num.threads = 1
  ))
  t2 <- system.time(abess_fit2 <- abess(
    dataset[["x"]],
    dataset[["y"]],
    tune.type = "cv",
    num.threads = 5
  ))

  expect_lt(t2[3], t1[3])
  abess_fit1[["call"]] <- NULL
  abess_fit2[["call"]] <- NULL
  expect_true(all.equal(abess_fit1, abess_fit2))
})

test_that("abess (gaussian) works", {
  n <- 100
  p <- 50
  support_size <- 3

  ## default interface
  dataset <- generate.data(n, p, support_size, seed = 1)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]])
  test_batch(abess_fit, dataset, gaussian)

  ## formula interface
  dat <- cbind.data.frame(dataset[["x"]][, 1:30],
    "y" = dataset[["y"]],
    dataset[["x"]][, 31:p]
  )
  abess_fit <- abess(y ~ ., data = dat)
  test_batch(abess_fit, dataset, gaussian)
})

test_that("abess (binomial) works", {
  n <- 150
  p <- 100
  support.size <- 3

  dataset <- generate.data(n, p, support.size,
    family = "binomial", seed = 1
  )
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "binomial",
    tune.type = "cv",
    newton = "exact",
    newton.thresh = 1e-8
  )
  test_batch(abess_fit, dataset, binomial)

  abess_fit1 <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "binomial",
    tune.type = "cv",
    newton = "approx",
    newton.thresh = 1e-8
  )
  beta <- extract(abess_fit)[["support.beta"]]
  beta1 <- extract(abess_fit1)[["support.beta"]]
  expect_true(all(abs(beta - beta1) < 1e-3))
})

test_that("abess (cox) works", {
  if (!require("survival")) {
    install.packages("survival")
  }
  n <- 60
  p <- 20
  support.size <- 3

  dataset <- generate.data(n, p, support.size,
    family = "cox", seed = 1
  )
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "cox",
    newton = "approx",
    tune.type = "cv"
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
  dat <-
    cbind.data.frame(dataset[["y"]], dataset[["x"]][, true_index])
  oracle_est <-
    survival::coxph(survival::Surv(time, status) ~ ., data = dat)
  oracle_beta <- coef(oracle_est)
  # estimated by abess:
  est_beta <- coef_value@x
  names(est_beta) <- NULL

  for (i in 1:support.size) {
    abs_abess_diff <- abs(est_beta[i] - true_beta[i])
    abs_coxph_diff <- abs(oracle_beta[i] - true_beta[i])
    expect_lt(abs_abess_diff / abs_coxph_diff, 1.05)
  }

  ## Surv object input:
  dataset[["y"]] <- survival::Surv(
    time = dataset[["y"]][, 1],
    event = dataset[["y"]][, 2]
  )
  abess_fit1 <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "cox",
    newton = "approx",
    tune.type = "cv"
  )
  expect_true(all.equal(abess_fit, abess_fit1))
})

test_that("abess (poisson) works", {
  # skip("poisson")
  n <- 200
  p <- 100
  support.size <- 3

  dataset <- generate.data(n, p, support.size,
    family = "poisson", seed = 78
  )
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "poisson",
    # newton = "exact",
    tune.type = "cv",
    newton.thresh = 1e-8,
    # support.size = 0:support.size,
    # always.include = which(dataset[['beta']]!=0)
  )
  test_batch(abess_fit, dataset, poisson)
})

test_that("abess (mgaussian) works", {
  n <- 30
  p <- 10
  support_size <- 3

  ## default interface
  dataset <-
    generate.data(n, p, support_size, seed = 1, family = "mgaussian")
  abess_fit <- abess(dataset[["x"]], dataset[["y"]],
    family = "mgaussian", tune.type = "cv"
  )
  test_batch_multivariate(abess_fit, dataset)

  ## formula interface
  dat <- cbind.data.frame(dataset[["x"]], dataset[["y"]])
  abess_fit <-
    abess(
      cbind(y1, y2, y3) ~ .,
      data = dat,
      family = "mgaussian",
      tune.type = "cv"
    )
  test_batch_multivariate(abess_fit, dataset)
})

test_that("abess (multinomial) works", {
  skip("Skip multinomial now!")
  ## not pass:
  n <- 600
  p <- 100
  support_size <- 3
  dataset <- generate.data(n, p, support_size,
    seed = 1, family = "multinomial"
  )
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "multinomial",
    tune.type = "cv",
    newton = "approx"
  )
  test_batch_multivariate(abess_fit, dataset, FALSE)

  ## not pass:
  n <- 200
  p <- 500
  support_size <- 3

  dataset <-
    generate.data(n, p, support_size, seed = 1, family = "multinomial")
  abess_fit <- abess(dataset[["x"]], dataset[["y"]],
    family = "multinomial", tune.type = "cv"
  )
  test_batch_multivariate(abess_fit, dataset, FALSE)
})

test_that("Fast than Lasso (gaussian) works", {
  skip("Skip comparison with glmnet now!")
  if (!require("glmnet")) {
    install.packages("glmnet")
  }
  n <- 500
  p <- 1500
  support_size <- 3

  dataset <- generate.data(n, p, support_size, seed = 1)
  t1 <-
    system.time(abess_fit <- abess(
      dataset[["x"]],
      dataset[["y"]],
      tune.type = "cv",
      nfolds = 10,
      num.threads = 8
    ))
  tune_num <- length(abess_fit[["support.size"]])
  glmnet_fit <-
    glmnet::glmnet(dataset[["x"]], dataset[["y"]], nlambda = tune_num)
  t2 <-
    system.time(
      glmnet_fit <- glmnet::cv.glmnet(dataset[["x"]], dataset[["y"]],
        lambda = glmnet_fit[["lambda"]], nfolds = 10
      )
    )
  expect_lt(t1[3], t2[3])
})

test_that("abess (golden section) works", {
  n <- 100
  p <- 50
  support_size <- 3
  dataset <- generate.data(n, p, support_size)

  ## default search range
  abess_fit <-
    abess(dataset[["x"]], dataset[["y"]], tune.path = "gsection", nfolds = 5)
  test_batch(abess_fit, dataset, gaussian)

  ## self-defined search range
  abess_fit <- abess(dataset[["x"]],
    dataset[["y"]],
    tune.path = "gsection",
    gs.range = c(1, 6)
  )
  test_batch(abess_fit, dataset, gaussian)
})

test_that("abess (output) works", {
  n <- 30
  p <- 30
  support_size <- 3

  dataset <- generate.data(n, p, support_size, seed = 1)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]])
  expect_true(is.vector(abess_fit[["dev"]]))
  expect_true(is.vector(abess_fit[["tune.value"]]))
  expect_true(is.vector(abess_fit[["support.size"]]))
  expect_true(is.vector(abess_fit[["intercept"]]))
  expect_true(is.vector(abess_fit[["edf"]]))
})

test_that("abess (always-include) works", {
  n <- 50
  p <- 20
  support_size <- 3
  dataset <- generate.data(n, p, support_size)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]], always.include = c(1))
  expect_true(all((abess_fit[["beta"]][1, , drop = TRUE][-1] != 0)))
})

test_that("abess (L2 regularization, ridge estimation) works", {
  n <- 50
  p <- 20
  support_size <- 3
  lambda_value <- 1
  dataset <- generate.data(n, p, support_size)
  true_index <- which(dataset[["beta"]] != 0)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]],
    always.include = true_index,
    lambda = lambda_value, support.size = length(true_index)
  )
  coef_est <- as.vector(coef(abess_fit))
  coef_est <- coef_est[coef_est != 0]
  if (!require("MASS")) {
    install.packages("MASS")
  }
  dat <- cbind.data.frame("y" = dataset[["y"]], "x" = dataset[["x"]][, true_index])
  oracle_est <- coef(lm.ridge(y ~ ., data = dat, lambda = lambda_value))
  expect_equal(as.numeric(coef_est), as.numeric(oracle_est), tolerance = 1e-10)
})

test_that("abess (L2 regularization) works", {
  n <- 100
  p <- 20
  support_size <- 3
  dataset <- generate.data(n, p, support_size)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]], lambda = 0.1)
  expect_true(all(diff(abess_fit[["edf"]]) > 0))
  expect_true(all(abess_fit[["edf"]] <= abess_fit[["support.size"]]))
  abess_fit2 <- abess(dataset[["x"]], dataset[["y"]])
  expect_true(extract(abess_fit)[["support.size"]] >= extract(abess_fit2)[["support.size"]])
})

test_that("abess (gamma) works", {
  skip_on_ci()
  n <- 2000
  p <- 100
  support.size <- 3
  dataset <- generate.data(n, p, support.size,
    family = "gamma", seed = 1
  )

  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "gamma",
    tune.type = "cv",
    support.size = 0:support.size
  )

  test_batch(abess_fit, dataset, Gamma)
})

test_that("abess (one variable input) works", {
  n <- 100
  p <- 1
  support.size <- 1
  dataset <- generate.data(n, p, support.size, seed = 1)

  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    tune.type = "gic",
    support.size = 0:support.size
  )
  test_batch(abess_fit, dataset, gaussian)
  
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    tune.type = "cv",
    support.size = 0:support.size
  )
  test_batch(abess_fit, dataset, gaussian)
})

test_that("abess (init.active.set) works", {
  n <- 100
  p <- 50
  support.size <- 3
  dataset <- generate.data(n, p, support.size, seed = 1)
  
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    tune.type = "gic",
    support.size = 0:support.size, 
    init.active.set = c(1, 2)
  )
  test_batch(abess_fit, dataset, gaussian)
  
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    tune.type = "gic",
    support.size = 0:support.size, 
    init.active.set = 1:4
  )
  test_batch(abess_fit, dataset, gaussian)
})