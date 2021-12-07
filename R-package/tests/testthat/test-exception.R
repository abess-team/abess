library(abess)
library(testthat)
require(Matrix)

test_that("abess (data) works", {
  n <- 100
  p <- 20
  support_size <- 3
  dataset <- generate.data(n, p, support_size)

  ########### Exception for x ############
  dat <- cbind.data.frame("y" = dataset[["y"]], dataset[["x"]])
  dat[["x1"]] <- as.character(dat[["x1"]])
  expect_error(abess(y ~ ., data = dat), regexp = "Some columns in data are character")
  dat[["x1"]] <- as.factor(as.numeric(dat[["x1"]]))
  expect_error(abess(y ~ ., data = dat), regexp = "Some columns in data are factor")

  dataset[["x"]][1, 1] <- NA
  expect_error(abess(dataset[["x"]], dataset[["y"]]), regexp = "x has missing")

  dataset[["x"]][1, 1] <- Inf
  expect_error(abess(dataset[["x"]], dataset[["y"]]), regexp = "x has missing")

  dataset[["x"]][1, 1] <- 1
  dataset[["x"]] <- as.list(as.data.frame(dataset[["x"]]))
  expect_error(abess(dataset[["x"]], dataset[["y"]]))

  dataset <- generate.data(n, p, support_size)
  dataset[["x"]] <- Matrix(dataset[["x"]])
  expect_error(abess(dataset[["x"]], dataset[["y"]]))

  dataset[["x"]] <- as.matrix(dataset[["x"]])
  dataset[["x"]] <- matrix(as.character(dataset[["x"]]), nrow = n)
  expect_error(abess(dataset[["x"]], dataset[["y"]]), regexp = "x must")

  dataset[["x"]] <- matrix(as.numeric(dataset[["x"]]), nrow = n)
  dataset[["x"]] <- dataset[["x"]][, 1, drop = FALSE]
  expect_error(abess(dataset[["x"]], dataset[["y"]]), regexp = "x should")

  ########### Exception for y ############
  dataset <- generate.data(n, p, support_size)
  dataset[["y"]][1, 1] <- NA
  expect_error(abess(dataset[["x"]], dataset[["y"]]), regexp = "y has missing")
  dataset[["y"]][1, 1] <- Inf
  expect_error(abess(dataset[["x"]], dataset[["y"]]), regexp = "y has infinite")
  dataset[["y"]][1, 1] <- 0.3
  expect_error(abess(dataset[["x"]], dataset[["y"]], family = "binomial"),
    regexp = "Input binary y"
  )
  expect_error(abess(dataset[["x"]], dataset[["y"]], family = "poisson"),
    regexp = "y must be positive integer value"
  )
  expect_error(abess(dataset[["x"]], dataset[["y"]], family = "cox"),
    regexp = "y must be"
  )
  expect_error(abess(dataset[["x"]], dataset[["y"]], family = "mgaussian"),
    regexp = "y must be"
  )
  expect_error(abess(dataset[["x"]], dataset[["y"]], family = "multinomial"),
    regexp = "All of y value"
  )
  dataset[["y"]] <- cbind(dataset[["y"]], sort(dataset[["y"]]))
  expect_error(abess(dataset[["x"]], dataset[["y"]]),
    regexp = "The dimension of y"
  )

  dataset[["y"]] <- dataset[["y"]][, 1]
  dataset[["y"]] <- c(dataset[["y"]], sort(dataset[["y"]]))
  expect_error(abess(dataset[["x"]], dataset[["y"]]),
    regexp = "Rows of x"
  )

  ########### Exception for y ############
  dataset <- generate.data(n, p, support_size, family = "binomial")
  expect_warning(abess(dataset[["x"]], dataset[["y"]], family = "multinomial"),
    regexp = "We change to family = 'binomial'"
  )
})

test_that("abess (option) works", {
  n <- 20
  p <- 100
  support_size <- 3
  dataset <- generate.data(n, p, support_size)

  ## weight
  expect_error(abess(dataset[["x"]], dataset[["y"]], weight = rep(1, p)),
    regexp = "Rows of x"
  )

  ## c.max
  expect_warning(abess(dataset[["x"]], dataset[["y"]], c.max = 2.2),
    regexp = "c.max"
  )

  ## screening.num
  expect_error(abess(dataset[["x"]], dataset[["y"]], screening.num = p + 1),
    regexp = "screening"
  )
  expect_error(abess(dataset[["x"]], dataset[["y"]], screening.num = 1),
    regexp = "screening"
  )
  expect_error(abess(
    dataset[["x"]],
    dataset[["y"]],
    tune.path = "gsection",
    screening.num = 1
  ),
  regexp = "screening"
  )
  expect_warning(abess(dataset[["x"]], dataset[["y"]], screening.num = 19.2),
    regexp = "screening"
  )

  ## always.include
  expect_error(abess(dataset[["x"]], dataset[["y"]], always.include = c(-1)),
    regexp = "always.include"
  )
  expect_error(abess(dataset[["x"]], dataset[["y"]], always.include = c(1.2)),
    regexp = "always.include"
  )
  expect_error(abess(
    dataset[["x"]],
    dataset[["y"]],
    support.size = c(1, 2),
    always.include = c(1, 2, 3)
  ),
  regexp = "always.include"
  )
  expect_error(
    abess(
      dataset[["x"]],
      dataset[["y"]],
      tune.path = "gsection",
      gs.range = c(1, 2),
      always.include = c(1, 2, 3)
    ),
    regexp = "always.include"
  )

  ## nfold
  expect_warning(abess(
    dataset[["x"]],
    dataset[["y"]],
    nfolds = c(2.2),
    tune.type = "cv"
  ),
  regexp = "nfolds"
  )

  ## gs.range
  expect_warning(abess(
    dataset[["x"]],
    dataset[["y"]],
    tune.path = "gsection",
    gs.range = c(1.2, 8.3)
  ),
  regexp = "gs.range"
  )

  ## max.splicing.iter
  expect_warning(abess(dataset[["x"]], dataset[["y"]], max.splicing.iter = 3.2),
    regexp = "max.splicing.iter"
  )
})

test_that("abesspca exception handling works", {
  n <- 100
  p <- 20
  support_size <- 3
  dataset <- generate.data(n, p, support_size)

  ########### Exception for input matrix ############
  dataset[["x"]][1, 1] <- NA
  expect_error(abesspca(dataset[["x"]]), regexp = "x has missing")

  dataset[["x"]][1, 1] <- Inf
  expect_error(abesspca(dataset[["x"]]), regexp = "x has missing")

  dataset[["x"]][1, 1] <- 1
  dataset[["x"]] <- as.list(as.data.frame(dataset[["x"]]))
  expect_error(abesspca(dataset[["x"]]))

  dataset <- generate.data(n, p, support_size)
  dataset[["x"]] <- Matrix(dataset[["x"]])
  expect_error(abesspca(dataset[["x"]]))

  dataset[["x"]] <- as.matrix(dataset[["x"]])
  dataset[["x"]] <- matrix(as.character(dataset[["x"]]), nrow = n)
  expect_error(abess(dataset[["x"]], dataset[["y"]]), regexp = "x must")

  dataset[["x"]] <- matrix(as.numeric(dataset[["x"]]), nrow = n)
  dataset[["x"]] <- dataset[["x"]][, 1, drop = FALSE]
  expect_error(abess(dataset[["x"]], dataset[["y"]]), regexp = "x should")
})
