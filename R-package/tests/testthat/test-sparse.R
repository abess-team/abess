require(Matrix)
library(abess)
library(testthat)

sparse_batch_check <- function(abess_fit1, abess_fit2) {
  expect_equal(abess_fit1[["tune.value"]], abess_fit2[["tune.value"]],
    tolerance = 1e-2
  )
  expect_equal(abess_fit1[["dev"]], abess_fit2[["dev"]],
    tolerance = 1e-2
  )
  if (abess_fit1[["family"]] %in% c("mgaussian", "multinomial")) {
    support_size_num <- length(abess_fit1[["beta"]])
    for (i in 1:support_size_num) {
      if (length(abess_fit1[["beta"]][[1]]@i) != 0) {
        expect_lt(mean(abess_fit1[["beta"]][[i]]@i - abess_fit2[["beta"]][[i]]@i), 1e-3)
        expect_equal(abess_fit1[["beta"]][[i]]@x, abess_fit2[["beta"]][[i]]@x,
          tolerance = 1e-2
        )
      }
    }
  } else {
    expect_lt(mean(abess_fit1[["beta"]]@i - abess_fit2[["beta"]]@i), 1e-3)
    expect_equal(abess_fit1[["beta"]]@x, abess_fit2[["beta"]]@x,
      tolerance = 1e-2
    )
  }
}

test_that("Sparse matrix (gaussian) works", {
  n <- 30
  p <- 20
  support.size <- 3
  dataset <- generate.data(n, p, support.size)
  abess_fit1 <- abess(dataset[["x"]], dataset[["y"]])
  abess_fit2 <- abess(as.matrix(dataset[["x"]]), dataset[["y"]])

  abess_fit1[["call"]] <- NULL
  abess_fit2[["call"]] <- NULL
  expect_true(all.equal(abess_fit1, abess_fit2))
})

test_that("Sparse matrix (logistic) works", {
  n <- 200
  p <- 20
  support.size <- 3
  dataset <- generate.data(n, p, support.size, family = "binomial")
  dataset[["x"]][abs(dataset[["x"]]) < 1] <- 0
  dataset[["x"]] <- Matrix(dataset[["x"]])
  abess_fit1 <-
    abess(dataset[["x"]], dataset[["y"]], family = "binomial")
  abess_fit2 <-
    abess(as.matrix(dataset[["x"]]), dataset[["y"]], family = "binomial")

  abess_fit1[["call"]] <- NULL
  abess_fit2[["call"]] <- NULL
  sparse_batch_check(abess_fit1, abess_fit2)
})


test_that("Sparse matrix (poisson) works", {
  n <- 100
  p <- 20
  support.size <- 3
  dataset <- generate.data(n, p, support.size, family = "poisson")
  dataset[["x"]][abs(dataset[["x"]]) < 1] <- 0
  dataset[["x"]] <- Matrix(dataset[["x"]])
  abess_fit1 <-
    abess(dataset[["x"]], dataset[["y"]], family = "poisson")
  abess_fit2 <-
    abess(as.matrix(dataset[["x"]]), dataset[["y"]], family = "poisson")

  abess_fit1[["call"]] <- NULL
  abess_fit2[["call"]] <- NULL
  sparse_batch_check(abess_fit1, abess_fit2)
})

test_that("Sparse matrix (cox) works", {
  n <- 100
  p <- 20
  support.size <- 3
  dataset <- generate.data(n, p, support.size, family = "poisson")
  dataset[["x"]][abs(dataset[["x"]]) < 1] <- 0
  dataset[["x"]] <- Matrix(dataset[["x"]])
  abess_fit1 <-
    abess(dataset[["x"]], dataset[["y"]], family = "poisson")
  abess_fit2 <-
    abess(as.matrix(dataset[["x"]]), dataset[["y"]], family = "poisson")

  abess_fit1[["call"]] <- NULL
  abess_fit2[["call"]] <- NULL
  sparse_batch_check(abess_fit1, abess_fit2)
})

test_that("Sparse matrix (mgaussian) works", {
  n <- 100
  p <- 20
  support.size <- 3
  dataset <- generate.data(n, p, support.size, family = "mgaussian")
  dataset[["x"]][abs(dataset[["x"]]) < 1] <- 0
  dataset[["x"]] <- Matrix(dataset[["x"]])
  abess_fit1 <-
    abess(dataset[["x"]], dataset[["y"]], family = "mgaussian")
  abess_fit2 <-
    abess(as.matrix(dataset[["x"]]), dataset[["y"]], family = "mgaussian")

  abess_fit1[["call"]] <- NULL
  abess_fit2[["call"]] <- NULL
  sparse_batch_check(abess_fit1, abess_fit2)
})


test_that("Sparse matrix (multinomial) works", {
  n <- 200
  p <- 20
  support.size <- 3
  dataset <-
    generate.data(n, p, support.size, family = "multinomial")
  dataset[["x"]][abs(dataset[["x"]]) < 1] <- 0
  dataset[["x"]] <- Matrix(dataset[["x"]])
  abess_fit1 <-
    abess(dataset[["x"]], dataset[["y"]], family = "multinomial")
  abess_fit2 <-
    abess(as.matrix(dataset[["x"]]), dataset[["y"]], family = "multinomial")

  abess_fit1[["call"]] <- NULL
  abess_fit2[["call"]] <- NULL
  sparse_batch_check(abess_fit1, abess_fit2)
})
