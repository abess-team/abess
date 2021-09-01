library(testthat)
library(abess)
require(Matrix)

test_that("abesspca (FPC) works", {
  data(USArrests)
  
  spca_fit <- abesspca(USArrests)
  
  ## Reasonablity of abesspca
  ev <- spca_fit[["ev"]]
  ev_len <- length(ev)
  expect_true(all(ev[1:(ev_len - 1)] < ev[2:ev_len]))
  
  ev_diff <- as.vector(diff(ev))
  ev_diff_len <- ev_len - 1
  expect_true(all(ev_diff[1:(ev_diff_len - 1)] > ev_diff[2:ev_diff_len]))
  
  expect_true(all(spca_fit[["pev"]] <= 1))
  
  ## oracle estimation by svd function:
  svdobj <- svd(cov(USArrests))
  expect_equal(spca_fit[["var.all"]], svdobj[["d"]][1])
  
  ## check identity:
  spca_fit1 <- abesspca(USArrests)
  spca_fit2 <- abesspca(cov(USArrests), type = "gram")
  spca_fit1[["call"]] <- NULL
  spca_fit2[["call"]] <- NULL
  expect_true(all.equal(spca_fit1, spca_fit2))
  
  ## check identity:
  spca_fit1 <- abesspca(USArrests, cor = TRUE)
  spca_fit2 <- abesspca(cor(USArrests), type = "gram")
  spca_fit1[["call"]] <- NULL
  spca_fit2[["call"]] <- NULL
  expect_true(all.equal(spca_fit1, spca_fit2))
  
  ## check identity:
  spca_fit1 <- abesspca(USArrests)
  spca_fit2 <- abesspca(USArrests, support.size = 1:4)
  spca_fit1[["call"]] <- NULL
  spca_fit2[["call"]] <- NULL
  expect_true(all.equal(spca_fit1, spca_fit2))
})

test_that("abesspca (KPC) works", {
  data(USArrests)
  
  ## Input 1:
  spca_fit <- abesspca(USArrests,
                       support.size = rep(1, ncol(USArrests)),
                       sparse.type = "kpc")
  
  ## Reasonablity of abesspca
  skip_on_os("linux")
  ev <- spca_fit[["ev"]]
  ev_len <- length(ev)
  expect_true(all(ev[1:(ev_len - 1)] < ev[2:ev_len]))
  
  ev_diff <- as.vector(diff(ev))
  ev_diff_len <- ev_len - 1
  expect_true(all(ev_diff[1:(ev_diff_len - 1)] > ev_diff[2:ev_diff_len]))
  
  expect_true(all(spca_fit[["pev"]] <= 1))
  
  ## oracle estimation by svd function:
  svdobj <- svd(cov(USArrests))
  expect_equal(spca_fit[["var.all"]], sum(svdobj[["d"]]))
  
  ## Input 2:
  spca_fit1 <- abesspca(USArrests,
                        sparse.type = "kpc",
                        support.size = rep(4, 4))
  expect_true(all(spca_fit1[["pev"]] <= (1 + 1e-8)))
  
  ## Input 3 (default input):
  spca_fit1 <- abesspca(USArrests, sparse.type = "kpc")
  expect_true(all(spca_fit1[["pev"]] <= 1))
})

test_that("abesspca (group) works", {
  data(USArrests)
  
  spca_fit <- abesspca(USArrests, group.index = c(1, 1, 2, 3))
  expect_true(max(spca_fit[["support.size"]]) == 3)
  
  spca_fit1 <- abesspca(USArrests)
  expect_true(all(coef(spca_fit)[, 3] == coef(spca_fit1)[, 4]))
  
  spca_fit2 <- abesspca(USArrests,
                        group.index = c(1, 1, 2, 3),
                        sparse.type = "kpc")
  expect_true(all(spca_fit[["pev"]] <= 1))
})

test_that("abesspca (sparse) works", {
  data(USArrests)
  set.seed(123)
  zero_matrix <-
    sample(0:1, size = prod(dim(USArrests)), replace = TRUE)
  zero_matrix <- matrix(zero_matrix, nrow = nrow(USArrests))
  USArrests[zero_matrix == 0] <- 0
  ## covariance matrix:
  USArrests <- as.matrix(USArrests)
  spca_fit1 <- abesspca(USArrests)
  USArrests <- Matrix(USArrests, sparse = TRUE)
  spca_fit2 <- abesspca(USArrests)
  expect_true(all.equal(spca_fit1, spca_fit2))
  
  ## correlation matrix:
  USArrests <- as.matrix(USArrests)
  spca_fit1 <- abesspca(USArrests, cor = TRUE)
  USArrests <- Matrix(USArrests, sparse = TRUE)
  spca_fit2 <- abesspca(USArrests, cor = TRUE)
  expect_true(all.equal(spca_fit1, spca_fit2))
})

test_that("abesspca (always.include) works", {
  data(USArrests)
  spca_fit <- abesspca(USArrests, always.include = c(2))
  expect_true(all(coef(spca_fit)[2, , drop = TRUE] != 0))
})
