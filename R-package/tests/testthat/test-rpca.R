library(abess)
require(Matrix)

test_that("primary model fit for RPCA", {
})

test_that("RPCA works", {
  n <- 30
  p <- 30
  true_S_size <- 60
  true_L_rank <- 2
  dataset <- generate.matrix(n, p, support.size = true_S_size, 
                             rank = true_L_rank)
  abess_fit <- abessrpca(dataset[["x"]], rank = true_L_rank)
  expect_s3_class(abess_fit, class = "abessrpca")
})

test_that("RPCA (group) works", {
})

test_that("RPCA (sparse) works", {
})

test_that("RPCA (always.include) works", {
})
