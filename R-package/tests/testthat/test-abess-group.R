library(abess)
library(testthat)

test_that("Group selection: abess (gaussian) works", {
  n <- 500
  p <- 1500
  support_size <- 8
  
  dataset <- generate.data(n, p, support_size, seed = 1)
  true_index <- which(dataset[["beta"]] != 0)
  
  tmp <- dataset[["x"]][, 1:support_size]
  dataset[["x"]][, 1:support_size] <- dataset[["x"]][, true_index]
  dataset[["x"]][, true_index] <- tmp
  
  ## set group index
  group_index <- rep(1:(p / 2), each = 2)
  
  ## default interface
  abess_fit <- abess(dataset[["x"]], dataset[["y"]], group.index = group_index)
  expect_equal(extract(abess_fit)[["support.vars"]], paste0("x", 1:support_size))
  
  ## formula interface
  dat <- cbind.data.frame(dataset[["x"]][, 1:30], "y" = dataset[["y"]], 
                          dataset[["x"]][, 31:p])
  abess_fit <- abess(y ~ ., data = dat, group.index = group_index)
  expect_equal(extract(abess_fit)[["support.vars"]], paste0("x", 1:support_size))
})