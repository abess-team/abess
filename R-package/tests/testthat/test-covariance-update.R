library(abess)
library(testthat)

test_that("Covariance update works", {
  skip('Skip Covariance update!')
  n <- 10000
  p <- 100
  support_size <- 3
  
  ## default interface
  dataset <- generate.data(n, p, support_size, seed = 1)
  t1 <- system.time(abess_fit <- abess(dataset[["x"]], 
                                       dataset[["y"]],
                                       tune.type = "cv", 
                                       num.threads = 1))
  t2 <- system.time(abess_fit <- abess(dataset[["x"]], 
                                       dataset[["y"]], 
                                       type.gaussian = "covariance", 
                                       num.threads = 1))
  
  expect_lt(t2[3], t1[3])
  
})