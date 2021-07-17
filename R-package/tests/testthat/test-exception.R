library(abess)
library(testthat)
require(Matrix)

test_that("abess exception handling works", {
  n <- 100
  p <- 20
  support_size <- 3
  dataset <- generate.data(n, p, support_size)
  
  ########### Exception for x ############
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
               regexp = "Input binary y")
  expect_error(abess(dataset[["x"]], dataset[["y"]], family = "poisson"), 
               regexp = "y must be positive integer value")
  expect_error(abess(dataset[["x"]], dataset[["y"]], family = "cox"), 
               regexp = "y must be")
  expect_error(abess(dataset[["x"]], dataset[["y"]], family = "mgaussian"), 
               regexp = "y must be")
  expect_error(abess(dataset[["x"]], dataset[["y"]], family = "multinomial"), 
               regexp = "All of y value")
  dataset[["y"]] <- cbind(dataset[["y"]], sort(dataset[["y"]]))
  expect_error(abess(dataset[["x"]], dataset[["y"]]), 
               regexp = "The dimension of y")
  
  dataset[["y"]] <- dataset[["y"]][, 1]
  dataset[["y"]] <- c(dataset[["y"]], sort(dataset[["y"]]))
  expect_error(abess(dataset[["x"]], dataset[["y"]]), 
               regexp = "Rows of x")
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