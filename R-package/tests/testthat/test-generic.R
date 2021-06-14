library(abess)
library(testthat)

test_that("generic (univariate) works", {
  n <- 100
  p <- 200
  support_size <- 3
  
  dataset <- generate.data(n, p, support_size, seed = 1)
  abess_fit <- abess(dataset[["x"]], dataset[["y"]])
  
  expect_invisible(print(abess_fit))
  
  expect_invisible(plot(abess_fit))
  expect_invisible(plot(abess_fit, type = "l2norm"))
  expect_invisible(plot(abess_fit, type = "dev"))
  expect_invisible(plot(abess_fit, type = "tune"))
  
  expect_visible(coef(abess_fit))
  expect_visible(coef(abess_fit, support.size = 2))
  expect_visible(coef(abess_fit, sparse = FALSE))
  
  expect_visible(predict(abess_fit, newx = dataset[["x"]][1:10, ]))
  expect_visible(predict(abess_fit, newx = dataset[["x"]][1:10, ], support.size = c(3, 4)))
  
  expect_visible(extract(abess_fit))
  expect_visible(extract(abess_fit, support.size = 4))
  
  expect_visible(deviance(abess_fit))
  expect_visible(deviance(abess_fit, type = "gic"))
  expect_visible(deviance(abess_fit, type = "aic"))
  expect_visible(deviance(abess_fit, type = "bic"))
  expect_visible(deviance(abess_fit, type = "ebic"))
  
})


test_that("generic (multivariate) works", {
  n <- 100
  p <- 200
  support_size <- 3
  
  dataset <- generate.data(n, p, support_size, seed = 1, family = "mgaussian")
  abess_fit <- abess(dataset[["x"]], dataset[["y"]], 
                     family = "mgaussian", tune.type = "gic")
  
  expect_invisible(print(abess_fit))
  
  expect_invisible(plot(abess_fit))
  expect_invisible(plot(abess_fit, type = "l2norm"))
  expect_invisible(plot(abess_fit, type = "dev"))
  expect_invisible(plot(abess_fit, type = "tune"))
  
  expect_visible(coef(abess_fit))
  expect_visible(coef(abess_fit, support.size = 2))
  expect_visible(coef(abess_fit, sparse = FALSE))
  
  expect_visible(predict(abess_fit, newx = dataset[["x"]][1:10, ]))
  expect_visible(predict(abess_fit, newx = dataset[["x"]][1:10, ], support.size = c(3, 4)))
  
  expect_visible(extract(abess_fit))
  expect_visible(extract(abess_fit, support.size = 4))
  
  expect_visible(deviance(abess_fit))
  expect_visible(deviance(abess_fit, type = "gic"))
  expect_visible(deviance(abess_fit, type = "aic"))
  expect_visible(deviance(abess_fit, type = "bic"))
  expect_visible(deviance(abess_fit, type = "ebic"))
})


test_that("generic (univariate) works", {
  n <- 100
  p <- 200
  support_size <- 3
  
  dataset <- generate.data(n, p, support_size, seed = 1)
  abess_fit <- abesspca(dataset[["x"]])
  
  expect_invisible(print(abess_fit))
  
  expect_invisible(plot(abess_fit))
  expect_invisible(plot(abess_fit, type = "variance"))
  
  expect_visible(loadings(abess_fit))
  expect_visible(loadings(abess_fit, support.size = 2))
  expect_visible(loadings(abess_fit, sparse = FALSE))
})
