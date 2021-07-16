library(testthat)
library(abess)

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
  
  ## oracle estimation by svd function:
  svdobj <- svd(cov(USArrests))
  expect_equal(spca_fit[["var.all"]], svdobj[["d"]][1])
  
  ## check identity:
  spca_fit1 <- abesspca(USArrests)
  spca_fit2 <- abesspca(cov(USArrests), type = "gram")
  spca_fit1[["call"]] <- NULL
  spca_fit2[["call"]] <- NULL
  expect_true(all.equal(spca_fit1, spca_fit2))
})

test_that("abesspca (KPC) works", {
  data(USArrests)
  
  spca_fit <- abesspca(USArrests, 
                       support.size = rep(1, ncol(USArrests)), 
                       sparse.type = "kpc")
  
  ## Reasonablity of abesspca
  ev <- spca_fit[["ev"]]
  ev_len <- length(ev)
  expect_true(all(ev[1:(ev_len - 1)] < ev[2:ev_len]))
  
  ev_diff <- as.vector(diff(ev))
  ev_diff_len <- ev_len - 1
  expect_true(all(ev_diff[1:(ev_diff_len - 1)] > ev_diff[2:ev_diff_len]))
  
  ## oracle estimation by svd function:
  svdobj <- svd(cov(USArrests))
  expect_equal(spca_fit[["var.all"]], sum(svdobj[["d"]]))
})

test_that("abesspca (group) works", {
  data(USArrests)
  
  spca_fit <- abesspca(USArrests, group.index = c(1, 1, 2, 3))
  expect_true(max(spca_fit[["support.size"]]) == 3)
  
  spca_fit1 <- abesspca(USArrests)
  expect_true(all(coef(spca_fit)[, 3] == coef(spca_fit1)[, 4]))
})

test_that("abesspca (always.include) works", {
  skip("Skip always.include now!")
  data(USArrests)
  spca_fit <- abesspca(USArrests, always.include = c(1))
  expect_true(all(coef(spca_fit)[1, , drop] == 0))
})
