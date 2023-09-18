Sys.setenv("OMP_THREAD_LIMIT" = 2)

library(testthat)
library(abess)

test_check("abess")
