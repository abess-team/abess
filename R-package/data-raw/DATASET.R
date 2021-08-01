library(readr)
library(SIS)


################## Trim 32 dataset ##################
trim32 <- readRDS("./data-raw/trim32.rds")

sis_fit <- SIS(as.matrix(trim32[["x"]]), as.vector(trim32[["y"]]),
  family = "gaussian", iter = FALSE, nsis = 500
)

trim32[["x"]] <- trim32[["x"]]
trim32_x <- trim32[["x"]][, sis_fit[["sis.ix0"]]]

trim32 <- cbind.data.frame("y" = trim32[["y"]], trim32_x)
usethis::use_data(trim32, trim32, overwrite = TRUE)
