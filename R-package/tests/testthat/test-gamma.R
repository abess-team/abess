library(abess)

for(seed in 1:10){
  n <- 2000
  p <- 100
  support.size <- 3
  dataset <- generate.data(n, p, support.size,
                           family = "gamma", seed = seed*18)
  
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "gamma",
    tune.type = "cv",
    #newton = "exact",
    newton.thresh = 1e-8,
    support.size = support.size,
    #always.include = which(dataset[["beta"]] != 0)
  )
  
  
  ## support size
  fit_s_size <- abess_fit[["best.size"]]
  coef_value <- coef(abess_fit, support.size = fit_s_size)
  est_index <- coef_value@i[-1]
  true_index <- which(dataset[["beta"]]!= 0)
  
  cat(true_index-est_index,";")
  
  start_point <- rep(1,length(true_index))
  coef0 <- abs(min(dataset[["x"]][, true_index] %*% start_point)) + 1
  #oracle_est <- glm(y ~ ., data = dat, family = "Gamma",start = c(coef0,start_point))
  
  #cat(oracle_est)
}



