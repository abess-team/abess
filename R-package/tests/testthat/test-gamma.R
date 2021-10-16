library(abess)

num <- 0
#for(seed in 1:100){
  n <- 1000
  p <- 100
  support.size <- 3
  dataset <- generate.data(n, p, support.size,
                           family = "gamma", seed = 16)
  
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "gamma",
    tune.type = "cv",
    newton.thresh = 1e-8,
    support.size = 0:10,
    #always.include = which(dataset[["beta"]] != 0)
  )
  
  
  ## support size
  fit_s_size <- abess_fit[["best.size"]]
  coef_value <- coef(abess_fit, support.size = fit_s_size)
  est_index <- coef_value@i[-1]
  true_index <- which(dataset[["beta"]]!= 0)
  
  cat(true_index,':',est_index,";")
  
  if(length(est_index)==support.size && true_index==est_index){
    cat(seed," ")
    num = num+1
  }
#}



