metrics <- function(beta.fit, dat.test)
{
  coef.err <- norm(beta.fit - c(0, dat.test$beta), "2")
  y.pred <- dat.test$x %*% beta.fit[-1] + beta.fit[1]
  pe <- norm(y.pred - dat.test$y, "2")
  nonzero.fit <- as.numeric(abs(beta.fit[-1]) > 1e-05)
  nonzero.true <- as.numeric(abs(dat.test$beta) > 1e-05)
  tpr <- length(which(nonzero.fit > 0 & nonzero.true > 0))/sum(nonzero.true)
  fpr <- length(which(nonzero.fit > 0 & nonzero.true == 0))/sum(!nonzero.true)
  mcc <- mccr(nonzero.true, nonzero.fit)
  return(c(coef.err = coef.err, pe = pe, tpr = tpr, fpr = fpr, mcc = mcc))
}


simu.glmnet <- function(dat, dat.test)
{
  ptm <- proc.time()
  res <- cv.glmnet(dat$x, dat$y, nfolds = 5)
  t <- (proc.time() - ptm)[3]
  beta.fit <- coef(res, s = res$lambda.min)
  metrics.glmnet <- metrics(beta.fit, dat.test)
  return(c(metrics.glmnet, t = t))
}

simu.ncvreg <- function(dat, dat.test, penalty)
{
  ptm <- proc.time()
  res <- cv.ncvreg(dat$x, dat$y, penalty = penalty, nfolds = 5)
  t <- (proc.time() - ptm)[3]
  beta.fit <- coef(res, lambda = res$lambda.min)
  metrics.ncvreg <- metrics(beta.fit, dat.test)
  return(c(metrics.ncvreg, t = t))
}

simu.L0learn <- function(dat, dat.test, algorithm)
{
  ptm <- proc.time()
  res <- L0Learn.cvfit(dat$x, dat$y, algorithm = algorithm, nFolds = 5)
  t <- (proc.time() - ptm)[3]
  lambda <- print(res)[, 1]
  beta.fit <- coef(res, lambda = lambda[which.min(res$cvMeans[[1]])])
  metrics.L0Learn <- metrics(beta.fit, dat.test)
  return(c(metrics.L0Learn, t = t))
}

simu.abess <- function(dat, dat.test)
{
  ptm <- proc.time()
  res <- abess(dat$x, dat$y, support.size = 0:99, tune.type = "cv", num.threads = 5)
  t <- (proc.time() - ptm)[3]
  beta.fit <- coef(res, support.size = res$support.size[which.min(res$tune.value)])
  metrics.abess <- metrics(beta.fit, dat.test)
  return(c(metrics.abess, t = t))
}

simu <- function(i, n, p, rho, cortype, method = c("glmnet", "ncvreg.MCP", 
                                                   "ncvreg.SCAD", "ncvreg.lasso", "L0Learn.CD", "L0Learn.CDPSI", "abess"))
{
  set.seed(i)
  dat <- generate.data(n, p, rho = rho, cortype = cortype, support.size = 10, 
                       sigma = 1, seed = i)
  dat.test <- generate.data(n, p, rho = rho, cortype = cortype, beta = dat$beta, 
                            sigma = 1, seed = i + 100)
  res.default <- rep(0, 6)
  if ("glmnet" %in% method)
  {
    res.glmnet <- simu.glmnet(dat, dat.test)
  } else
  {
    res.glmnet <- res.default
  }
  if ("ncvreg.MCP" %in% method)
  {
    res.ncvreg.MCP <- simu.ncvreg(dat, dat.test, penalty = "MCP")
  } else
  {
    res.ncvreg.MCP <- res.default
  }
  if ("ncvreg.SCAD" %in% method)
  {
    res.ncvreg.SCAD <- simu.ncvreg(dat, dat.test, penalty = "SCAD")
  } else
  {
    res.ncvreg.SCAD <- res.default
  }
  if ("ncvreg.lasso" %in% method)
  {
    res.ncvreg.lasso <- simu.ncvreg(dat, dat.test, penalty = "lasso")
  } else
  {
    res.ncvreg.lasso <- res.default
  }
  if ("L0Learn.CD" %in% method)
  {
    res.L0Learn.CD <- simu.L0learn(dat, dat.test, algorithm = "CD")
  } else
  {
    res.L0Learn.CD <- res.default
  }
  # if ("L0Learn.CDPSI" %in% method)
  # {
  #   res.L0Learn.CDPSI <- simu.L0learn(dat, dat.test, algorithm = "CDPSI")
  # } else
  # {
  #   res.L0Learn.CDPSI <- res.default
  # }
  if ("abess" %in% method)
  {
    res.abess <- simu.abess(dat, dat.test)
  } else
  {
    res.abess <- res.default
  }
  
  # print(rbind(res.glmnet, res.ncvreg.MCP, res.ncvreg.SCAD, res.ncvreg.lasso, 
  #             res.L0Learn.CD, res.abess))
  return(rbind(res.glmnet, res.ncvreg.MCP, res.ncvreg.SCAD, res.ncvreg.lasso, 
               res.L0Learn.CD, res.abess))
}
