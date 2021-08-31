library(glmnet)
library(abess)
library(ncvreg)
library(L0Learn)
library(mccr)
library(pROC)
M <- 100
p <- 1000
n <- 500
rho_list <- c(0.7, 0.1)
model_list <- c("linear", "logistic")
for (model in model_list)
{
  source(paste0(model, "_source.R"))
  for (rho in rho_list)
  {
    cortype <- ifelse(rho <= 0.5, 2, 3)
    if (model != "poisson")
    {
      res <- array(0, dim = c(6, 6, M), dimnames = list(c("glmnet", 
                                                          "ncvreg.MCP", "ncvreg.SCAD", "ncvreg.lasso", "L0Learn.CD", 
                                                          "abess"), c("coef.err", "pred.prf", "tpr", "fpr", "mcc", 
                                                                      "time")))
    } else
    {
      res <- array(0, dim = c(5, 6, M), dimnames = list(c("glmnet", 
                                                          "ncvreg.MCP", "ncvreg.SCAD", "ncvreg.lasso", "abess"), 
                                                        c("coef.err", "pred.prf", "tpr", "fpr", "mcc", "time")))
    }
    for (i in 1:M)
    {
      print(paste("i", i))
      tmp <- simu(i, n, p, rho, cortype = cortype, method = c("glmnet", 
                                                              "ncvreg.MCP", "ncvreg.SCAD", "ncvreg.lasso", "L0Learn.CD", 
                                                              "abess"))
      res[, , i] <- tmp
    }
    file_name <- paste(model, "n", n, "p", p, "rho", rho, "cortype", 
                       cortype, "M", M, sep = "_")
    file_name <- paste0(file_name, ".rda")
    save(res, file = file_name)
  }
}
