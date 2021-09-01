library(glmnet)
library(abess)
library(ncvreg)
library(L0Learn)
library(mccr)
library(pROC)
library(ggplot2)
library(tidyr)
M <- 100
p <- 1000
rho_list <- c(0.1, 0.7)
model_list <- c("linear", "logistic")
for (model in model_list)
{
  if (model == "logistic") {
    n <- 1000
  } else {
    n <- 500
  }
  source(paste0(model, "_source.R"))
  for (rho in rho_list)
  {
    cortype <- ifelse(rho <= 0.5, 2, 3)
    if (model != "poisson")
    {
      res <- array(0, dim = c(6, 6, M), dimnames = list(
        c(
          "glmnet",
          "ncvreg.MCP",
          "ncvreg.SCAD",
          "ncvreg.lasso",
          "L0Learn.CD",
          "abess"
        ),
        c("coef.err", "pred.prf", "tpr", "fpr", "mcc",
          "time")
      ))
    } else
    {
      res <- array(0, dim = c(5, 6, M), dimnames = list(
        c(
          "glmnet",
          "ncvreg.MCP",
          "ncvreg.SCAD",
          "ncvreg.lasso",
          "abess"
        ),
        c("coef.err", "pred.prf", "tpr", "fpr", "mcc", "time")
      ))
    }
    for (i in 1:M)
    {
      print(paste("i", i))
      tmp <-
        simu(
          i,
          n,
          p,
          rho,
          cortype = cortype,
          method = c(
            "glmnet",
            "ncvreg.MCP",
            "ncvreg.SCAD",
            "ncvreg.lasso",
            "L0Learn.CD",
            "abess"
          )
        )
      res[, , i] <- tmp
    }
    file_name <-
      paste(model,
            "n",
            n,
            "p",
            p,
            "rho",
            rho,
            "cortype",
            cortype,
            "M",
            M,
            sep = "_")
    file_name <- paste0(file_name, ".rda")
    save(res, file = file_name)
  }
}


################ Visualize result ################
M = 100
p = 1000
rho_list = c(0.1, 0.7)
model_list = c("linear", "logistic")
plot_data = NULL
for (model in model_list)
{
  if (model == "logistic") {
    n <- 1000
  } else {
    n <- 500
  }
  for (rho in rho_list)
  {
    cortype = ifelse(rho < 0.5, 2, 3)
    if (model == "logistic") {
      # n =1000
      file_name = paste(model,
                        "n",
                        n,
                        "p",
                        p,
                        "rho",
                        rho,
                        "cortype",
                        cortype,
                        "M",
                        M,
                        sep = "_")
      file_name = paste0(file_name, ".rda")
      load(file_name)
    } else{
      n = 500
      file_name = paste(model,
                        "n",
                        n,
                        "p",
                        p,
                        "rho",
                        rho,
                        "cortype",
                        cortype,
                        "M",
                        M,
                        sep = "_")
      file_name = paste0(file_name, ".rda")
      load(file_name)
    }
    
    if (model != "poisson")
    {
      res = res[, which(dimnames(res)[[2]] == "time"),]
      res = res[c(1, 4, 3, 2, 5, 6), ]
      dimnames(res)[[1]] = c(
        "glmnet-LASSO",
        "ncvreg-LASSO",
        "ncvreg-SCAD",
        "ncvreg-MCP",
        "L0Learn-CD",
        "ABESS"
      )
      
    } else{
      res = res[, which(dimnames(res)[[2]] == "time"),]
      res = res[c(1, 4, 3, 2, 5),]
      dimnames(res)[[1]] = c("glmnet-LASSO",
                             "ncvreg-LASSO",
                             "ncvreg-SCAD",
                             "ncvreg-MCP",
                             "ABESS")
    }
    
    res_mean = apply(res, 1, mean)
    res_sd = apply(res, 1, sd)
    tmp = cbind(value = res_mean, sd = res_sd)
    tmp = data.frame(tmp)
    tmp$method = rownames(tmp)
    tmp$rho = rho
    tmp$model = model
    plot_data = rbind(plot_data, tmp)
  }
}


model_names = c("linear" = "Linear",
                "logistic" = "Logistic")

color = rev(c(
  '#b2182b',
  '#ef8a62',
  '#fddbc7',
  '#bcbcbc',
  '#d1e5f0',
  '#2166ac'
))

plot_data$rho = as.factor(plot_data$rho)
plot_data$method = factor(
  plot_data$method,
  levels = c(
    "glmnet-LASSO",
    "ncvreg-LASSO",
    "ncvreg-SCAD",
    "ncvreg-MCP",
    "L0Learn-CD",
    "ABESS"
  )
)
p <-
  ggplot(plot_data, aes(x = rho, y = value, fill = method), coef = 5) +
  geom_bar(
    stat = "identity",
    color = "black",
    position = position_dodge(),
    alpha = 1
  ) +
  geom_errorbar(aes(ymin = value - sd, ymax = value + sd),
                width = .2,
                position = position_dodge(.9)) +
  facet_wrap(
    ~ model,
    scales = "free",
    nrow = 1,
    labeller = as_labeller(model_names)
  ) +
  labs(y = "Run Time (s)") +
  scale_x_discrete(labels = c("Low Corr", "High Corr")) +
  scale_fill_manual(
    values = color,
    labels = c(
      "glmnet-LASSO",
      "ncvreg-LASSO",
      "ncvreg-SCAD",
      "ncvreg-MCP",
      "L0Learn-CD",
      "ABESS"
    )
  ) +
  theme_bw() +
  guides(fill = guide_legend(nrow = 1)) +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.title = element_blank(),
    legend.text.align = 0
  )

ggsave(p,
       filename = "r_runtime.png",
       width = 11,
       height = 4)

###### variable selection and estimation plot ######
plot_fun <- function(model)
{
  plot_data = NULL
  M = 100
  n = 500
  p = 1000
  rho_list = c(0.1, 0.7)
  model_list = c("linear", "logistic")
  for (rho in rho_list)
  {
    cortype = ifelse(rho < 0.5, 2, 3)
    if (model == "logistic") {
      n = 1000
      file_name = paste(model,
                        "n",
                        n,
                        "p",
                        p,
                        "rho",
                        rho,
                        "cortype",
                        cortype,
                        "M",
                        M,
                        sep = "_")
      file_name = paste0(file_name, ".rda")
      load(file_name)
    } else{
      n = 500
      file_name = paste(model,
                        "n",
                        n,
                        "p",
                        p,
                        "rho",
                        rho,
                        "cortype",
                        cortype,
                        "M",
                        M,
                        sep = "_")
      file_name = paste0(file_name, ".rda")
      load(file_name)
    }
    
    if (model != "poisson")
    {
      res = res[,-which(dimnames(res)[[2]] %in% c("time", "mcc", 'coef.err')),]
      res = res[c(1, 4, 3, 2, 5, 6), ,]
      dimnames(res)[[1]] = c(
        "glmnet-LASSO",
        "ncvreg-LASSO",
        "ncvreg-SCAD",
        "ncvreg-MCP",
        "L0Learn-CD",
        "ABESS"
      )
      
    } else{
      res = res[,-which(dimnames(res)[[2]] %in% c("time", "mcc", 'coef.err')),]
      res = res[c(1, 4, 3, 2, 5),]
      dimnames(res)[[1]] = c("glmnet-LASSO",
                             "ncvreg-LASSO",
                             "ncvreg-SCAD",
                             "ncvreg-MCP",
                             "ABESS")
    }
    res_tmp <- res
    size <- dim(res_tmp)[3]
    res_list <- list()
    for (i in 1:size) {
      res_list[[i]] <- res_tmp[, , i]
    }
    res_list <- do.call("rbind", res_list)
    col_name <- dimnames(res)[[2]]
    colnames(res_list) <- col_name
    res_list <- as.data.frame(res_list)
    rownames(res_list) <- NULL
    row_name <-  dimnames(res)[[1]]
    
    res_list[["method"]] <- rep(row_name, size)
    res_list[["method"]] <- factor(res_list[["method"]],
                                   levels =   row_name)
    
    plot_data_tmp = gather(res_list, metric, value, -method)
    plot_data_tmp$metric = factor(plot_data_tmp$metric , levels = col_name)
    plot_data_tmp$method = factor(plot_data_tmp$method, levels = row_name)
    plot_data_tmp$rho = rho
    plot_data = rbind(plot_data, plot_data_tmp)
  }
  
  if (model == "logistic")
  {
    metric_names <- c('pred.prf' = "AUC",
                      'tpr' = "TPR",
                      'fpr' = "FPR")
  } else{
    metric_names <- c('pred.prf' = "SSE",
                      'tpr' = "TPR",
                      'fpr' = "FPR")
  }
  
  plot_data$rho = as.factor(plot_data$rho)
  p = ggplot(plot_data, aes(x = rho, y = value, fill = method), coef = 5) +
    geom_boxplot() +
    facet_wrap(
      ~ metric,
      scales = "free",
      ncol = 1,
      labeller = as_labeller(metric_names)
    ) +
    scale_fill_manual(
      values = color,
      labels = c(
        "glmnet-LASSO",
        "ncvreg-LASSO",
        "ncvreg-SCAD",
        "ncvreg-MCP",
        "L0Learn-CD",
        "ABESS"
      )
    ) +
    scale_x_discrete(labels = c("Low Corr", "High Corr")) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      panel.grid = element_blank(),
      axis.title = element_blank(),
      legend.text.align = 0
    )
  
  p
}
p1 = plot_fun("linear")
p2 = plot_fun(("logistic"))
p = ggpubr::ggarrange(p1,
                      p2,
                      ncol = 2,
                      common.legend = T,
                      legend = "bottom")

ggsave(p,
       filename = "r_performance.png",
       width = 6,
       height = 8)
