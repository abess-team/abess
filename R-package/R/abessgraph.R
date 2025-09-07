#' 
#' Nodewise logistic regression for inverse Ising problem
#' 
#' @inheritParams abess.default
#'
#' @param max.support.size 
#'
#' @return a sparse interaction matrix estimation
#' @export
#'
#' @examples
#' p <- 16
#' n <- 10
#' library(abess)
#' train <- generate.bmn.data(n, p, type = 3, graph.seed = 1, seed = 1, beta = 0.4)
#' res <- slide(train[["data"]], train[["weight"]], tune.type = "gic", 
#'              max.support.size = rep(4, p), support.size = rep(4, p))
#' all((res[[1]] != 0) == (train[["theta"]] != 0))
#' 
#' ## use cross validation to nodewisely select support.size
#' valid <- generate.bmn.data(n, p, type = 3, graph.seed = 1, seed = 10000, beta = 0.4)
#' all(train[["theta"]] == valid[["theta"]])
#' x <- rbind(train[["data"]], valid[["data"]])
#' sample_weight <- c(train[["weight"]], valid[["weight"]])
#' fold_id <- c(rep(1, length(train[["weight"]])), rep(2, length(valid[["weight"]])))
#' res <- slide(x, sample_weight, tune.type = "cv", foldid = fold_id, graph.threshold = 0.2)
#' all((res[[1]] != 0) == (train[["theta"]] != 0))
#' 
#' ## use IC to nodewisely select support.size (without post-thresholding)
#' res <- slide(x, sample_weight, tune.type = "gic", ic.scale = 2)
#' all((res[[1]] != 0) == (train[["theta"]] != 0))
#' 
#' res <- slide(x, sample_weight, tune.type = "gic", ic.scale = 1, graph.threshold = 0.2)
#' all((res[[1]] != 0) == (train[["theta"]] != 0))
#' 
#' #' res <- slide(x, sample_weight, tune.type = "bic")
#' all((res[[1]] != 0) == (train[["theta"]] != 0))
#' 
slide <- function(x, weight = NULL, c.max = 8, max.support.size = NULL, tune.type = "cv", foldid = NULL, support.size = NULL, ic.scale = 1, graph.threshold = 0.0) 
{
  p <- ncol(x)
  if (is.null(max.support.size)) {
    max.support.size <- min(c(p - 2, 100))
    max.support.size <- rep(max.support.size, p)
  }
  if (is.null(foldid) && tune.type == "cv") {
    foldid <- c()
    nfolds <- 2
  } else if (tune.type == "cv") {
    nfolds <- length(unique(foldid))
  } else {
    nfolds <- 1
  }
  
  theta <- matrix(0, p, p)
  for (node in 1:p) {
    model_node <-
      abess::abess(
        x = x[, -node],
        y = x[, node],
        weight = weight,
        family = "binomial",
        tune.path = "sequence",
        support.size = 0:max.support.size[node],
        tune.type = tune.type, 
        ic.scale = ic.scale, 
        nfolds = nfolds,
        foldid = foldid,
        c.max = as.integer(min(round(max.support.size[node] / 2), c.max)),
        max.splicing.iter = 100,
        newton = "approx",
        newton.thresh = 1e-10,
        max.newton.iter = 100,
        num.threads = nfolds, 
        seed = 1
      )
    if (is.null(support.size)) {
      est_theta_node <- as.vector(extract(model_node)[["beta"]])
    } else {
      est_theta_node <- as.vector(extract(model_node, support.size = support.size[node])[["beta"]])
    }
    theta[node, -node] <- (est_theta_node / 2)
  }
  theta <- (t(theta) + theta) / 2
  
  if (graph.threshold > 0.0 && is.null(support.size)) {
    theta <- thres_bmn_est(theta, graph.threshold)
  }
  
  res_out <- list(
    omega = theta
  )
  class(res_out) <- "abessbmn"
  res_out
}


thres_bmn_est <- function(theta, thres) {
  if (thres > 0) {
    theta[abs(theta) <= thres] <- 0
  } else if (thres < 0) {
    theta_vec <- as.vector(theta)
    ## TODO: use finite mixture model to cluster
  } 
  theta
}
