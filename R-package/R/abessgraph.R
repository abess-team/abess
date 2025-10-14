#' 
#' Sparsity Learning for Ising moDel rEconstruction (SLIDE)
#' 
#' @inheritParams abess.default
#'
#' @param max.support.size The maximum node degree in the estimated graph. If prior information is available, we recommend setting this value accordingly. Otherwise, it is internally set to \eqn{n / (\log p \log \log n)} by default. 
#' @param graph.threshold A numeric value specifying the post-thresholding level for graph estimation. If prior knowledge about the minimum signal strength is available, this can be set to approximately half of that value. The default is \code{0.0}, which means no thresholding is applied.
#'
#' @return a sparse interaction matrix estimation
#' @export
#' @references Reconstruct Ising Model with Global Optimality via SLIDE. Xuanyu Chen, Jin Zhu, Junxian Zhu, Xueqin Wang, Heping Zhang (2025). Journal of the American Statistical Association (Accepted)
#'
#' @examples
#' p <- 16
#' n <- 1e3
#' library(abess)
#' train <- generate.bmn.data(n, p, type = 3, graph.seed = 1, seed = 1, beta = 0.4)
#' res <- slide(train[["data"]], train[["weight"]], tune.type = "gic", 
#'              max.support.size = rep(4, p), support.size = rep(4, p))
#' all((res[[1]] != 0) == (train[["theta"]] != 0))
#' 
slide <- function(x, weight = NULL, c.max = 8, max.support.size = NULL, tune.type = "cv", foldid = NULL, support.size = NULL, ic.scale = 1, graph.threshold = 0.0, newton = 'approx') 
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
    init.active.set <- NULL
    selected_edge <- which(abs(theta[-node, node]) > 0)
    if (length(selected_edge) > 0) {
      init.active.set <- selected_edge
    }
    model_node <-
      abess::abess(
        x = x[, -node],
        y = x[, node],   ## it would be convert to {0, 1} vector in the `abess` function
        weight = weight,
        family = "binomial",
        tune.path = "sequence",
        support.size = 0:max.support.size[node],
        tune.type = tune.type, 
        normalize = 0,   ## to facilitate the subsequently graphical thresholding 
        init.active.set = init.active.set,
        ic.scale = ic.scale, 
        nfolds = nfolds,
        foldid = foldid,
        c.max = as.integer(min(round(max.support.size[node] / 2), c.max)),
        max.splicing.iter = 100,
        newton = newton,
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
    theta[node, -node] <- est_theta_node
  }
  theta <- (t(theta) + theta) / 2
  theta <- theta / 2    # convert the coefficient from {0, 1} to {-1, 1}
  
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
