#' Adaptive best subset selection for principal component analysis
#' 
#' @inheritParams abess
#' @param x 
#' @param type 
#' @param weight 
#' @param normalize 
#' @param c.max 
#' @param gs.range 
#' @param lambda 
#' @param always.include 
#' @param group.index 
#' @param max.splicing.iter 
#' @param warm.start 
#' @param ... 
#'
#' @return
#' @export
#'
#' @examples
#' 
#' library(elasticnet)
#' data(pitprops)
#' out1<-spca(pitprops,K=6,type="Gram",sparse="varnum",trace=TRUE,para=c(6, 5, 4, 3, 2, 1))
#' ## print the object out1
#' out1
#' 
abesspca <- function(x, 
                     type = c("cov", "cor", "rob.cov", "rob.cor"), 
                     support.size = NULL, 
                     weight = rep(1, nrow(x)),
                     normalize = NULL,
                     c.max = 2,
                     gs.range = NULL, 
                     lambda = 0,
                     always.include = NULL,
                     group.index = NULL, 
                     max.splicing.iter = 20,
                     warm.start = TRUE,
                     ...)
{
  if (type == "cov") {
    gram_x <- cov(x)
  } else if (type == "cor") {
    
  } else if (type == "rob.cov") {
    
  } else if (type == "rob.cor") {
    
  }
  
  
  svdobj <- svd(gram_x)
  v <- svdobj$v
  total_variance <- sum((svdobj$d)^2)
  
  result <- sapply(1:100, function(x) SPCA_sequential(Sigma, sparsity, maxchange, random))
  result <- result[, which.max(sapply(result[1,],sum))]
  
  ve <- variance_explained(X, result$evector)
  pev <- ve / total_variance
  
  rob_info <- NULL
  
  out <- list(
    "cov" = gram_x, 
    "center" = center, 
    "loading" = result$evector,
    "pev" = pev,
    "var.all" = total_variance, 
    "call" = match.call(),
    "rob.info" = rob_info
  )
  
  class(out) <- "abesspca"
}

variance_explained <- function(X, loading){
  Z <- qr(X %*% loading)
  result <- sum(diag(qr.R(Z))^2)
  return(result)
}

