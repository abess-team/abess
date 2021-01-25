#' Provides estimated coefficients from a fitted "abess" object.
#'
#' This function provides estimated
#' coefficients from a fitted "\code{abess}" object.
#'
#'
#' @param object A "\code{abess}" project.
#' @param support.size An integer vector specify 
#' the coefficient fitted at given \code{support.size}. 
#' If \code{support.size = NULL}, then all coefficients would be returned. 
#' Default: \code{support.size = NULL}.
#' @param sparse A logical value, specifying whether the coefficients should be
#' presented as sparse matrix or not. Default: \code{sparse = TRUE}.
#' @param ... Other arguments.
#' 
#' @seealso \code{\link{abess}}, \code{\link{print.abess}}.
#' 
#' @export
#'
coef.abess <- function(object, 
                       support.size = NULL, 
                       sparse = TRUE, ...)
{
  coef <- object[["beta"]]
  beta0 <- t(as.matrix(object[["intercept"]]))
  rownames(beta0) <- "(intercept)"
  coef <- methods::rbind2(beta0, coef)
  
  if (!is.null(support.size)) {
    supp_size_index <- match_support_size(object, support.size)
    coef <- coef[, supp_size_index, drop = FALSE]
  }
  rownames(coef)[1] <- "(intercept)"
  
  stopifnot(is.logical(sparse))
  if (!sparse) {
    coef <- as.matrix(coef)
  }
  
  coef
}