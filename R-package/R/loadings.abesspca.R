#' @rdname loadings.abesspca
#' @export
loadings <- function(object, support.size = NULL, sparse = TRUE, ...) UseMethod("loadings")

#' Extract Sparse Loadings from a fitted "\code{abesspca}" object.
#'
#' This function provides estimated
#' coefficients from a fitted "\code{abesspca}" object.
#' @rdname loadings.abesspca
#'
#' @param object An "\code{abesspca}" project.
#' @param support.size An integer vector specifies 
#' the coefficient fitted at given \code{support.size}. 
#' If \code{support.size = NULL}, then all coefficients would be returned. 
#' Default: \code{support.size = NULL}.
#' 
#' @param sparse A logical value, specifying whether the coefficients should be
#' presented as sparse matrix or not. Default: \code{sparse = TRUE}.
#' @param ... Other arguments.
#' 
#' @return A matrix with \code{length(support.size)} columns. 
#' Each column corresponds to a sparse loading for the first principal component, 
#' where the number of non-zeros entries depends on the \code{support.size}. 
#' 
#' @inherit abesspca seealso
#' 
#' @method loadings abesspca
#' 
#' @export
#'
loadings.abesspca <- function(object, 
                              support.size = NULL, 
                              sparse = TRUE, ...)
{
  supp_size_index <- NULL
  if (!is.null(support.size)) {
    supp_size_index <- match_support_size(object, support.size)
  } else {
    supp_size_index <- match_support_size(object, object[["support.size"]])
  }
  
  stopifnot(is.logical(sparse))
  coef <- object[["loadings"]]
  if (!is.null(supp_size_index)) {
    coef <- coef[, supp_size_index, drop = FALSE]
  }
  
  if (!sparse) {
    coef <- as.matrix(coef)
  }
  
  coef
}

