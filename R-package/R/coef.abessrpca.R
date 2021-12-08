#' Extract sparse component from a fitted "\code{abessrpca}" object.
#'
#' This function provides estimated
#' coefficients from a fitted "\code{abessrpca}" object.
#' @rdname coef.abessrpca
#'
#' @inheritParams coef.abess
#' @param object An "\code{abessrpca}" project.
#' @param support.size An integer vector specifies
#' the sparse matrix fitted at given \code{support.size} to be returned.
#' If \code{support.size = NULL}, then the sparse matrix with 
#' the least tuning value would be returned.
#' Default: \code{support.size = NULL}.
#'
#' @return A list with \code{length(support.size)} number of dgCMatrix,
#' each of which is the estimation the sparse component.
#'
#' @inherit abessrpca seealso
#'
#' @method coef abessrpca
#'
#' @export
#'
coef.abessrpca <- function(object,
                           support.size = NULL,
                           sparse = TRUE,
                           ...) {
  supp_size_index <- NULL
  if (!is.null(support.size)) {
    supp_size_index <- match_support_size(object, support.size)
  } else {
    min_ic_index <- which.min(object[["tune.value"]])
    supp_size_index <- match_support_size(object, 
                                          object[["support.size"]][min_ic_index])
  }

  stopifnot(is.logical(sparse))
  coef <- object[["S"]]
  if (!is.null(supp_size_index)) {
    coef <- coef[[supp_size_index]]
  }
  if (!sparse) {
    coef <- lapply(coef, as.matrix)
  }

  coef
}
