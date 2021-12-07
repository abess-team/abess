#' Extract sparse component from a fitted "\code{abessrpca}" object.
#'
#' This function provides estimated
#' coefficients from a fitted "\code{abessrpca}" object.
#' @rdname coef.abessrpca
#'
#' @inheritParams coef.abess
#' @param object An "\code{abessrpca}" project.
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
    supp_size_index <-
      match_support_size(object, object[["support.size"]])
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
