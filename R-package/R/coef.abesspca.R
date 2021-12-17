#' Extract Sparse Loadings from a fitted "\code{abesspca}" object.
#'
#' This function provides estimated
#' coefficients from a fitted "\code{abesspca}" object.
#' @rdname coef.abesspca
#'
#' @param object An "\code{abesspca}" project.
#' @param support.size An integer vector specifies
#' the coefficient fitted at given \code{support.size}.
#' If \code{support.size = NULL}, then all coefficients would be returned.
#' Default: \code{support.size = NULL}.
#' This parameter is omitted if {sparse.type = "kpc"}.
#' @param kpc An integer vector specifies
#' the coefficient fitted at given principal component.
#' If \code{kpc = NULL}, then all coefficients would be returned.
#' Default: \code{kpc = NULL}.
#' This parameter is omitted if {sparse.type = "fpc"}.
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
#' @method coef abesspca
#'
#' @export
#'
coef.abesspca <- function(object,
                          support.size = NULL,
                          kpc = NULL,
                          sparse = TRUE, ...) {
  supp_size_index <- NULL
  if (object[["sparse.type"]] == "fpc") {
    if (!is.null(support.size)) {
      supp_size_index <- match_support_size(object, support.size)
    } else {
      supp_size_index <- match_support_size(object, object[["support.size"]])
    }
  } else {
    if (is.null(kpc)) {
      supp_size_index <- 1:length(object[["support.size"]])
    } else {
      supp_size_index <- kpc
    }
  }

  stopifnot(is.logical(sparse))
  coef <- object[["coef"]]
  if (object[["sparse.type"]] == "fpc") {
    if (!is.null(supp_size_index)) {
      coef <- coef[, supp_size_index, drop = FALSE]
    }
    if (!sparse) {
      coef <- as.matrix(coef)
    }
  } else {
    coef <- coef[supp_size_index]
    if (!sparse) {
      coef <- lapply(coef, as.matrix)
    }
  }

  coef
}
