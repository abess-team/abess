#' Extract Model Coefficients from a fitted "\code{abess}" object.
#'
#' This function provides estimated
#' coefficients from a fitted "\code{abess}" object.
#' @rdname coef.abess
#'
#' @param object An "\code{abess}" project.
#' @param support.size An integer vector specifies
#' the coefficient fitted at given \code{support.size}.
#' If \code{support.size = NULL}, then all coefficients would be returned.
#' Default: \code{support.size = NULL}.
#'
#' @param sparse A logical value, specifying whether the coefficients should be
#' presented as sparse matrix or not. Default: \code{sparse = TRUE}.
#' @param ... Other arguments.
#'
#' @return A coefficient matrix when fitting an univariate model including gaussian, binomial, poisson, and cox;
#' otherwise, a list containing coefficient matrices.
#' For a coefficient matrix, each row is a variable, and each column is a support size.
#'
#' @inherit abess.default seealso
#'
#' @method coef abess
#'
#' @export
#'
coef.abess <- function(object,
                       support.size = NULL,
                       sparse = TRUE, ...) {
  supp_size_index <- NULL
  if (!is.null(support.size)) {
    supp_size_index <- match_support_size(object, support.size)
  } else {
    supp_size_index <- match_support_size(object, object[["support.size"]])
  }
  stopifnot(is.logical(sparse))
  multi_y <- object[["family"]] %in% MULTIVARIATE_RESPONSE

  if (multi_y) {
    coef <- list()
    for (i in 1:length(supp_size_index)) {
      coef[[i]] <- combine_beta_intercept(
        object[["beta"]][[supp_size_index[i]]],
        object[["intercept"]][[supp_size_index[i]]]
      )
    }
  } else {
    coef <- combine_beta_intercept(object[["beta"]], object[["intercept"]])
    if (!is.null(supp_size_index)) {
      coef <- coef[, supp_size_index, drop = FALSE]
    }
  }

  if (!sparse) {
    if (multi_y) {
      coef <- lapply(coef, as.matrix)
    } else {
      coef <- as.matrix(coef)
    }
  }

  coef
}


combine_beta_intercept <- function(beta_mat, intercept_vec) {
  coef <- beta_mat
  beta0 <- t(as.matrix(intercept_vec))
  rownames(beta0) <- "(intercept)"
  coef <- methods::rbind2(beta0, coef)
  coef
}
