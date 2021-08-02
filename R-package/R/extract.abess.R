#' @rdname extract.abess
#' @export
extract <- function(object, support.size = NULL, ...) UseMethod("extract")

#'
#' @title Extract one model from a fitted "\code{abess}" object.
#'
#' @description Extract the fixed-support-size
#' model's information such as the selected
#' predictors, coefficient estimation, and so on.
#'
#' @rdname extract.abess
#'
#' @inheritParams coef.abess
#' @param support.size An integer value specifies
#' the model size fitted at given \code{support.size}.
#' If \code{support.size = NULL}, then the model with
#' the best tuning value would be returned.
#' Default: \code{support.size = NULL}.
#'
#' @return A \code{list} object including the following components:
#' \item{beta}{A \eqn{p}-by-1 matrix of sparse matrix, stored in column format.}
#' \item{intercept}{The fitted intercept value.}
#' \item{support.size}{The \code{support.size} used in the function.}
#' \item{support.beta}{The \code{support.size}-length vector of fitted
#' coefficients on the support set.}
#' \item{support.vars}{The character vector gives
#' variables in the support set.}
#' \item{tune.value}{The tuning value of the model.}
#' \item{dev}{The deviance of the model.}
# \item{dev.explained}{The percentage deviance explained
# (relative to the null deviance).}
#'
#' @inherit abess.default seealso
#'
#' @export
#'
extract.abess <- function(object,
                          support.size = NULL,
                          ...) {
  if (is.null(support.size)) {
    s_value <- object[["best.size"]]
  } else {
    stopifnot(length(support.size) == 1)
    stopifnot(is.numeric(support.size))
    s_value <- support.size
  }

  support_size_index <- match(s_value, object[["support.size"]])
  best_coef <- coef.abess(object, s_value)

  multi_y <- object[["family"]] %in% MULTIVARIATE_RESPONSE

  if (multi_y) {
    best_coef <- best_coef[[1]]
    beta <- best_coef[-1, , drop = FALSE]
    intercept <- best_coef[1, , drop = TRUE]
  } else {
    beta <- best_coef[-1, , drop = FALSE]
    intercept <- best_coef[1, 1]
  }
  vars_name <- best_coef@Dimnames[[1]][-1]

  if (multi_y) {
    best_coef_sum <- Matrix::rowSums(best_coef, sparseResult = TRUE)
    if (any(intercept != 0)) {
      support_index <- best_coef_sum@i[-1]
      support_beta <- as.matrix(best_coef[support_index, , drop = FALSE])
      support_index <- support_index - 1
    } else {
      support_index <- best_coef_sum@i
      support_beta <- as.matrix(best_coef_sum@x[support_index, , drop = FALSE])
    }
  } else {
    if (intercept != 0.0) {
      support_index <- best_coef@i[-1]
      support_beta <- best_coef@x[-1]
    } else {
      support_index <- best_coef@i
      support_beta <- best_coef@x
    }
  }

  support_vars <- vars_name[support_index]
  support_size <- s_value
  dev <- object[["dev"]][support_size_index]
  dev_explain <- 0
  tune_value <- object[["tune.value"]][support_size_index]

  list(
    "beta" = beta,
    "intercept" = intercept,
    "support.size" = support_size,
    "support.vars" = support_vars,
    "support.beta" = support_beta,
    "dev" = dev,
    "tune.value" = tune_value
  )
}
