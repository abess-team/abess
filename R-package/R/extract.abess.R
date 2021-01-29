#' @export
extract <- function(x, ...) UseMethod("extract")

#' 
#' @title Extract one model from a fitted "abess" object.
#' 
#' @description Extract the fixed-support-size 
#' model's information such as the selected 
#' predictors, coefficient estimation, and so on.
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
                          ...) 
{
  if (is.null(support.size)) {
    s_value <- abess_fit[["best.size"]]
  } else {
    stopifnot(length(support.size) != 1)
    stopifnot(is.integer(support.size))
    s_value <- support.size
  }
  
  support_size_index <- match(s_value, object[["support.size"]])
  best_coef <- coef(abess_fit, s_value)
  
  beta <- best_coef[-1, , drop = FALSE]
  intercept <- best_coef[1, 1]
  support_vars <- best_coef@Dimnames[[1]][-1][best_coef@i[-1]]
  support_beta <- best_coef@x[-1]
  support_size <- s_value
  dev <- object[["dev"]][support_size_index]
  dev_explain <- 0
  tune_value <- object[["tune.value"]][support_size_index]
  
  list("beta" = beta, 
       "intercept" = intercept, 
       "support.size" = support_size, 
       "support.vars" = support_vars, 
       "support.beta" = support_beta, 
       "dev" = dev, 
       "tune.value" = tune_value
       )
}
