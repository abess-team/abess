#' Extract the deviance from a fitted "abess" object.
#'
#' Similar to other deviance methods, 
#' which returns deviance from a fitted "\code{abess}" object.
#'
#'
#' @param object A "\code{abess}" object.
#' @param type The type of deviance. 
#' One of the following: \code{"standard"}, 
#' \code{"gic"}, \code{"ebic"}, \code{"bic"} and \code{"aic"}. 
#' Default is \code{"standard"}.
#' @param ... additional arguments
#' 
#' @return A numeric vector.
#' 
#' @inherit abess.default seealso
#' 
#' @export
deviance.abess <- function(object, 
                           type = c("standard", "gic", "ebic", "bic", "aic"), 
                           ...)
{
  num <- object[["nobs"]]
  nvars <- object[["nvars"]]
  
  type <- match.arg(type)
  
  if (type == "standard") {
    dev <- object[["dev"]]
  } else {
    if (type == object[["tune.type"]]) {
      dev <- object[["tune.value"]]
    } else {
      dev <- object[["dev"]]
      supp_size <- object[["support.size"]]
      if (type == "aic") {
        adjust_term <- supp_size
      } else if (type == "bic") {
        adjust_term <- supp_size
      } else if (type == "ebic") {
        adjust_term <- supp_size
      } else if (type == "gic") {
        adjust_term <- supp_size
      }
      dev <- dev + adjust_term
    }
  }
  
  dev
}