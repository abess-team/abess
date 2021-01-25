#' print method for a "bess" object
#'
#' Print the primary elements of the "\code{abess}" object.
#'
#' prints the fitted model and returns it invisibly.
#'
#' @param x A "\code{abess}" object.
#' @param digits Minimum number of significant digits to be used.
#' @param ... additional print arguments
#' 
#' @return print a \code{data.frame} with three columns:
#' \item 1. support size 
#' \item 2. deviance of model
# \item 2. the percentage of deviance explained, which is defined as 
# \eqn{1 - dev / nulldev}, where \eqn{nulldev} is deviance of 
# the intercept model.
#' \item 3. the tuning value
#' 
#' @seealso \code{\link{abess}}, \code{\link{coef.abess}}, 
#' 
#' @export
#'
print.abess <- function(x,
                        digits = max(5, getOption("digits") - 5),
                        ...)
{
  cat("Call:\n", paste(deparse(x[["call"]]), sep = "\n", collapse = "\n"),
      "\n\n", sep = "")
  out <- data.frame("support.size" = x[["support.size"]], 
                    "dev" = x[["dev"]], 
                    "tune.value" = x[["tune.value"]], row.names = NULL)
  colnames(out)[3] <- x[["tune.type"]]
  print(out)
}