#' Print method for a fitted "\code{abess}" object
#' 
#' Prints the fitted model and returns it invisibly.
#' 
#' @rdname print.abess
#'
#' @param x A "\code{abess}" object.
#' @param digits Minimum number of significant digits to be used.
#' @param ... additional print arguments
#' 
#' @details Print a \code{data.frame} with three columns:
#' the first column is support size of model; 
#' the second column is deviance of model;
#' the last column is the tuning value of the certain tuning type.
#' 
#' @return No return value, called for side effects
#' 
#' @inherit abess.default seealso
#' 
#' @method print abess
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