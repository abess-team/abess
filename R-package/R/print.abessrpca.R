#' Print method for a fitted "\code{abessrpca}" object
#'
#' Prints the fitted model and returns it invisibly.
#'
#' @rdname print.abessrpca
#'
#' @param x A "\code{abessrpca}" object.
#' @param digits Minimum number of significant digits to be used.
#' @param ... additional print arguments
#'
#' @details Print a \code{data.frame} with three columns:
#' the first column is support size of model;
#' the second column is the explained variance of model;
#' the last column is the percent of explained variance of model.
#'
#' @return No return value, called for side effects
#'
#' @inherit abessrpca seealso
#'
#' @method print abessrpca
#'
#' @export
#'
print.abessrpca <- function(x,
                            digits = max(5, getOption("digits") - 5),
                            ...) {
  cat("Call:\n", paste(deparse(x[["call"]]), sep = "\n", collapse = "\n"),
    "\n\n",
    sep = ""
  )
  support_size <- unlist(x[["support.size"]])
  out <- data.frame(
    "support.size" = unlist(x[["support.size"]]),
    "loss" = unlist(x[["loss"]]),
    "tune" = unlist(x[["tune.value"]]),
    row.names = NULL
  )
  colnames(out)[3] <- x[["tune.type"]]
  print(out)
}
