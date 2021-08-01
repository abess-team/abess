#' Print method for a fitted "\code{abesspca}" object
#'
#' Prints the fitted model and returns it invisibly.
#'
#' @rdname print.abesspca
#'
#' @param x A "\code{abesspca}" object.
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
#' @inherit abesspca seealso
#'
#' @method print abesspca
#'
#' @export
#'
print.abesspca <- function(x,
                           digits = max(5, getOption("digits") - 5),
                           ...) {
  cat("Call:\n", paste(deparse(x[["call"]]), sep = "\n", collapse = "\n"),
    "\n\n",
    sep = ""
  )
  out <- data.frame(
    "support.size" = x[["support.size"]],
    "ev" = x[["ev"]],
    "pev" = x[["pev"]], row.names = NULL
  )
  print(out)
}
