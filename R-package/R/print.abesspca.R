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
  support_size <- unlist(x[["support.size"]])
  pc_list <- list()
  for (i in 1:x[["kpc.num"]]) {
    pc_list[[i]] <- rep(i, length(x[["support.size"]][[i]]))
  }
  out <- data.frame(
    "PC" = unlist(pc_list),
    "support.size" = unlist(x[["support.size"]]),
    "ev" = unlist(x[["ev"]]),
    "pev" = unlist(x[["pev"]]),
    row.names = NULL
  )
  print(out)
}
