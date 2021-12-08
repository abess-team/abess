#' @title Creat plot from a fitted "\code{abessrpca}" object
#'
#' @description Produces a sparse-matrix/loss/tuning-value plot
#' for a fitted "\code{abessrpca}" object.
#'
#' @inheritParams coef.abessrpca
#' @inheritParams print.abessrpca
#'
#' @param type The plot type.
#' One of the following:
#' \code{"S"} (i.e., a heatmap for the sparse matrix estimation),
#' \code{"loss"} (i.e., a support.size versus loss plot),
#' and \code{"tune"} (i.e., , a support.size versus tuning value plot).
#' Default is \code{"coef"}.
#' @param label A logical value.
#' If \code{label = TRUE} (the default),
#' label the curves with variable sequence numbers.
#' @param ... Other graphical parameters to plot
#'
#' @return No return value, called for side effects.
#'
#' @inherit abessrpca seealso
#'
#' @method plot abessrpca
#'
#' @export
#'
plot.abessrpca <- function(x,
                           type = c("S", "loss", "tune"),
                           support.size = NULL,
                           label = TRUE,
                           ...) {
  user_default_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(user_default_par))

  type <- match.arg(type)
  if (type == "S") {
    stats::heatmap(as.matrix(x[["S"]]), Rowv = NA, Colv = NA, scale = "none", col = grDevices::cm.colors(256), frame.plot = TRUE, margins = c(2.4, 2.4), ...)
  } else {
    if (type == "loss") {
      y_value <- x[["loss"]]
    } else {
      y_value <- x[["tune.value"]]
    }
    x_value <- x[["support.size"]]
    ic_type <- type
    if (ic_type == "tune") {
      ic_type <- x[["tune.type"]]
    }
    default_mar <- c(5, 4, 3, 2) + 0.1
    y_value <- as.vector(y_value)
    plot_loss(y_value, x_value, mar = default_mar, ic.type = ic_type)
  }
}
