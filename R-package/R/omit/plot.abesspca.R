#' @title Creat plot from a fitted "\code{abesspca}" object
#'
#' @description Produces a coefficient/deviance/tuning-value plot
#' for a fitted "\code{abesspca}" object.
#'
#' @inheritParams print.abesspca
#' @param type The type of terms to be plot in the y-axis.
#' One of \code{"coef"} and \code{"variance"}.
#' Default is \code{"coef"}.
#' @param label A logical value.
#' If \code{label = TRUE} (the default),
#' label the curves with variable sequence numbers.
#' @param ... Other graphical parameters to plot
#'
#' @return No return value, called for side effects.
#'
# @note
# If \code{family = "mgaussian"} or \code{family = "multinomial"},
# a coefficient plot is produced for
# each dimension of multivariate response.
#'
#' @inherit abesspca seealso
#'
#' @method plot abesspca
#'
#' @export
#'
#' @examples
#' library(abess)
#' head(USArrests)
#' pca_fit <- abesspca(USArrests)
#' plot(pca_fit)
#' plot(pca_fit, "variance")
plot.abesspca <- function(x,
                          type = c("coef", "variance"),
                          label = FALSE,
                          ...) {
  user_default_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(user_default_par))

  stopifnot(is.logical(label))

  if (x[["sparse.type"]] == "fpc") {
    type <- match.arg(type)
  } else {
    if (length(type) == 2) {
      type <- "variance"
    }
  }
  df_list <- x[["support.size"]]
  df_list <- c(0, df_list)
  if (type == "variance") {
    y_value <- c(0, x[["ev"]])
  } else if (type == "coef") {
    y_value <- x[["coef"]]
    y_value <- cbind(0, y_value)
  } else {
  }

  if (type %in% c("variance")) {
    plot_loss(y_value, df_list,
      mar = c(3, 4, 3, 4),
      ic.type = "variance"
    )
  }
  if (type %in% c("coef")) {
    if (x[["sparse.type"]] == "coef") {
      stop("Best subset selection for K (>=2) principal component analysis does not supports plotting loadings matrix solution path.")
    }
    plot_solution(y_value, df_list,
      mar = c(3, 4, 3, 4), label
    )
  }
}

plot_loss <- function(loss, df,
                      mar = c(0, 4, 2, 4),
                      ic.type) {
  graphics::plot.new() # empty plot
  graphics::plot.window(range(df), range(loss), xaxs = "i")
  oldpar <- graphics::par(mar = mar, lend = "square") # square line ends
  graphics::par(new = TRUE) # add to the plot
  graphics::plot(df, loss,
    type = "b",
    ylab = ifelse(ic.type == "cv",
      "cross validation deviance",
      ic.type
    ),
    xlim = c(0, max(df))
  )
  graphics::title(xlab = "Support size", line = 2)
  graphics::grid()
  graphics::axis(2)
  # axis(4, pos=par("usr")[1], line=0.5)  # this would plot them 'inside'
  # box()                                 # outer box

  graphics::par(oldpar)
}

# plot_solution_one <- function(beta, df, mar, label) {
#   beta <- as.matrix(beta)
#   p <- nrow(beta)
#   graphics::plot.new()                            # empty plot
#   graphics::plot.window(range(df), range(beta), xaxs="i")
#
#   oldpar <- graphics::par(mar = mar, lend = "square") # square line ends
#   graphics::par(new = TRUE)                           # add to the plot
#
#   graphics::plot(df, beta[1, , drop = TRUE],
#                  type = "l", col = 1,
#                  xlim = c(0, max(df)), xlab = "",
#                  ylim = range(beta), ylab = "Coefficients")
#   graphics::title(xlab = 'Support size', line = 2)
#   for (i in 2:p) {
#     graphics::lines(df, beta[i, , drop = TRUE],
#                     col = i, xlim = c(0, p + 1))
#   }
#
#   if (label) {
#     nnz <- p
#     xpos <- max(df) - 0.8
#     pos <- 4
#     xpos <- rep(xpos, nnz)
#     ypos <- beta[, ncol(beta)]
#     graphics::text(xpos, ypos, 1:p, cex = 0.8, pos = pos)
#   }
#
#   graphics::grid()
#   graphics::axis(2)
#   graphics::box()                             # outer box
#   graphics::par(oldpar)
# }
#
#
# plot_solution <- function(beta, df, mar = c(3, 4, 0, 4), label = FALSE) {
#   if (is.list(beta)) {
#     dim_y <- ncol(beta[[1]])
#     size_df <- length(df)
#     beta_plot <- lapply(1:dim_y, function(i) {
#       sapply(1:size_df, function(j) {
#         beta[[j]][, i]
#       })
#     })
#     for (i in 1:length(beta_plot)) {
#       plot_solution_one(beta_plot[[i]], df, mar, label)
#     }
#   } else {
#     plot_solution_one(beta, df, mar, label)
#   }
# }
