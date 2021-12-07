#' @title Creat plot from a fitted "\code{abess}" object
#'
#' @description Produces a coefficient/deviance/tuning-value plot
#' for a fitted "abess" object.
#'
#' @inheritParams print.abess
#' @param type The type of terms to be plot in the y-axis.
#' One of the following: \code{"coef"} (i.e., coefficients),
#' \code{"l2norm"} (i.e., L2-norm of coefficients),
#' \code{"dev"} (i.e., deviance),
#' and \code{"tune"} (i.e., tuning value).
#' Default is \code{"coef"}.
#' @param label A logical value.
#' If \code{label = TRUE} (the default),
#' label the curves with variable sequence numbers.
#' @param ... Other graphical parameters to plot
#'
#' @return No return value, called for side effects.
#'
#' @note
#' If \code{family = "mgaussian"} or \code{family = "multinomial"},
#' a coefficient plot is produced for
#' each dimension of multivariate response.
#'
#' @inherit abess.default seealso
#'
#' @method plot abess
#'
#' @export
#'
#' @examples
#' dataset <- generate.data(100, 20, 3)
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]])
#' plot(abess_fit)
#' plot(abess_fit, type = "l2norm")
#' plot(abess_fit, type = "dev")
#' plot(abess_fit, type = "tune")
plot.abess <- function(x,
                       type = c(
                         "coef", "l2norm",
                         "dev", "dev.ratio", "tune"
                       ),
                       label = FALSE,
                       ...) {
  user_default_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(user_default_par))

  stopifnot(is.logical(label))

  type <- match.arg(type)
  if (type == "tune") {
    y_value <- x[["tune.value"]]
  } else if (type == "dev") {
    y_value <- x[["dev"]]
  } else {
    y_value <- x[["beta"]]
    if (type == "l2norm") {
      if (class(y_value) == "list") {
        for (i in 1:length(y_value)) {
          y_value[[i]] <- Matrix::rowSums(y_value[[i]]^2)
        }
        y_value <- do.call("cbind", y_value)
        y_value <- sqrt(y_value)
      } else {
        y_value <- abs(y_value)
      }
    }
  }
  df_list <- x[["support.size"]]

  default_mar <- c(5, 4, 3, 2) + 0.1

  if (type %in% c("dev", "tune")) {
    plot_loss(y_value, df_list,
      mar = default_mar,
      ic.type = ifelse(type == "dev", "dev",
        x[["tune.type"]]
      )
    )
  }
  if (type %in% c("coef", "l2norm")) {
    plot_solution(y_value, df_list,
      mar = default_mar, label
    )
  }
}

plot_loss <- function(loss, df,
                      mar = c(0, 4, 2, 4),
                      ic.type, ...) {
  graphics::plot.new() # empty plot
  graphics::plot.window(range(df), range(loss), xaxs = "i")
  oldpar <- graphics::par(mar = mar, lend = "square") # square line ends
  graphics::par(new = TRUE) # add to the plot
  ylab_display <- ifelse(ic.type == "cv",
    "cross validation",
    ic.type
  )
  if (ic.type %in% c("aic", "bic", "ebic", "gic")) {
    ylab_display <- toupper(ylab_display)
  }
  graphics::plot(df, loss,
    type = "o", pch = 16,
    col = "#3182bd",
    ylab = ylab_display,
    xlim = c(min(df), max(df)), xlab = "Support size",
    ...
  )
  graphics::grid()
  graphics::axis(2)
  # axis(4, pos=par("usr")[1], line=0.5)  # this would plot them 'inside'
  # box()                                 # outer box

  graphics::par(oldpar)
}

plot_solution_one <- function(beta, df, mar, label, start = 0, ...) {
  beta <- as.matrix(beta)
  p <- nrow(beta)
  graphics::plot.new() # empty plot
  graphics::plot.window(range(df), range(beta), xaxs = "i")

  oldpar <- graphics::par(mar = mar, lend = "square") # square line ends
  graphics::par(new = TRUE) # add to the plot

  graphics::plot(df, beta[1, , drop = TRUE],
    type = "l", col = 1,
    xlim = c(start, max(df)), xlab = "Support size",
    ylim = range(beta), ylab = "Coefficients",
    ...
  )
  for (i in 2:p) {
    graphics::lines(df, beta[i, , drop = TRUE],
      col = i, xlim = c(start, p + 1)
    )
  }

  if (label) {
    nnz <- p
    xpos <- max(df) - 0.8
    pos <- 4
    xpos <- rep(xpos, nnz)
    ypos <- beta[, ncol(beta)]
    graphics::text(xpos, ypos, 1:p, cex = 0.8, pos = pos)
  }

  graphics::grid()
  graphics::axis(2)
  graphics::box() # outer box
  graphics::par(oldpar)
}


plot_solution <- function(beta, df, mar = c(3, 4, 0, 4), label = FALSE) {
  if (is.list(beta)) {
    dim_y <- ncol(beta[[1]])
    size_df <- length(df)
    beta_plot <- lapply(1:dim_y, function(i) {
      sapply(1:size_df, function(j) {
        beta[[j]][, i]
      })
    })
    for (i in 1:length(beta_plot)) {
      plot_solution_one(beta_plot[[i]], df, mar, label)
    }
  } else {
    plot_solution_one(beta, df, mar, label)
  }
}
