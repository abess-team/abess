#' Title
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
#' @details A coefficient profile plot is produced. 
#' Especially, 
#' if \code{family = "mgaussian"} or \code{family = "multinomial"}, 
#' a coefficient plot is produced for each class.
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
plot.abess <- function (x, 
                        type = c("coef", "l2norm", 
                                 "dev", "dev.ratio", "tune"), 
                        label = FALSE, 
                        ...) 
{
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
        y_value_tmp <- y_value[[1]]^2
        for (i in 2:length(y_value)) {
          y_value_tmp <- y_value_tmp + y_value[[i]]^2
        }
        y_value <- sqrt(y_value_tmp)
      } else {
        y_value <- abs(y_value)
      }
    }
  }
  df_list <- abess_fit[["support.size"]]
  
  if (type %in% c("dev", "tune")) {
    plot_loss(y_value, df_list, 
              mar = c(3, 4, 3, 4), 
              ic.type = ifelse(type == "dev", "dev", 
                               x[["tune.type"]]))
  }
  if (type %in% c("coef", "l2norm")) {
    plot_solution(y_value, df_list, 
                  mar = c(3, 4, 3, 4), label)
  }
}

plot_loss <- function(loss, df, 
                      mar = c(0, 4, 2, 4), 
                      ic.type) {
  
  graphics::plot.new()                            # empty plot
  graphics::plot.window(range(df), range(loss), xaxs="i")
  oldpar <- graphics::par(mar = mar, 
                          lend="square")          # square line ends
  graphics::par(new = TRUE)                         # add to the plot
  graphics::plot(df, loss, type = "b", 
                 ylab = ifelse(ic.type == "cv", 
                               "cross validation deviance", 
                               ic.type),
                 xlim = c(0, max(df)))
  graphics::title(xlab = "Support size", line = 2)
  graphics::grid()
  graphics::axis(2)
  #axis(4, pos=par("usr")[1], line=0.5)  # this would plot them 'inside'
  # box()                                 # outer box
  
  graphics::par(oldpar)
}


plot_solution <- function (beta, df, 
                           mar = c(3, 4, 0, 4), 
                           label = FALSE) {
  beta <- as.matrix(beta)
  p <- nrow(beta)
  graphics::plot.new()                            # empty plot
  
  graphics::plot.window(range(df), range(beta), xaxs="i")
  
  oldpar <- graphics::par(mar = mar, lend = "square") # square line ends
  graphics::par(new = TRUE)                           # add to the plot
  
  graphics::plot(df, beta[1, , drop = TRUE], 
                 type = "l", col = 1, 
                 xlim = c(0, max(df)), xlab = "",
                 ylim = range(beta), ylab = "Coefficients")
  graphics::title(xlab = 'Support size', line = 2)
  for (i in 2:p) {
    graphics::lines(df, beta[i, , drop = TRUE], 
                    col = i, xlim = c(0, p + 1))
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
  graphics::box()                             # outer box
  
  graphics::par(oldpar)
}
