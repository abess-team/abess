#' @title Creat plot from a fitted "\code{abess}" object
#'
#' @description Produces a coefficient/deviance/tuning-value plot
#' for a fitted "abess" object.
#'
#' @inheritParams print.abesspca
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
#' @inherit abesspca seealso
#'
#' @method plot abesspca
#'
#' @export
#'
#' @examples
#' abess_fit <- abesspca(USArrests, support.size = 1:4, sparse.type = "kpc")
#' plot(abess_fit)
#' plot(abess_fit, type = "coef")
#' 
plot.abesspca <- function(x,
                          type = c("pev", "coef", "tune"),
                          label = FALSE,
                          ...) {
  user_default_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(user_default_par))
  
  stopifnot(is.logical(label))
  
  type <- match.arg(type)
  if (type == "tune") {
    y_value <- x[["tune.value"]]
  } else if (type == "pev") {
    y_value <- x[["pev"]]
  } else {
    y_value <- x[["coef"]]
  }
  df_list <- x[["support.size"]]
  
  default_mar <- c(5, 4, 3, 2) + 0.1
  
  if (type == "pev") {
    plot_pca_pev(y_value,
                 df_list,
                 mar = default_mar)
  }
  if (type == "tune") {
    
  }
  if (type %in% c("coef", "l2norm")) {
    plot_solution_pca(y_value, df_list,
                      mar = default_mar, label)
  }
}

plot_pca_pev <- function(pev, df_list, mar) {
  df_max <- sapply(df_list, max)
  df_max <- c(0, df_max)
  df_max <- cumsum(df_max)
  plot_df <- lapply(1:length(df_list), function(i) {
    df_list[[i]] + df_max[i]
  })
  pev <- unlist(pev)
  plot_df <- unlist(plot_df)
  
  graphics::plot.new() # empty plot
  graphics::plot.window(range(plot_df), range(pev), xaxs = "i")
  oldpar <- graphics::par(mar = mar, lend = "square") # square line ends
  graphics::par(new = TRUE) # add to the plot
  graphics::plot(plot_df, pev,
                 type = "o", pch = 16,
                 col = "#3182bd",
                 ylab = "Percent of explained variance",
                 xlim = c(1, max(plot_df)), xlab = "Cumulative support size", 
                 ylim = c(0, 1)
  )
  graphics::grid()
  graphics::axis(2)
  # axis(4, pos=par("usr")[1], line=0.5)  # this would plot them 'inside'
  # box()                                 # outer box
  
  graphics::par(oldpar)
}

plot_solution_pca <- function(beta, df, mar = c(3, 4, 0, 4), label = FALSE) {
  if (is.list(beta)) {
    dim_y <- ncol(beta[[1]])
    kpc_num <- length(df)
    for (i in 1:kpc_num) {
      plot_solution_one(beta[[i]], df[[i]], mar, label, start = 1)
    }
  } else {
    plot_solution_one(beta, df, mar, label, start = 1)
  }
}
