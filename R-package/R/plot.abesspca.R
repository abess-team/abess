#' @title Creat plot from a fitted "\code{abess}" object
#'
#' @description Produces a coefficient/deviance/tuning-value plot
#' for a fitted "abess" object.
#'
#' @inheritParams print.abess
#' @param type The type of terms to be plot in the y-axis.
#' One of the following:
#' \code{"pev"} (i.e., percent of explained variance),
#' \code{"coef"} (i.e., coefficients),
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
#' plot(abess_fit, type = "tune")
plot.abesspca <- function(x,
                          type = c("pev", "coef", "tune"),
                          label = FALSE,
                          ...) {

  total.variance <- ifelse(x[["sparse.type"]] == "kpc", TRUE, FALSE) # very slight different (not informative)

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
  sparese_type <- x[["sparse.type"]]

  default_mar <- c(5, 4, 3, 2) + 0.1

  if (type == "pev") {
    if (!total.variance && sparese_type == "kpc") {
      for (j in 1:x[["kpc.num"]]) {
        plot_pca_pev(
          x[["pev.pc"]][[j]],
          df_list[[j]],
          default_mar,
          sparese_type,
          total.variance,
          j,
          ...
        )
      }
    } else {
      plot_pca_pev(
        y_value,
        df_list,
        default_mar,
        sparese_type,
        total.variance,
        1,
        ...
      )
    }
  }
  if (type == "tune") {
    plot_tune_pca(
      y_value,
      df_list,
      x[["tune.type"]],
      default_mar
    )
  }
  if (type %in% c("coef")) {
    plot_solution_pca(y_value, df_list,
      mar = default_mar, label
    )
  }
}

plot_pca_pev <- function(pev, df_list, mar, type, total.variance, i, ...) {
  if (type == "kpc" && total.variance) {
    df_max <- sapply(df_list, max)
    df_max <- c(0, df_max)
    df_max <- cumsum(df_max)
    plot_df <- lapply(1:length(df_list), function(i) {
      df_list[[i]] + df_max[i]
    })
    plot_df_max <- sapply(plot_df, max)
    plot_title <- "Sequential PCA"
    x_lab <- "Cumulative support size"
  } else {
    plot_df <- df_list
    plot_title <- sprintf("PC %s", i)
    x_lab <- "Support size"
  }
  pev <- unlist(pev)
  plot_df <- unlist(plot_df)

  graphics::plot.new() # empty plot
  graphics::plot.window(range(plot_df), range(pev), xaxs = "i")
  oldpar <- graphics::par(mar = mar, lend = "square") # square line ends
  graphics::par(new = TRUE) # add to the plot
  graphics::plot(plot_df, pev,
    type = "o", pch = 16,
    col = "#3182bd",
    xlab = x_lab,
    ylab = "Percent of explained variance",
    xlim = c(min(plot_df), max(plot_df)),
    main = plot_title,
    ...
  )
  if (type == "kpc" && total.variance) {
    for (v_df in plot_df_max) {
      graphics::abline(v = v_df, col = "#d95f0e")
    }
  }
  graphics::grid()
  graphics::axis(2)
  # axis(4, pos=par("usr")[1], line=0.5)  # this would plot them 'inside'
  # box()                                 # outer box

  graphics::par(oldpar)
}

plot_tune_pca <- function(tune_value, df, ic.type, mar = c(3, 4, 0, 4), ...) {
  if (is.list(tune_value)) {
    dim_y <- ncol(tune_value[[1]])
    kpc_num <- length(df)
    for (i in 1:kpc_num) {
      title_name <- sprintf("PC %s", i)
      plot_loss(tune_value[[i]], df[[i]], mar, ic.type,
        main = title_name,
        ...
      )
    }
  } else {
    title_name <- "PC 1"
    plot_loss(tune_value, df, mar, ic.type,
      main = title_name,
      ...
    )
  }
}

plot_solution_pca <- function(beta, df, mar = c(3, 4, 0, 4), label = FALSE, ...) {
  if (is.list(beta)) {
    dim_y <- ncol(beta[[1]])
    kpc_num <- length(df)
    for (i in 1:kpc_num) {
      title_name <- sprintf("PC %s", i)
      plot_solution_one(beta[[i]], df[[i]], mar, label,
        start = min(df[[i]]),
        main = title_name
      )
    }
  } else {
    title_name <- "PC 1"
    plot_solution_one(beta, df, mar, label,
      start = min(df),
      main = title_name
    )
  }
}
