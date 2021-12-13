#' @title Make predictions from a fitted "\code{abess}" object.
#'
#' @inheritParams extract.abess
#' @param newx New data used for prediction. If omitted, the fitted linear predictors are used.
#' @param type \code{type = "link"} gives the linear predictors for \code{"binomial"},
#' \code{"poisson"} or \code{"cox"} models; for \code{"gaussian"} models it gives the
#' fitted values. \code{type = "response"} gives the fitted probabilities for
#' \code{"binomial"}, fitted mean for \code{"poisson"} and the fitted relative-risk for
#' \code{"cox"}; for \code{"gaussian"}, \code{type = "response"} is equivalent to \code{type = "link"}.
#' @param ... Additional arguments affecting the predictions produced.
#'
#' @return The object returned depends on the types of family.
#'
#' @inherit abess.default seealso
#'
#' @method predict abess
#'
#' @export
#'
predict.abess <- function(object, newx,
                          type = c("link", "response"),
                          support.size = NULL,
                          ...) {
  if (missing(newx)) {
    stop("You need to supply a value for newx")
  }
  newx <- as.matrix(newx)
  multi_y <- object[["family"]] %in% MULTIVARIATE_RESPONSE

  if (!is.null(colnames(newx))) {
    if (multi_y) {
      vn <- rownames(object[["beta"]][[1]])
    } else {
      vn <- rownames(object[["beta"]])
    }
    if (any(is.na(match(vn, colnames(newx))))) {
      stop("names of newx don't match training data!")
    }
    newx <- newx[, vn]
  }
  type <- match.arg(type)

  if (is.null(support.size)) {
    supp_size_index <- which.min(object[["tune.value"]])
  } else {
    supp_size_index <- match_support_size(object, support.size)
  }

  if (multi_y) {
    y <- list()
    i <- 1
    for (index in supp_size_index) {
      y_tmp <- newx %*% object[["beta"]][[index]]
      y[[i]] <- sweep(as.matrix(y_tmp), 2,
        FUN = "+",
        STATS = object[["intercept"]][[index]]
      )
      i <- i + 1
    }
    names(y) <- support.size
  } else {
    y <- newx %*% object[["beta"]][, supp_size_index, drop = FALSE]
    y <- sweep(as.matrix(y), 2,
      FUN = "+",
      STATS = object[["intercept"]][supp_size_index]
    )
  }

  if (object[["family"]] %in% c("gaussian", "mgaussian")) {
  } else if (object[["family"]] == "binomial") {
    if (type == "link") {
    } else if (type == "response") {
      bi <- stats::binomial()
      y <- bi[["linkinv"]](y)
    } else if (type == "class") {

    }
  } else if (object[["family"]] == "poisson") {
    if (type == "link") {
    } else {
      poi <- stats::poisson()
      y <- poi[["linkinv"]](y)
    }
  } else if (object[["family"]] == "cox") {
    if (type == "link") {
    } else {
      y <- exp(y)
    }
  } else if (object[["family"]] == "multinomial") {
    if (type == "link") {
    } else if (type == "response") {
      for (i in 1:length(supp_size_index)) {
        y_exp <- exp(cbind(y[[i]], 0))
        total_exp <- rowSums(y_exp)
        y[[i]] <- sweep(y_exp, MARGIN = 1, total_exp, FUN = "/")
      }
    }
  }

  y
}
