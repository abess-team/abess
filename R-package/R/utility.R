.onAttach <- 
  function(libname, pkgname) {
    packageStartupMessage("\n Thank you for using abess! To acknowledge our work, please cite the package:")
    packageStartupMessage("\n Zhu J, Wang X, Hu L, Huang J, Jiang K, Zhang Y, Lin S, Zhu J (2022). 'abess: A Fast Best Subset Selection Library in Python and R.' Journal of Machine Learning Research, 23(202), 1-7. https://www.jmlr.org/papers/v23/21-1060.html.")
  }

match_support_size <- function(object, support.size) {
  supp_size_index <- match(support.size, object[["support.size"]])
  if (anyNA(supp_size_index)) {
    stop("Arugments support.size comprises support sizes that are not in the abess object.")
  }
  supp_size_index
}

check_integer <- function(x, message) {
  if (any(x %% 1 != 0)) {
    stop(message)
  }
}

check_integer_warning <- function(x, message) {
  if (any(x %% 1 != 0)) {
    warning(message)
  }
}

check_integer_warning_variable <- function(x, var_name) {
  if (any(x %% 1 != 0)) {
    message <-
      sprintf("%s should be an integer. It is coerced to as.integer(%s).",
              var_name,
              var_name)
    warning(message)
  }
}

abess_model_matrix <- function(object,
                               data = environment(object),
                               contrasts.arg = NULL,
                               xlev = NULL,
                               ...) {
  ############################################################
  # The wrapped code refers to model.matrix.default function
  t <- if (missing(data)) {
    stats::terms(object)
  } else {
    stats::terms(object, data = data)
  }
  if (is.null(attr(data, "terms"))) {
    data <- stats::model.frame(object, data, xlev = xlev)
  } else {
    deparse2 <- function(x) {
      paste(deparse(x, width.cutoff = 500L), collapse = " ")
    }
    reorder <- match(vapply(attr(t, "variables"), deparse2, "")[-1L],
                     names(data))
    if (anyNA(reorder)) {
      stop("model frame and formula mismatch in model.matrix()")
    }
    if (!identical(reorder, seq_len(ncol(data)))) {
      data <- data[, reorder, drop = FALSE]
    }
  }
  ############################################################
  y_name <- strsplit(deparse(t), split = " ~ ")[[1]][1]
  if (length(data)) {
    namD <- names(data)
    namD <- setdiff(namD, y_name)
    for (i in namD) {
      if (is.character(data[[i]])) {
        stop(
          "Some columns in data are character! You may convert these columns to a dummy variable via model.matrix function or discard them."
        )
      } else if (is.factor(data[[i]])) {
        stop(
          "Some columns in data are factor!. You may convert these columns to a dummy variable via model.matrix function or discard them."
        )
      }
    }
  }
  data
}

map_tunetype2numeric <- function(tune.type) {
  ic_type <- switch(
    tune.type,
    "aic" = 1,
    "bic" = 2,
    "gic" = 3,
    "ebic" = 4,
    "cv" = 0
  )
  ic_type
}

check_foldid <- function(foldid, nobs) {
  stopifnot(is.vector(foldid))
  stopifnot(is.numeric(foldid))
  stopifnot(length(foldid) == nobs)
  check_integer_warning(foldid,
                        "nfolds should be an integer value. It is coerced to be as.integer(foldid). ")
  foldid <- as.integer(foldid)
  cv_fold_id <- foldid
  cv_fold_id
}

check_nfold <- function(nfolds) {
  stopifnot(is.numeric(nfolds) & nfolds >= 2)
  check_integer_warning(nfolds,
                        "nfolds should be an integer value. It is coerced to be as.integer(nfolds). ")
  nfolds <- as.integer(nfolds)
  nfolds
}

map_dgCMatrix2entry <- function(x) {
  x <- summary(x)
  x[, 1:2] <- x[, 1:2] - 1
  x <- as.matrix(x)
  x <- x[, c(3, 1, 2)]
  x
}

MULTIVARIATE_RESPONSE <- c("mgaussian", "multinomial", "ordinal")

.onUnload <- function(libpath) {
  library.dynam.unload("abess", libpath)
}
