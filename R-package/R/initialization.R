Initialization <- function(c.max,
                           support.size,
                           always.include,
                           group.index,
                           splicing.type,
                           max.splicing.iter,
                           warm.start,
                           ic.scale,
                           num.threads,
                           tune.type,
                           important.search)
{
  para <- list(
    c.max = c.max,
    support.size = support.size,
    always.include = always.include,
    group.index = group.index,
    splicing.type = splicing.type,
    max.splicing.iter = max.splicing.iter,
    warm.start = warm.start,
    ic.scale = ic.scale,
    num.threads = num.threads,
    tune.type = tune.type,
    important.search = important.search
  )
  
  class(para) <- append("Initialization", class(para))
  return(para)
}

Initialization_GLM <- function(c.max,
                               support.size,
                               always.include,
                               group.index,
                               splicing.type,
                               max.splicing.iter,
                               warm.start,
                               ic.scale,
                               num.threads,
                               tune.type,
                               important.search,
                               newton.thresh,
                               tune.path,
                               max.newton.iter,
                               lambda,
                               family,
                               screening.num,
                               gs.range,
                               early.stop,
                               weight,
                               cov.update,
                               normalize,
                               init.active.set,
                               newton,
                               foldid,
                               nfolds)
{
  para <- Initialization(
    c.max = c.max,
    support.size = support.size,
    always.include = always.include,
    group.index = group.index,
    splicing.type = splicing.type,
    max.splicing.iter = max.splicing.iter,
    warm.start = warm.start,
    ic.scale = ic.scale,
    num.threads = num.threads,
    tune.type = tune.type,
    important.search = important.search
  )
  para$newton.thresh <- newton.thresh
  para$tune.path <- tune.path
  para$max.newton.iter <- max.newton.iter
  para$lambda <- lambda
  para$family <- family
  para$screening.num <- screening.num
  para$gs.range <- gs.range
  para$early.stop <- early.stop
  para$weight <- weight
  para$cov.update <- cov.update
  para$normalize <- normalize
  para$init.active.set <- init.active.set
  para$newton <- newton
  para$foldid <- foldid
  para$nfolds <- nfolds
  
  class(para) <- append("glm", class(para))
  return(para)
}

Initialization_PCA <- function(c.max,
                               support.size,
                               always.include,
                               group.index,
                               splicing.type,
                               max.splicing.iter,
                               warm.start,
                               ic.scale,
                               num.threads,
                               tune.type,
                               kpc.num,
                               cor,
                               type,
                               support.num,
                               important.search,
                               sparse.type,
                               nfolds,
                               foldid)
{
  para <- Initialization(
    c.max = c.max,
    support.size = support.size,
    always.include = always.include,
    group.index = group.index,
    splicing.type = splicing.type,
    max.splicing.iter = max.splicing.iter,
    warm.start = warm.start,
    ic.scale = ic.scale,
    num.threads = num.threads,
    tune.type = tune.type,
    important.search = important.search
  )
  para$kpc.num <- kpc.num
  para$support.num <- support.num
  para$cor <- cor
  para$type <- type
  para$sparse.type <- sparse.type
  para$nfolds <- nfolds
  para$foldid <- foldid
  
  class(para) <- append("pca", class(para))
  return(para)
}

Initialization_RPCA <- function(rank,
                                support.size,
                                tune.path,
                                gs.range,
                                tune.type,
                                ic.scale,
                                lambda,
                                always.include,
                                group.index,
                                c.max,
                                splicing.type,
                                max.splicing.iter,
                                warm.start,
                                important.search,
                                max.newton.iter,
                                newton.thresh,
                                num.threads)
{
  para <- Initialization(
    c.max = c.max,
    support.size = support.size,
    always.include = always.include,
    group.index = group.index,
    splicing.type = splicing.type,
    max.splicing.iter = max.splicing.iter,
    warm.start = warm.start,
    ic.scale = ic.scale,
    num.threads = num.threads,
    tune.type = tune.type,
    important.search = important.search
  )
  para$rank <- rank
  para$tune.path <- tune.path
  para$gs.range <- gs.range
  para$max.newton.iter <- max.newton.iter
  para$newton.thresh <- newton.thresh
  para$lambda <- lambda
  
  class(para) <- append("rpca", class(para))
  return(para)
}

strategy_for_tuning <-
  function(para)
    UseMethod("strategy_for_tuning")

strategy_for_tuning_private <- function(para) {
  if (para$tune.path == "gsection") {
    para$path_type <- 2
  } else if (para$tune.path == "sequence") {
    para$path_type <- 1
  }
  para
}

strategy_for_tuning.rpca <- strategy_for_tuning_private

strategy_for_tuning.glm <- strategy_for_tuning_private

rank <- function(para)
  UseMethod("rank")

rank.rpca <- function(para) {
  stopifnot(!anyNA(para$rank))
  stopifnot(all(para$rank >= 0))
  para
}


number_of_thread <- function(para)
  UseMethod("number_of_thread")

number_of_thread.Initialization <- function(para) {
  stopifnot(is.numeric(para$num.threads) & para$num.threads >= 0)
  para$num_threads <- as.integer(para$num.threads)
  para
}


newton_thresh <- function(para)
  UseMethod("newton_thresh")

newton_thresh_private <- function(para) {
  stopifnot(is.numeric(para$newton.thresh) & para$newton.thresh > 0)
  para$newton_thresh <- as.double(para$newton.thresh)
  para
}

newton_thresh.glm <- newton_thresh_private

newton_thresh.rpca <- newton_thresh_private


max_newton_iter <- function(para)
  UseMethod("max_newton_iter")

max_newton_iter_private <- function(default) {
  function(para) {
    if (!is.null(para$max.newton.iter)) {
      stopifnot(is.numeric(para$max.newton.iter) &
                  para$max.newton.iter >= 1)
      para$max_newton_iter <- as.integer(para$max.newton.iter)
    } else {
      para$max_newton_iter <- ifelse(para$newton_type == 0, 10, default)
      if (para$family == "gamma" && para$newton_type == 1) {
        para$max_newton_iter <- 200
      }
    }
    para
  }
  
}

max_newton_iter.rpca <- max_newton_iter_private(100)

max_newton_iter.glm <- max_newton_iter_private(60)


lambda <- function(para)
  UseMethod("lambda")

lambda_private <- function(para) {
  stopifnot(length(para$lambda) == 1)
  stopifnot(!anyNA(para$lambda))
  stopifnot(all(para$lambda >= 0))
  
  para
}

lambda.rpca <- lambda_private

lambda.glm <- lambda_private


warm_start <- function(para)
  UseMethod("warm_start")

warm_start.Initialization <- function(para) {
  stopifnot(is.logical(para$warm.start))
  
  para
}


splicing_type <- function(para)
  UseMethod("splicing_type")

splicing_type.Initialization <- function(para) {
  stopifnot(length(para$splicing.type) == 1)
  stopifnot(para$splicing.type %in% c(1, 2))
  para$splicing_type <-
    2 - as.integer(para$splicing.type) # adapt requirements of cpp
  
  para
}


max_splicing_iter <- function(para)
  UseMethod("max_splicing_iter")

max_splicing_iter.Initialization <- function(para) {
  stopifnot(is.numeric(para$max.splicing.iter) &
              para$max.splicing.iter >= 1)
  check_integer_warning(
    para$max.splicing.iter,
    "max.splicing.iter should be an integer value.
                        It is coerced to as.integer(max.splicing.iter)."
  )
  para$max_splicing_iter <- as.integer(para$max.splicing.iter)
  
  para
}


x_matrix_info <- function(para, data)
  UseMethod("x_matrix_info")

x_matrix_info.Initialization <- function(para, data) {
  stopifnot(class(data$x)[1] %in% c("data.frame", "matrix", "dgCMatrix"))
  para$nvars <- ncol(data$x)
  para$nobs <- nrow(data$x)
  # if x is not a matrix type object, it will return NULL:
  para$vn <- colnames(data$x)
  if (is.null(para$vn)) {
    para$vn <- paste0("x", 1:para$nvars)
  }
  
  para
}


x_matrix_content <-
  function(para, data)
    UseMethod("x_matrix_content")

x_matrix_content_private <- function(least_col) {
  function(para, data) {
    if (ncol(data$x) < least_col) {
      stop("x should have at least two columns!")
    }
    para$sparse_X <- class(data$x)[1] == "dgCMatrix"
    if (!para$sparse_X) {
      if (is.data.frame(data$x)) {
        data$x <- as.matrix(data$x)
      }
      if (!is.numeric(data$x)) {
        warning(
          "x should be a *numeric* matrix/data.frame!
              The factor value are coerced to as.numeric(x)."
        )
        data$x <- apply(data$x, 2, as.numeric)
      }
    }
    if (anyNA(data$x) || any(is.infinite(data$x))) {
      stop("x has missing value or infinite value!")
    }
    list(para = para, data = data)
  }
}

x_matrix_content.rpca <- x_matrix_content_private(2)

x_matrix_content.pca <- x_matrix_content_private(2)

x_matrix_content.glm <- x_matrix_content_private(0)

y_matrix <- function(para, data)
  UseMethod("y_matrix")

y_matrix.glm <- function(para, data) {
  if (anyNA(data$y)) {
    stop("y has missing value!")
  }
  if (any(is.infinite(data$y))) {
    stop("y has infinite value!")
  }
  if (para$family == "gaussian") {
    if (is.matrix(data$y)) {
      if (dim(data$y)[2] > 1) {
        stop("The dimension of y should not exceed 1 when family = 'gaussian'!")
      }
    }
  }
  if (para$family %in% c("binomial", "multinomial", "ordinal")) {
    if (length(unique(data$y)) == 2 &&
        para$family %in% c("multinomial", "ordinal")) {
      warning(
        "y is a binary variable and is not match to family = 'multinomial' or 'ordinal'.
              We change to family = 'binomial'"
      )
      para$model_type <- 2
      para$family <- "binomial"
    }
    if (length(unique(data$y)) > 2 && para$family == "binomial") {
      stop(
        "Input binary y when family = 'binomial'; otherwise,
           change the option for family to 'multinomial'. "
      )
    }
    if (length(unique(data$y)) == para$nobs &&
        para$family %in% c("multinomial", "ordinal")) {
      stop(
        "All of y value are distinct.
           Please input categorial y when family = 'multinomial' or 'ordinal'."
      )
    }
    if ((para$nobs / length(unique(data$y))) < 5 &&
        para$family %in% c("multinomial", "ordinal")) {
      warning(
        "The number of the category of y is relative large compare to nvars.
              The numerical result might be unstable."
      )
    }
    if (!is.factor(data$y)) {
      data$y <- as.factor(data$y)
    }
    class.name <- levels(data$y)
    para$y_vn <- class.name
    
    if (para$family == "binomial") {
      data$y <- as.numeric(data$y) - 1
    }
    if (para$family %in% c("multinomial", "ordinal")) {
      data$y <- model.matrix( ~ factor(as.numeric(data$y) - 1) + 0)
      colnames(data$y) <- NULL
    }
  }
  if (para$family == "poisson") {
    if (any(data$y < 0)) {
      stop("y must be positive integer value when family = 'poisson'.")
    }
  }
  if (para$family == "gamma") {
    if (any(data$y < 0)) {
      stop("y must be positive value when family = 'gamma'.")
    }
  }
  if (para$family == "cox") {
    if (!is.matrix(data$y)) {
      data$y <- as.matrix(data$y)
    }
    if (ncol(data$y) != 2) {
      stop("y must be a Surv object or a matrix with two columns when family = 'cox'!")
    }
    stopifnot(length(unique(data$y[, 2])) == 2)
    # pre-process data for cox para
    sort_y <- order(data$y[, 1])
    data$y <- data$y[sort_y,]
    data$x <- data$x[sort_y,]
    data$y <- data$y[, 2]
  }
  if (para$family == "mgaussian") {
    if (!is.matrix(data$y) || dim(data$y)[2] <= 1) {
      stop("y must be a n-by-q matrix (q > 1) when family = 'mgaussian'!")
    }
    para$y_vn <- colnames(data$y)
    if (is.null(para$y_vn)) {
      para$y_vn <- colnames("y", 1:dim(data$y)[2])
    }
  }
  data$y <- as.matrix(data$y)
  para$y_dim <- ncol(data$y)
  para$multi_y <- para$family %in% MULTIVARIATE_RESPONSE
  
  list(para = para, data = data)
}

screening_num <- function(para)
  UseMethod("screening_num")

screening_num.rpca <- function(para) {
  para$screening_num <- para$nobs * para$nvars
  
  para
}

screening_num.glm <- function(para) {
  if (is.null(para$screening.num)) {
    para$screening <- FALSE
    para$screening_num <- para$nvars
  } else {
    stopifnot(is.numeric(para$screening.num))
    stopifnot(para$screening.num >= 1)
    check_integer_warning(
      para$screening.num,
      "screening.num should be a integer. It is coerced to as.integer(screening.num)."
    )
    para$screening.num <- as.integer(para$screening.num)
    if (para$screening.num > para$nvars) {
      stop("The number of screening features must be equal or less than that of the column of x!")
    }
    if (para$path_type == 1) {
      if (para$screening.num < max(para$s_list)) {
        stop(
          "The number of screening features must be equal or greater than the maximum one in support.size!"
        )
      }
    } else {
      if (para$screening.num < para$s_max) {
        stop(
          "The number of screening features must be equal or greater than the max(gs.range)!"
        )
      }
    }
    para$screening <- TRUE
    para$screening_num <- para$screening.num
  }
  
  
  para
}


group_variable <- function(para)
  UseMethod("group_variable")

group_variable_private <- function(para, screening_num) {
  if (is.null(para$group.index)) {
    para$group_select <- FALSE
    para$g_index <- 1:screening_num - 1
    para$ngroup <- 1
    para$max_group_size <- 1
  } else {
    stopifnot(all(!is.na(para$group.index)))
    stopifnot(all(is.finite(para$group.index)))
    stopifnot(diff(para$group.index) >= 0)
    check_integer(para$group.index,
                  "group.index must be a vector with integer value.")
    para$group_select <- TRUE
    gi <- unique(para$group.index)
    para$g_index <- match(gi, para$group.index) - 1
    g_df <- c(diff(para$g_index), para$nvars - max(para$g_index))
    para$ngroup <- length(para$g_index)
    para$max_group_size <- max(g_df)
  }
  para
}

group_variable.glm <- function(para) {
  group_variable_private(para, para$nvars)
}

group_variable.pca <- function(para) {
  group_variable_private(para, para$nvars)
}

group_variable.rpca <- function(para) {
  group_variable_private(para, para$nvars * para$nobs)
}

sparse_level_list <- function(para)
  UseMethod("sparse_level_list")

sparse_level_list.rpca <- function(para) {
  max_rank <- max(c(para$nvars, para$nobs))
  if (is.null(para$support.size)) {
    if (para$group_select) {
      para$s_list <-
        0:min(c(
          para$ngroup,
          round(para$nobs / para$max_group_size / log(para$ngroup))
        ))
    } else {
      min_support_set <- max(c(3 * para$rank, max_rank / 2))
      para$s_list <- min_support_set:round(1.5 * max(max_rank))
    }
  } else {
    stopifnot(any(is.numeric(para$support.size) &
                    para$support.size >= 0))
    check_integer(para$support.size,
                  "support.size must be a vector with integer value.")
    if (para$group_select) {
      stopifnot(max(para$support.size) <= para$ngroup)
    }
    stopifnot(max(para$support.size) <= para$nvars * para$nobs)
    para$support.size <- sort(para$support.size)
    para$support.size <- unique(para$support.size)
    para$s_list <- para$support.size
  }
  
  para
}

sparse_level_list.pca <- function(para) {
  if (para$group_select) {
    para$s_max <- para$ngroup
  } else {
    para$s_max <- para$nvars
  }
  if (is.null(para$support.size)) {
    if (para$kpc.num == 1) {
      if (is.null(para$support.num)) {
        if (para$group_select) {
          s_num <- min(para$ngroup, 100)
        } else {
          s_num <- min(para$nvars, 100)
        }
      }
      para$s_list <-
        round(seq.int(
          from = 1,
          to = para$s_max,
          length.out = s_num
        ))
      para$s_list <- unique(para$s_list)
    } else {
      if (para$group_select) {
        s_num <- min(para$ngroup, 100)
      } else {
        s_num <- min(para$nvars, 100)
      }
      para$s_list <- as.list(rep(s_num, para$kpc.num))
    }
  } else {
    stopifnot(any(is.numeric(para$support.size) &
                    para$support.size >= 0))
    if (para$group_select) {
      stopifnot(max(para$support.size) <= para$ngroup)
    } else {
      stopifnot(max(para$support.size) <= para$nvars)
    }
    if (para$kpc.num == 1) {
      para$support.size <- unique(sort(para$support.size))
    } else {
      if (class(para$support.size) == "list") {
        stopifnot(length(para$support.size) == para$kpc.num)
        para$support.size <- lapply(support.size, unique)
      } else if (is.vector(para$support.size)) {
        para$support.size <-
          rep(list(unique(sort(
            para$support.size
          ))), para$kpc.num)
      } else {
        stop("support.size must be vector or list.")
      }
    }
    para$s_list <- para$support.size
  }
  s_list_bool_nrow <-
    ifelse(para$group_select, para$ngroup, para$nvars)
  if (class(para$s_list) == "list") {
    para$s_list_bool <-
      matrix(0, nrow = s_list_bool_nrow, ncol = para$kpc.num)
    for (i in 1:para$kpc.num) {
      para$s_list_bool[para$s_list[[i]],] <- 1
    }
  } else {
    para$s_list_bool <- matrix(0, nrow = s_list_bool_nrow, ncol = 1)
    para$s_list_bool[para$s_list,] <- 1
  }
  
  
  para
}

sparse_level_list.glm <- function(para) {
  if (is.null(para$support.size)) {
    if (para$group_select) {
      para$s_list <-
        0:min(c(
          para$ngroup,
          round(para$nobs / para$max_group_size / log(para$ngroup))
        ))
    } else {
      para$s_list <-
        0:min(c(para$nvars, round(
          para$nobs / log(log(para$nobs)) / log(para$nvars)
        )))
    }
  } else {
    stopifnot(any(is.numeric(para$support.size) &
                    para$support.size >= 0))
    check_integer(para$support.size,
                  "support.size must be a vector with integer value.")
    if (para$group_select) {
      stopifnot(max(para$support.size) <= para$ngroup)
    } else {
      stopifnot(max(para$support.size) <= para$nvars)
    }
    stopifnot(max(para$support.size) < para$nobs)
    para$support.size <- sort(para$support.size)
    para$support.size <- unique(para$support.size)
    para$s_list <- para$support.size
  }
  
  
  para
}


C_max <- function(para)
  UseMethod("C_max")

C_max_private <- function(default) {
  function(para) {
    if (is.null(para$c.max)) {
      para$c_max <- max(c(2, default))
    } else {
      stopifnot(is.numeric(para$c.max))
      stopifnot(para$c.max >= 1)
      check_integer_warning(para$c.max,
                            "c.max should be an integer. It is coerced to as.integer(c.max).")
      para$c_max <- as.integer(para$c.max)
    }
    para
  }
  
}
C_max.pca <- function(para) {
  C_max_private(round(max(unlist(para$s_list)) / 2))(para)
}

C_max.rpca <- function(para) {
  C_max_private(round(max(unlist(para$s_list)) / 2))(para)
}

C_max.glm <- C_max_private(2)


tune_support_size_method <-
  function(para)
    UseMethod("tune_support_size_method")

tune_support_size_method_private <- function(para) {
  para$ic_type <- map_tunetype2numeric(para$tune.type)
  para$is_cv <- para$tune.type == "cv"
  if (para$is_cv) {
    if (is.null(para$foldid)) {
      para$cv_fold_id <- integer(0)
      para$nfolds <- check_nfold(para$nfolds)
    } else {
      para$cv_fold_id <- check_foldid(para$foldid, para$nobs)
      para$nfolds <- length(unique(para$foldid))
    }
  } else {
    para$cv_fold_id <- integer(0)
    para$nfolds <- 1
  }
  para
}

tune_support_size_method.glm <- tune_support_size_method_private

tune_support_size_method.pca <- function(para) {
  para$tune_type <- para$tune.type
  if (para$cov_type == "gram" && para$tune_type == "cv") {
    warnings("Cross validation is not allow when input a gram matrix.
             Coerce into tune.type = 'gic'.")
    para$tune_type <- "gic"
  }
  
  tune_support_size_method.glm(para)
}

tune_support_size_method.rpca <- function(para) {
  para$ic_type <- map_tunetype2numeric(para$tune.type)
  para$is_cv <- FALSE
  para$cv_fold_id <- integer(0)
  
  para
}


information_criterion <-
  function(para)
    UseMethod("information_criterion")

information_criterion.Initialization <- function(para) {
  stopifnot(is.numeric(para$ic.scale))
  stopifnot(para$ic.scale >= 0)
  para$ic_scale <- as.integer(para$ic.scale)
  
  para
}


important_searching <-
  function(para)
    UseMethod("important_searching")

important_searching_private <- function(default) {
  function(para) {
    if (is.null(para$important.search)) {
      para$important_search <- as.integer(min(c(para$nvars, default)))
    } else {
      stopifnot(is.numeric(para$important.search))
      stopifnot(para$important.search >= 0)
      check_integer_warning(para$important.search)
      para$important_search <- as.integer(para$important.search)
    }
    
    para
  }
}

important_searching.Initialization <-
  important_searching_private(128)

important_searching.pca <- important_searching_private(0)


sparse_range <- function(para)
  UseMethod("sparse_range")

sparse_range.Initialization <- function(para) {
  if (is.null(para$gs.range)) {
    para$s_min <- 1
    if (para$group_select) {
      para$s_max <-
        min(c(
          para$ngroup,
          round(para$nobs / para$max_group_size / log(para$ngroup))
        ))
    } else {
      para$s_max <-
        min(c(para$nvars, round(
          para$nobs / log(log(para$nobs)) / log(para$nvars)
        )))
    }
  } else {
    stopifnot(length(para$gs.range) == 2)
    stopifnot(all(is.numeric(para$gs.range)))
    stopifnot(all(para$gs.range > 0))
    check_integer_warning(
      para$gs.range,
      "gs.range should be a vector with integer.
                          It is coerced to as.integer(gs.range)."
    )
    stopifnot(as.integer(para$gs.range)[1] != as.integer(para$gs.range)[2])
    if (para$group_select) {
      stopifnot(max(para$gs.range) < para$ngroup)
    } else {
      stopifnot(max(para$gs.range) < para$nvars)
    }
    para$gs.range <- as.integer(para$gs.range)
    para$s_min <- min(para$gs.range)
    para$s_max <- max(para$gs.range)
  }
  
  para
}

sparse_type <- function(para)
  UseMethod("sparse_type")

sparse_type.pca <- function(para) {
  if (is.null(para$kpc.num)) {
    para$kpc.num <- ifelse(para$sparse.type == "fpc", 1, 2)
  }
  else{
    stopifnot(para$kpc.num >= 1)
    check_integer_warning(para$kpc.num,
                          "kpc.num should be an integer. It is coerced to as.integer(kpc.num).")
    para$kpc.num <- as.integer(para$kpc.num)
    para$sparse.type <- ifelse(para$kpc.num == 1, "fpc", "kpc")
  }
  para
}

always_included_variables <-
  function(para)
    UseMethod("always_included_variables")

always_included_variables.Initialization <- function(para) {
  if (is.null(para$always.include)) {
    para$always_include <- numeric(0)
  } else {
    if (anyNA(para$always.include) ||
        any(is.infinite(para$always.include))) {
      stop("always.include has missing values or infinite values.")
    }
    stopifnot(para$always.include %in% 1:para$nvars)
    stopifnot(para$always.include > 0)
    check_integer(para$always.include,
                  "always.include must be a vector with integer value.")
    para$always.include <- as.integer(para$always.include) - 1
    always_include_num <- length(para$always.include)
    if (always_include_num > para$screening_num) {
      stop("The number of variables in always.include must not exceed the screening.num")
    }
    if (para$path_type == 1) {
      if (always_include_num > max(para$s_list)) {
        stop(
          "always.include containing too many variables.
           The length of it must not exceed the maximum in support.size."
        )
      }
      if (always_include_num > min(para$s_list)) {
        if (is.null(para$support.size)) {
          para$s_list <- para$s_list[para$s_list >= always_include_num]
        } else {
          stop(
            sprintf(
              "always.include containing %s variables. The min(support.size) must be equal or greater than this.",
              always_include_num
            )
          )
        }
      }
    } else {
      if (always_include_num > para$s_max) {
        stop(
          "always.include containing too many variables. The length of it must not exceed the max(gs.range)."
        )
      }
      if (always_include_num > para$s_min) {
        if (is.null(para$support.size)) {
          para$s_min <- always_include_num
        } else {
          stop(
            sprintf(
              "always.include containing %s variables. The min(gs.range) must be equal or greater than this.",
              always_include_num
            )
          )
        }
      }
    }
    para$always_include <- para$always.include
  }
  
  para
}

always_included_variables.pca <- function(para) {
  if (is.null(para$always.include)) {
    para$always_include <- numeric(0)
  } else {
    if (anyNA(para$always.include)) {
      stop("always.include has missing values.")
    }
    if (any(para$always.include <= 0)) {
      stop(
        "always.include should be an vector containing variable indexes which is positive."
      )
    }
    para$always.include <- as.integer(para$always.include) - 1
    if (length(para$always.include) > max(unlist(para$s_list))) {
      stop(
        "always.include containing too many variables.
             The length of it should not exceed the maximum in support.size."
      )
    }
    
    para$always_include <- para$always.include
  }
  
  
  para
}


sparse.cov <- function(x, cor = FALSE) {
  n <- nrow(x)
  cMeans <- colMeans(x)
  covmat <-
    (as.matrix(crossprod(x)) - n * tcrossprod(cMeans)) / (n - 1)
  
  if (cor) {
    sdvec <- sqrt(diag(covmat))
    covmat <- covmat / crossprod(t(sdvec))
  }
  
  as.matrix(covmat)
}

compute_gram_matrix <-
  function(para, data)
    UseMethod("compute_gram_matrix")

compute_gram_matrix.pca <- function(para, data) {
  para$cov_type <- para$type
  para$sparse_matrix <- FALSE
  if (para$cov_type == "gram") {
    stopifnot(dim(data$x)[1] == dim(data$x)[2])
    stopifnot(all(t(data$x) == data$x))
    # eigen values:
    eigen_value <- eigen(data$x, only.values = TRUE)[["values"]]
    eigen_value <- (eigen_value + abs(eigen_value)) / 2
    para$gram_x <- data$x
    data$x <- matrix(0, ncol = para$nvars, nrow = 1)
  } else {
    stopifnot(length(para$cor) == 1)
    stopifnot(is.logical(para$cor))
    # eigen values:
    if (!para$cor) {
      singular_value <-
        (svd(scale(
          data$x, center = TRUE, scale = FALSE
        ))[["d"]]) ^ 2 # improve runtimes
      eigen_value <- singular_value / para$nobs
    } else {
      singular_value <-
        (svd(scale(
          data$x, center = TRUE, scale = TRUE
        ))[["d"]]) ^ 2 # improve runtimes
      eigen_value <- singular_value / (para$nobs - 1)
    }
    
    if (para$sparse_X) {
      para$gram_x <- sparse.cov(data$x, cor = para$cor)
      data$x <- map_dgCMatrix2entry(data$x)
      para$sparse_matrix <- TRUE
    } else {
      if (para$cor) {
        para$gram_x <- stats::cor(data$x)
      } else {
        para$gram_x <- stats::cov(data$x)
      }
    }
    if (!para$cor) {
      para$gram_x <- ((para$nobs - 1) / para$nobs) * para$gram_x
    }
    # x <- round(x, digits = 13)
  }
  
  # if (sparse.type == "fpc") {
  #   eigen_value <- eigen_value[1]
  # }
  para$pc_variance <- eigen_value
  para$total_variance <- sum(eigen_value)
  
  list(para = para, data = data)
}


early_stop <- function(para)
  UseMethod("early_stop")

early_stop.glm <- function(para) {
  stopifnot(is.logical(para$early.stop))
  para$early_stop <- para$early.stop
  
  para
}


model_type <- function(para)
  UseMethod("model_type")

model_type.glm <- function(para) {
  para$model_type <- switch(
    para$family,
    "gaussian" = 1,
    "binomial" = 2,
    "poisson" = 3,
    "cox" = 4,
    "mgaussian" = 5,
    "multinomial" = 6,
    "gamma" = 8,
    "ordinal" = 9
    
  )
  
  para
}


x_y_matching <- function(para, data)
  UseMethod("x_y_matching")

x_y_matching.glm <- function(para, data) {
  if (para$nobs != nrow(data$y)) {
    stop("Rows of x must be the same as rows of y!")
  }
  if (para$sparse_X) {
    data$x <- map_dgCMatrix2entry(data$x)
  }
  list(para = para, data = data)
}


weight <- function(para)
  UseMethod("weight")

weight.glm <- function(para) {
  if (is.null(para$weight)) {
    para$weight <- rep(1, para$nobs)
  }
  else{
    stopifnot(is.vector(para$weight))
    if (length(para$weight) != para$nobs) {
      stop("Rows of x must be the same as length of weight!")
    }
    stopifnot(all(is.numeric(para$weight)), all(para$weight >= 0))
  }
  
  para
}


covariance_update <- function(para)
  UseMethod("covariance_update")

covariance_update.glm <- function(para) {
  stopifnot(is.logical(para$cov.update))
  if (para$model_type == 1) {
    para$covariance_update <- para$cov.update
  } else {
    para$covariance_update <- FALSE
  }
  
  para
}


normalize_strategy <- function(para)
  UseMethod("normalize_strategy")

normalize_strategy.glm <- function(para) {
  if (is.null(para$normalize)) {
    para$normalize <- switch(
      para$family,
      "gaussian" = 1,
      "binomial" = 2,
      "poisson" = 2,
      "cox" = 3,
      "mgaussian" = 1,
      "multinomial" = 2,
      "gamma" = 2,
      "ordinal" = 2
    )
  } else {
    stopifnot(para$normalize %in% 0:3)
  }
  
  para
}


init_active_set <- function(para)
  UseMethod("init_active_set")

init_active_set.glm <- function(para) {
  if (!is.null(para$init.active.set)) {
    stopifnot(para$init.active.set >= 1)
    stopifnot(all(para$init.active.set <= para$nvars))
    check_integer_warning(
      para$init.active.set,
      "init.active.set should be a vector with integer.
                          It is coerced to as.integer(init.active.set)."
    )
    para$init.active.set <- as.integer(para$init.active.set)
    para$init.active.set <- sort(unique(para$model_type)) - 1
  }
  
  para
}


newton_type <- function(para)
  UseMethod("newton_type")

newton_type.glm <- function(para) {
  if (length(para$newton) == 2) {
    if (para$family %in% c("binomial", "cox", "multinomial", "gamma", "poisson")) {
      para$newton <- "approx"
    }
    else{
      para$newton <- "exact"
    }
  }
  stopifnot(length(para$newton) == 1)
  stopifnot(para$newton %in% c("exact", "approx"))
  
  if (para$family %in% c("gaussian", "mgaussian")) {
    para$newton <- "exact"
  }
  para$newton_type <- switch(
    para$newton,
    "exact" = 0,
    "approx" = 1,
    "auto" = 2
  )
  para$approximate_newton <- para$newton_type == 1
  
  para
}


initializate <- function(para, data)
  UseMethod("initializate")

initializate.glm <- function(para, data) {
  para <- lambda(para)
  para <- number_of_thread(para)
  para <- early_stop(para)
  para <- warm_start(para)
  para <- splicing_type(para)
  para <- max_splicing_iter(para)
  para <- model_type(para)
  para <- x_matrix_info(para, data)
  model <- x_matrix_content(para, data)
  para <- model$para
  data <- model$data
  para <- weight(para)
  model <- y_matrix(para, data)
  para <- model$para
  data <- model$data
  model <- x_y_matching(para, data)
  para <- model$para
  data <- model$data
  para <- strategy_for_tuning(para)
  para <- group_variable(para)
  para <- sparse_level_list(para)
  para <- sparse_range(para)
  para <- C_max(para)
  para <- covariance_update(para)
  para <- newton_type(para)
  para <- max_newton_iter(para)
  para <- newton_thresh(para)
  para <- tune_support_size_method(para)
  para <- information_criterion(para)
  para <- normalize_strategy(para)
  para <- screening_num(para)
  para <- important_searching(para)
  para <- always_included_variables(para)
  para <- init_active_set(para)
  
  list(para = para, data = data)
}

initializate.pca <- function(para, data) {
  para <- information_criterion(para)
  para <- number_of_thread(para)
  para <- warm_start(para)
  para <- splicing_type(para)
  para <- max_splicing_iter(para)
  para <- x_matrix_info(para, data)
  model <- x_matrix_content(para, data)
  para <- model$para
  data <- model$data
  para <- sparse_type(para)
  model <- compute_gram_matrix(para, data)
  para <- model$para
  data <- model$data
  para <- group_variable(para)
  para <- sparse_level_list(para)
  para <- C_max(para)
  para <- always_included_variables(para)
  para <- important_searching(para)
  para <- tune_support_size_method(para)
  
  list(para = para, data = data)
}

initializate.rpca <- function(para, data) {
  para <- strategy_for_tuning(para)
  para <- rank(para)
  para <- number_of_thread(para)
  para <- max_newton_iter(para)
  para <- newton_thresh(para)
  para <- lambda(para)
  para <- warm_start(para)
  para <- splicing_type(para)
  para <- max_splicing_iter(para)
  para <- x_matrix_info(para, data)
  model <- x_matrix_content(para, data)
  para <- model$para
  data <- model$data
  para <- screening_num(para)
  para <- group_variable(para)
  para <- sparse_level_list(para)
  para <- C_max(para)
  para <- tune_support_size_method(para)
  para <- information_criterion(para)
  para <- important_searching(para)
  para <- sparse_range(para)
  para <- always_included_variables(para)
  
  list(para = para, data = data)
}
