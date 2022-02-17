Initialization <- function(
  x,
  c.max,
  support.size,
  always.include,
  group.index,
  splicing.type,
  max.splicing.iter,
  warm.start,
  ic.scale,
  num.threads,
  tune.type,
  important.search
)
{
  para <- list(
    x=x,
    c.max=c.max,
    support.size=support.size,
    always.include=always.include,
    group.index=group.index,
    splicing.type=splicing.type,
    max.splicing.iter=max.splicing.iter,
    warm.start=warm.start,
    ic.scale=ic.scale,
    num.threads=num.threads,
    tune.type=tune.type,
    important.search=important.search
  )
  
  class(para) <- append("Initialization",class(para))
  return(para)
}

Initialization_GLM <- function(
  x,
  c.max,
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
  y,
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
  nfolds
)
{
  para <- Initialization(
    x,
    c.max,
    support.size,
    always.include,
    group.index,
    splicing.type,
    max.splicing.iter,
    warm.start,
    ic.scale,
    num.threads,
    tune.type,
    important.search
  )
  para$newton.thresh <- newton.thresh
  para$y <- y
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
  
  class(para) <- append("glm",class(para))
  return(para)
}



strategy_for_tuning <- function(model) UseMethod("strategy_for_tuning")


strategy_for_tuning.rpca <- function(model){
  if (model$tune.path == "gsection") {
    model$path_type <- 2
  } else if (model$tune.path == "sequence") {
    model$path_type <- 1
  }
  model
}

strategy_for_tuning.glm <- function(model){
  if (model$tune.path == "gsection") {
    model$path_type <- 2
  } else if (model$tune.path == "sequence") {
    model$path_type <- 1
  }
  model
}

rank <- function(model) UseMethod("rank")

rank.rpca <- function(model){
  stopifnot(!missing(model$rank))
  stopifnot(!anyNA(model$rank))
  stopifnot(all(model$rank >= 0))
  model
}


number_of_thread <- function(model) UseMethod("number_of_thread")

number_of_thread.Initialization <- function(model){
  stopifnot(is.numeric(model$num.threads) & model$num.threads >= 0)
  model$num_threads <- as.integer(model$num.threads)
  model
}


newton_thresh <- function(model) UseMethod("newton_thresh")

newton_thresh_private <- function(model){
  stopifnot(is.numeric(model$newton.thresh) & model$newton.thresh > 0)
  model$newton_thresh <- as.double(model$newton.thresh)
  model
}

newton_thresh.glm <- function(model){
  model <- newton_thresh_private(model)
  model
}

newton_thresh.rpca <- function(model){
  model <- newton_thresh_private(model)
  model
}


max_newton_iter <- function(model) UseMethod("max_newton_iter")

max_newton_iter.rpca <- function(model){
  if (!is.null(model$max.newton.iter)) {
    stopifnot(is.numeric(model$max.newton.iter) & model$max.newton.iter >= 1)
    model$max_newton_iter <- as.integer(model$max.newton.iter)
  } else {
    model$max_newton_iter <- 100
  }
  model
}

max_newton_iter.glm <- function(model){
  if (!is.null(model$max.newton.iter)) {
    stopifnot(is.numeric(model$max.newton.iter) & model$max.newton.iter >= 1)
    model$max_newton_iter <- as.integer(model$max.newton.iter)
  } else {
    model$max_newton_iter <- ifelse(model$newton_type == 0, 10, 60)
    if (model$family == "gamma" && model$newton_type == 1) {
      model$max_newton_iter <- 200
    }
  }
  model
}

## 
lambda <- function(model) UseMethod("lambda")

lambda.rpca <- function(model){
  
  stopifnot(!anyNA(model$lambda))
  stopifnot(all(model$lambda >= 0))
  
  model
}

lambda.glm <- function(model){
  
  stopifnot(length(model$lambda) == 1)
  stopifnot(!anyNA(model$lambda))
  stopifnot(all(model$lambda >= 0))
  
  model
}


warm_start <- function(model) UseMethod("warm_start")

warm_start.Initialization <- function(model){
  
  stopifnot(is.logical(model$warm.start))
  
  model
}

##
splicing_type <- function(model) UseMethod("splicing_type")

splicing_type.Initialization <- function(model){
  
  stopifnot(length(model$splicing.type) == 1)
  stopifnot(model$splicing.type %in% c(1, 2))
  model$splicing_type <- as.integer(model$splicing.type)
  
  model
}

splicing_type.glm <- function(model){
  
  stopifnot(length(model$splicing.type) == 1)
  stopifnot(model$splicing.type %in% c(1, 2)) 
  model$splicing.type <- 2 - model$splicing.type 
  model$splicing_type <- as.integer(model$splicing.type)
  
  model
}


max_splicing_iter <- function(model) UseMethod("max_splicing_iter")

max_splicing_iter.Initialization <- function(model){
  
  stopifnot(is.numeric(model$max.splicing.iter) & model$max.splicing.iter >= 1)
  check_integer_warning(
    model$max.splicing.iter,
    "max.splicing.iter should be an integer value.
                        It is coerced to as.integer(max.splicing.iter)."
  )
  model$max_splicing_iter <- as.integer(model$max.splicing.iter)
  
  model
}


x_matrix_info <- function(model) UseMethod("x_matrix_info")

x_matrix_info.Initialization <- function(model){
  
  stopifnot(class(model$x)[1] %in% c("data.frame", "matrix", "dgCMatrix"))
  model$nvars <- ncol(model$x)
  model$nobs <- nrow(model$x)
  model$vn <- colnames(model$x) # if x is not a matrix type object, it will return NULL.
  if (is.null(model$vn)) {
    model$vn <- paste0("x", 1:model$nvars)
  }
  
  model
}


x_matrix_content <- function(model) UseMethod("x_matrix_content")

x_matrix_content_private <- function(model){
  
  ##? don't check sparse matrix, why? Don't accept sqarse matrix
  model$sparse_X <- class(model$x)[1] == "dgCMatrix"
  if (!model$sparse_X) {
    if (is.data.frame(model$x)) {
      model$x <- as.matrix(model$x)
    }
    if (!is.numeric(model$x)) {
      stop("x must be a *numeric* matrix/data.frame!")
    }
    if (ncol(model$x) == 1) {
      stop("x should have at least two columns!")
    }
    if (anyNA(model$x) || any(is.infinite(model$x))) {
      stop("x has missing value or infinite value!")
    }
  }
  
  
  model
}

x_matrix_content.rpca <- function(model){
  model <- x_matrix_content_private(model)
  model
}

x_matrix_content.pca <- function(model){
  model <- x_matrix_content_private(model)
  model
}

x_matrix_content.glm <- function(model){
  
  model$sparse_X <- class(model$x)[1] == "dgCMatrix"
  if (model$sparse_X) {
    if (class(model$x) == "dgCMatrix") {
      model$x <- map_dgCMatrix2entry(model$x)
    }
  } else {
    if (is.data.frame(model$x)) {
      model$x <- as.matrix(model$x)
    }
    if (!is.numeric(model$x)) {
      warning("x should be a *numeric* matrix/data.frame! 
              The factor value are coerced to as.numeric(x).")
      model$x <- apply(model$x, 2, as.numeric)
    }
  }
  if (anyNA(model$x) || any(is.infinite(model$x))) {
    stop("x has missing value or infinite value!")
  }
  
  
  model
}

y_matrix <- function(model) UseMethod("y_matrix")

y_matrix.glm <- function(model){
  
  if (anyNA(model$y)) {
    stop("y has missing value!")
  }
  if (any(is.infinite(model$y))) {
    stop("y has infinite value!")
  }
  if (model$family == "gaussian") {
    if (is.matrix(model$y)) {
      if (dim(model$y)[2] > 1) {
        stop("The dimension of y should not exceed 1 when family = 'gaussian'!")
      }
    }
  }
  if (model$family == "binomial" || model$family == "multinomial") {
    if (length(unique(model$y)) == 2 && model$family == "multinomial") {
      warning("y is a binary variable and is not match to family = 'multinomial'.
              We change to family = 'binomial'")
      model$model_type <- 2
      model$family <- "binomial"
    }
    if (length(unique(model$y)) > 2 && model$family == "binomial") {
      stop("Input binary y when family = 'binomial'; otherwise,
           change the option for family to 'multinomial'. ")
    }
    if (length(unique(model$y)) == model$nobs && model$family == "multinomial") {
      stop("All of y value are distinct.
           Please input categorial y when family = 'multinomial'.")
    }
    if ((model$nobs / length(unique(model$y))) < 5 && model$family == "multinomial") {
      warning("The number of the category of y is relative large compare to nvars.
              The numerical result might be unstable.")
    }
    if (!is.factor(model$y)) {
      model$y <- as.factor(model$y)
    }
    class.name <- levels(model$y)
    model$y_vn <- class.name
    
    if (model$family == "binomial") {
      model$y <- as.numeric(model$y) - 1
    }
    if (model$family == "multinomial") {
      model$y <- model.matrix(~ factor(as.numeric(model$y) - 1) + 0)
      colnames(model$y) <- NULL
    }
  }
  if (model$family == "poisson") {
    if (any(model$y < 0)) {
      stop("y must be positive integer value when family = 'poisson'.")
    }
  }
  if (model$family == "gamma") {
    if (any(model$y < 0)) {
      stop("y must be positive value when family = 'gamma'.")
    }
  }
  if (model$family == "cox") {
    if (!is.matrix(model$y)) {
      model$y <- as.matrix(model$y)
    }
    if (ncol(model$y) != 2) {
      stop("y must be a Surv object or a matrix with two columns when family = 'cox'!")
    }
    stopifnot(length(unique(model$y[, 2])) == 2)
    # pre-process data for cox model
    sort_y <- order(model$y[, 1])
    model$y <- model$y[sort_y, ]
    model$x <- model$x[sort_y, ]
    model$y <- model$y[, 2]
  }
  if (model$family == "mgaussian") {
    if (!is.matrix(model$y) || dim(model$y)[2] <= 1) {
      stop("y must be a n-by-q matrix (q > 1) when family = 'mgaussian'!")
    }
    model$y_vn <- colnames(model$y)
    if (is.null(model$y_vn)) {
      model$y_vn <- colnames("y", 1:dim(model$y)[2])
    }
  }
  model$y <- as.matrix(model$y)
  model$y_dim <- ncol(model$y)
  model$multi_y <- model$family %in% MULTIVARIATE_RESPONSE
  
  
  model
}

screening_num <- function(model) UseMethod("screening_num")

screening_num.Initialization <- function(model){
  
  model$screening_num <- model$nobs * model$nvars
  
  model
}

screening_num.glm <- function(model){
  
  if (is.null(model$screening.num)) {
    model$screening <- FALSE
    model$screening_num <- model$nvars
  } else {
    stopifnot(is.numeric(model$screening.num))
    stopifnot(model$screening.num >= 1)
    check_integer_warning(
      model$screening.num,
      "screening.num should be a integer.
                          It is coerced to as.integer(screening.num)."
    )
    model$screening.num <- as.integer(model$screening.num)
    if (model$screening.num > model$nvars) {
      stop("The number of screening features must be equal or less than that of the column of x!")
    }
    if (model$path_type == 1) {
      if (model$screening.num < max(model$s_list)) {
        stop("The number of screening features must be equal or greater than the maximum one in support.size!")
      }
    } else {
      if (model$screening.num < model$s_max) {
        stop("The number of screening features must be equal or greater than the max(gs.range)!")
      }
    }
    model$screening <- TRUE
    model$screening_num <- model$screening.num
  }
  
  
  model
}


group_variable <- function(model) UseMethod("group_variable")

group_variable.Initialization <- function(model){
  
  if (is.null(model$group.index)) {
    model$group_select <- FALSE
    model$g_index <- 1:model$screening_num - 1
    model$ngroup <- 1
    model$max_group_size <- 1
    # g_df <- rep(1, nvars)
  } else {
    stopifnot(all(!is.na(model$group.index)))
    stopifnot(all(is.finite(model$group.index)))
    stopifnot(diff(model$group.index) >= 0)
    check_integer(model$group.index, "group.index must be a vector with integer value.")
    model$group_select <- TRUE
    gi <- unique(model$group.index)
    model$g_index <- match(gi, model$group.index) - 1
    g_df <- c(diff(model$g_index), model$nvars - max(model$g_index))
    model$ngroup <- length(model$g_index)
    model$max_group_size <- max(g_df)
  }
  
  
  model
}


sparse_level_list <- function(model) UseMethod("sparse_level_list")

sparse_level_list.rpca <- function(model){
  
  max_rank <- max(c(model$nvars, model$nobs))
  if (is.null(model$support.size)) {
    if (model$group_select) {
      model$s_list <- 0:min(c(model$ngroup, round(model$nobs / model$max_group_size / log(model$ngroup))))
    } else {
      min_support_set <- max(c(3 * model$rank, max_rank / 2))
      model$s_list <- min_support_set:round(1.5 * max(max_rank))
    }
  } else {
    stopifnot(any(is.numeric(model$support.size) & model$support.size >= 0))
    check_integer(model$support.size, "support.size must be a vector with integer value.")
    if (model$group_select) {
      stopifnot(max(model$support.size) <= model$ngroup)
    }
    stopifnot(max(model$support.size) <= model$nvars * model$nobs)
    model$support.size <- sort(model$support.size)
    model$support.size <- unique(model$support.size)
    model$s_list <- model$support.size
  }
  
  model
}

sparse_level_list.pca <- function(model){
  
  if (model$group_select) {
    model$s_max <- model$ngroup
  } else {
    model$s_max <- model$nvars
  }
  if (is.null(model$support.size)) {
    if (model$kpc.num == 1) {
      if (is.null(model$support.num)) {
        if (model$group_select) {
          s_num <- min(model$ngroup, 100)
        } else {
          s_num <- min(model$nvars, 100)
        }
      }
      model$s_list <- round(seq.int(from = 1, to = model$s_max, length.out = s_num))
      model$s_list <- unique(model$s_list)
    } else {
      if (model$group_select) {
        s_num <- min(model$ngroup, 100)
      } else {
        s_num <- min(model$nvars, 100)
      }
      model$s_list <- as.list(rep(s_num, model$kpc.num))
    }
  } else {
    stopifnot(any(is.numeric(model$support.size) & model$support.size >= 0))
    if (model$group_select) {
      stopifnot(max(model$support.size) <= model$ngroup)
    } else {
      stopifnot(max(model$support.size) <= model$nvars)
    }
    if (model$kpc.num == 1) {
      model$support.size <- sort(model$support.size)
      model$support.size <- unique(model$support.size)
    } else {
      if (class(model$support.size) == "list") {
        stopifnot(length(model$support.size) == model$kpc.num)
        ##? 
        model$support.size <- lapply(support.size, function(x) {
          x <- unique(x)
          x
        })
      } else if (is.vector(model$support.size)) {
        model$support.size <- sort(model$support.size)
        model$support.size <- unique(model$support.size)
        model$support.size <- rep(list(model$support.size), model$kpc.num)
      } else {
        stop("support.size must be vector or list.")
      }
    }
    model$s_list <- model$support.size
  }
  s_list_bool_nrow <- ifelse(model$group_select, model$ngroup, model$nvars)
  if (class(model$s_list) == "list") {
    model$s_list_bool <- matrix(0, nrow = s_list_bool_nrow, ncol = model$kpc.num)
    for (i in 1:model$kpc.num) {
      model$s_list_bool[model$[[i]], ] <- 1
    }
  } else {
    model$s_list_bool <- matrix(0, nrow = s_list_bool_nrow, ncol = 1)
    model$s_list_bool[model$s_list, ] <- 1
  }
  
  
  model
}

sparse_level_list.glm <- function(model){
  
  if (is.null(model$support.size)) {
    if (model$group_select) {
      model$s_list <- 0:min(c(model$ngroup, round(model$nobs / model$max_group_size / log(model$ngroup))))
    } else {
      model$s_list <- 0:min(c(model$nvars, round(model$nobs / log(log(model$nobs)) / log(model$nvars))))
    }
  } else {
    stopifnot(any(is.numeric(model$support.size) & model$support.size >= 0))
    check_integer(model$support.size, "support.size must be a vector with integer value.")
    if (model$group_select) {
      stopifnot(max(model$support.size) <= model$ngroup)
    } else {
      stopifnot(max(model$support.size) <= model$nvars)
    }
    stopifnot(max(model$support.size) < model$nobs)
    model$support.size <- sort(model$support.size)
    model$support.size <- unique(model$support.size)
    model$s_list <- model$support.size
  }
  
  
  model
}

##
C_max <- function(model) UseMethod("C_max")

C_max.pca <- function(model){
  
  model$s_list_max <- max(unlist(model$s_list))
  if (is.null(model$c.max)) {
    model$c_max <- max(c(2, round(model$s_list_max / 2)))
  } else {
    model <- NextMethod(model) ##?
  }
  
  
  model
}

C_max.rpca <- function(model){
  
  model$s_list_max <- max(unlist(model$s_list))
  if (is.null(model$c.max)) {
    model$c_max <- max(c(2, round(model$s_list_max / 2)))
  } else {
    model <- NextMethod(model) ##?
  }
  
  
  model
}

C_max.Initialization <- function(model){
  
  stopifnot(is.numeric(model$c.max))
  stopifnot(model$c.max >= 1)
  check_integer_warning(
    model$c.max,
    "c.max should be an integer. It is coerced to as.integer(c.max)."
  )
  model$c_max <- as.integer(model$c.max)
  
  
  model
}

## TODO 
tune_support_size_method <- function(model) UseMethod("tune_support_size_method")

tune_support_size_method.glm <- function(model){
  
  model$ic_type <- map_tunetype2numeric(model$tune.type)
  model$is_cv <- ifelse(model$tune.type == "cv", TRUE, FALSE)
  if (model$is_cv) {
    if (is.null(model$foldid)) {
      model$cv_fold_id <- integer(0)
      model$nfolds <- check_nfold(model$nfolds)
    } else {
      model$cv_fold_id <- check_foldid(model$foldid, model$nobs)
      model$nfolds <- length(unique(model$foldid))
    }
  } else {
    model$cv_fold_id <- integer(0)
    model$nfolds <- 1
  }
  
  
  model
}

tune_support_size_method.pca <- function(model){
  
  if (model$cov_type == "gram" && model$tune_type == "cv") {
    warnings("Cross validation is not allow when input a gram matrix.
             Coerce into tune.type = 'gic'.")
    model$tune_type <- "gic"
  }
  ## 和GLM本质一样
  model$ic_type <- map_tunetype2numeric(model$tune_type)
  if (model$tune_type != "cv") {
    model$nfolds <- 1
    model$cv_fold_id <- integer(0)
  } else {
    if (is.null(model$foldid)) {
      model$cv_fold_id <- integer(0)
      model$nfolds <- check_nfold(model$nfolds)
    } else {
      model$cv_fold_id <- check_foldid(model$foldid, model$nobs)
      model$nfolds <- length(unique(model$nfolds))
    }
  }

  
  model
}

tune_support_size_method.rpca <- function(model){
  
  model$ic_type <- map_tunetype2numeric(model$tune.type)
  model$is_cv <- FALSE
  model$cv_fold_id <- integer(0)
  
  
  model
}


information_criterion <- function(model) UseMethod("information_criterion")

information_criterion.Initialization <- function(model){
  
  stopifnot(is.numeric(model$ic.scale))
  stopifnot(model$ic.scale >= 0)
  model$ic_scale <- as.integer(model$ic.scale)
  
  
  model
}

##
important_searching <- function(model) UseMethod("important_searching")

important_searching.Initialization <- function(model){
  
  if (is.null(model$important.search)) {
    model$important_search <- as.integer(min(c(model$nvars, 128)))
  } else {
    stopifnot(is.numeric(model$important.search))
    stopifnot(model$important.search >= 0)
    check_integer_warning(model$important.search)
    model$important_search <- as.integer(model$important.search)
  }
  
  
  model
}

important_searching.pca <- function(model){
  
  if (is.null(model$important.search)) {
    model$important_search <- as.integer(0)
  } else {
    stopifnot(is.numeric(model$important.search))
    stopifnot(model$important.search >= 0)
    check_integer_warning(model$important.search)
    model$important_search <- as.integer(model$important.search)
  }
  
  
  model
}


sparse_range <- function(model) UseMethod("sparse_range")

sparse_range.Initialization <- function(model){
  
  if (is.null(model$gs.range)) {
    model$s_min <- 1
    if (model$group_select) {
      model$s_max <- min(c(model$ngroup, round(model$nobs / model$max_group_size / log(model$ngroup))))
    } else {
      model$s_max <- min(c(model$nvars, round(model$nobs / log(log(model$nobs)) / log(model$nvars))))
    }
  } else {
    stopifnot(length(model$gs.range) == 2)
    stopifnot(all(is.numeric(model$gs.range)))
    stopifnot(all(model$gs.range > 0))
    check_integer_warning(
      model$gs.range,
      "gs.range should be a vector with integer.
                          It is coerced to as.integer(gs.range)."
    )
    stopifnot(as.integer(model$gs.range)[1] != as.integer(model$gs.range)[2])
    if (model$group_select) {
      stopifnot(max(model$gs.range) < model$ngroup)
    } else {
      stopifnot(max(model$gs.range) < model$nvars)
    }
    model$gs.range <- as.integer(model$gs.range)
    model$s_min <- min(model$gs.range)
    model$s_max <- max(model$gs.range)
  }
  
  model
}



always_included_variables <- function(model) UseMethod("always_included_variables")
## glm and rpca
always_included_variables.Initialization <- function(model){
  
  if (is.null(model$always.include)) {
    model$always_include <- numeric(0)
  } else {
    if (anyNA(model$always.include) || any(is.infinite(model$always.include))) {
      stop("always.include has missing values or infinite values.")
    }
    stopifnot(model$always.include %in% 1:model$nvars)
    stopifnot(model$always.include > 0)
    check_integer(model$always.include, "always.include must be a vector with integer value.")
    model$always.include <- as.integer(model$always.include) - 1
    always_include_num <- length(model$always.include)
    if (always_include_num > model$screening_num) {
      stop("The number of variables in always.include must not exceed the screening.num")
    }
    if (model$path_type == 1) {
      if (always_include_num > max(model$s_list)) {
        stop("always.include containing too many variables.
           The length of it must not exceed the maximum in support.size.")
      }
      if (always_include_num > min(model$s_list)) {
        if (is.null(model$support.size)) {
          model$s_list <- model$s_list[model$s_list >= always_include_num]
        } else {
          stop(sprintf("always.include containing %s variables. The min(support.size) must be equal or greater than this.", always_include_num))
        }
      }
    } else {
      if (always_include_num > model$s_max) {
        stop("always.include containing too many variables. The length of it must not exceed the max(gs.range).")
      }
      if (always_include_num > model$s_min) {
        if (is.null(model$support.size)) {
          model$s_min <- always_include_num
        } else {
          stop(sprintf("always.include containing %s variables. The min(gs.range) must be equal or greater than this.", always_include_num))
        }
      }
    }
    model$always_include <- model$always.include
  }
  
  
  model
}

always_included_variables.pca <- function(model){
  
  if (is.null(model$always.include)) {
    model$always_include <- numeric(0)
  } else {
    if (anyNA(model$always.include)) {
      stop("always.include has missing values.")
    }
    if (any(model$always.include <= 0)) {
      stop("always.include should be an vector containing variable indexes which is positive.")
    }
    model$always.include <- as.integer(model$always.include) - 1
    if (length(model$always.include) > model$s_list_max) {
      stop("always.include containing too many variables.
             The length of it should not exceed the maximum in support.size.")
    }
    
    model$always_include <- model$always.include
  }
  
  
  model
}

## 可以合并
sparse_type <- function(model) UseMethod("sparse_type")

sparse_type.pca <- function(model){
  
  stopifnot(model$kpc.num >= 1)
  check_integer_warning(model$kpc.num, "kpc.num should be an integer. It is coerced to as.integer(kpc.num).")
  model$sparse.type <- ifelse(model$kpc.num == 1, "fpc", "kpc")
  
  
  model
}


compute_gram_matrix <- function(model) UseMethod("compute_gram_matrix")

compute_gram_matrix.pca <- function(model){
  
  model$sparse_matrix <- FALSE
  if (model$cov_type == "gram") {
    stopifnot(dim(model$x)[1] == dim(model$x)[2])
    stopifnot(all(t(model$x) == model$x))
    # eigen values:
    eigen_value <- eigen(model$x, only.values = TRUE)[["values"]]
    eigen_value <- (eigen_value + abs(eigen_value)) / 2
    model$gram_x <- model$x
    model$x <- matrix(0, ncol = model$nvars, nrow = 1)
  } else {
    stopifnot(length(model$cor) == 1)
    stopifnot(is.logical(model$cor))
    # eigen values:
    if (!model$cor) {
      singular_value <- (svd(scale(model$x, center = TRUE, scale = FALSE))[["d"]])^2 # improve runtimes
      eigen_value <- singular_value / model$nobs
    } else {
      singular_value <- (svd(scale(model$x, center = TRUE, scale = TRUE))[["d"]])^2 # improve runtimes
      eigen_value <- singular_value / (model$nobs - 1)
    }
    
    if (model$sparse_X) {
      if (model$cor) {
        model$gram_x <- sparse.cov(model$x, cor = TRUE)
      } else {
        model$gram_x <- sparse.cov(model$x)
      }
      model$x <- map_dgCMatrix2entry(model$x)
      model$sparse_matrix <- TRUE
    } else {
      if (model$cor) {
        model$gram_x <- stats::cor(model$x)
      } else {
        model$gram_x <- stats::cov(model$x)
      }
    }
    if (!model$cor) {
      model$gram_x <- ((model$nobs - 1) / model$nobs) * model$gram_x
    }
    # x <- round(x, digits = 13)
  }
  
  # if (sparse.type == "fpc") {
  #   eigen_value <- eigen_value[1]
  # }
  model$pc_variance <- eigen_value
  model$total_variance <- sum(eigen_value)
 
  
  model
}


early_stop <- function(model) UseMethod("early_stop")

early_stop.glm <- function(model){
  
  stopifnot(is.logical(model$early.stop))
  
  model
}


model_type <- function(model) UseMethod("model_type")

model_type.glm <- function(model){
  
  model$model_type <- switch(model$family,
                             "gaussian" = 1,
                             "binomial" = 2,
                             "poisson" = 3,
                             "cox" = 4,
                             "mgaussian" = 5,
                             "multinomial" = 6,
                             "gamma" = 8
  )
  
  
  model
}


x_y_matching <- function(model) UseMethod("x_y_matching")

x_y_matching.glm <- function(model){
  
  if (nrow(model$x) != nrow(model$y)) {
    stop("Rows of x must be the same as rows of y!")
  }
  
  model
}


weight <- function(model) UseMethod("weight")

weight.glm <- function(model){
  
  if (is.null(model$weight)) {
    model$weight <- rep(1, model$nobs)
  }
  else{
    stopifnot(is.vector(model$weight))
    if (length(model$weight) != model$nobs) {
      stop("Rows of x must be the same as length of weight!")
    }
    stopifnot(all(is.numeric(model$weight)), all(model$weight >= 0))
  }
  
  
  model
}


covariance_update <- function(model) UseMethod("covariance_update")

covariance_update.glm <- function(model){
  
  stopifnot(is.logical(model$cov.update))
  if (model$model_type == 1) {
    model$covariance_update <- model$cov.update
  } else {
    model$covariance_update <- FALSE
  }
  
  
  model
}


normalize_strategy <- function(model) UseMethod("normalize_strategy")

normalize_strategy.glm <- function(model){
  
  if (is.null(model$normalize)) {
    model$is_normal <- TRUE
    model$normalize <- switch(model$family,
                              "gaussian" = 1,
                              "binomial" = 2,
                              "poisson" = 2,
                              "cox" = 3,
                              "mgaussian" = 1,
                              "multinomial" = 2,
                              "gamma" = 2
    )
  } else {
    stopifnot(model$normalize %in% 0:3)
    if (model$normalize != 0) {
      if (model$normalize == 1) {
        model$normalize <- 2
      } else if (model$normalize == 2) {
        model$normalize <- 3
      } else if (model$normalize == 3) {
        model$normalize <- 1
      } else {
      }
      model$is_normal <- TRUE
    } else {
      model$is_normal <- FALSE
      model$normalize <- 0
    }
  }
  
  
  model
}


init_active_set <- function(model) UseMethod("init_active_set")

init_active_set.glm <- function(model){
  
  if (!is.null(model$init.active.set)) {
    stopifnot(model$init.active.set >= 1)
    stopifnot(all(model$init.active.set <= model$nvars))
    check_integer_warning(model$init.active.set, "init.active.set should be a vector with integer.
                          It is coerced to as.integer(init.active.set).")
    model$init.active.set <- as.integer(model$init.active.set)
    model$init.active.set <- sort(unique(model$init.active.set)) - 1
  }
  
  
  model
}


newton_type <- function(model) UseMethod("newton_type")

newton_type.glm <- function(model){
  
  if (length(model$newton) == 2) {
    if (model$family %in% c("binomial", "cox", "multinomial", "gamma", "poisson")) {
      model$newton <- "approx"
    }
    else{
      model$newton <- "exact"
    }
  }
  stopifnot(length(model$newton) == 1)
  stopifnot(model$newton %in% c("exact", "approx"))
  
  if (model$family %in% c("gaussian", "mgaussian")) {
    model$newton <- "exact"
  }
  model$newton_type <- switch(model$newton,
                              "exact" = 0,
                              "approx" = 1,
                              "auto" = 2
  )
  model$approximate_newton <- ifelse(model$newton_type == 1, TRUE, FALSE)
  
  
  model
}


initializate <- function(model) UseMethod("initializate")

initializate.glm <- function(model){
  model <- lambda(model)
  model <- number_of_thread(model)
  model <- early_stop(model)
  model <- warm_start(model)
  model <- splicing_type(model)
  model <- max_splicing_iter(model)
  model <- model_type(model)
  model <- x_matrix_info(model)
  model <- x_matrix_content(model)
  model <- weight(model)
  model <- y_matrix(model)
  model <- x_y_matching(model)
  model <- strategy_for_tuning(model)
  model <- group_variable(model)
  model <- sparse_level_list(model)
  model <- sparse_range(model)
  model <- C_max(model)
  model <- covariance_update(model)
  model <- newton_type(model)
  model <- max_newton_iter(model)
  model <- newton_thresh(model)
  model <- tune_support_size_method(model)
  model <- information_criterion(model)
  model <- normalize_strategy(model)
  model <- screening_num(model)
  model <- important_searching(model)
  model <- always_included_variables(model)
  model <- init_active_set(model)
  model
}
