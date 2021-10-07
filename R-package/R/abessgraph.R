#' Title
#'
#' @inheritParams abess.default
#'
#' @return
#' @export
#'
#' @examples
#' library(abess)
abessbmn <- function(x,
                     tune.type = c("gic", "bic", "cv", "ebic", "aic"),
                     weight = NULL,
                     c.max = 5,
                     support.size = NULL,
                     gs.range = NULL,
                     always.include = NULL,
                     splicing.type = 0,
                     max.splicing.iter = 20,
                     warm.start = TRUE,
                     nfolds = 5,
                     foldid = NULL,
                     newton.thresh = 1e-6,
                     max.newton.iter = 500,
                     ic.scale = 1.0,
                     num.threads = 0,
                     seed = 1,
                     ...)
{
  early.stop <- FALSE
  screening.num <- NULL
  important.search <- NULL
  lambda <- 0
  group.index <- NULL
  
  nobs <- nrow(x)
  nvars <- ncol(x)
  maximum_degree <- nvars * (nvars - 1) / 2
  y <- matrix(0, nrow = nobs, ncol = 1)
  
  if (is.null(weight)) {
    weight <- rep(1, nobs)
  }
  
  if (is.null(support.size)) {
    support_size <- seq.int(0, maximum_degree, length.out = maximum_degree)
  } else {
    support_size <- support.size
  }
  
  tune.path <- c("sequence", "gsection")
  tune.path <- tune.path[1]
  # tune_path <- match.arg(tune.path)
  tune_path <- ifelse(tune.path == "sequence", 1, 2)
  tune_path <- as.integer(tune_path)
  
  tune.type <- match.arg(tune.type)
  ic_type <- switch(tune.type,
                    "aic" = 1,
                    "bic" = 2,
                    "gic" = 3,
                    "ebic" = 4,
                    "cv" = 1
  )
  is_cv <- ifelse(tune.type == "cv", TRUE, FALSE)
  if (is_cv) {
    stopifnot(is.numeric(nfolds) & nfolds >= 2)
    check_integer_warning(
      nfolds,
      "nfolds should be an integer value. It is coerced to be as.integer(nfolds). "
    )
    nfolds <- as.integer(nfolds)
    
    if (is.null(foldid)) {
      cv_fold_id <- integer(0)
    } else {
      stopifnot(is.vector(foldid))
      stopifnot(is.numeric(foldid))
      stopifnot(length(foldid) == nobs)
      check_integer_warning(
        foldid,
        "nfolds should be an integer value. It is coerced to be as.integer(foldid). "
      )
      foldid <- as.integer(foldid)
      cv_fold_id <- foldid
    }
  } else {
    cv_fold_id <- integer(0)
  }
  
  ## group variable:
  group_select <- FALSE
  if (is.null(group.index)) {
    g_index <- 1:nvars - 1
    ngroup <- 1
    max_group_size <- 1
    # g_df <- rep(1, nvars)
  } else {
    g_index <- group.index
  }
  
  # check always included variables:
  if (is.null(always.include)) {
    always_include <- integer(0)
  } else {
    always_include <- always.include - 1
  }
  
  # newton <- c("approx", "exact")
  # newton <- match.arg(newton)
  # newton_type <- switch(newton,
  #                       "exact" = 0,
  #                       "approx" = 1,
  #                       "auto" = 2
  # )
  # approximate_newton <- ifelse(newton_type == 1, TRUE, FALSE)
  approximate_newton <- FALSE
  
  result <- abessCpp2(
    x = x,
    y = y,
    n = nobs,
    p = nvars,
    data_type = as.integer(1),
    weight = as.double(weight),
    sigma = matrix(-1),
    is_normal = FALSE,
    algorithm_type = as.integer(6),
    model_type = as.integer(8),
    max_iter = as.integer(max.splicing.iter),
    exchange_num = as.integer(c.max),
    path_type = tune_path,
    is_warm_start = warm.start,
    ic_type = as.integer(ic_type),
    ic_coef = as.double(ic.scale),
    is_cv = is_cv,
    Kfold = nfolds,
    status = c(0),
    sequence = support_size,
    lambda_seq = lambda,
    s_min = as.integer(0),
    s_max = as.integer(0),
    K_max = as.integer(0),
    epsilon = as.double(0.0001),
    lambda_min = as.double(0),
    lambda_max = as.double(0),
    nlambda = as.integer(0),
    is_screening = FALSE,
    screening_size = -1,
    powell_path = as.integer(1),
    g_index = g_index,
    always_select = always_include,
    tau = 0,
    primary_model_fit_max_iter = as.integer(max.newton.iter),
    primary_model_fit_epsilon = as.double(newton.thresh),
    early_stop = FALSE,
    approximate_Newton = approximate_newton,
    thread = num.threads,
    covariance_update = FALSE,
    sparse_matrix = FALSE,
    splicing_type = as.integer(splicing.type),
    sub_search = as.integer(0),
    cv_fold_id = cv_fold_id
  )
  
  omega <- lapply(result[["beta_all"]], recovery_adjacent_matrix, p = nvars)
  omega <- simplify2array(omega)
  
  if (is_cv) {
    names(result)[which(names(result) == "test_loss_all")] <- "tune.value"
    result[["ic_all"]] <- NULL
  } else {
    names(result)[which(names(result) == "ic_all")] <- "tune.value"
    result[["test_loss_all"]] <- NULL
  }
  
  return(list(omega = omega, 
              support.size = support_size, 
              pseudo.loglik = result[["train_loss_all"]], 
              tune.value = result[["tune.value"]], 
              nobs = nobs, 
              nvars = nvars, 
              tune.type = tune.type))
}

recovery_adjacent_matrix <- function(x, p) {
  zero_mat <- matrix(data = 0, nrow = p, ncol = p)
  # zero_mat[lower.tri(zero_mat)] <- x
  # zero_mat <- zero_mat + t(zero_mat)
  # diag(zero_mat) <- diag(zero_mat) / 2
  i <- 1
  j <- 1
  for (k in 1:as.integer(p * (p - 1) / 2)) {
    if (i == j) {
      i <- 1
      j <- j + 1
    }
    zero_mat[j, i] <- zero_mat[i, j] <- x[k]
    i <- i + 1
  }
  zero_mat
}
