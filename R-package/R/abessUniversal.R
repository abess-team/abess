#' @title test abessUniversal
#' @export
abessUniversal <- function(
                          f,
                          model_size,
                          sample_size,
                          x,
                          y,
                          family = c(
                            "gaussian", "binomial", "poisson", "cox",
                            "mgaussian", "multinomial", "gamma","ordinal"
                          ),
                          tune.path = c("sequence", "gsection"),
                          tune.type = c("gic", "ebic", "bic", "aic", "cv"),
                          weight = NULL,
                          normalize = NULL,
                          c.max = 2,
                          support.size = NULL,
                          gs.range = NULL,
                          lambda = 0,
                          always.include = NULL,
                          group.index = NULL,
                          init.active.set = NULL, 
                          splicing.type = 2,
                          max.splicing.iter = 20,
                          screening.num = NULL,
                          important.search = NULL,
                          warm.start = TRUE,
                          nfolds = 5,
                          foldid = NULL,
                          cov.update = FALSE,
                          newton = c("exact", "approx"),
                          newton.thresh = 1e-6,
                          max.newton.iter = NULL,
                          early.stop = FALSE,
                          ic.scale = 1.0,
                          num.threads = 0,
                          seed = 1,
                          ...) {
  set.seed(seed)
  family <- match.arg(family)
  tune.path <- match.arg(tune.path)
  tune.type <- match.arg(tune.type)
  
  data <- list(
    x = x,
    y = y
  )

  para <- Initialization_GLM(
    c.max=c.max,
    support.size=support.size,
    always.include=always.include,
    group.index=group.index,
    splicing.type=splicing.type,
    max.splicing.iter=max.splicing.iter,
    warm.start=warm.start,
    ic.scale=ic.scale,
    num.threads=num.threads,
    newton.thresh=newton.thresh,
    tune.type=tune.type,
    important.search=important.search,
    tune.path=tune.path,
    max.newton.iter=max.newton.iter,
    lambda=lambda,
    family=family,
    screening.num=screening.num,
    gs.range=gs.range,
    early.stop=early.stop,
    weight=weight,
    cov.update=cov.update,
    normalize=normalize,
    init.active.set=init.active.set,
    newton=newton,
    foldid=foldid,
    nfolds=nfolds
  )

  model <- initializate(para,data)
  para <- model$para
  data <- model$data
  
  y <- data$y
  x <- data$x
  tune.path <- para$tune.path
  lambda <- para$lambda
  family <- para$family
  gs.range <- para$gs.range
  weight <- para$weight
  normalize <- para$normalize
  init.active.set <- para$init.active.set
  nfolds <- para$nfolds
  warm.start <- para$warm.start
  num_threads  <- para$num_threads 
  splicing_type  <- para$splicing_type 
  max_splicing_iter <- para$max_splicing_iter
  nobs <- para$nobs
  nvars <- para$nvars
  vn <- para$vn
  sparse_X <- para$sparse_X
  screening_num <- para$screening_num
  g_index <- para$g_index
  s_list <- para$s_list
  c_max <- para$c_max
  ic_scale <- para$ic_scale
  important_search <- para$important_search
  always_include   <- para$always_include  
  max_newton_iter   <- para$max_newton_iter  
  path_type <- para$path_type
  newton_thresh <- para$newton_thresh
  screening <- para$screening
  ic_type <- para$ic_type
  is_cv <- para$is_cv
  cv_fold_id <- para$cv_fold_id
  s_min <- para$s_min
  s_max <- para$s_max
  model_type <- para$model_type
  covariance_update <- para$covariance_update
  approximate_newton <- para$approximate_newton
  y_vn <- para$y_vn
  y_dim <- para$y_dim
  multi_y <- para$multi_y
  early_stop <- para$early_stop
  
  result <- abessUniversal_API(
    s_function = f,
    model_size = model_size,
    sample_size = sample_size,
    max_iter = max_splicing_iter,
    exchange_num = c_max,
    path_type = path_type,
    is_warm_start = warm.start,
    ic_type = ic_type,
    ic_coef = ic_scale,
    sequence = as.vector(s_list),
    lambda_seq = lambda,
    s_min = s_min,
    s_max = s_max,
    screening_size = ifelse(screening_num >= nvars, -1, screening_num),
    g_index = g_index,
    always_select = always_include,
    primary_model_fit_max_iter = max_newton_iter,
    primary_model_fit_epsilon = newton_thresh,
    early_stop = early_stop,
    thread = num_threads,
    splicing_type = splicing_type,
    sub_search = important_search,
    A_init = as.integer(init.active.set)
  )

  ## process result
  result[["beta"]] <- NULL
  result[["coef0"]] <- NULL
  result[["train_loss"]] <- NULL
  result[["ic"]] <- NULL
  result[["lambda"]] <- NULL

  result[["nobs"]] <- nobs
  result[["nvars"]] <- nvars
  result[["family"]] <- family
  result[["tune.path"]] <- tune.path
  result[["tune.type"]] <- ifelse(is_cv == TRUE, "cv",
    c("AIC", "BIC", "GIC", "EBIC")[ic_type]
  )
  result[["gs.range"]] <- gs.range

  ## preprocessing result in "gsection"
  if (tune.path == "gsection") {
    ## change the order:
    reserve_order <- rev(seq_len(length(result[["sequence"]])))
    result[["beta_all"]] <- result[["beta_all"]][reserve_order]
    if (is.matrix(result[["coef0_all"]])) {
      result[["coef0_all"]] <- result[["coef0_all"]][reserve_order, , drop = FALSE]
    } else {
      result[["coef0_all"]] <- as.matrix(result[["coef0_all"]][reserve_order])
    }
    result[["train_loss_all"]] <- result[["train_loss_all"]][reserve_order, , drop = FALSE]
    result[["ic_all"]] <- result[["ic_all"]][reserve_order, , drop = FALSE]
    result[["test_loss_all"]] <- result[["test_loss_all"]][reserve_order, , drop = FALSE]
    result[["sequence"]] <- result[["sequence"]][reserve_order]
    gs_unique_index <- match(sort(unique(result[["sequence"]])), result[["sequence"]])

    ## remove replicate support size:
    result[["beta_all"]] <- result[["beta_all"]][gs_unique_index]
    result[["coef0_all"]] <- result[["coef0_all"]][gs_unique_index, , drop = FALSE]
    result[["train_loss_all"]] <- result[["train_loss_all"]][gs_unique_index, , drop = FALSE]
    result[["ic_all"]] <- result[["ic_all"]][gs_unique_index, , drop = FALSE]
    result[["test_loss_all"]] <- result[["test_loss_all"]][gs_unique_index, , drop = FALSE]
    result[["sequence"]] <- result[["sequence"]][gs_unique_index]
    result[["support.size"]] <- result[["sequence"]]
    s_list <- result[["support.size"]]
    result[["sequence"]] <- NULL
  } else {
    result[["support.size"]] <- s_list
  }

  result[["edf"]] <- result[["effective_number_all"]][, 1]
  result[["effective_number_all"]] <- NULL
  names(result)[which(names(result) == "train_loss_all")] <- "dev"
  result[["dev"]] <- result[["dev"]][, 1]
  if (is_cv) {
    names(result)[which(names(result) == "test_loss_all")] <- "tune.value"
    result[["ic_all"]] <- NULL
  } else {
    names(result)[which(names(result) == "ic_all")] <- "tune.value"
    result[["test_loss_all"]] <- NULL
  }
  result[["tune.value"]] <- result[["tune.value"]][, 1]

  ############ restore intercept ############
  result[["best.size"]] <- s_list[which.min(result[["tune.value"]])]
  names(result)[which(names(result) == "coef0_all")] <- "intercept"
  if (family %in% MULTIVARIATE_RESPONSE) {
    if (family %in% c("multinomial", "ordinal")) {
      result[["intercept"]] <- lapply(result[["intercept"]], function(x) {
        x <- x[-y_dim]
      })
    }
  } else {
    result[["intercept"]] <- as.vector(result[["intercept"]])
  }

  ############ restore intercept ############
  names(result)[which(names(result) == "beta_all")] <- "beta"
  if (multi_y) {
    if (screening) {
      for (i in seq_len(length(result[["beta"]]))) {
        beta_all <- matrix(0, nrow = nvars, ncol = y_dim)
        beta_all[result[["screening_A"]] + 1, ] <- result[["beta"]][[i]]
        result[["beta"]][[i]] <- beta_all
      }
    }
    names(result[["beta"]]) <- as.character(s_list)
    if (family == "mgaussian") {
      result[["beta"]] <- lapply(result[["beta"]], Matrix::Matrix,
        sparse = TRUE, dimnames = list(vn, y_vn)
      )
    } else if (family %in% c("multinomial", "ordinal")) {
      result[["beta"]] <- lapply(result[["beta"]], function(x) {
        Matrix::Matrix(x[, -y_dim], sparse = TRUE, dimnames = list(vn, y_vn[-1]))
      })
    }
  } else {
    result[["beta"]] <- do.call("cbind", result[["beta"]])
    if (screening) {
      beta_all <- matrix(0,
        nrow = nvars,
        ncol = length(s_list)
      )
      beta_all[result[["screening_A"]] + 1, ] <- result[["beta"]]
      result[["beta"]] <- beta_all
    }
    result[["beta"]] <- Matrix::Matrix(result[["beta"]],
      sparse = TRUE,
      dimnames = list(vn, as.character(s_list))
    )
  }

  result[["screening.vars"]] <- vn[result[["screening_A"]] + 1]
  result[["screening_A"]] <- NULL

  result[["call"]] <- match.call()
  class(result) <- "abess"

  set.seed(NULL)
  return(result)
}

