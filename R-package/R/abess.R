#' @title Adaptive Best-Subset Selection via splicing algorithm
#'
#' @description Performs the nonparametric two-sample or \eqn{K}-sample Ball Divergence test for
#' equality of multivariate distributions
#'
#' @aliases abess
#'
#' @author Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang
#'
#' @param x Input matrix, of dimension \eqn{n \times p}; each row is an observation
#' vector and each column is a predictor/feature/variable.
#' @param y The response variable, of \code{n} observations. For \code{family = "binomial"} should be
#' a factor with two levels. For \code{family="poisson"}, \code{y} should be a vector with positive integer.
#'  For \code{family = "cox"}, \code{y} should be a two-column matrix
#' with columns named \code{time} and \code{status}.
#' @param type One of the two types of problems.
#' \code{type = "bss"} for the best subset selection,
#' and \code{type = "bsrr"} for the best subset ridge regression.
#' @param family One of the following models: \code{"gaussian"}, \code{"binomial"},
#' \code{"poisson"}, or \code{"cox"}. Depending on the response. Any unambiguous substring can be given.
#' @param method The method to be used to select the optimal model size and \eqn{L_2} shrinkage. For
#' \code{method = "sequential"}, we solve the best subset selection and the best subset ridge regression
#' problem for each \code{s} in \code{1,2,...,s.max} and \eqn{\lambda} in \code{lambda.list}. For \code{method =
#' "gsection"}, which is only valid for \code{type = "bss"},
#' we solve the best subset selection problem with model size ranged between s.min and s.max,
#' where the specific model size to be considered is determined by golden section. we
#' solve the best subset selection problem with a range of non-continuous model
#' sizes. For \code{method = "pgsection"} and \code{"psequential"}, the Powell method is used to
#' solve the best subset ridge regression problem. Any unambiguous substring can be given.
#' @param tune The criterion for choosing the model size and \eqn{L_2} shrinkage
#' parameters. Available options are \code{"gic"}, \code{"ebic"}, \code{"bic"}, \code{"aic"} and \code{"cv"}.
#' Default is \code{"gic"}.
#' @param support.size An increasing list of sequential values representing the model
#' sizes. Only used for \code{method = "sequential"}. Default is \code{1:min(p,
#' round(n/log(n)))}.
#' @param lambda.list A lambda sequence for \code{"bsrr"}. Default is
#' \code{exp(seq(log(100), log(0.01), length.out = 100))}.
#' @param s.min The minimum value of model sizes. Only used for \code{method =
#' "gsection"}, \code{"psequential"} and \code{"pgsection"}. Default is 1.
#' @param s.max The maximum value of model sizes. Only used for \code{method =
#' "gsection"}, \code{"psequential"} and \code{"pgsection"}. Default is \code{min(p, round(n/log(n)))}.
#' @param lambda.min The minimum value of lambda. Only used for \code{method =
#' "powell"}. Default is \code{0.001}.
#' @param lambda.max The maximum value of lambda. Only used for \code{method =
#' "powell"}. Default is \code{100}.
#' @param nlambda The number of \eqn{\lambda}s for the Powell path with sequential line search method.
#' Only valid for \code{method = "psequential"}.
#' @param always.include An integer vector containing the indexes of variables that should always be included in the model.
#' @param screening.num Users can pre-exclude some irrelevant variables according to maximum marginal likelihood estimators before fitting a
#' model by passing an integer to \code{screening.num} and the sure independence screening will choose a set of variables of this size.
#' Then the active set updates are restricted on this subset.
#' @param normalize Options for normalization. \code{normalize = 0} for
#' no normalization. Setting \code{normalize = 1} will
#' only subtract the mean of columns of \code{x}.
#' \code{normalize = 2} for scaling the columns of \code{x} to have \eqn{\sqrt n} norm.
#' \code{normalize = 3} for subtracting the means of the columns of \code{x} and \code{y}, and also
#' normalizing the columns of \code{x} to have \eqn{\sqrt n} norm.
#' If \code{normalize = NULL}, by default, \code{normalize} will be set \code{1} for \code{"gaussian"},
#' \code{2} for \code{"binomial"} and \code{"poisson"}, \code{3} for \code{"cox"}.
#' @param weight Observation weights. Default is \code{1} for each observation.
#' @param max.iter The maximum number of iterations in the bess function.
#' In most of the case, only a few steps can guarantee the convergence. Default
#' is \code{20}.
#' @param warm.start Whether to use the last solution as a warm start. Default
#' is \code{TRUE}.
#' @param nfolds The number of folds in cross-validation. Default is \code{5}.
#' @param group.index A vector of integers indicating the which group each variable is in.
#' For variables in the same group, they should be located in adjacent columns of \code{x}
#' and their corresponding index in \code{group.index} should be the same.
#' Denote the first group as \code{1}, the second \code{2}, etc.
#' If you do not fit a model with a group structure,
#' please set \code{group.index = NULL}. Default is \code{NULL}.
#' @param seed Seed to be used to devide the sample into K cross-validation folds. Default is \code{NULL}.
#' @param ... further arguments to be passed to or from methods.
#'
#' @return A list with class attribute 'bess' and named components:
#' \item{beta}{The best fitting coefficients.}
#' \item{coef0}{The best fitting
#' intercept.}
#' \item{bestmodel}{The best fitted model for \code{type = "bss"}, the class of which is \code{"lm"}, \code{"glm"} or \code{"coxph"}.}
#' \item{loss}{The training loss of the best fitting model.}
#' \item{ic}{The information criterion of the best fitting model when model
#' selection is based on a certain information criterion.} \item{cvm}{The mean
#' cross-validated error for the best fitting model when model selection is
#' based on the cross-validation.}
#' \item{lambda}{The lambda chosen for the best fitting model}
#' \item{beta.all}{For \code{bess} objects obtained by \code{gsection}, \code{pgsection}
#' and \code{psequential}, \code{beta.all} is a matrix with each column be the coefficients
#' of the model in each iterative step in the tuning path.
#' For \code{bess} objects obtained by \code{sequential} method,
#' A list of the best fitting coefficients of size
#' \code{s=0,1,...,p} and \eqn{\lambda} in \code{lambda.list} with the
#' smallest loss function. For \code{"bess"} objects of \code{"bsrr"} type, the fitting coefficients of the
#' \eqn{i^{th} \lambda} and the \eqn{j^{th}} \code{s} are at the \eqn{i^{th}}
#' list component's \eqn{j^{th}} column.}
#' \item{coef0.all}{For \code{bess} objects obtained from \code{gsection}, \code{pgsection} and \code{psequential},
#' \code{coef0.all} contains the intercept for the model in each iterative step in the tuning path.
#' For \code{bess} objects obtained from \code{sequential} path,
#' \code{coef0.all} contains the best fitting
#' intercepts of size \eqn{s=0,1,\dots,p} and \eqn{\lambda} in
#' \code{lambda.list} with the smallest loss function.}
#' \item{loss.all}{For \code{bess} objects obtained from \code{gsection}, \code{pgsection} and \code{psequential},
#' \code{loss.all} contains the training loss of the model in each iterative step in the tuning path.
#' For \code{bess} objects obtained from \code{sequential} path, this is a
#' list of the training loss of the best fitting intercepts of model size
#' \eqn{s=0,1,\dots,p} and \eqn{\lambda} in \code{lambda.list}. For \code{"bess"} object obtained by \code{"bsrr"},
#' the training loss of the \eqn{i^{th} \lambda} and the \eqn{j^{th}} \code{s}
#' is at the \eqn{i^{th}} list component's \eqn{j^{th}} entry.}
#' \item{ic.all}{For \code{bess} objects obtained from \code{gsection}, \code{pgsection} and \code{psequential},
#' \code{ic.all} contains the values of the chosen information criterion of the model in each iterative step in the tuning path.
#' For \code{bess} objects obtained from \code{sequential} path, this is a
#' matrix of the values of the chosen information criterion of model size \eqn{s=0,1,\dots,p}
#' and \eqn{\lambda} in \code{lambda.list} with the smallest loss function. For \code{"bess"} object obtained by \code{"bsrr"},
#' the training loss of the \eqn{i^{th} \lambda} and the \eqn{j^{th}}
#' \code{s} is at the \eqn{i^{th}} row \eqn{j^{th}} column. Only available when
#' model selection is based on a certain information criterion.}
#'
#' \item{cvm.all}{For \code{bess} objects obtained from \code{gsection}, \code{pgsection} and \code{psequential},
#' \code{cvm.all} contains the mean cross-validation error of the model in each iterative step in the tuning path.
#' For \code{bess} objects obtained from \code{sequential} path, this is a
#'  matrix of the mean cross-validation error of model size
#' \eqn{s=0,1,\dots,p} and \eqn{\lambda} in \code{lambda.list} with the
#' smallest loss function. For \code{"bess"} object obtained by \code{"bsrr"}, the training loss of the \eqn{i^{th}
#' \lambda} and the \eqn{j^{th}} \code{s} is at the \eqn{i^{th}} row
#' \eqn{j^{th}} column. Only available when model selection is based on the
#' cross-validation.}
#' \item{lambda.all}{The lambda chosen for each step in \code{pgsection} and \code{psequential}.}
#' \item{family}{Type of the model.}
#' \item{support.size}{The input
#' \code{support.size}.} \item{nsample}{The sample size.}
#' \item{type}{Either \code{"bss"} or \code{"bsrr"}.}
#' \item{method}{Method used for tuning parameters selection.}
#' \item{ic.type}{The criterion of model selection.}
#'
#' @seealso
#' \code{\link{bess.fix}}
#'
#' @references A polynomial algorithm for best-subset selection problem. Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang. Proceedings of the National Academy of Sciences Dec 2020, 117 (52) 33117-33123; DOI: 10.1073/pnas.2014241117
#'
#' @export
#' @examples
#' 
#' n <- 200
#' p <- 300
#' support.size <- 3
#' #-------------------linear model----------------------#
#' dataset <- generate.data(n, p, support.size)
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]])
#' abess_fit[["best.model"]]
#' 
#' #-------------------logistic model----------------------#
#' dataset <- generate.data(n, p, support.size, family = "binomial")
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]], family = "binomial")
#' abess_fit[["best.model"]]
#' 
abess <- function(x, ...) UseMethod("abess")


#' @rdname abess
#' @export
#' @method abess default
abess.default <- function(x, y,
                  family = c("gaussian", "binomial"),
                  method = c("sequential", "gsection"),
                  tune.type = c("gic", "ebic", "bic", "aic", "cv"),
                  weight = rep(1, nrow(x)),
                  normalize = NULL,
                  c.max = 2,
                  support.size = NULL,
                  gs.range = NULL, 
                  always.include = NULL,
                  max.splicing.iter = 20,
                  screening.num = NULL,
                  group.index = NULL,
                  warm.start = TRUE,
                  nfolds = 5,
                  newton = c("exact", "approx"), 
                  newton.thresh = 1e-6, 
                  max.newton.iter = NULL, 
                  early.stop = FALSE, 
                  num.threads = 0, 
                  seed = 1, 
                  ...)
{
  tau <- NULL
  
  ## TODO:
  type <- c("bss", "bsrr")
  # type <- match.arg(type)
  type <- type[1]
  algorithm_type = switch(type,
                          "bss" = "GPDAS",
                          "bsrr" = "GL0L2")
  
  lambda.list <- 0
  lambda.min <- 0.001
  lambda.max <- 100
  nlambda <- 100
  
  set.seed(seed)
  
  ## check number of thread:
  stopifnot(is.numeric(num.threads) & num.threads >= 0)
  num_threads <- as.integer(num.threads)
  
  ## check early stop:
  stopifnot(is.logical(early.stop))
  early_stop <- early.stop
  
  ## check warm start:
  stopifnot(is.logical(warm.start))
  
  ## check max splicing iteration
  stopifnot(is.numeric(max.splicing.iter) & max.splicing.iter >= 1)
  max_splicing_iter <- as.integer(max.splicing.iter)
  
  ## task type:
  family <- match.arg(family)
  model_type <- switch(
    family,
    "gaussian" = 1,
    "binomial" = 2,
    "poisson" = 3,
    "cox" = 4
  )
  
  ## check predictors:
  # if (anyNA(x)) {
  #   stop("x has missing value!")
  # }
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  nvars <- ncol(x)
  nobs <- nrow(x)
  if (nvars == 1) {
    stop("x should have two columns at least!")
  }
  vn <- colnames(x)
  if (is.null(vn)) {
    vn <- paste("x", 1:nvars, sep = "")
  }
  
  ## check C-max:
  stopifnot(is.numeric(c.max) & c.max >= 1)
  if (c.max >= nvars) {
    stop("c.max should smaller than the number of predictors!")
  }
  c_max <- as.integer(c.max)
  
  ## check response:
  if (anyNA(y)) {
    stop("y has missing value!")
  }
  if (family == "binomial")
  {
    if (is.factor(y)) {
      y <- as.character(y)
    }
    if (length(unique(y)) != 2) {
      stop("Please input binary variable!")
    } else if (setequal(y_names <- unique(y), c(0, 1)) == FALSE)
    {
      y[which(y == unique(y)[1])] = 0
      y[which(y == unique(y)[2])] = 1
      y <- as.numeric(y)
    }
  }
  if (family == "cox")
  {
    if (!is.matrix(y)) {
      y <- as.matrix(y)
    }
    if (ncol(y) != 2) {
      stop("Please input y with two columns!")
    }
    ## pre-process data for cox model
    sort_y <- order(y[, 1])
    y <- y[sort_y, ]
    x <- x[sort_y, ]
    y <- y[, 2]
  }
  
  # check whether x and y are matching:
  if (is.vector(y)) {
    if (nrow(x) != length(y))
      stop("Rows of x must be the same as length of y!")
  } else {
    if (nrow(x) != nrow(y))
      stop("Rows of x must be the same as rows of y!")
  }
  
  # check weight
  stopifnot(is.vector(weight))
  if (length(weight) != nobs) {
    stop("Rows of x must be the same as length of weight!")
  }
  stopifnot(all(is.numeric(weight)), all(weight >= 0))
  
  ## check parameters for sub-optimization:
  # 1:
  newton <- match.arg(newton)
  newton_type <- switch(
    newton,
    "exact" = 0,
    "approx" = 1,
    "auto" = 2
  )
  approximate_newton <- ifelse(newton_type == 1, TRUE, FALSE)
  # 2:
  if (!is.null(max.newton.iter)) {
    stopifnot(is.numeric(max.newton.iter) & max.newton.iter >= 1)
    max_newton_iter <- as.integer(max.newton.iter)
  } else {
    max_newton_iter <- ifelse(newton_type == 0, 10, 60)
  }
  # 3:
  stopifnot(is.numeric(newton.thresh) & newton.thresh > 0)
  newton_thresh <- as.double(newton.thresh)
  
  # sparse level list (sequential):
  if (is.null(support.size)) {
    s_list <- 1:min(c(nvars, round(nobs / log(log(nobs)) / log(p))))
  } else {
    stopifnot(any(is.numeric(support.size) & support.size > 0))
    stopifnot(max(support.size) > nvars)
    support.size <- sort(support.size)
    support.size <- unique(support.size)
    s_list <- support.size
  }
  
  # sparse range (golden-section):
  if (is.null(gs.range)) {
    s_min <- 1
    s_max <- min(c(nvars, round(nobs / log(log(nobs)) / log(p))))
  } else {
    stopifnot(length(gs.range) == 2)
    stopifnot(any(is.numeric(gs.range) & gs.range > 0))
    stopifnot(as.integer(gs.range)[1] != as.integer(gs.range)[2])
    stopifnot(as.integer(max(gs.range)) <= nvars)
    gs.range <- as.integer(gs.range)
    s_min <- min(gs.range)
    s_max <- max(gs.range)
  }
  
  # tune model size method:
  tune.type <- match.arg(tune.type)
  ic_type <- switch(
    tune.type,
    "aic" = 1,
    "bic" = 2,
    "gic" = 3,
    "ebic" = 4,
    "cv" = 1
  )
  is_cv <- ifelse(tune.type == "cv", TRUE, FALSE)
  if (is_cv) {
    stopifnot(is.numeric(nfolds) & nfolds >= 2)
    nfolds <- as.integer(nfolds)
  }
  
  ## strategy for tunning
  method <- match.arg(method)
  if (method == "pgsection") {
    path_type <- 2
  } else if (method == "psequential") {
    path_type <- 2
  } else if (method == "sequential") {
    path_type <- 1
  } else {
    path_type <- 2
  }
  
  ## group variable:
  if (!is.null(group.index)) {
    if (path_type == 1 & max(support.size) > length(group.index))
      stop("The maximum one support.size should not be larger than the number of groups!")
    if (path_type == 2 & s.max > length(group.index))
      stop("s.max is too large. Should be smaller than the number of groups!")
    
    gi <- unique(group.index)
    g_index <- match(gi, group.index) - 1
    g_df <- c(diff(g_index), 
              length(group.index) - g_index[length(g_index)])
  } else {
    g_index <- 1:nvars - 1
  }
  
  ## normalize strategy: 
  if (is.null(normalize)) {
    is_normal <- TRUE
    normalize <- switch(
      family,
      "gaussian" = 1,
      "binomial" = 2,
      "poisson" = 2,
      "cox" = 3
    )
  } else {
    stopifnot(normalize %in% 0:3)
    if (normalize != 0) {
      # normalize <- as.character(normalize)
      # normalize <- switch (normalize,
      #                      '1' <- 2,
      #                      '2' <- 3,
      #                      '3' <- 1
      # )
      if (normalize == 1) {
        normalize <- 2
      } else if (normalize == 2) {
        normalize <- 3
      } else if (normalize == 3) {
        normalize <- 1
      } else {
      }
      is_normal <- TRUE
    } else {
      is_normal <- FALSE
      normalize <- 0
    }
  }

  if (is.null(screening.num)) {
    screening <- FALSE
    screening_num <- nvars
  } else {
    stopifnot(is.numeric(screening.num))
    stopifnot(screening.num >= 1)
    screening.num <- as.integer(screening.num)
    if (screening.num > nvars)
      stop("The number of screening features must be equal or less than that of the column of x!")
    if (path_type == 1) {
      if (screening.num < max(support.size))
        stop(
          "The number of screening features must be equal or greater than the maximum one in support.size!"
        )
    } else{
      if (screening.num < s.max)
        stop("The number of screening features must be equal or greater than the s.max!")
    }
    screening <- TRUE
    screening_num <- screening.num
  }
  
  # check always included varibles:
  if (is.null(always.include)) {
    always_include <- numeric(0)
  } else {
    if (anyNA(always.include)) {
      stop("always.include has missing values.")
    }
    if (any(always.include <= 0)) {
      stop("always.include should be an vector containing variable indexes which is positive.")
    }
    always.include <- as.integer(always.include) - 1
    if (length(always.include) > screening.num)
      stop("The number of variables in always.include should not exceed the screening.num")
    if (path_type == 1) {
      if (length(always.include) > max(support.size))
        stop("always.include containing too many variables. 
             The length of it should not exceed the maximum in support.size.")
    } else{
      if (length(always.include) > s.max)
        stop(
          "always.include containing too many variables. The length of it should not exceed the s.max."
        )
    }
    always_include <- always.include
  }
  
  result <- abessCpp(
    x = x,
    y = y,
    data_type = normalize,
    weight = weight,
    is_normal = is_normal,
    algorithm_type = 6,
    model_type = model_type,
    max_iter = max_splicing_iter,
    exchange_num = c_max,
    path_type = path_type,
    is_warm_start = warm.start,
    ic_type = ic_type,
    ic_coef = 1.0,
    is_cv = is_cv,
    Kfold = nfolds,
    state = rep(2, 10),
    sequence = s_list,
    lambda_seq = 0,
    s_min = s_min,
    s_max = s_max,
    K_max = as.integer(20),
    epsilon = 0,
    lambda_max = 0,
    lambda_min = 0,
    nlambda = 10,
    is_screening = screening,
    screening_size = screening_num,
    powell_path = 1,
    g_index = g_index,
    always_select = always_include,
    tau = 0.0,
    primary_model_fit_max_iter = max_newton_iter,
    primary_model_fit_epsilon = newton_thresh,
    early_stop = early_stop,
    approximate_Newton = approximate_newton,
    thread = num_threads
  )
  names(result[["beta"]]) <- vn
  best_model <- list("beta" = result[["beta"]], 
                    "coef0" = result[["coef0"]], 
                    "support.size" = sum(result[["beta"]] != 0.0), 
                    "dev" = result[["train_loss"]], 
                    "tune.value" = result[["ic"]])
  result[["beta"]] <- NULL
  result[["coef0"]] <- NULL
  result[["train_loss"]] <- NULL
  result[["ic"]] <- NULL
  result[["lambda"]] <- NULL
  result[["best.model"]] <- best_model
  
  # names(result)[which(names(result) == "train_loss_all")] <- "dev"
  names(result)[which(names(result) == 'ic_all')] <- 'tune.value'
  names(result)[which(names(result) == "coef0_all")] <- "coef0"
  names(result)[which(names(result) == 'beta_all')] <- "beta"
  result[["beta"]] <- do.call("cbind", result[["beta"]])
  rownames(result[["beta"]]) <- vn
  
  result[["family"]] <- family
  result[["tune.path"]] <- method
  result[["support.size"]] <- s_list
  result[["tune.type"]] <- ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "EBIC")[ic_type])
  result[["gs.range"]] <- gs.range
  result[["screening.index"]] <- result[["screening_A"]] + 1
  
  result[["call"]] <- match.call()
  class(result) <- "abess"
  
  result[["beta"]] <- recover(result, sparse = FALSE)
    
  set.seed(NULL)
  
  return(result)
}

#' @rdname abess
#'
#' @param formula an object of class "\code{formula}": 
#' a symbolic description of the model to be fitted. 
#' The details of model specification are given in the "Details" section of "\code{\link{formula}}".
#' @param data a data frame containing the variables in the \code{formula}. 
#' @param subset an optional vector specifying a subset of observations to be used.
#' @param na.action a function which indicates 
#' what should happen when the data contain \code{NA}s. 
#' Defaults to \code{getOption("na.action")}.
#' @method abess formula
#' @export
#' @examples
#' 
#' ################  Formula interface  ################
#' data("trim32")
#' abess_fit <- abess(y ~ ., data = trim32)
#' abess_fit
abess.formula <- function(formula, data, subset, na.action, ...) {
  contrasts <- NULL   ## for sparse X matrix
  cl <- match.call()
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "na.action"), 
             names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  
  y <- model.response(mf, "numeric")
  x <- model.matrix(mt, mf, contrasts)[, -1]
  
  abess_res <- abess.default(x, y, ...)
  abess_res
}
