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
#' @param s.list An increasing list of sequential values representing the model
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
#' \item{s.list}{The input
#' \code{s.list}.} \item{nsample}{The sample size.}
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
#' #-------------------linear model----------------------#
#' # Generate simulated data
#' n <- 200
#' p <- 20
#' k <- 5
#' rho <- 0.4
#' seed <- 10
#' Tbeta <- rep(0, p)
#' Tbeta[1:k*floor(p/k):floor(p/k)] <- rep(1, k)
#' Data <- gen.data(n, p, k, rho, family = "gaussian", beta = Tbeta, seed = seed)
#' x <- Data$x[1:140, ]
#' y <- Data$y[1:140]
#' x_new <- Data$x[141:200, ]
#' y_new <- Data$y[141:200]
#' lm.bss <- bess(x, y)
#' lambda.list <- exp(seq(log(5), log(0.1), length.out = 10))
#' lm.bsrr <- bess(x, y, type = "bsrr", method = "pgsection")
#' coef(lm.bss)
#' coef(lm.bsrr)
#' print(lm.bss)
#' print(lm.bsrr)
#' summary(lm.bss)
#' summary(lm.bsrr)
#' pred.bss <- predict(lm.bss, newx = x_new)
#' pred.bsrr <- predict(lm.bsrr, newx = x_new)
#'
#' # generate plots
#' plot(lm.bss, type = "both", breaks = TRUE)
#' plot(lm.bsrr)
#' #-------------------logistic model----------------------#
#' #Generate simulated data
#' Data <- gen.data(n, p, k, rho, family = "binomial", beta = Tbeta, seed = seed)
#' logi.bss <- bess(x, y, family = "binomial")
#' coef(logi.bss)
#' print(logi.bss)
#' summary(logi.bss)
#' pred.bss <- predict(logi.bss, newx = x_new)
abess <- function(x, ...)
  UseMethod("abess")


#' @rdname abess
#' @export
#' @method abess default
bess <- function(x,
                 y,
                 family = c("gaussian", "binomial"),
                 method = c("gsection", "sequential"),
                 tune = c("gic", "ebic", "bic", "aic", "cv"),
                 weight = rep(1, nrow(x)),
                 normalize = NULL,
                 c.max = 2,
                 tau = NULL,
                 s.list = NULL,
                 gs.range = NULL, 
                 always.include = NULL,
                 max.iter = 20,
                 screening.num = NULL,
                 group.index = NULL,
                 warm.start = TRUE,
                 nfolds = 5,
                 newton = c("auto", "exact", "approx"), 
                 newton.thresh = , 
                 max.newton.iter = 50, 
                 early.stop = FALSE, 
                 seed = 1, 
                 ...)
{
  type <- c("bss", "bsrr")
  type <- match.arg(type)
  algorithm_type = switch(type,
                          "bss" = "GPDAS",
                          "bsrr" = "GL0L2")
  
  lambda.list <- 0
  lambda.min <- 0.001
  lambda.max <- 100
  nlambda <- 100
  
  set.seed(seed)
  
  ## check predictors:
  # if (anyNA(x)) {
  #   stop("x has missing value!")
  # }
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  if (ncol(x) == 1) {
    stop("x should have two columns at least!")
  }
  vn <- colnames(x)
  if (is.null(vn)) {
    vn <- paste("X", 1:ncol(x), sep = "")
  }
  
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
  }
  
  # check whether x and y are matching:
  if (is.vector(y)) {
    if (nrow(x) != length(y))
      stop("Rows of x must be the same as length of y!")
  } else {
    if (nrow(x) != nrow(y))
      stop("Rows of x must be the same as rows of y!")
  }
  
  ## task type:
  family <- match.arg(family)
  model_type <- switch(
    family,
    "gaussian" = 1,
    "binomial" = 2,
    "poisson" = 3,
    "cox" = 4
  )
  
  ## check parameters for sub-optimization:
  newton <- match.arg(newton)
  newton_type <- switch(
    newton,
    "exact" = 0,
    "approx" = 1,
    "auto" = 2
  )
  approximate_Newton <- ifelse(newton_type == 1, TRUE, FALSE)
  stopifnot(max.newton.iter > 0 & is.integer(max.newton.iter))
  stopifnot(newton.thresh > 0)
  
  # sparse level:
  if (is.null(s.list)) {
    s.list <- 1:min(ncol(x), round(nrow(x) / log(nrow(x))))
  }
  if (is.null(gs.range)) {
    s.min <- 1
    s.max <- min(ncol(x), round(nrow(x) / log(nrow(x))))
  }
  
  # tune model size method:
  tune <- match.arg(tune)
  ic_type <- switch(
    tune,
    "aic" = 1,
    "bic" = 2,
    "gic" = 3,
    "ebic" = 4,
    "cv" = 1
  )
  is_cv <- ifelse(tune == "cv", TRUE, FALSE)
  if (is_cv) {
    stopifnot(is.integer(nfolds) & nfolds > 1)
  }
  
  ## strategy for tunning
  method <- match.arg(method)
  if (method == "pgsection") {
    path_type <- 2
    line.search <- 1
  } else if (method == "psequential") {
    path_type <- 2
    line.search <- 2
  } else if (method == "sequential") {
    path_type <- 1
    line.search <- 1
  } else{
    path_type <- 2
    line.search <- 1
  }
  
  ## group variable:
  if (!is.null(group.index)) {
    if (path_type == 1 &
        s.list[length(s.list)] > length(group.index))
      stop("The maximum one s.list should not be larger than the number of groups!")
    if (path_type == 2 &
        s.max > length(group.index))
      stop("s.max is too large. Should be smaller than the number of groups!")
  } else{
    if (path_type == 1 &
        s.list[length(s.list)] > ncol(x))
      stop("The maximum one in s.list is too large!")
    if (path_type == 2 & s.max > ncol(x))
      stop("s.max is too large")
  }
  if (!is.null(group.index)) {
    gi <- unique(group.index)
    g_index <- match(gi, group.index) - 1
    g_df <- c(diff(g_index), 
              length(group.index) - g_index[length(g_index)])
    # g_df <- NULL
    # g_index <- NULL
    # group_set <- unique(group.index)
    # j <- 1
    # k <- 0
    # for(i in group_set){
    #   while(group.index[j] != i){
    #     j <- j+1
    #     k <- k+1
    #   }
    #   g_index <- c(g_index, j - 1)
    #   g_df <- c(g_df, k)
    # }
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
  } else if (normalize != 0) {
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
    } else{
      normalize <- 1
    }
    is_normal <- TRUE
  } else{
    is_normal <- FALSE
    normalize <- 0
  }

  if (is.null(screening.num)) {
    screening <- FALSE
    screening.num <- ncol(x)
  } else{
    stopifnot(is.integer(screening.num))
    stopifnot(screening.num > 0)
    screening <- TRUE
    if (screening.num > ncol(x))
      stop("The number of screening features must be equal or less than that of the column of x!")
    if (path_type == 1) {
      if (screening.num < s.list[length(s.list)])
        stop(
          "The number of screening features must be equal or greater than the maximum one in s.list!"
        )
    } else{
      if (screening.num < s.max)
        stop("The number of screening features must be equal or greater than the s.max!")
    }
  }
  
  # check always included varibles:
  if (is.null(always.include)) {
    always.include <- numeric(0)
  } else{
    if (is.na(sum(as.integer(always.include))))
      stop("always.include should be an integer vector")
    if (sum(always.include <= 0))
      stop("always.include should be an vector containing variable indexes which is possitive.")
    always.include <- as.integer(always.include) - 1
    if (length(always.include) > screening.num)
      stop("The number of variables in always.include should not exceed the sc")
    if (path_type == 1) {
      if (length(always.include) > s.list[length(s.list)])
        stop(
          "always.include containing too many variables. The length of it should not exceed the maximum in s.list."
        )
    } else{
      if (length(always.include) > s.max)
        stop(
          "always.include containing too many variables. The length of it should not exceed the s.max."
        )
    }
  }
  
  if (algorithm_type == "PDAS") {
    if (model_type == 4) {
      ys <- y
      xs <- x
      sort_y <- order(y[, 1])
      y <- y[sort_y, ]
      x <- x[sort_y, ]
      y <- y[, 2]
    }
    res.pdas <-
      abessCpp(x = x, y = y, data_type = normalize,
        weight = weight, is_normal = is_normal,
        algorithm_type = 1, model_type =  model_type,
        max_iter = max.iter, exchange_num = 2,
        path_type = path_type, is_warm_start = warm.start,
        ic_type = ic_type, is_cv = is_cv,
        K = nfolds, state = rep(2, 10),
        sequence = s.list, lambda_seq = 0,
        s_min = gs.min, s_max = gs.max,
        K_max = 10, epsilon = 10,
        lambda_max = 0, lambda_min = 0,
        nlambda = nlambda, is_screening = screening,
        screening_size = screening.num,
        powell_path = 1, g_index = (1:ncol(x) - 1),
        always_select = always.include,
        tao = 1.1, 
        primary_model_fit_max_iter = , 
        primary_model_fit_epsilon = , 
        early_stop = early_stop, 
        approximate_Newton = approximate_Newton
      )
    beta.pdas <- res.pdas$beta
    names(beta.pdas) <- vn
    res.pdas$beta <- beta.pdas
    if (is_cv == TRUE) {
      names(res.pdas)[which(names(res.pdas) == "ic")] <- "cvm"
      names(res.pdas)[which(names(res.pdas) == "ic_all")] <-
        "cvm.all"
    } else{
      names(res.pdas)[which(names(res.pdas) == 'ic_all')] <- 'ic.all'
    }
    res.pdas$x <- ifelse(family == "cox", xs, x)
    res.pdas$y <- ifelse(family == "cox", ys, y)
    res.pdas$family <- family
    names(res.pdas)[which(names(res.pdas) == "train_loss")] <-
      "loss"
    names(res.pdas)[which(names(res.pdas) == "train_loss_all")] <-
      "loss.all"
    names(res.pdas)[which(names(res.pdas) == 'beta_all')] <-
      'beta.all'
    names(res.pdas)[which(names(res.pdas) == "coef0_all")] <-
      'coef0.all'
    res.pdas$s.list <- s.list
    res.pdas$nsample <- nrow(x)
    res.pdas$algorithm_type <- "ABESS"
    res.pdas$method <- method
    res.pdas$type <- type
    res.pdas$ic.type <-
      ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "EBIC")[ic_type])
    res.pdas$s.max <- s.max
    res.pdas$s.min <- s.min
    if (screening)
      res.pdas$screening_A <-  res.pdas$screening_A + 1
    
    res.pdas$call <- match.call()
    class(res.pdas) <- 'abess'
    #res.pdas$beta_all <- res.pdas$beta.all
    res.pdas$beta.all <- recover(res.pdas, F)
    
  }
  
  set.seed(NULL)
  return(res.pdas)
}

#' @rdname abess
#'
#' @param formula a formula of the form \code{response ~ group} where \code{response} gives the data values and \code{group} a vector or factor of the corresponding groups.
#' @param data an optional matrix or data frame (or similar: see \code{model.frame}) containing the variables in the formula \code{formula}. By default the variables are taken from \code{environment(formula)}.
#' @param subset an optional vector specifying a subset of observations to be used.
#' @param na.action a function which indicates what should happen when the data contain \code{NA}s. Defaults to \code{getOption("na.action")}.
#' @method bd.test formula
#' @export
#' @examples
#'
#' ################  Formula interface  ################
#'
abess.formula <- function(formula, data, subset, na.action, ...) {
  if (missing(formula)
      || (length(formula) != 3L)
      || (length(attr(terms(formula[-2L]), "term.labels")) != 1L))
    stop("'formula' missing or incorrect")
  m <- match.call(expand.dots = FALSE)
  if (is.matrix(eval(m$data, parent.frame())))
    m$data <- as.data.frame(data)
  ## need stats:: for non-standard evaluation
  m[[1L]] <- quote(stats::model.frame)
  m$... <- NULL
  mf <- eval(m, parent.frame())
  DNAME <- paste(names(mf), collapse = " by ")
  names(mf) <- NULL
  response <- attr(attr(mf, "terms"), "response")
  g <- factor(mf[[-response]])
  if (nlevels(g) < 2L)
    stop("grouping factor must contain at least two levels")
  DATA <- list()
  DATA[["x"]] <- split(mf[[response]], g)
  y <- do.call("bd.test", c(DATA, list(...)))
  remind_info <-
    strsplit(y$data.name, split = "number of observations")[[1]][2]
  DNAME <- paste0(DNAME, "\nnumber of observations")
  y$data.name <- paste0(DNAME, remind_info)
  y
}
