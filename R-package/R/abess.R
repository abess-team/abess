#' @export
abess <- function(x, ...) UseMethod("abess")

#' @title Adaptive best subset selection (for generalized linear model)
#'
#' @description Adaptive best-subset selection for regression,
#' (multi-class) classification, counting-response, censored-response,
#' positive response, multi-response modeling in polynomial times.
#'
#' @aliases abess
#'
#' @author Jin Zhu, Junxian Zhu, Canhong Wen, Heping Zhang, Xueqin Wang
#'
#' @param x Input matrix, of dimension \eqn{n \times p}; each row is an observation
#' vector and each column is a predictor/feature/variable.
#' Can be in sparse matrix format (inherit from class \code{"dgCMatrix"} in package \code{Matrix}).
#' @param y The response variable, of \code{n} observations.
#' For \code{family = "binomial"} should have two levels.
#' For \code{family="poisson"}, \code{y} should be a vector with positive integer.
#' For \code{family = "cox"}, \code{y} should be a \code{Surv} object returned
#' by the \code{survival} package (recommended) or
#' a two-column matrix with columns named \code{"time"} and \code{"status"}.
#' For \code{family = "mgaussian"}, \code{y} should be a matrix of quantitative responses.
#' For \code{family = "multinomial"}, \code{y} should be a factor of at least three levels.
#' Note that, for either \code{"binomial"} or \code{"multinomial"},
#' if y is presented as a numerical vector, it will be coerced into a factor.
#' @param family One of the following models:
#' \code{"gaussian"} (continuous response),
#' \code{"binomial"} (binary response),
#' \code{"poisson"} (non-negative count),
#' \code{"cox"} (left-censored response),
#' \code{"mgaussian"} (multivariate continuous response),
#' \code{"multinomial"} (multi-class response),
#' \code{"gamma"} (positive continuous response).
#' Depending on the response. Any unambiguous substring can be given.
#' @param tune.path The method to be used to select the optimal support size. For
#' \code{tune.path = "sequence"}, we solve the best subset selection problem for each size in \code{support.size}.
#' For \code{tune.path = "gsection"}, we solve the best subset selection problem with support size ranged in \code{gs.range},
#' where the specific support size to be considered is determined by golden section.
#' @param tune.type The type of criterion for choosing the support size.
#' Available options are \code{"gic"}, \code{"ebic"}, \code{"bic"}, \code{"aic"} and \code{"cv"}.
#' Default is \code{"gic"}.
#' @param support.size An integer vector representing the alternative support sizes.
#' Only used for \code{tune.path = "sequence"}. Default is \code{0:min(n, round(n/(log(log(n))log(p))))}.
#' @param gs.range A integer vector with two elements.
#' The first element is the minimum model size considered by golden-section,
#' the later one is the maximum one. Default is \code{gs.range = c(1, min(n, round(n/(log(log(n))log(p)))))}.
#' Not available now.
#' @param lambda A single lambda value for regularized best subset selection. Default is 0.
#' @param always.include An integer vector containing the indexes of variables that should always be included in the model.
#' @param group.index A vector of integers indicating the which group each variable is in.
#' For variables in the same group, they should be located in adjacent columns of \code{x}
#' and their corresponding index in \code{group.index} should be the same.
#' Denote the first group as \code{1}, the second \code{2}, etc.
#' If you do not fit a model with a group structure,
#' please set \code{group.index = NULL} (the default).
#' @param splicing.type Optional type for splicing.
#' If \code{splicing.type = 1}, the number of variables to be spliced is
#' \code{c.max}, ..., \code{1}; if \code{splicing.type = 2},
#' the number of variables to be spliced is \code{c.max}, \code{c.max/2}, ..., \code{1}.
#' (Default: \code{splicing.type = 2}.)
#' @param screening.num An integer number. Preserve \code{screening.num} number of predictors with the largest
#' marginal maximum likelihood estimator before running algorithm.
#' @param important.search An integer number indicating the number of
#' important variables to be splicing.
#' When \code{important.search} \eqn{\ll} \code{p} variables,
#' it would greatly reduce runtimes. Default: \code{important.search = 128}.
#' @param normalize Options for normalization. \code{normalize = 0} for no normalization.
#' \code{normalize = 1} for subtracting the mean of columns of \code{x}.
#' \code{normalize = 2} for scaling the columns of \code{x} to have \eqn{\sqrt n} norm.
#' \code{normalize = 3} for subtracting the means of the columns of \code{x} and \code{y}, and also
#' normalizing the columns of \code{x} to have \eqn{\sqrt n} norm.
#' If \code{normalize = NULL}, \code{normalize} will be set \code{1} for \code{"gaussian"},
#' \code{2} for \code{"binomial"}. Default is \code{normalize = NULL}.
#' @param c.max an integer splicing size. Default is: \code{c.max = 2}.
#' @param weight Observation weights. When \code{weight = NULL},
#' we set \code{weight = 1} for each observation as default.
#' @param max.splicing.iter The maximum number of performing splicing algorithm.
#' In most of the case, only a few times of splicing iteration can guarantee the convergence.
#' Default is \code{max.splicing.iter = 20}.
#' @param warm.start Whether to use the last solution as a warm start. Default is \code{warm.start = TRUE}.
#' @param nfolds The number of folds in cross-validation. Default is \code{nfolds = 5}.
#' @param foldid an optional integer vector of values between 1, ..., nfolds identifying what fold each observation is in.
#' The default \code{foldid = NULL} would generate a random foldid.
#' @param cov.update A logical value only used for \code{family = "gaussian"}. If \code{cov.update = TRUE},
#' use a covariance-based implementation; otherwise, a naive implementation.
#' The naive method is more computational efficient than covariance-based method when \eqn{p >> n} and \code{important.search} is much large than its default value.
#' Default: \code{cov.update = FALSE}.
#' @param newton A character specify the Newton's method for fitting generalized linear models,
#' it should be either \code{newton = "exact"} or \code{newton = "approx"}.
#' If \code{newton = "exact"}, then the exact hessian is used,
#' while \code{newton = "approx"} uses diagonal entry of the hessian,
#' and can be faster (especially when \code{family = "cox"}).
#' @param newton.thresh a numeric value for controlling positive convergence tolerance.
#' The Newton's iterations converge when \eqn{|dev - dev_{old}|/(|dev| + 0.1)<} \code{newton.thresh}.
#' @param max.newton.iter a integer giving the maximal number of Newton's iteration iterations.
#' Default is \code{max.newton.iter = 10} if \code{newton = "exact"}, and \code{max.newton.iter = 60} if \code{newton = "approx"}.
#' @param early.stop A boolean value decide whether early stopping.
#' If \code{early.stop = TRUE}, algorithm will stop if the last tuning value less than the existing one.
#' Default: \code{early.stop = FALSE}.
#' @param ic.scale A non-negative value used for multiplying the penalty term
#' in information criterion. Default: \code{ic.scale = 1}.
#' @param num.threads An integer decide the number of threads to be
#' concurrently used for cross-validation (i.e., \code{tune.type = "cv"}).
#' If \code{num.threads = 0}, then all of available cores will be used.
#' Default: \code{num.threads = 0}.
#' @param seed Seed to be used to divide the sample into cross-validation folds.
#' Default is \code{seed = 1}.
#' @param ... further arguments to be passed to or from methods.
#'
#' @return A S3 \code{abess} class object, which is a \code{list} with the following components:
#' \item{beta}{A \eqn{p}-by-\code{length(support.size)} matrix of coefficients for univariate family, stored in column format;
#' while a list of \code{length(support.size)} coefficients matrix (with size \eqn{p}-by-\code{ncol(y)}) for multivariate family.}
#' \item{intercept}{An intercept vector of length \code{length(support.size)} for univariate family;
#' while a list of \code{length(support.size)} intercept vector (with size \code{ncol(y)}) for multivariate family.}
#' \item{dev}{the deviance of length \code{length(support.size)}.}
#' \item{tune.value}{A value of tuning criterion of length \code{length(support.size)}.}
#' \item{nobs}{The number of sample used for training.}
#' \item{nvars}{The number of variables used for training.}
#' \item{family}{Type of the model.}
#' \item{tune.path}{The path type for tuning parameters.}
#' \item{support.size}{The actual \code{support.size} values used.
#' Note that it is not necessary the same as the input
#' if the later have non-integer values or duplicated values.}
#' \item{edf}{The effective degree of freedom.
#' It is the same as \code{support.size} when \code{lambda = 0}.}
#' \item{best.size}{The best support size selected by the tuning value.}
#' \item{tune.type}{The criterion type for tuning parameters.}
#' \item{tune.path}{The strategy for tuning parameters.}
#' \item{screening.vars}{The character vector specify the feature
#' selected by feature screening.
#' It would be an empty character vector if \code{screening.num = 0}.}
#' \item{call}{The original call to \code{abess}.}
#'
#' @md
#' 
#' @details
#' Best-subset selection aims to find a small subset of predictors,
#' so that the resulting model is expected to have the most desirable prediction accuracy.
#' Best-subset selection problem under the support size \eqn{s} is
#' \deqn{\min_\beta -2 \log L(\beta) \;\;{\rm s.t.}\;\; \|\beta\|_0 \leq s,}
#' where \eqn{L(\beta)} is arbitrary convex functions. In
#' the GLM case, \eqn{\log L(\beta)} is the log-likelihood function; in the Cox
#' model, \eqn{\log L(\beta)} is the log partial-likelihood function. 
#' The best subset selection problem is solved by the splicing algorithm in this package, see Zhu (2020) for details.
#' Under mild conditions, the algorithm exactly solve this problem in polynomial time.
#' This algorithm exploits the idea of sequencing and splicing to reach a stable solution in finite steps when \eqn{s} is fixed.
#' The parameters \code{c.max}, \code{splicing.type} and \code{max.splicing.iter} allow user control the splicing technique flexibly. 
#' On the basis of our numerical experiment results, we assign properly parameters to the these parameters as the default 
#' such that the precision and runtime are well balanced, we suggest users keep the default values unchanged. 
#' Please see [this online page](https://abess-team.github.io/abess/articles/v10-algorithm.html) for more details about the splicing algorithm. 
#' 
#' To find the optimal support size \eqn{s},
#' we provide various criterion like GIC, AIC, BIC and cross-validation error to determine it. 
#' More specifically, the sequence of models implied by \code{support.size} are fit by the splicing algorithm. 
#' And the solved model with least information criterion or cross-validation error is the optimal model. 
#' The sequential searching for the optimal model is somehow time-wasting. 
#' A faster strategy is golden section (GS), which only need to specify \code{gs.range}. 
#' More details about GS is referred to Zhang et al (2021). 
#' 
#' It is worthy to note that the parameters \code{newton}, \code{max.newton.iter} and \code{newton.thresh} allows 
#' user control the parameter estimation in non-guassian models. 
#' The parameter estimation procedure use Newton method or approximated Newton method (only consider the diagonal elements in the Hessian matrix). 
#' Again, we suggest to use the default values unchanged because the same reason for the parameter \code{c.max}. 
#' 
#' \code{abess} support some well-known advanced statistical methods to analyze data, including 
#' \itemize{
#'   \item{sure independent screening: } {helpful for ultra-high dimensional predictors (i.e., \eqn{p \gg n}). Use the parameter \code{screening.num} to retain the marginally most important predictors. See Fan et al (2008) for more details. }
#'   \item{best subset of group selection: } {helpful when predictors have group structure. Use the parameter \code{group.index} to specify the group structure of predictors. See Zhang et al (2021) for more details. }
#'   \item{\eqn{l_2} regularization best subset selection: } {helpful when signal-to-ratio is relatively small. Use the parameter \code{lambda} to control the magnitude of the regularization term.}
#'   \item{nuisance selection: } {helpful when the prior knowledge of important predictors is available. Use the parameter \code{always.include} to retain the important predictors.}
#' }
#' The arbitrary combination of the four methods are definitely support. 
#' Please see [online vignettes](https://abess-team.github.io/abess/articles/v07-advancedFeatures.html) for more details about the advanced features support by \code{abess}. 
#' 
#' @references A polynomial algorithm for best-subset selection problem. Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang. Proceedings of the National Academy of Sciences Dec 2020, 117 (52) 33117-33123; \doi{10.1073/pnas.2014241117}
#' @references Certifiably Polynomial Algorithm for Best Group Subset Selection. Zhang, Yanhang, Junxian Zhu, Jin Zhu, and Xueqin Wang (2021). arXiv preprint arXiv:2104.12576.
#' @references abess: A Fast Best Subset Selection Library in Python and R. Jin Zhu, Liyuan Hu, Junhao Huang, Kangkang Jiang, Yanhang Zhang, Shiyun Lin, Junxian Zhu, Xueqin Wang (2021). arXiv preprint arXiv:2110.09697.
#' @references Sure independence screening for ultrahigh dimensional feature space. Fan, J. and Lv, J. (2008), Journal of the Royal Statistical Society: Series B (Statistical Methodology), 70: 849-911. \doi{10.1111/j.1467-9868.2008.00674.x}
#' @references Targeted Inference Involving High-Dimensional Data Using Nuisance Penalized Regression. Qiang Sun & Heping Zhang (2020). Journal of the American Statistical Association, \doi{10.1080/01621459.2020.1737079}
#' 
#'
#' @seealso \code{\link{print.abess}},
#' \code{\link{predict.abess}},
#' \code{\link{coef.abess}},
#' \code{\link{extract.abess}},
#' \code{\link{plot.abess}},
#' \code{\link{deviance.abess}}.
#'
#' @export
#' @rdname abess
#' @method abess default
#' @examples
#' \donttest{
#' library(abess)
#' n <- 100
#' p <- 20
#' support.size <- 3
#'
#' ################ linear model ################
#' dataset <- generate.data(n, p, support.size)
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]])
#' ## helpful generic functions:
#' print(abess_fit)
#' coef(abess_fit, support.size = 3)
#' predict(abess_fit,
#'   newx = dataset[["x"]][1:10, ],
#'   support.size = c(3, 4)
#' )
#' str(extract(abess_fit, 3))
#' deviance(abess_fit)
#' plot(abess_fit)
#' plot(abess_fit, type = "tune")
#'
#' ################ logistic model ################
#' dataset <- generate.data(n, p, support.size, family = "binomial")
#' ## allow cross-validation to tuning
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]],
#'   family = "binomial", tune.type = "cv"
#' )
#' abess_fit
#'
#' ################ poisson model ################
#' dataset <- generate.data(n, p, support.size, family = "poisson")
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]],
#'   family = "poisson", tune.type = "cv"
#' )
#' abess_fit
#'
#' ################ Cox model ################
#' dataset <- generate.data(n, p, support.size, family = "cox")
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]],
#'   family = "cox", tune.type = "cv"
#' )
#'
#' ################ Multivariate gaussian model ################
#' dataset <- generate.data(n, p, support.size, family = "mgaussian")
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]],
#'   family = "mgaussian", tune.type = "cv"
#' )
#' plot(abess_fit, type = "l2norm")
#'
#' ################ Multinomial model (multi-classification) ################
#' dataset <- generate.data(n, p, support.size, family = "multinomial")
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]],
#'   family = "multinomial", tune.type = "cv"
#' )
#' predict(abess_fit,
#'   newx = dataset[["x"]][1:10, ],
#'   support.size = c(3, 4), type = "response"
#' )
#'
#' ########## Best group subset selection #############
#' dataset <- generate.data(n, p, support.size)
#' group_index <- rep(1:10, each = 2)
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]], group.index = group_index)
#' str(extract(abess_fit))
#'
#' ################ Golden section searching ################
#' dataset <- generate.data(n, p, support.size)
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]], tune.path = "gsection")
#' abess_fit
#'
#' ################ Feature screening ################
#' p <- 1000
#' dataset <- generate.data(n, p, support.size)
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]],
#'   screening.num = 100
#' )
#' str(extract(abess_fit))
#'
#' ################ Sparse predictor ################
#' require(Matrix)
#' p <- 1000
#' dataset <- generate.data(n, p, support.size)
#' dataset[["x"]][abs(dataset[["x"]]) < 1] <- 0
#' dataset[["x"]] <- Matrix(dataset[["x"]])
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]])
#' str(extract(abess_fit))
#' }
abess.default <- function(x,
                          y,
                          family = c(
                            "gaussian", "binomial", "poisson", "cox",
                            "mgaussian", "multinomial", "gamma"
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
  tau <- NULL

  ## check lambda
  stopifnot(length(lambda) == 1)
  stopifnot(!anyNA(lambda))
  stopifnot(all(lambda >= 0))
  lambda.list <- lambda
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

  ## check splicing type
  stopifnot(length(splicing.type) == 1)
  stopifnot(splicing.type %in% c(1, 2))
  splicing_type <- 2 - splicing.type
  splicing_type <- as.integer(splicing_type)

  ## check max splicing iteration
  stopifnot(is.numeric(max.splicing.iter) & max.splicing.iter >= 1)
  check_integer_warning(
    max.splicing.iter,
    "max.splicing.iter should be an integer value.
                        It is coerced to as.integer(max.splicing.iter)."
  )
  max_splicing_iter <- as.integer(max.splicing.iter)

  ## task type:
  family <- match.arg(family)
  model_type <- switch(family,
    "gaussian" = 1,
    "binomial" = 2,
    "poisson" = 3,
    "cox" = 4,
    "mgaussian" = 5,
    "multinomial" = 6,
    "gamma" = 8
  )

  ## check predictors:
  stopifnot(class(x)[1] %in% c("data.frame", "matrix", "dgCMatrix"))
  vn <- colnames(x) ## if x is not a matrix type object, it will return NULL.
  nvars <- ncol(x)
  nobs <- nrow(x)
  if (nvars == 1) {
    stop("x should have at least two columns!")
  }
  if (is.null(vn)) {
    vn <- paste0("x", 1:nvars)
  }
  sparse_X <- ifelse(class(x)[1] %in% c("matrix", "data.frame"), FALSE, TRUE)
  if (sparse_X) {
    if (class(x) == "dgCMatrix") {
      x <- map_dgCMatrix2entry(x)
    }
  } else {
    if (is.data.frame(x)) {
      x <- as.matrix(x)
    }
    if (!is.numeric(x)) {
      stop("x must be a *numeric* matrix/data.frame!")
    }
  }
  if (anyNA(x) || any(is.infinite(x))) {
    stop("x has missing value or infinite value!")
  }

  # check weight
  if (is.null(weight)) {
    weight <- rep(1, nobs)
  }
  stopifnot(is.vector(weight))
  if (length(weight) != nobs) {
    stop("Rows of x must be the same as length of weight!")
  }
  stopifnot(all(is.numeric(weight)), all(weight >= 0))

  ## check response:
  if (anyNA(y)) {
    stop("y has missing value!")
  }
  if (any(is.infinite(y))) {
    stop("y has infinite value!")
  }
  if (family == "gaussian") {
    if (is.matrix(y)) {
      if (dim(y)[2] > 1) {
        stop("The dimension of y should not exceed 1 when family = 'gaussian'!")
      }
    }
  }
  if (family == "binomial" || family == "multinomial") {
    if (length(unique(y)) == 2 && family == "multinomial") {
      warning("y is a binary variable and is not match to family = 'multinomial'.
              We change to family = 'binomial'")
      model_type <- 2
      family <- "binomial"
    }
    if (length(unique(y)) > 2 && family == "binomial") {
      stop("Input binary y when family = 'binomial'; otherwise,
           change the option for family to 'multinomial'. ")
    }
    if (length(unique(y)) == nobs && family == "multinomial") {
      stop("All of y value are distinct.
           Please input categorial y when family = 'multinomial'.")
    }
    if ((nobs / length(unique(y))) < 5 && family == "multinomial") {
      warning("The number of the category of y is relative large compare to nvars.
              The numerical result might be unstable.")
    }
    if (!is.factor(y)) {
      y <- as.factor(y)
    }
    class.name <- levels(y)
    y_vn <- class.name

    if (family == "binomial") {
      y <- as.numeric(y) - 1
    }
    if (family == "multinomial") {
      y <- model.matrix(~ factor(as.numeric(y) - 1) + 0)
      colnames(y) <- NULL
    }
  }
  if (family == "poisson") {
    if (any(y < 0)) {
      stop("y must be positive integer value when family = 'poisson'.")
    }
  }
  if (family == "gamma") {
    if (any(y < 0)) {
      stop("y must be positive value when family = 'gamma'.")
    }
  }
  if (family == "cox") {
    if (!is.matrix(y)) {
      y <- as.matrix(y)
    }
    if (ncol(y) != 2) {
      stop("y must be a Surv object or a matrix with two columns when family = 'cox'!")
    }
    stopifnot(length(unique(y[, 2])) == 2)
    ## pre-process data for cox model
    sort_y <- order(y[, 1])
    y <- y[sort_y, ]
    x <- x[sort_y, ]
    y <- y[, 2]
  }
  if (family == "mgaussian") {
    if (!is.matrix(y) || dim(y)[2] <= 1) {
      stop("y must be a n-by-q matrix (q > 1) when family = 'mgaussian'!")
    }
    y_vn <- colnames(y)
    if (is.null(y_vn)) {
      y_vn <- colnames("y", 1:dim(y)[2])
    }
  }
  y <- as.matrix(y)
  y_dim <- ncol(y)
  multi_y <- family %in% MULTIVARIATE_RESPONSE

  # check whether x and y are matching:
  if (nobs != nrow(y)) {
    stop("Rows of x must be the same as rows of y!")
  }

  ## strategy for tuning
  tune.path <- match.arg(tune.path)
  if (tune.path == "gsection") {
    path_type <- 2
  } else if (tune.path == "sequence") {
    path_type <- 1
  }

  ## group variable:
  group_select <- FALSE
  if (is.null(group.index)) {
    g_index <- 1:nvars - 1
    ngroup <- 1
    max_group_size <- 1
    # g_df <- rep(1, nvars)
  } else {
    stopifnot(all(!is.na(group.index)))
    stopifnot(all(is.finite(group.index)))
    stopifnot(diff(group.index) >= 0)
    check_integer(group.index, "group.index must be a vector with integer value.")
    group_select <- TRUE
    gi <- unique(group.index)
    g_index <- match(gi, group.index) - 1
    g_df <- c(diff(g_index), nvars - max(g_index))
    ngroup <- length(g_index)
    max_group_size <- max(g_df)
  }

  # sparse level list (sequence):
  if (is.null(support.size)) {
    if (group_select) {
      s_list <- 0:min(c(ngroup, round(nobs / max_group_size / log(ngroup))))
    } else {
      s_list <- 0:min(c(nvars, round(nobs / log(log(nobs)) / log(nvars))))
    }
  } else {
    stopifnot(any(is.numeric(support.size) & support.size >= 0))
    check_integer(support.size, "support.size must be a vector with integer value.")
    if (group_select) {
      stopifnot(max(support.size) <= ngroup)
    } else {
      stopifnot(max(support.size) <= nvars)
    }
    stopifnot(max(support.size) < nobs)
    support.size <- sort(support.size)
    support.size <- unique(support.size)
    s_list <- support.size
    # if (s_list[1] == 0) {
    #   zero_size <- TRUE
    # } else {
    #   zero_size <- FALSE
    #   s_list <- c(0, s_list)
    # }
  }

  # sparse range (golden-section):
  if (is.null(gs.range)) {
    s_min <- 1
    if (group_select) {
      s_max <- min(c(ngroup, round(nobs / max_group_size / log(ngroup))))
    } else {
      s_max <- min(c(nvars, round(nobs / log(log(nobs)) / log(nvars))))
    }
  } else {
    stopifnot(length(gs.range) == 2)
    stopifnot(all(is.numeric(gs.range)))
    stopifnot(all(gs.range > 0))
    check_integer_warning(
      gs.range,
      "gs.range should be a vector with integer.
                          It is coerced to as.integer(gs.range)."
    )
    gs.range <- as.integer(gs.range)
    stopifnot(as.integer(gs.range)[1] != as.integer(gs.range)[2])
    if (group_select) {
      stopifnot(max(gs.range) < ngroup)
    } else {
      stopifnot(max(gs.range) < nvars)
    }
    gs.range <- as.integer(gs.range)
    s_min <- min(gs.range)
    s_max <- max(gs.range)
  }

  ## check C-max:
  stopifnot(is.numeric(c.max))
  stopifnot(c.max >= 1)
  check_integer_warning(
    c.max,
    "c.max should be an integer. It is coerced to as.integer(c.max)."
  )
  c_max <- as.integer(c.max)

  ## check compatible between group selection and support size
  # if (group_select) {
  #   if (path_type == 1 & max(s_list) > length(gi))
  #     stop("The maximum one support.size should not be larger than the number of groups!")
  #   if (path_type == 2 & s_max > length(gi))
  #     stop("max(gs.range) is too large. Should be smaller than the number of groups!")
  # }

  ## check covariance update
  stopifnot(is.logical(cov.update))
  if (model_type == 1) {
    covariance_update <- cov.update
  } else {
    covariance_update <- FALSE
  }

  ## check parameters for sub-optimization:
  # 1:
  if (length(newton) == 2) {
    if (family %in% c("binomial", "cox", "multinomial", "gamma", "poisson")) {
      newton <- "approx"
    }
  }
  newton <- match.arg(newton)
  # if (newton == "auto") {
  #   if (family == "cox") {
  #     newton <- "approx"
  #   } else if (family == "logistic") {
  #     newton <- "auto"
  #   }
  # }
  if (family %in% c("gaussian", "mgaussian")) {
    newton <- "exact"
  }
  newton_type <- switch(newton,
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
    if (family == "gamma" && newton_type == 1) {
      max_newton_iter <- 200
    }
  }
  # 3:
  stopifnot(is.numeric(newton.thresh) & newton.thresh > 0)
  newton_thresh <- as.double(newton.thresh)

  # tune support size method:
  tune.type <- match.arg(tune.type)
  ic_type <- map_tunetype2numeric(tune.type)
  is_cv <- ifelse(tune.type == "cv", TRUE, FALSE)
  if (is_cv) {
    if (is.null(foldid)) {
      cv_fold_id <- integer(0)
      nfolds <- check_nfold(nfolds)
    } else {
      cv_fold_id <- check_foldid(foldid, nobs)
      nfolds <- length(unique(nfolds))
    }
  } else {
    cv_fold_id <- integer(0)
    # nfolds <- 1
  }

  ## information criterion
  stopifnot(is.numeric(ic.scale))
  stopifnot(ic.scale >= 0)
  ic_scale <- as.integer(ic.scale)

  ## normalize strategy:
  if (is.null(normalize)) {
    is_normal <- TRUE
    normalize <- switch(family,
      "gaussian" = 1,
      "binomial" = 2,
      "poisson" = 2,
      "cox" = 3,
      "mgaussian" = 1,
      "multinomial" = 2,
      "gamma" = 2
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
    check_integer_warning(
      screening.num,
      "screening.num should be a integer.
                          It is coerced to as.integer(screening.num)."
    )
    screening.num <- as.integer(screening.num)
    if (screening.num > nvars) {
      stop("The number of screening features must be equal or less than that of the column of x!")
    }
    if (path_type == 1) {
      if (screening.num < max(s_list)) {
        stop("The number of screening features must be equal or greater than the maximum one in support.size!")
      }
    } else {
      if (screening.num < s_max) {
        stop("The number of screening features must be equal or greater than the max(gs.range)!")
      }
    }
    screening <- TRUE
    screening_num <- screening.num
  }

  # check important searching:
  if (is.null(important.search)) {
    important_search <- min(c(nvars, 128))
    important_search <- as.integer(important_search)
  } else {
    stopifnot(is.numeric(important.search))
    stopifnot(important.search >= 0)
    check_integer_warning(important.search)
    important_search <- as.integer(important.search)
  }

  # check always included variables:
  if (is.null(always.include)) {
    always_include <- numeric(0)
  } else {
    if (anyNA(always.include) || any(is.infinite(always.include))) {
      stop("always.include has missing values or infinite values.")
    }
    stopifnot(always.include %in% 1:nvars)
    stopifnot(always.include > 0)
    check_integer(always.include, "always.include must be a vector with integer value.")
    always.include <- as.integer(always.include) - 1
    always_include_num <- length(always.include)
    if (always_include_num > screening_num) {
      stop("The number of variables in always.include must not exceed the screening.num")
    }
    if (path_type == 1) {
      if (always_include_num > max(s_list)) {
        stop("always.include containing too many variables.
           The length of it must not exceed the maximum in support.size.")
      }
      if (always_include_num > min(s_list)) {
        if (is.null(support.size)) {
          s_list <- s_list[s_list >= always_include_num]
        } else {
          stop(sprintf("always.include containing %s variables. The min(support.size) must be equal or greater than this.", always_include_num))
        }
      }
    } else {
      if (always_include_num > s_max) {
        stop("always.include containing too many variables. The length of it must not exceed the max(gs.range).")
      }
      if (always_include_num > s_min) {
        if (is.null(support.size)) {
          s_min <- always_include_num
        } else {
          stop(sprintf("always.include containing %s variables. The min(gs.range) must be equal or greater than this.", always_include_num))
        }
      }
    }
    always_include <- always.include
  }

  t1 <- proc.time()
  result <- abessGLM_API(
    x = x,
    y = y,
    n = nobs,
    p = nvars,
    normalize_type = normalize,
    weight = weight,
    algorithm_type = 6,
    model_type = model_type,
    max_iter = max_splicing_iter,
    exchange_num = c_max,
    path_type = path_type,
    is_warm_start = warm.start,
    ic_type = ic_type,
    ic_coef = ic_scale,
    Kfold = nfolds,
    sequence = as.vector(s_list),
    lambda_seq = lambda,
    s_min = s_min,
    s_max = s_max,
    lambda_max = 0,
    lambda_min = 0,
    nlambda = 10,
    screening_size = ifelse(screening_num >= nvars, -1, screening_num),
    g_index = g_index,
    always_select = always_include,
    primary_model_fit_max_iter = max_newton_iter,
    primary_model_fit_epsilon = newton_thresh,
    early_stop = early_stop,
    approximate_Newton = approximate_newton,
    thread = num_threads,
    covariance_update = covariance_update,
    sparse_matrix = sparse_X,
    splicing_type = splicing_type,
    sub_search = important_search,
    cv_fold_id = cv_fold_id, 
    A_init = as.integer(c())
  )
  t2 <- proc.time()
  # print(t2 - t1)

  ## process result

  ### process best model (abandon):
  # support.index <- which(result[["beta"]] != 0.0)
  # names(result[["beta"]]) <- vn
  # best_model <- list("beta" = result[["beta"]],
  #                    "coef0" = result[["coef0"]],
  #                    "support.index" = support.index,
  #                    "support.size" = sum(result[["beta"]] != 0.0),
  #                    "dev" = result[["train_loss"]],
  #                    "tune.value" = result[["ic"]])
  # result[["best.model"]] <- best_model

  result[["beta"]] <- NULL
  result[["coef0"]] <- NULL
  result[["train_loss"]] <- NULL
  result[["ic"]] <- NULL
  result[["lambda"]] <- NULL

  result[["nobs"]] <- nobs
  result[["nvars"]] <- nvars
  result[["family"]] <- family
  result[["tune.path"]] <- tune.path
  # result[["support.df"]] <- g_df
  result[["tune.type"]] <- ifelse(is_cv == TRUE, "cv",
    c("AIC", "BIC", "GIC", "EBIC")[ic_type]
  )
  result[["gs.range"]] <- gs.range

  ## preprocessing result in "gsection"
  if (tune.path == "gsection") {
    ## change the order:
    reserve_order <- length(result[["sequence"]]):1
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

  result[["best.size"]] <- s_list[which.min(result[["tune.value"]])]
  names(result)[which(names(result) == "coef0_all")] <- "intercept"
  if (family %in% MULTIVARIATE_RESPONSE) {
    if (family == "multinomial") {
      result[["intercept"]] <- lapply(result[["intercept"]], function(x) {
        x <- x[-y_dim]
      })
    }
  } else {
    result[["intercept"]] <- as.vector(result[["intercept"]])
  }

  names(result)[which(names(result) == "beta_all")] <- "beta"
  # names(result)[which(names(result) == 'screening_A')] <- "screening.index"
  # result[["screening.index"]] <- result[["screening.index"]] + 1

  if (multi_y) {
    if (screening) {
      for (i in 1:length(result[["beta"]])) {
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
    } else {
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

  # if (s_list[0] == 0) {
  #   nulldev <- result[["dev"]][1]
  # } else {
  #   f <- switch(
  #     family,
  #     "gaussian" = gaussian(),
  #     "binomial" = binomial(),
  #     "poisson" = poisson()
  #   )
  #   if (family != "cox") {
  #     nulldev <- deviance(glm(y ~ .,
  #                             data = cbind.data.frame(y, 1),
  #                             family = f))
  #   } else {
  #     nulldev <- 0
  #   }
  # }
  # result[["nulldev"]] <- 0

  result[["call"]] <- match.call()
  class(result) <- "abess"

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
#' \donttest{
#' ################  Formula interface  ################
#' data("trim32")
#' abess_fit <- abess(y ~ ., data = trim32)
#' abess_fit
#' }
abess.formula <- function(formula, data, subset, na.action, ...) {
  contrasts <- NULL ## for sparse X matrix
  cl <- match.call()
  mf <- match.call(expand.dots = FALSE)
  m <- match(
    c("formula", "data", "subset", "na.action"),
    names(mf), 0L
  )
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")

  suppressWarnings(y <- model.response(mf, "numeric"))
  x <- abess_model_matrix(mt, mf, contrasts)[, -1]
  x <- as.matrix(x)

  # all_name <- all.vars(mt)
  # if (abess_res[["family"]] == "cox") {
  #   response_name <- all_name[1:2]
  # } else {
  #   response_name <- all_name[1]
  # }

  abess_res <- abess.default(x, y, ...)
  abess_res[["call"]] <- cl

  # best_support <- abess_res[["best.model"]][["support.index"]]
  # support_name <- colnames(x)[best_support]
  #
  # response_index <- match(response_name, all_name)
  # x_index <- match(support_name, all_name)
  # abess_res[["best.model"]][["support.index"]] <- x_index
  # names(abess_res[["best.model"]][["support.index"]]) <- support_name

  abess_res
}
