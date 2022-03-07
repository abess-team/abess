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
#' For \code{family = "multinomial"} or \code{"ordinal"}, \code{y} should be a factor of at least three levels.
#' Note that, for either \code{"binomial"}, \code{"ordinal"} or \code{"multinomial"},
#' if y is presented as a numerical vector, it will be coerced into a factor.
#' @param family One of the following models:
#' \code{"gaussian"} (continuous response),
#' \code{"binomial"} (binary response),
#' \code{"poisson"} (non-negative count),
#' \code{"cox"} (left-censored response),
#' \code{"mgaussian"} (multivariate continuous response),
#' \code{"multinomial"} (multi-class response),
#' \code{"ordinal"} (multi-class ordinal response),
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
#' @param init.active.set A vector of integers indicating the initial active set. 
#' Default: \code{init.active.set = NULL}. 
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
#' @param normalize Options for normalization. 
#' \code{normalize = 0} for no normalization.
#' \code{normalize = 1} for subtracting the means of the columns of \code{x} and \code{y}, and also
#' normalizing the columns of \code{x} to have \eqn{\sqrt n} norm.
#' \code{normalize = 2} for subtracting the mean of columns of \code{x} and 
#' scaling the columns of \code{x} to have \eqn{\sqrt n} norm.
#' \code{normalize = 3} for scaling the columns of \code{x} to have \eqn{\sqrt n} norm.
#' If \code{normalize = NULL}, \code{normalize} will be set \code{1} for \code{"gaussian"} and \code{"mgaussian"},
#' \code{3} for \code{"cox"}. Default is \code{normalize = NULL}.
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
#' ################ Ordinal regression  ################
#' dataset <- generate.data(n, p, support.size, family = "ordinal", class.num = 4)
#' abess_fit <- abess(dataset[["x"]], dataset[["y"]],
#'   family = "ordinal", tune.type = "cv"
#' )
#' coef <- coef(abess_fit, support.size = abess_fit[["best.size"]])[[1]]
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
