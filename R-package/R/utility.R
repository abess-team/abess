#' @title Generate simulated data
#'
#' @description Generate simulated data under the
#' generalized linear model and Cox proportional hazard model.
#'
#' @param n The number of observations.
#' @param p The number of predictors of interest.
#' @param support.size The number of nonzero coefficients in the underlying regression
#' model. Can be omitted if \code{beta} is supplied.
#' @param rho A parameter used to characterize the pairwise correlation in
#' predictors. Default is \code{0}.
#' @param family The distribution of the simulated response. \code{"gaussian"} for
#' univariate quantitative response, \code{"binomial"} for binary classification response,
#' \code{"poisson"} for counting response, \code{"cox"} for left-censored response,
#' \code{"mgaussian"} for multivariate quantitative response,
#' \code{"mgaussian"} for multi-classification response.
#' @param beta The coefficient values in the underlying regression model.
#' If it is supplied, \code{support.size} would be omitted.
#' @param cortype The correlation structure.
#' \code{cortype = 1} denotes the independence structure,
#' where the covariance matrix has \eqn{(i,j)} entry equals \eqn{I(i \neq j)}.
#' \code{cortype = 2} denotes the exponential structure,
#' where the covariance matrix has \eqn{(i,j)} entry equals \eqn{rho^{|i-j|}}.
#' code{cortype = 3} denotes the constant structure,
#' where the non-diagonal entries of covariance
#' matrix are \eqn{rho} and diagonal entries are 1.
#' @param snr A numerical value controlling the signal-to-noise ratio (SNR). The SNR is defined as
#' as the variance of \eqn{x\beta} divided
#' by the variance of a gaussian noise: \eqn{\frac{Var(x\beta)}{\sigma^2}}.
#' The gaussian noise \eqn{\epsilon} is set with mean 0 and variance.
#' The noise is added to the linear predictor \eqn{\eta} = \eqn{x\beta}. Default is \code{snr = 10}.
#' Note that this arguments's effect is overridden if \code{sigma} is supplied with a non-null value.
#' @param sigma The variance of the gaussian noise. Default \code{sigma = NULL} implies it is determined by \code{snr}.
#' @param weibull.shape The shape parameter of the Weibull distribution.
#' It works only when \code{family = "cox"}.
#' Default: \code{weibull.shape = 1}.
#' @param uniform.max A parameter controlling censored rate.
#' A large value implies a small censored rate;
#' otherwise, a large censored rate.
#' It works only when \code{family = "cox"}.
#' Default is \code{uniform.max = 1}.
# @param sigma A parameter used to control the signal-to-noise ratio. For linear regression,
# it is the error variance \eqn{\sigma^2}. For logistic regression,
# the larger the value of sigma, the higher the signal-to-noise ratio.
# Valid only for \code{cortype = 3}.
#' @param y.dim Response's Dimension. It works only when \code{family = "mgaussian"}. Default: \code{y.dim = 3}.
#' @param class.num The number of class. It works only when \code{family = "multinomial"}. Default: \code{class.num = 3}.
#' @param seed random seed. Default: \code{seed = 1}.
#' @return A \code{list} object comprising:
#' \item{x}{Design matrix of predictors.}
#' \item{y}{Response variable.}
#' \item{beta}{The coefficients used in the underlying regression model.}
#'
#' @details
# We generate an \eqn{n \times p} random Gaussian matrix
# \eqn{X} with mean 0 and a covariance matrix with an exponential structure
# or a constant structure. For the exponential structure, the covariance matrix
# has \eqn{(i,j)} entry equals \eqn{rho^{|i-j|}}. For the constant structure,
# the \eqn{(i,j)} entry of the covariance matrix is \eqn{rho} for every \eqn{i
# \neq j} and 1 elsewhere. For the moving average structure,  For the design matrix \eqn{X},
# we first generate an \eqn{n \times p} random Gaussian matrix \eqn{\bar{X}}
# whose entries are i.i.d. \eqn{\sim N(0,1)} and then normalize its columns
# to the \eqn{\sqrt n} length. Then the design matrix \eqn{X} is generated with
# \eqn{X_j = \bar{X}_j + \rho(\bar{X}_{j+1}+\bar{X}_{j-1})} for \eqn{j=2,\dots,p-1}.
#'
#' For \code{family = "gaussian"}, the data model is
#' \deqn{Y = X \beta + \epsilon.}
#' The underlying regression coefficient \eqn{\beta} has
#' uniform distribution [m, 100m] and \eqn{m=5 \sqrt{2log(p)/n}.}
#'
#' For \code{family= "binomial"}, the data model is \deqn{Prob(Y = 1) = \exp(X
#' \beta + \epsilon)/(1 + \exp(X \beta + \epsilon)).}
#' The underlying regression coefficient \eqn{\beta} has
#' uniform distribution [2m, 10m] and \eqn{m = 5 \sqrt{2log(p)/n}.}
#'
#' For \code{family = "poisson"}, the data is modeled to have
#' an exponential distribution:
#' \deqn{Y = Exp(\exp(X \beta + \epsilon)).}
#' The underlying regression coefficient \eqn{\beta} has
#' uniform distribution [2m, 10m] and \eqn{m = \sqrt{2log(p)/n}/3.}
#'
#' For \code{family = "cox"}, the model for failure time \eqn{T} is
#' \deqn{T = (-\log(U / \exp(X \beta)))^{1/weibull.shape},}
#' where \eqn{U} is a uniform random variable with range [0, 1].
#' The centering time \eqn{C} is generated from
#' uniform distribution \eqn{[0, uniform.max]},
#' then we define the censor status as
#' \eqn{\delta = I(T \le C)} and observed time as \eqn{R = \min\{T, C\}}.
#' The underlying regression coefficient \eqn{\beta} has
#' uniform distribution [2m, 10m],
#' where \eqn{m = 5 \sqrt{2log(p)/n}}.
#'
#' For \code{family = "mgaussian"}, the data model is
#' \deqn{Y = X \beta + E.}
#' The non-zero values of regression matrix \eqn{\beta} are sampled from
#' uniform distribution [m, 100m] and \eqn{m=5 \sqrt{2log(p)/n}.}
#'
#' For \code{family= "multinomial"}, the data model is \deqn{Prob(Y = 1) = \exp(X \beta + E)/(1 + \exp(X \beta + E)).}
#' The non-zero values of regression coefficient \eqn{\beta} has
#' uniform distribution [2m, 10m] and \eqn{m = 5 \sqrt{2log(p)/n}.}
#'
#' In the above models, \eqn{\epsilon \sim N(0, \sigma^2 )} and \eqn{E \sim MVN(0, \sigma^2 \times I_{q \times q})},
#' where \eqn{\sigma^2} is determined by the \code{snr} and q is \code{y.dim}.
#'
#' @author Jin Zhu
#'
#' @export
#'
#' @examples
#'
#' # Generate simulated data
#' n <- 200
#' p <- 20
#' support.size <- 5
#' dataset <- generate.data(n, p, support.size)
#' str(dataset)
generate.data <- function(n,
                          p,
                          support.size = NULL,
                          rho = 0,
                          family = c(
                            "gaussian", "binomial", "poisson",
                            "cox", "mgaussian", "multinomial",
                            "gamma"
                          ),
                          beta = NULL,
                          cortype = 1,
                          snr = 10,
                          sigma = NULL,
                          weibull.shape = 1,
                          uniform.max = 1,
                          y.dim = 3,
                          class.num = 3,
                          seed = 1) {
  # sigma <- 1

  family <- match.arg(family)
  if (family == "mgaussian") {
    y_dim <- y.dim
  } else if (family == "multinomial") {
    y_dim <- class.num
  } else {
    y_dim <- 1
  }
  y_cor <- diag(y_dim)

  set.seed(seed)
  # if(is.null(beta)){
  #   beta <- rep(0, p)
  #   beta[1:support.size*floor(p/support.size):floor(p/support.size)] <- rep(1, support.size)
  # } else{
  #   beta <- beta
  # }

  multi_y <- FALSE
  if (family %in% c("mgaussian", "multinomial")) {
    multi_y <- TRUE
  }

  if (!is.null(beta)) {
    if (multi_y) {
      stopifnot(is.matrix(beta))
      support.size <- sum(apply(abs(beta) > 1e-5, 1, sum) != 0)
    } else {
      stopifnot(is.vector(beta))
      support.size <- sum(abs(beta) > 1e-5)
    }

    beta[abs(beta) <= 1e-5] <- 0
  } else {
    if (is.null(support.size)) {
      stop("Please provide an integer to support.size.")
    }
    stopifnot(is.numeric(support.size) & support.size >= 1)
  }

  if (cortype == 1) {
    Sigma <- diag(p)
  } else if (cortype == 2) {
    Sigma <- matrix(0, p, p)
    Sigma <- rho^(abs(row(Sigma) - col(Sigma)))
  } else if (cortype == 3) {
    Sigma <- matrix(rho, p, p)
    diag(Sigma) <- 1
  }
  if (cortype == 1) {
    x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  } else {
    x <- MASS::mvrnorm(n, rep(0, p), Sigma)
  }

  ### pre-treatment for beta ###
  input_beta <- beta
  if (multi_y) {
    beta <- matrix(0, p, y_dim)
  } else {
    beta <- rep(0, p)
  }
  ### pre-treatment for beta ###
  nonzero <- sample(1:p, support.size)
  if (family == "gaussian") {
    if (p > 1) {
      m <- 5 * sqrt(2 * log(p) / n)
      M <- 100 * m
    } else {
      m <- 5 * sqrt(2 * log(2) / n)
      M <- 100 * m
    }
    if (is.null(input_beta)) {
      beta[nonzero] <- stats::runif(support.size, m, M)
    } else {
      beta <- input_beta
    }
    if (is.null(sigma)) {
      sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
    }

    y <- x %*% beta + rnorm(n, 0, sigma)
  }
  if (family == "binomial") {
    m <- 5 * sqrt(2 * log(p) / n)
    if (is.null(input_beta)) {
      beta[nonzero] <- stats::runif(support.size, 2 * m, 10 * m)
    } else {
      beta <- input_beta
    }
    if (is.null(sigma)) {
      sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
    }

    eta <- x %*% beta + rnorm(n, 0, sigma)
    PB <- apply(eta, 1, generatedata2)
    y <- stats::rbinom(n, 1, PB)
  }
  if (family == "cox") {
    m <- 5 * sqrt(2 * log(p) / n)
    if (is.null(input_beta)) {
      beta[nonzero] <- stats::runif(support.size, 2 * m, 10 * m)
    } else {
      beta <- input_beta
    }
    if (is.null(sigma)) {
      sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
    }

    eta <- x %*% beta + rnorm(n, 0, sigma)
    time <- (-log(stats::runif(n)) / drop(exp(eta)))^(1 / weibull.shape)
    ctime <- stats::runif(n, max = uniform.max)
    status <- (time < ctime) * 1
    censoringrate <- 1 - mean(status)
    # cat("censoring rate:", censoringrate, "\n")
    time <- pmin(time, ctime)
    y <- cbind(time = time, status = status)
  }
  if (family == "poisson") {
    m <- 5 * sqrt(2 * log(p) / n)
    # m <- sigma * sqrt(2 * log(p) / n) / 3
    if (is.null(input_beta)) {
      beta[nonzero] <- stats::runif(support.size, -2 * m, 2 * m)
      # beta[nonzero] <- stats::rnorm(support.size, sd = m)
    } else {
      beta <- input_beta
    }
    if (is.null(sigma)) {
      sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
    }

    sigma <- 0
    eta <- x %*% beta + stats::rnorm(n, 0, sigma)
    eta <- ifelse(eta > 30, 30, eta)
    eta <- ifelse(eta < -30, -30, eta)
    eta <- exp(eta)
    # eta[eta<0.0001] <- 0.0001
    # eta[eta>1e5] <- 1e5
    # y <- stats::rpois(n, eta)
    y <- sapply(eta, stats::rpois, n = 1)
  }
  if (family == "mgaussian") {
    m <- 5 * sqrt(2 * log(p) / n)
    M <- 100 * m
    if (is.null(input_beta)) {
      beta[nonzero, ] <- matrix(stats::runif(support.size * y_dim, m, M),
        ncol = y_dim
      )
    } else {
      beta <- input_beta
    }
    if (is.null(sigma)) {
      sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
    }
    sigma <- diag(sigma)
    sigma <- sigma * y_cor
    epsilon <- MASS::mvrnorm(n = n, mu = rep(0, y_dim), Sigma = sigma)
    y <- x %*% beta + epsilon
    colnames(y) <- paste0("y", 1:y_dim)
  }
  if (family == "multinomial") {
    m <- 2.5 * sqrt(2 * log(p) / n)
    M <- 50 * m
    if (is.null(input_beta)) {
      beta[nonzero, ] <- matrix(stats::runif(support.size * y_dim, m, M),
        ncol = y_dim
      )
    } else {
      beta <- input_beta
    }
    if (is.null(sigma)) {
      sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
    }
    sigma <- diag(sigma)
    sigma <- sigma * y_cor
    epsilon <- MASS::mvrnorm(n, rep(0, y_dim), sigma)
    prob_y <- x %*% beta
    prob_y <- exp(prob_y)
    prob_y <- sweep(prob_y, MARGIN = 1, STATS = rowSums(prob_y), FUN = "/")
    y <- apply(prob_y, 1, function(x) {
      sample(0:(length(x) - 1), size = 1, prob = x)
    })
  }
  if (family == "gamma") {
    m <- 5 * sqrt(2 * log(p) / n)
    if (is.null(input_beta)) {
      # this is the true value of beta
      beta[nonzero] <- stats::runif(support.size, m, 100 * m)
    } else {
      beta <- input_beta
    }
    # add noise
    sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
    eta <- x %*% beta + stats::rnorm(n, 0, sigma)
    # set coef_0 as + abs(min(eta)) + 1
    eta <- eta + abs(min(eta)) + 10
    # set the shape para of gamma uniformly in [0.1,100.1]
    shape_para <- 100 * runif(n) + 0.1
    y <- stats::rgamma(n, shape = shape_para, rate = shape_para * eta)
  }
  set.seed(NULL)

  colnames(x) <- paste0("x", 1:p)
  return(list(x = x, y = y, beta = beta))
}

generatedata2 <- function(eta) {
  a <- exp(eta) / (1 + exp(eta))
  if (is.infinite(exp(eta))) {
    a <- 1
  }
  return(a)
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

abess_model_matrix <- function(object, data = environment(object),
                               contrasts.arg = NULL,
                               xlev = NULL, ...) {
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
    reorder <- match(
      vapply(attr(t, "variables"), deparse2, "")[-1L],
      names(data)
    )
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
        stop("Some columns in data are character! You may convert these columns to a dummy variable via model.matrix function or discard them.")
      } else if (is.factor(data[[i]])) {
        stop("Some columns in data are factor!. You may convert these columns to a dummy variable via model.matrix function or discard them.")
      }
    }
  }
  data
}

map_tunetype2numeric <- function(tune.type) {
  ic_type <- switch(tune.type,
    "aic" = 1,
    "bic" = 2,
    "gic" = 3,
    "ebic" = 4,
    "cv" = 1
  )
  ic_type
}

check_foldid <- function(foldid, nobs) {
  stopifnot(is.vector(foldid))
  stopifnot(is.numeric(foldid))
  stopifnot(length(foldid) == nobs)
  check_integer_warning(
    foldid,
    "nfolds should be an integer value. It is coerced to be as.integer(foldid). "
  )
  foldid <- as.integer(foldid)
  cv_fold_id <- foldid
  cv_fold_id
}

check_nfold <- function(nfolds) {
  stopifnot(is.numeric(nfolds) & nfolds >= 2)
  check_integer_warning(
    nfolds,
    "nfolds should be an integer value. It is coerced to be as.integer(nfolds). "
  )
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

MULTIVARIATE_RESPONSE <- c("mgaussian", "multinomial")

.onUnload <- function(libpath) {
  library.dynam.unload("abess", libpath)
}
