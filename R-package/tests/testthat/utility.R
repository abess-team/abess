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
# This option is invalid for \code{cortype = 3}.
# @param censoring Whether data is censored or not. Valid only for \code{family = "cox"}. Default is \code{TRUE}.
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
#' For \code{family = "gaussian"} , the data model is
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
#' In the above models, \eqn{\epsilon \sim N(0,
#' \sigma^2 ),} where \eqn{\sigma^2} is determined by the \code{snr}.
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
                            "gaussian",
                            "binomial",
                            "poisson",
                            "cox",
                            "mgaussian",
                            "multinomial",
                            "gamma"
                          ),
                          beta = NULL,
                          cortype = 1,
                          snr = 10,
                          weibull.shape = 1,
                          uniform.max = 1,
                          y.dim = 3,
                          class.num = 3,
                          seed = 1) {
  sigma <- 1

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
    m <- 5 * sqrt(2 * log(p) / n)
    M <- 100 * m
    if (is.null(input_beta)) {
      beta[nonzero] <- stats::runif(support.size, m, M)
    } else {
      beta <- input_beta
    }
    sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)

    y <- x %*% beta + rnorm(n, 0, sigma)
  }
  if (family == "binomial") {
    m <- 5 * sqrt(2 * log(p) / n)
    if (is.null(input_beta)) {
      beta[nonzero] <- stats::runif(support.size, 2 * m, 10 * m)
    } else {
      beta <- input_beta
    }
    sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)

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
    sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)

    eta <- x %*% beta + rnorm(n, 0, sigma)
    time <-
      (-log(stats::runif(n)) / drop(exp(eta)))^(1 / weibull.shape)
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
    sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
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
    sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
    sigma <- diag(sigma)
    sigma <- sigma * y_cor
    epsilon <-
      MASS::mvrnorm(
        n = n,
        mu = rep(0, y_dim),
        Sigma = sigma
      )
    y <- x %*% beta + epsilon
    colnames(y) <- paste0("y", 1:y_dim)
  }
  if (family == "multinomial") {
    m <- 5 * sqrt(2 * log(p) / n)
    M <- 100 * m
    if (is.null(input_beta)) {
      beta[nonzero, ] <- matrix(stats::runif(support.size * y_dim, m, M),
        ncol = y_dim
      )
    } else {
      beta <- input_beta
    }
    sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
    sigma <- diag(sigma)
    sigma <- sigma * y_cor
    epsilon <- MASS::mvrnorm(n, rep(0, y_dim), sigma)
    prob_y <- x %*% beta
    prob_y <- exp(prob_y)
    prob_y <-
      sweep(prob_y,
        MARGIN = 1,
        STATS = rowSums(prob_y),
        FUN = "/"
      )
    y <- apply(prob_y, 1, function(x) {
      sample(0:2, size = 1, prob = x)
    })
  }
  if (family == "gamma") {
    m <- 5 * sqrt(2 * log(p) / n)
    if (is.null(input_beta)) {
      # TODO
      beta[nonzero] <- stats::runif(support.size, m, 100 * m)
    } else {
      beta <- input_beta
    }
    sigma <- sqrt((t(beta) %*% Sigma %*% beta) / snr)
    eta <- x %*% beta + stats::rnorm(n, 0, sigma)
    eta <- abs(eta)
    # TODO the shape para of gamma is uniform in [0.1,100.1]
    shape_para <- 100 * runif(n) + 0.1
    y <- stats::rgamma(n, shape = shape_para, rate = shape_para * eta)
  }
  set.seed(NULL)

  colnames(x) <- paste0("x", 1:p)
  return(list(x = x, y = y, beta = beta))
}

#' @title Generate simulated data
#'
#' @description Generate simulated data under the
#' group generalized linear model and group Cox proportional hazard model.
#'
#' @param n The number of observations.
#' @param J The number of groups.
#' @param k The group sizes. Assume each group includes the same number of variables.
#' @param support.size The number of nonzero groups in the underlying regression
#' model.
#' @param rho A parameter used to characterize the inter-group correlation. Default is \code{0}.
#' @param family The distribution of the simulated response. \code{"gaussian"} for
#' univariate quantitative response, \code{"binomial"} for binary classification response,
#' \code{"poisson"} for counting response, \code{"cox"} for left-censored response.
#' @param cortype The correlation structure.
#' \code{cortype = 1} denotes the independence structure,
#' where the covariance matrix has \eqn{(i,j)} entry equals \eqn{I(i \neq j)}.
#' \code{cortype = 2} denotes the exponential structure,
#' where the covariance matrix has \eqn{(i,j)} entry equals \eqn{rho^{|i-j|}}.
#' code{cortype = 3} denotes the constant structure,
#' where the non-diagonal entries of covariance
#' matrix are \eqn{rho} and diagonal entries are 1.
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
#' @param seed Random seed. Default: \code{seed = 1}.
#' @param sigma1 The standard deviation of error term which corresponds to gaussian distribution. Default: \code{sigma1 = 1}.
#' @param sigma2 A parameter controlling the strength of coefficients. Assume \eqn{\gamma_i \sim N(0, sigma2^2)}, the \code{i}th coefficients of \code{j}th group are generated by
#' \eqn{\beta_{G_j,i}=\gamma_i-\frac{1}{k+1}\sum_{i=1}^{k+1} \gamma_i}.  Default: \code{sigma2 = 1}.
#' @return A \code{list} object comprising:
#' \item{x}{Design matrix of predictors.}
#' \item{y}{Response variable.}
#' \item{beta}{The coefficients used in the underlying group regression model.}
#' \item{group.index}{A vector of integers indicating the which group each variable is in.}
#' \item{true.group}{A vector of integers indicating the true group subset.}
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
#' For \code{family = "gaussian"} , the data model is
#' \deqn{Y = X \beta + \epsilon.}
#'
#' For \code{family= "binomial"}, the data model is \deqn{Prob(Y = 1) = \exp(X
#' \beta + \epsilon)/(1 + \exp(X \beta + \epsilon)).}
#'
#' For \code{family = "poisson"}, the data is modeled to have
#' an exponential distribution:
#' \deqn{Y = Exp(\exp(X \beta + \epsilon)).}
#'
#' For \code{family = "cox"}, the model for failure time \eqn{T} is
#' \deqn{T = (-\log(U / \exp(X \beta)))^{1/weibull.shape},}
#' where \eqn{U} is a uniform random variable with range [0, 1].
#' The centering time \eqn{C} is generated from
#' uniform distribution \eqn{[0, uniform.max]},
#' then we define the censor status as
#' \eqn{\delta = I(T \le C)} and observed time as \eqn{R = \min\{T, C\}}.
#'
#' In the above models, error term \eqn{\epsilon \sim N(0,
#' sigma1^2 )}.
#'
#' @author Yanhang Zhang
#'
#' @export
#'
#' @examples
#'
#' # Generate simulated data
#' n <- 500
#' J <- 10
#' k <- 5
#' support.size <- 5
#' dataset <- generate.group(n, J, k, support.size)
#' str(dataset)
generate.group <- function(n,
                           J,
                           k,
                           support.size = NULL,
                           rho = 0.5,
                           family = c("gaussian", "binomial", "poisson", "cox"),
                           cortype = 1,
                           weibull.shape = 1,
                           uniform.max = 1,
                           sigma1 = 1,
                           sigma2 = 1,
                           seed = 1) {
  family <- match.arg(family)
  set.seed(seed)
  group.index <- rep(1:J, each = k)

  if (is.null(support.size)) {
    stop("Please provide an integer to support.size.")
  }
  stopifnot(is.numeric(support.size) & support.size >= 1)

  if (cortype == 1) {
    Sigma <- diag(J)
  } else if (cortype == 2) {
    Sigma <- matrix(0, J, J)
    Sigma <- rho^(abs(row(Sigma) - col(Sigma)))
  } else if (cortype == 3) {
    Sigma <- matrix(rho, J, J)
    diag(Sigma) <- 1
  }
  if (cortype == 1) {
    x <- matrix(rnorm(n * J), nrow = n, ncol = J)
  } else {
    x <- MASS::mvrnorm(n, rep(0, J), Sigma)
  }
  z <- matrix(rnorm(n * J * k), nrow = n)
  z <- sapply(1:(J * k), function(i) {
    g <- floor((i - 1) / k) + 1
    return((x[, g] + z[, i]) / sqrt(2))
  })
  x <- matrix(unlist(z), n)
  beta <- rep(0, J * k)
  true.group <- sort(sample(1:J, support.size))
  nonzero <- as.vector(sapply(true.group, function(i) {
    return(((i - 1) * k + 1):(i * k))
  }))
  beta[nonzero] <- gen.coef(support.size, k, sigma2, seed)
  if (family == "gaussian") {
    y <- x %*% beta + rnorm(n, 0, sigma1)
  }
  if (family == "binomial") {
    eta <- x %*% beta + rnorm(n, 0, sigma1)
    PB <- apply(eta, 1, generatedata2)
    y <- stats::rbinom(n, 1, PB)
  }
  if (family == "cox") {
    eta <- x %*% beta + rnorm(n, 0, sigma1)
    time <-
      (-log(stats::runif(n)) / drop(exp(eta)))^(1 / weibull.shape)
    ctime <- stats::runif(n, max = uniform.max)
    status <- (time < ctime) * 1
    censoringrate <- 1 - mean(status)
    time <- pmin(time, ctime)
    y <- cbind(time = time, status = status)
  }
  if (family == "poisson") {
    eta <- x %*% beta + stats::rnorm(n, 0, sigma1)
    eta <- ifelse(eta > 50, 50, eta)
    eta <- ifelse(eta < -50, -50, eta)
    eta <- exp(eta)
    y <- sapply(eta, stats::rpois, n = 1)
  }
  set.seed(NULL)

  colnames(x) <- paste0("x", 1:(J * k))
  return(list(
    x = x,
    y = y,
    beta = beta,
    group.index = group.index,
    true.group = true.group
  ))
}

gen.coef <- function(size, k, sigma, seed) {
  set.seed(seed)
  coef <- rep(0, size * k)
  for (i in 1:size) {
    temp <- stats::rnorm(k + 1, 0, sigma)
    coef[((i - 1) * k + 1):(i * k)] <- (temp - mean(temp))[-1]
  }
  return(coef)
}

generatedata2 <- function(eta) {
  a <- exp(eta) / (1 + exp(eta))
  if (is.infinite(exp(eta))) {
    a <- 1
  }
  return(a)
}

# #' Title
# #' @description recover beta after feature screening
# #' @note run this function
# #' before \code{object[["screening_A"]]} is removed.
# #' @noRd
# recover_beta <- function(object) {
#   if (length(object[["screening_A"]]) != 0) {
#     beta_all <- matrix(0, nrow = object[["nvars"]],
#                        ncol = length(object[["support.size"]]))
#     beta_all[object[["screening_A"]] + 1, ] <- object[["beta"]]
#     object[["beta"]] <- beta_all
#   }
# }


list.beta <- function(beta.mat, object, sparse) {
  beta.all <- matrix(0,
    nrow = length(object[["best.model"]][["beta"]]),
    ncol = ncol(beta.mat)
  )
  beta.all[object[["screening.index"]], ] <- beta.mat[[1]]
  if (sparse) {
    beta.all <- Matrix::Matrix(beta.all)
  }
  return(beta.all)
}

match_support_size <- function(object, support.size) {
  supp_size_index <- match(support.size, object[["support.size"]])
  if (anyNA(supp_size_index)) {
    stop("Arugments support.size comprises support sizes that are not in the abess object.")
  }
  supp_size_index
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
  if (length(data)) {
    namD <- names(data)

    for (i in namD) {
      if (is.character(data[[i]])) {
        stop(
          "Some columns in data are character!
             You may convert these columns to a dummy variable via
             model.matrix function or discard them."
        )
      } else if (is.factor(data[[i]])) {
        stop(
          "Some columns in data are factor!.
        You may convert these columns to a dummy variable via
             model.matrix function or discard them."
        )
      }
    }
  }
  data
}

MULTIVARIATE_RESPONSE <- c("mgaussian", "multinomial")

.onUnload <- function(libpath) {
  library.dynam.unload("abess", libpath)
}
