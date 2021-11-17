#' Adaptive best subset selection for principal component analysis
#'
#' @inheritParams abess.default
#' @param x A matrix object. It can be either a predictor matrix
#' where each row is an observation and each column is a predictor or
#' a sample covariance/correlation matrix.
#' If \code{x} is a predictor matrix, it can be in sparse matrix format
#' (inherit from class \code{"dgCMatrix"} in package \code{Matrix}).
#' @param type If \code{type = "predictor"}, \code{x} is considered as the predictor matrix.
#' If \code{type = "gram"}, \code{x} is considered as a sample covariance or correlation matrix.
#' @param c.max an integer splicing size. The default of \code{c.max} is the maximum of 2 and \code{max(support.size) / 2}.
#' @param sparse.type If \code{sparse.type = "fpc"}, then best subset selection performs on the first principal component;
#' If \code{sparse.type = "kpc"}, then best subset selection performs on the first \eqn{K} principal components.
#' (The parameter will be discard in future version.)
#' @param cor A logical value. If \code{cor = TRUE}, perform PCA on the correlation matrix;
#' otherwise, the covariance matrix.
#' This option is available only if \code{type = "predictor"}.
#' Default: \code{cor = FALSE}.
#' @param support.size An integer vector. It represents the alternative support sizes when \code{sparse.type = "fpc"},
#' while each support size controls the sparsity of a principal component when \code{sparse.type = "kpc"}.
#' When \code{sparse.type = "fpc"} but \code{support.size} is not supplied,
#' it is set as \code{support.size = 1:min(ncol(x), 100)} if \code{group.index = NULL};
#' otherwise, \code{support.size = 1:min(length(unique(group.index)), 100)}.
#' When \code{sparse.type = "kpc"} but \code{support.size} is not supplied,
#' then for 20\% principal components,
#' it is set as \code{min(ncol(x), 100)} if \code{group.index = NULL};
#' otherwise, \code{min(length(unique(group.index)), 100)}.
#' @param splicing.type Optional type for splicing.
#' If \code{splicing.type = 1}, the number of variables to be spliced is
#' \code{c.max}, ..., \code{1}; if \code{splicing.type = 2},
#' the number of variables to be spliced is \code{c.max}, \code{c.max/2}, ..., \code{1}.
#' Default: \code{splicing.type = 1}.
#'
#' @details Adaptive best subset selection for principal component analysis aim
#' to solve the non-convex optimization problem:
#' \deqn{\arg\max_{v} v^\top \Sigma v, s.t.\quad v^\top v=1, \|v\|_0 \leq s, }
#' where \eqn{s} is support size. 
#' Here, \eqn{\Sigma} is covariance matrix, i.e., 
#' \deqn{\Sigma = \frac{1}{n} X^{\top} X.}
#' A generic splicing technique is implemented to
#' solve this problem.
#' By exploiting the warm-start initialization, the non-convex optimization
#' problem at different support size (specified by \code{support.size})
#' can be efficiently solved.
#'
#'
#' @return A S3 \code{abesspca} class object, which is a \code{list} with the following components:
#' \item{coef}{A \eqn{p}-by-\code{length(support.size)} loading matrix of sparse principal components (PC),
#' where each row is a variable and each column is a support size;}
#' \item{nvars}{The number of variables.}
#' \item{sparse.type}{The same as input.}
#' \item{support.size}{The actual support.size values used. Note that it is not necessary the same as the input if the later have non-integer values or duplicated values.}
#' \item{ev}{A vector with size \code{length(support.size)}. It records the explained variance at each support size.}
#' \item{cum.ev}{Cumulative sums of explained variance.}
#' \item{pev}{A vector with the same length as \code{ev}. It records the percentage of explained variance at each support size.}
#' \item{cum.pev}{Cumulative sums of the percentage of explained variance.}
#' \item{var.all}{If \code{sparse.type = "fpc"},
#' it is the total variance of the explained by first principal component;
#' otherwise, the total standard deviations of all principal components.}
#' \item{call}{The original call to \code{abess}.}
#'
#' @author Jin Zhu, Junxian Zhu, Ruihuang Liu, Junhao Huang, Xueqin Wang
#'
#' @references A polynomial algorithm for best-subset selection problem. Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang. Proceedings of the National Academy of Sciences Dec 2020, 117 (52) 33117-33123; DOI: 10.1073/pnas.2014241117
#' @references Sparse principal component analysis. Hui Zou, Hastie Trevor, and Tibshirani Robert. Journal of computational and graphical statistics 15.2 (2006): 265-286.
#'
#' @export
#'
#' @seealso \code{\link{print.abesspca}},
#' \code{\link{coef.abesspca}},
# \code{\link{plot.abesspca}}.
#'
#' @examples
#' \donttest{
#' library(abess)
#'
#' ## predictor matrix input:
#' head(USArrests)
#' pca_fit <- abesspca(USArrests)
#' pca_fit
#'
#' ## covariance matrix input:
#' cov_mat <- stats::cov(USArrests) * (nrow(USArrests) - 1) / nrow(USArrests)
#' pca_fit <- abesspca(cov_mat, type = "gram")
#' pca_fit  
#'
#' ## robust covariance matrix input:
#' rob_cov <- MASS::cov.rob(USArrests)[["cov"]]
#' rob_cov <- (rob_cov + t(rob_cov)) / 2
#' pca_fit <- abesspca(rob_cov, type = "gram")
#' pca_fit
#'
#' ## K-component principal component analysis
#' pca_fit <- abesspca(USArrests,
#'   sparse.type = "kpc",
#'   support.size = c(1, 2)
#' )
#' coef(pca_fit)
#' }
abesspca <- function(x,
                     type = c("predictor", "gram"),
                     sparse.type = c("fpc", "kpc"),
                     cor = FALSE,
                     support.size = NULL,
                     K = 1, 
                     tune.type = c("cv", "aic", "bic", "gic", "ebic"), 
                     seq.tune = NULL, 
                     nfolds = 5,
                     foldid = NULL, 
                     ic.scale = 1.0,
                     c.max = NULL,
                     lambda = 0,
                     always.include = NULL,
                     group.index = NULL,
                     splicing.type = 1,
                     max.splicing.iter = 20,
                     warm.start = TRUE, 
                     ...) {
  support.num <- NULL
  important.search <- NULL

  ## check warm start:
  stopifnot(is.logical(warm.start))

  ## check splicing type
  stopifnot(length(splicing.type) == 1)
  stopifnot(splicing.type %in% c(1, 2))
  splicing_type <- as.integer(splicing.type)

  ## check max splicing iteration
  stopifnot(is.numeric(max.splicing.iter) & max.splicing.iter >= 1)
  max_splicing_iter <- as.integer(max.splicing.iter)

  ## check x matrix:
  stopifnot(class(x)[1] %in% c("matrix", "data.frame", "dgCMatrix"))
  nvars <- ncol(x)
  nobs <- nrow(x)
  sparse_X <- ifelse(class(x)[1] %in% c("matrix", "data.frame"), FALSE, TRUE)
  if (sparse_X) {
  } else {
    if (is.data.frame(x)) {
      x <- as.matrix(x)
    }
    if (!is.numeric(x)) {
      stop("x must be a *numeric* matrix/data.frame!")
    }
    if (nvars == 1) {
      stop("x should have at least two columns!")
    }
    if (anyNA(x) || any(is.infinite(x))) {
      stop("x has missing value or infinite value!")
    }
  }
  vn <- colnames(x)
  if (is.null(vn)) {
    vn <- paste0("x", 1:nvars)
  }

  ## check sparse.type
  sparse_type <- match.arg(sparse.type)
  stopifnot(K >= 1)
  check_integer_warning(K, "K should be an integer. It is coerced to as.integer(K).")
  sparse_type <- ifelse(K == 1, "fpc", "kpc")
  if (is.null(seq.tune)) {
    if (K == 1) {
      seq.tune <- TRUE
    } else {
      seq.tune <- FALSE
    }
  }
  
  ## compute gram matrix
  cov_type <- match.arg(type)
  if (cov_type == "gram") {
    stopifnot(dim(x)[1] == dim(x)[2])
    stopifnot(all(t(x) == x))
    ## eigen values:
    eigen_value <- eigen(x, only.values = TRUE)[["values"]]
    eigen_value <- (eigen_value + abs(eigen_value)) / 2
    # singular_value <- sqrt(eigen_value)
  } else {
    stopifnot(length(cor) == 1)
    stopifnot(is.logical(cor))
    ## eigen values:
    if (!cor) {
      singular_value <- (svd(scale(x, center = TRUE, scale = FALSE))[["d"]])^2 # improve runtimes
      eigen_value <- singular_value / nobs
    } else {
      eigen_value <- rep(1, nvars)
    }

    if (sparse_X) {
      if (cor) {
        x <- sparse.cov(x, cor = TRUE)
      } else {
        x <- sparse.cov(x)
      }
    } else {
      if (cor) {
        x <- stats::cor(x)
      } else {
        x <- stats::cov(x)
      }
    }
    if (!cor) {
      x <- ((nobs - 1) / nobs) * x
    }
    # x <- round(x, digits = 13)
  }

  # if (sparse_type == "fpc") {
  #   eigen_value <- eigen_value[1]
  # }
  total_variance <- sum(eigen_value)

  ## total variance:
  # svdobj <- svd(x)
  # stopifnot(all(svdobj[["d"]] > 0))
  # total_variance <- sum((svdobj[["d"]])^2)
  # v <- svdobj[["v"]]

  ## check lambda:
  stopifnot(!anyNA(lambda))
  stopifnot(all(lambda >= 0))

  ## group variable:
  group_select <- FALSE
  if (is.null(group.index)) {
    g_index <- 1:nvars - 1
    ngroup <- 1
  } else {
    group_select <- TRUE
    gi <- unique(group.index)
    g_index <- match(gi, group.index) - 1
    g_df <- c(diff(g_index), nvars - max(g_index))
    ngroup <- length(g_index)
    max_group_size <- max(g_df)
  }

  # sparse level list (sequence):
  if (group_select) {
    s_max <- ngroup
  } else {
    s_max <- nvars
  }
  if (is.null(support.size)) {
    if (sparse_type == "fpc") {
      if (is.null(support.num)) {
        if (group_select) {
          s_num <- min(ngroup, 100)
        } else {
          s_num <- min(nvars, 100)
        }
      }
      # else {
      #   s_num <- support.num
      #   if (group_select) {
      #     stopifnot(s_num <= ngroup)
      #   } else {
      #     stopifnot(s_num <= nvars)
      #   }
      # }
      s_list <- round(seq.int(from = 1, to = s_max, length.out = s_num))
      s_list <- unique(s_list)
      k_num <- 1
    } else {
      if (group_select) {
        s_num <- min(ngroup, 100)
        k_num <- round(ngroup * 0.2)
      } else {
        s_num <- min(nvars, 100)
        k_num <- round(nvars * 0.2)
      }
      s_list <- rep(s_num, k_num)
    }
  } else {
    stopifnot(any(is.numeric(support.size) & support.size >= 0))
    if (group_select) {
      stopifnot(max(support.size) <= ngroup)
    } else {
      stopifnot(max(support.size) <= nvars)
    }
    if (sparse_type == "fpc") {
      support.size <- sort(support.size)
      support.size <- unique(support.size)
      k_num <- 1
    } else {
      k_num <- length(support.size)
    }
    s_list <- support.size
  }

  ## check C-max:
  if (is.null(c.max)) {
    c_max <- max(c(2, round(max(s_list) / 2)))
  } else {
    stopifnot(is.numeric(c.max) & c.max >= 1)
    check_integer_warning(
      c.max,
      "c.max should be an integer. It is coerced to as.integer(c.max)."
    )
    c_max <- as.integer(c.max)
  }

  ## check always included variables:
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
    if (length(always.include) > max(s_list)) {
      stop("always.include containing too many variables.
             The length of it should not exceed the maximum in support.size.")
    }

    always_include <- always.include
  }
  
  ## important searching: 
  if (is.null(important.search)) {
    important_search <- as.integer(0)
  } else {
    stopifnot(is.numeric(important.search))
    stopifnot(important.search >= 0)
    important_search <- as.integer(important.search)
  }

  ## Cpp interface:
  result <- abessPCA_API(
    x = matrix(0, ncol = nvars, nrow = 1),
    n = nobs,
    p = nvars,
    normalize_type = 1,
    weight = c(1),
    sigma = x,
    is_normal = FALSE,
    algorithm_type = 6,
    max_iter = max_splicing_iter,
    exchange_num = c_max,
    path_type = 1,
    is_warm_start = warm.start,
    is_tune = TRUE, 
    ic_type = 1,
    ic_coef = ic.scale,
    is_cv = FALSE,
    Kfold = 2,
    sequence = as.vector(s_list),
    lambda_seq = lambda,
    s_min = 0,
    s_max = 10,
    K_max = as.integer(20),
    epsilon = 0.0001,
    lambda_max = 0,
    lambda_min = 0,
    nlambda = 10,
    screening_size = -1,
    powell_path = 1,
    g_index = g_index,
    always_select = always_include,
    tau = 0.0,
    early_stop = FALSE,
    thread = 1,
    sparse_matrix = FALSE, ### to change
    splicing_type = splicing_type, 
    sub_search = important_search, 
    cv_fold_id = integer(0), 
    pca_num = K
  )

  # result[["beta"]] <- NULL
  # result[["coef0"]] <- NULL
  # result[["train_loss"]] <- NULL
  # result[["ic"]] <- NULL
  # result[["coef0_all"]] <- NULL
  # result[["ic_all"]] <- NULL
  # result[["test_loss_all"]] <- NULL

  result[["nvars"]] <- nvars
  result[["support.size"]] <- s_list
  result[["sparse.type"]] <- sparse_type
  if (sparse_type == "fpc") {
    result[["coef"]] <- result[["beta_all"]]
    result[["ev"]] <- -result[["train_loss_all"]][, 1]
    # names(result)[which(names(result) == "train_loss_all")] <- "ev"
    # result[["ev"]] <- - result[["ev"]][, 1]
  } else {
    result[["coef"]] <- result[["beta"]][[1]]
  }

  # names(result)[which(names(result) == 'beta_all')] <- "coef"
  result[["coef"]] <- do.call("cbind", result[["coef"]])
  result[["coef"]] <- Matrix::Matrix(result[["coef"]],
    sparse = TRUE,
    dimnames = list(vn, as.character(s_list))
  )

  if (sparse_type == "kpc") {
    # k_num <- ncol(result[["coef"]])
    # ev_vec <- numeric(k_num)
    # for (i in 1:k_num) {
    #   ev_vec[i] <- variance_explained(
    #     x_copy,
    #     result[["coef"]][, 1:i, drop = FALSE]
    #   )
    # }
    ev_vec <- adjusted_variance_explained(x_copy, result[["coef"]])
    result[["ev"]] <- ev_vec
    result[["cum.ev"]] <- cumsum(ev_vec)
  }

  result[["pev"]] <- result[["ev"]] / total_variance
  result[["cum.pev"]] <- cumsum(result[["pev"]])
  result[["var.all"]] <- total_variance

  result[["call"]] <- match.call()
  out <- result

  class(out) <- "abesspca"
  out
}

adjusted_variance_explained <- function(covmat, loading) {
  loading <- as.matrix(loading)
  normloading <- sqrt(apply(loading^2, 2, sum))
  pc <- covmat %*% t(t(loading)/normloading)
  # since it is covariance matrix, it is no need to square the diagonal values.
  ev <- abs(diag(qr.R(qr(pc))))   
  # ev <- (diag(qr.R(qr(pc))))^2
  ev
}

sparse.cov <- function(x, cor = FALSE) {
  n <- nrow(x)
  cMeans <- colMeans(x)
  covmat <- (as.matrix(crossprod(x)) - n * tcrossprod(cMeans)) / (n - 1)

  if (cor) {
    sdvec <- sqrt(diag(covmat))
    covmat <- covmat / crossprod(t(sdvec))
  }

  as.matrix(covmat)
}

project_cov <- function(cov_mat, direction) {
  term2 <- (t(direction) %*% (cov_mat %*% as.matrix(direction)))[1, 1] * as.matrix(direction) %*% t(direction)
  term3 <- as.matrix(direction) %*% (t(direction) %*% cov_mat)
  term4 <- (cov_mat %*% as.matrix(direction)) %*% t(direction)
  cov_mat + term2 - term3 - term4
}
