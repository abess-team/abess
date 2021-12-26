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
#' @param kpc.num A integer decide the number of principal components to be sequentially considered.
#' @param c.max an integer splicing size. The default of \code{c.max} is the maximum of 2 and \code{max(support.size) / 2}.
#' @param sparse.type If \code{sparse.type = "fpc"}, then best subset selection performs on the first principal component;
#' If \code{sparse.type = "kpc"}, then best subset selection would be sequentially performed on the first \code{kpc.num} number of principal components.
#' If \code{kpc.num} is supplied, the default is \code{sparse.type = "kpc"}; otherwise, is \code{sparse.type = "fpc"}.
#' @param tune.type The type of criterion for choosing the support size.
#' Available options are \code{"gic"}, \code{"ebic"}, \code{"bic"}, \code{"aic"} and \code{"cv"}.
#' Default is \code{"gic"}.
#' \code{tune.type = "cv"} is available only when \code{type = "predictor"}.
#' @param cor A logical value. If \code{cor = TRUE}, perform PCA on the correlation matrix;
#' otherwise, the covariance matrix.
#' This option is available only if \code{type = "predictor"}.
#' Default: \code{cor = FALSE}.
#' @param support.size It is a flexible input. If it is an integer vector.
#' It represents the support sizes to be considered for each principal component.
#' If it is a \code{list} object containing \code{kpc.num} number of integer vectors,
#' the i-th principal component consider the support size specified in the i-th element in the \code{list}.
#' The default is \code{support.size = NULL}, and some rules in details section are used to specify \code{support.size}.
#' @param splicing.type Optional type for splicing.
#' If \code{splicing.type = 1}, the number of variables to be spliced is
#' \code{c.max}, ..., \code{1}; if \code{splicing.type = 2},
#' the number of variables to be spliced is \code{c.max}, \code{c.max/2}, ..., \code{1}.
#' Default: \code{splicing.type = 1}.
#'
#' @details Adaptive best subset selection for principal component analysis (abessPCA) aim
#' to solve the non-convex optimization problem:
#' \deqn{-\arg\min_{v} v^\top \Sigma v, s.t.\quad v^\top v=1, \|v\|_0 \leq s, }
#' where \eqn{s} is support size.
#' Here, \eqn{\Sigma} is covariance matrix, i.e.,
#' \deqn{\Sigma = \frac{1}{n} X^{\top} X.}
#' A generic splicing technique is implemented to
#' solve this problem.
#' By exploiting the warm-start initialization, the non-convex optimization
#' problem at different support size (specified by \code{support.size})
#' can be efficiently solved.
#' 
#' The abessPCA can be conduct sequentially for each component. 
#' Please see the multiple principal components Section on the [webite](https://abess-team.github.io/abess/articles/v08-sPCA.html) 
#' for more details about this function. 
#' For \code{abesspca} function, the arguments \code{kpc.num} control the number of components to be consider. 
#'
#' When \code{sparse.type = "fpc"} but \code{support.size} is not supplied,
#' it is set as \code{support.size = 1:min(ncol(x), 100)} if \code{group.index = NULL};
#' otherwise, \code{support.size = 1:min(length(unique(group.index)), 100)}.
#' When \code{sparse.type = "kpc"} but \code{support.size} is not supplied,
#' then for 20\% principal components,
#' it is set as \code{min(ncol(x), 100)} if \code{group.index = NULL};
#' otherwise, \code{min(length(unique(group.index)), 100)}.
#'
#' @return A S3 \code{abesspca} class object, which is a \code{list} with the following components:
#' \item{coef}{A \eqn{p}-by-\code{length(support.size)} loading matrix of sparse principal components (PC),
#' where each row is a variable and each column is a support size;}
#' \item{nvars}{The number of variables.}
#' \item{sparse.type}{The same as input.}
#' \item{support.size}{The actual support.size values used. Note that it is not necessary the same as the input if the later have non-integer values or duplicated values.}
#' \item{ev}{A vector with size \code{length(support.size)}. It records the explained variance at each support size.}
#' \item{tune.value}{A value of tuning criterion of length \code{length(support.size)}.}
#' \item{kpc.num}{The number of principal component being considered.}
#' \item{var.pc}{The variance of principal components obtained by performing standard PCA.}
#' \item{cum.var.pc}{Cumulative sums of \code{var.pc}.}
#' \item{var.all}{If \code{sparse.type = "fpc"},
#' it is the total standard deviations of all principal components.}
#' \item{cum.ev}{Cumulative sums of explained variance.}
#' \item{pev}{A vector with the same length as \code{ev}. It records the percent of explained variance (compared to \code{var.all}) at each support size.}
#' \item{pev.pc}{It records the percent of explained variance (compared to \code{var.pc}) at each support size.}
#' \item{call}{The original call to \code{abess}.}
#' It is worthy to note that, if \code{sparse.type == "kpc"}, the \code{ev}, \code{tune.value}, \code{pev} and \code{pev.pc} in list are \code{list} objects.
#'
#' @author Jin Zhu, Junxian Zhu, Ruihuang Liu, Junhao Huang, Xueqin Wang
#' 
#' @note Some parameters not described in the Details Section is explained in the document for \code{\link{abess}} 
#' because the meaning of these parameters are very similar. 
#' 
#' @references A polynomial algorithm for best-subset selection problem. Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang. Proceedings of the National Academy of Sciences Dec 2020, 117 (52) 33117-33123; \doi{10.1073/pnas.2014241117}
#' @references Sparse principal component analysis. Hui Zou, Hastie Trevor, and Tibshirani Robert. Journal of computational and graphical statistics 15.2 (2006): 265-286. \doi{10.1198/106186006X113430}
#'
#' @export
#' 
#' @md
#'
#' @seealso \code{\link{print.abesspca}},
#' \code{\link{coef.abesspca}},
#' \code{\link{plot.abesspca}}.
#'
#' @examples
#' \donttest{
#' library(abess)
#'
#' ## predictor matrix input:
#' head(USArrests)
#' pca_fit <- abesspca(USArrests)
#' pca_fit
#' plot(pca_fit)
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
#'   support.size = 1:4
#' )
#' coef(pca_fit)
#' plot(pca_fit)
#' plot(pca_fit, "coef")
#'
#' ## select support size via cross-validation ##
#' n <- 500
#' p <- 50
#' support_size <- 3
#' dataset <- generate.spc.matrix(n, p, support_size, snr = 20)
#' spca_fit <- abesspca(dataset[["x"]], tune.type = "cv", nfolds = 5)
#' plot(spca_fit, type = "tune")
#' }
abesspca <- function(x,
                     type = c("predictor", "gram"),
                     sparse.type = c("fpc", "kpc"),
                     cor = FALSE,
                     support.size = NULL,
                     kpc.num = ifelse(sparse.type == "fpc", 1, 2),
                     tune.type = c("gic", "aic", "bic", "ebic", "cv"),
                     nfolds = 5,
                     foldid = NULL,
                     ic.scale = 1.0,
                     c.max = NULL,
                     always.include = NULL,
                     group.index = NULL,
                     splicing.type = 1,
                     max.splicing.iter = 20,
                     warm.start = TRUE,
                     num.threads = 0,
                     ...) {
  support.num <- NULL
  important.search <- NULL

  ## check number of thread:
  stopifnot(is.numeric(num.threads) & num.threads >= 0)
  num_threads <- as.integer(num.threads)

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
  sparse.type <- match.arg(sparse.type)
  stopifnot(kpc.num >= 1)
  check_integer_warning(kpc.num, "kpc.num should be an integer. It is coerced to as.integer(kpc.num).")
  sparse.type <- ifelse(kpc.num == 1, "fpc", "kpc")

  ## compute gram matrix
  sparse_matrix <- FALSE
  cov_type <- match.arg(type)
  if (cov_type == "gram") {
    stopifnot(dim(x)[1] == dim(x)[2])
    stopifnot(all(t(x) == x))
    ## eigen values:
    eigen_value <- eigen(x, only.values = TRUE)[["values"]]
    eigen_value <- (eigen_value + abs(eigen_value)) / 2
    gram_x <- x
    x <- matrix(0, ncol = nvars, nrow = 1)
  } else {
    stopifnot(length(cor) == 1)
    stopifnot(is.logical(cor))
    ## eigen values:
    if (!cor) {
      singular_value <- (svd(scale(x, center = TRUE, scale = FALSE))[["d"]])^2 # improve runtimes
      eigen_value <- singular_value / nobs
    } else {
      singular_value <- (svd(scale(x, center = TRUE, scale = TRUE))[["d"]])^2 # improve runtimes
      eigen_value <- singular_value / (nobs - 1)
    }

    if (sparse_X) {
      if (cor) {
        gram_x <- sparse.cov(x, cor = TRUE)
      } else {
        gram_x <- sparse.cov(x)
      }
      x <- map_dgCMatrix2entry(x)
      sparse_matrix <- TRUE
    } else {
      if (cor) {
        gram_x <- stats::cor(x)
      } else {
        gram_x <- stats::cov(x)
      }
    }
    if (!cor) {
      gram_x <- ((nobs - 1) / nobs) * gram_x
    }
    # x <- round(x, digits = 13)
  }

  # if (sparse.type == "fpc") {
  #   eigen_value <- eigen_value[1]
  # }
  pc_variance <- eigen_value
  total_variance <- sum(eigen_value)

  ## total variance:
  # svdobj <- svd(x)
  # stopifnot(all(svdobj[["d"]] > 0))
  # total_variance <- sum((svdobj[["d"]])^2)
  # v <- svdobj[["v"]]

  ## check lambda:
  lambda <- 0
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
    if (kpc.num == 1) {
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
      } else {
        s_num <- min(nvars, 100)
      }
      s_list <- as.list(rep(s_num, kpc.num))
    }
  } else {
    stopifnot(any(is.numeric(support.size) & support.size >= 0))
    if (group_select) {
      stopifnot(max(support.size) <= ngroup)
    } else {
      stopifnot(max(support.size) <= nvars)
    }
    if (kpc.num == 1) {
      support.size <- sort(support.size)
      support.size <- unique(support.size)
    } else {
      if (class(support.size) == "list") {
        stopifnot(length(support.size) == kpc.num)
        support.size <- lapply(support.size, function(x) {
          support.size <- unique(support.size)
          support.size
        })
      } else if (is.vector(support.size)) {
        support.size <- sort(support.size)
        support.size <- unique(support.size)
        support.size <- rep(list(support.size), kpc.num)
      } else {
        stop("support.size must be vector or list.")
      }
    }
    s_list <- support.size
  }
  s_list_bool_nrow <- ifelse(group_select, ngroup, nvars)
  if (class(s_list) == "list") {
    s_list_bool <- matrix(0, nrow = s_list_bool_nrow, ncol = kpc.num)
    for (i in 1:kpc.num) {
      s_list_bool[s_list[[i]], ] <- 1
    }
  } else {
    s_list_bool <- matrix(0, nrow = s_list_bool_nrow, ncol = 1)
    s_list_bool[s_list, ] <- 1
  }

  ## check C-max:
  s_list_max <- max(unlist(s_list))
  if (is.null(c.max)) {
    c_max <- max(c(2, round(s_list_max / 2)))
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
    if (length(always.include) > s_list_max) {
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

  ##
  tune_type <- match.arg(tune.type)
  if (cov_type == "gram" && tune_type == "cv") {
    warnings("Cross validation is not allow when input a gram matrix.
             Coerce into tune.type = 'gic'.")
    tune_type <- "gic"
  }
  ic_type <- map_tunetype2numeric(tune_type)
  if (tune_type != "cv") {
    nfolds <- 1
    cv_fold_id <- integer(0)
  } else {
    if (is.null(foldid)) {
      cv_fold_id <- integer(0)
      nfolds <- check_nfold(nfolds)
    } else {
      cv_fold_id <- check_foldid(foldid, nobs)
      nfolds <- length(unique(nfolds))
    }
  }

  ## Cpp interface:
  result <- abessPCA_API(
    x = x,
    n = nobs,
    p = nvars,
    normalize_type = as.integer(0),
    weight = c(1.0),
    sigma = gram_x,
    # algorithm_type = 6,
    max_iter = max_splicing_iter,
    exchange_num = c_max,
    path_type = 1,
    is_warm_start = warm.start,
    ic_type = 1,
    ic_coef = ic.scale,
    Kfold = nfolds,
    sequence = s_list_bool,
    s_min = 0,
    s_max = 10,
    screening_size = -1,
    g_index = g_index,
    always_select = always_include,
    early_stop = FALSE,
    thread = num.threads,
    sparse_matrix = sparse_matrix,
    splicing_type = splicing_type,
    sub_search = important_search,
    cv_fold_id = cv_fold_id,
    pca_num = kpc.num, 
    A_init = as.integer(c())
  )

  # result[["beta"]] <- NULL
  # result[["coef0"]] <- NULL
  # result[["train_loss"]] <- NULL
  # result[["ic"]] <- NULL
  # result[["coef0_all"]] <- NULL
  # result[["ic_all"]] <- NULL
  # result[["test_loss_all"]] <- NULL

  if (sparse.type == "fpc") {
    names(result)[which(names(result) == "beta_all")] <- "coef"
    # result[["coef"]] <- result[["beta_all"]]
    result[["ev"]] <- -result[["train_loss_all"]][, 1]
    # names(result)[which(names(result) == "train_loss_all")] <- "ev"
    # result[["ev"]] <- - result[["ev"]][, 1]
    result[["coef"]] <- do.call("cbind", result[["coef"]])
    result[["coef"]] <- Matrix::Matrix(result[["coef"]],
      sparse = TRUE,
      dimnames = list(vn, as.character(s_list))
    )
    result[["beta"]] <- NULL
    result[["coef0"]] <- NULL
    result[["train_loss"]] <- NULL
    result[["ic"]] <- NULL
    result[["lambda"]] <- NULL
    result[["coef0_all"]] <- NULL
    result[["train_loss_all"]] <- NULL
    if (tune_type == "cv") {
      names(result)[which(names(result) == "ic_all")] <- "tune.value"
      result[["test_loss_all"]] <- NULL
    } else {
      names(result)[which(names(result) == "test_loss_all")] <- "tune.value"
      result[["ic_all"]] <- NULL
    }
    result[["tune.value"]] <- as.vector(result[["tune.value"]])
    result[["effective_number_all"]] <- NULL
  } else {
    coef_list <- lapply(result, function(x) {
      x[["beta_all"]]
    })
    for (i in 1:kpc.num) {
      coef_list[[i]] <- Matrix::Matrix(do.call("cbind", coef_list[[i]]),
        sparse = TRUE,
        dimnames = list(vn, as.character(s_list[[i]]))
      )
    }
    ev_list <- list(-result[[1]][["train_loss_all"]][, 1])
    for (i in 2:kpc.num) {
      ev_vec <- c()
      tmp <- coef_list[[1]]
      tmp <- tmp[, ncol(tmp), drop = FALSE]
      j <- 2
      while (j < i) {
        tmp2 <- coef_list[[j]]
        tmp <- cbind(tmp, tmp2[, ncol(tmp2), drop = FALSE])
        j <- j + 1
      }
      tmp2 <- coef_list[[j]]
      for (k in 1:ncol(tmp2)) {
        ev_vec <- c(ev_vec, sum(adjusted_variance_explained(gram_x, cbind(tmp, tmp2[, k]))))
      }
      ev_list[[i]] <- ev_vec
    }
    if (tune_type == "cv") {
      tune_value <- lapply(result, function(x) {
        x[["test_loss_all"]]
      })
    } else {
      tune_value <- lapply(result, function(x) {
        x[["ic_all"]]
      })
    }
    result <- NULL
    result[["coef"]] <- coef_list
    result[["ev"]] <- ev_list
    result[["tune.value"]] <- tune_value
  }
  result[["kpc.num"]] <- kpc.num
  result[["var.pc"]] <- pc_variance
  result[["cum.var.pc"]] <- cumsum(pc_variance)
  result[["var.all"]] <- total_variance
  if (sparse.type == "fpc") {
    result[["pev"]] <- result[["ev"]] / total_variance
    result[["pev.pc"]] <- result[["ev"]] / pc_variance[1]
  } else {
    result[["pev"]] <- lapply(result[["ev"]], function(x) {
      x / total_variance
    })
    result[["pev.pc"]] <- lapply(1:kpc.num, function(i) {
      result[["ev"]][[i]] / result[["cum.var.pc"]][i]
    })
  }
  result[["nvars"]] <- nvars
  result[["sparse.type"]] <- sparse.type
  result[["support.size"]] <- s_list
  result[["tune.type"]] <- tune_type

  result[["call"]] <- match.call()
  class(result) <- "abesspca"
  result
}

adjusted_variance_explained <- function(covmat, loading) {
  loading <- as.matrix(loading)
  normloading <- sqrt(apply(loading^2, 2, sum))
  pc <- covmat %*% t(t(loading) / normloading)
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
