#' Adaptive best subset selection for robust principal component analysis
#'
#' @description Decompose a matrix into the summation of
#' low-rank matrix and sparse matrix via the best subset selection approach
#'
#' @inheritParams abess.default
# ’
#' @param x A matrix object. 
#' @param rank A positive integer value specify the rank of the low-rank matrix. 
#' @param support.size An integer vector representing the alternative support sizes. 
#' Only used for tune.path = "sequence". Strongly suggest its minimum value larger than \code{min(dim(x))}.  
#' @param tune.type The type of criterion for choosing the support size. Available options are "gic", "ebic", "bic" and "aic". 
#' Default is "gic".
#' 
#' @note 
#' Some parameters not described in the Details Section is explained in the document for \code{\link{abess}} 
#' because the meaning of these parameters are very similar. 
#' 
#' At present, \eqn{l_2} regularization and group selection are not support, 
#' and thus, set \code{lambda} and \code{group.index} have no influence on the output. 
#' This feature will coming soon. 
#' 
#' @return A S3 \code{abessrpca} class object, which is a \code{list} with the following components:
#' \item{S}{A list with \code{length(support.size)} elements,
#' each of which is a sparse matrix estimation;}
#' \item{L}{The low rank matrix estimation.}
#' \item{nobs}{The number of sample used for training.}
#' \item{nvars}{The number of variables used for training.}
#' \item{rank}{The rank of matrix \code{L}.}
#' \item{loss}{The loss of objective function.}
#' \item{tune.value}{A value of tuning criterion of length \code{length(support.size)}.}
#' \item{support.size}{The actual support.size values used.
#' Note that it is not necessary the same as the input if the later have non-integer values or duplicated values.}
#' \item{tune.type}{The criterion type for tuning parameters.}
#' \item{call}{The original call to \code{abessrpca}.}
#' 
#' @details Adaptive best subset selection for robust principal component analysis aim to find two latent matrices \eqn{L} and \eqn{S} such that the original matrix \eqn{X} can be appropriately approximated:
#' \deqn{x = L + S + N,} 
#' where \eqn{L} is a low-rank matrix, \eqn{S} is a sparse matrix, \eqn{N} is a dense noise matrix. 
#' Generic splicing technique can be employed to solve this problem by iteratively improve the quality of the estimation of \eqn{S}. 
#' 
#' For a given support set \eqn{\Omega}, the optimization problem: 
#' \deqn{\min_S \| x - L - S\|_F^2 \;\;{\rm s.t.}\;\; S_{ij} = 0 {\rm for } (i, j) \in \Omega^c,}
#' still a non-convex optimization problem. We use the hard-impute algorithm proposed in one of the reference to solve this problem. 
#' The hard-impute algorithm is an iterative algorithm, people can set \code{max.newton.iter} and \code{newton.thresh} to 
#' control the solution precision of the optimization problem. 
#' (Here, the name of the two parameters are somehow abused to make the parameters cross functions have an unified name.) 
#' According to our experiments, 
#' we assign properly parameters to the two parameter as the default such that the precision and runtime are well balanced, 
#' we suggest users keep the default values unchanged. 
#' 
#' @export
#'
#' @references A polynomial algorithm for best-subset selection problem. Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang. Proceedings of the National Academy of Sciences Dec 2020, 117 (52) 33117-33123; \doi{10.1073/pnas.2014241117}
#' @references Emmanuel J. Candès, Xiaodong Li, Yi Ma, and John Wright. 2011. Robust principal component analysis? Journal of the ACM. 58, 3, Article 11 (May 2011), 37 pages. \doi{10.1145/1970392.1970395}
#' @references Mazumder, Rahul, Trevor Hastie, and Robert Tibshirani. Spectral regularization algorithms for learning large incomplete matrices. The Journal of Machine Learning Research 11 (2010): 2287-2322.
#'
#' @examples
#' \donttest{
#' library(abess)
#' n <- 30
#' p <- 30
#' true_S_size <- 60
#' true_L_rank <- 2
#' dataset <- generate.matrix(n, p, support.size = true_S_size, rank = true_L_rank)
#' res <- abessrpca(dataset[["x"]], rank = true_L_rank, support.size = 50:70)
#' print(res)
#' coef(res)
#' plot(res, type = "tune")
#' plot(res, type = "loss")
#' plot(res, type = "S")
#' }
abessrpca <- function(x,
                      rank,
                      support.size = NULL,
                      tune.path = c("sequence", "gsection"),
                      gs.range = NULL,
                      tune.type = c("gic", "aic", "bic", "ebic"),
                      ic.scale = 1.0,
                      lambda = 0,
                      always.include = NULL,
                      group.index = NULL,
                      c.max = NULL,
                      splicing.type = 2,
                      max.splicing.iter = 1,
                      warm.start = TRUE,
                      important.search = NULL,
                      max.newton.iter = 1,
                      newton.thresh = 1e-3,
                      num.threads = 0,
                      seed = 1,
                      ...) {
  ## strategy for tuning
  tune.path <- match.arg(tune.path)
  if (tune.path == "gsection") {
    path_type <- 2
  } else if (tune.path == "sequence") {
    path_type <- 1
  }

  ## check rank:
  stopifnot(!missing(rank))
  stopifnot(!anyNA(rank))
  stopifnot(all(rank >= 0))

  ## check number of thread:
  stopifnot(is.numeric(num.threads) & num.threads >= 0)
  num_threads <- as.integer(num.threads)

  ## check parameters for sub-optimization:
  # 1:
  if (!is.null(max.newton.iter)) {
    stopifnot(is.numeric(max.newton.iter) & max.newton.iter >= 1)
    max_newton_iter <- as.integer(max.newton.iter)
  } else {
    max_newton_iter <- 100
  }
  # 2:
  stopifnot(is.numeric(newton.thresh) & newton.thresh > 0)
  newton_thresh <- as.double(newton.thresh)

  ## check lambda:
  lambda <- 0
  stopifnot(!anyNA(lambda))
  stopifnot(all(lambda >= 0))

  ## check warm start:
  stopifnot(is.logical(warm.start))

  ## check splicing type
  stopifnot(length(splicing.type) == 1)
  stopifnot(splicing.type %in% c(1, 2))
  splicing_type <- as.integer(splicing.type)

  ## check max splicing iteration
  stopifnot(is.numeric(max.splicing.iter) & max.splicing.iter >= 1)
  max_splicing_iter <- as.integer(max.splicing.iter)

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
    if (anyNA(x) || any(is.infinite(x))) {
      stop("x has missing value or infinite value!")
    }
    if (nvars == 1) {
      ### some special handling
    }
  }
  vn <- colnames(x)
  if (is.null(vn)) {
    vn <- paste0("x", 1:nvars)
  }

  screening_num <- nobs * nvars

  ## group variable:
  group_select <- FALSE
  if (is.null(group.index)) {
    g_index <- 1:(nobs * nvars) - 1
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
  max_rank <- max(c(nvars, nobs))
  if (is.null(support.size)) {
    if (group_select) {
      s_list <- 0:min(c(ngroup, round(nobs / max_group_size / log(ngroup))))
    } else {
      min_support_set <- max(c(3 * rank, max_rank / 2))
      s_list <- min_support_set:round(1.5 * max(max_rank))
    }
  } else {
    stopifnot(any(is.numeric(support.size) & support.size >= 0))
    check_integer(support.size, "support.size must be a vector with integer value.")
    if (group_select) {
      stopifnot(max(support.size) <= ngroup)
    }
    stopifnot(max(support.size) <= nvars * nobs)
    support.size <- sort(support.size)
    support.size <- unique(support.size)
    s_list <- support.size
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

  # tune support size method:
  tune.type <- match.arg(tune.type)
  ic_type <- map_tunetype2numeric(tune.type)
  is_cv <- FALSE
  cv_fold_id <- integer(0)

  ## information criterion
  stopifnot(is.numeric(ic.scale))
  stopifnot(ic.scale >= 0)
  ic_scale <- as.integer(ic.scale)

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

  result_cpp <- abessRPCA_API(
    x = x,
    n = nobs,
    p = nvars,
    max_iter = max_splicing_iter,
    exchange_num = c_max,
    path_type = path_type,
    is_warm_start = warm.start,
    ic_type = 1,
    ic_coef = ic.scale,
    sequence = s_list,
    lambda_seq = rank,
    lambda_min = 0,
    lambda_max = 0,
    nlambda = 0,
    s_min = s_min,
    s_max = s_max,
    screening_size = -1,
    primary_model_fit_max_iter = max_newton_iter,
    primary_model_fit_epsilon = newton_thresh,
    g_index = g_index,
    always_select = always_include,
    early_stop = FALSE,
    thread = num.threads,
    sparse_matrix = sparse_X,
    splicing_type = splicing_type,
    sub_search = important_search
  )

  result_R <- list()
  S <- lapply(result_cpp[["beta_all"]], function(x) {
    non_zero_index <- which(x != 0)
    value <- x[non_zero_index]
    non_zero_index <- non_zero_index - 1
    col_index <- floor(non_zero_index / nobs)
    row_index <- non_zero_index %% nobs
    S_mat <- Matrix::sparseMatrix(
      dims = c(nobs, nvars),
      i = row_index, j = col_index,
      x = value, index1 = FALSE
    )
    S_mat
  })
  L <- lapply(S, function(y) {
    x - y
  })
  result_R[["S"]] <- S
  result_R[["L"]] <- L
  result_R[["nobs"]] <- nobs
  result_R[["nvars"]] <- nvars
  result_R[["rank"]] <- rank
  result_R[["loss"]] <- as.vector(result_cpp[["train_loss_all"]])
  result_R[["tune.value"]] <- as.vector(result_cpp[["ic_all"]])
  result_R[["support.size"]] <- s_list
  result_R[["tune.type"]] <- tune.type

  result_R[["call"]] <- match.call()
  class(result_R) <- "abessrpca"
  result_R
}
