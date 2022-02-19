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
                     kpc.num = NULL,
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
  
  sparse.type <- match.arg(sparse.type)
  tune.type <- match.arg(tune.type)
  type <- match.arg(type)

  data <- list(x=x)
  
  para <- Initialization_PCA(
    type = type,
    sparse.type = sparse.type,
    cor = cor,
    support.size = support.size,
    kpc.num = kpc.num,
    tune.type = tune.type,
    nfolds = nfolds,
    foldid = foldid,
    ic.scale = ic.scale,
    c.max = c.max,
    always.include = always.include,
    group.index = group.index,
    splicing.type = splicing.type,
    max.splicing.iter = max.splicing.iter,
    warm.start = warm.start,
    num.threads = num.threads,
    support.num = NULL,
    important.search = NULL
  )

  model <- initializate(para,data)
  para <- model$para
  data <- model$data
  
  x <- data$x

  kpc.num <- para$kpc.num
  warm.start <- para$warm.start
  num_threads  <- para$num_threads 
  splicing_type  <- para$splicing_type 
  max_splicing_iter <- para$max_splicing_iter
  nobs <- para$nobs
  nvars <- para$nvars
  vn <- para$vn
  g_index <- para$g_index
  s_list <- para$s_list
  c_max <- para$c_max
  ic_scale <- para$ic_scale
  important_search <- para$important_search
  always_include <- para$always_include
  s_max <- para$s_max
  s_list_bool <- para$s_list_bool
  tune_type <- para$tune_type
  ic_type <- para$ic_type
  cv_fold_id <- para$cv_fold_id
  nfolds <- para$nfolds
  sparse.type <- para$sparse.type
  sparse_matrix <- para$sparse_matrix
  gram_x <- para$gram_x
  pc_variance <- para$pc_variance
  total_variance  <- para$total_variance 

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
    ic_coef = ic_scale,
    Kfold = nfolds,
    sequence = s_list_bool,
    s_min = 0,
    s_max = 10,
    screening_size = -1,
    g_index = g_index,
    always_select = always_include,
    early_stop = FALSE,
    thread = num_threads,
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

project_cov <- function(cov_mat, direction) {
  term2 <- (t(direction) %*% (cov_mat %*% as.matrix(direction)))[1, 1] * as.matrix(direction) %*% t(direction)
  term3 <- as.matrix(direction) %*% (t(direction) %*% cov_mat)
  term4 <- (cov_mat %*% as.matrix(direction)) %*% t(direction)
  cov_mat + term2 - term3 - term4
}
