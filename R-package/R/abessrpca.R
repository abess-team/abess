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
#' Only used for \code{tune.path = "sequence"}. 
#' Strongly suggest its minimum value larger than \code{min(dim(x))}.  
#' @param tune.type The type of criterion for choosing the support size. 
#' Available options are "gic", "ebic", "bic" and "aic". 
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
  tune.path <- match.arg(tune.path)
  tune.type <- match.arg(tune.type)
  
  
  
  data <- list(x=x)
  para <- Initialization_RPCA(
    rank = rank,
    support.size = support.size,
    tune.path = tune.path,
    gs.range = gs.range,
    tune.type = tune.type,
    ic.scale = ic.scale,
    lambda = 0,
    always.include = always.include,
    group.index = group.index,
    c.max = c.max,
    splicing.type = splicing.type,
    max.splicing.iter = max.splicing.iter,
    warm.start = warm.start,
    important.search = important.search,
    max.newton.iter = max.newton.iter,
    newton.thresh = newton.thresh,
    num.threads = num.threads
  )
  
  model <- initializate(para,data)
  para <- model$para
  data <- model$data
  
  x <- data$x
  rank <- para$rank
  tune.type <- para$tune.type
  warm.start <- para$warm.start
  num.threads <- para$num.threads
  splicing_type  <- para$splicing_type 
  max_splicing_iter <- para$max_splicing_iter
  nobs <- para$nobs
  nvars <- para$nvars
  sparse_X <- para$sparse_X
  g_index <- para$g_index
  s_list <- para$s_list
  c_max <- para$c_max
  important_search <- para$important_search
  always_include <- para$always_include
  path_type  <- para$path_type 
  max_newton_iter <- para$max_newton_iter
  newton_thresh <- para$newton_thresh
  s_min <- para$s_min
  s_max <- para$s_max
  
  
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
    sub_search = important_search, 
    A_init = as.integer(c())
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
