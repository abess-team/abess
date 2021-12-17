#' @title Generate matrix with sparse principal component
#'
#' @description Generate simulated matrix that its principal component are
#' sparse linear combination of its columns.
#'
#' @inheritParams generate.matrix
#' @param support.size A integer specify the number of non-zero entries in the first column of loading matrix.
#' @param sparse.loading A \code{p}-by-\code{p} sparse orthogonal matrix. 
#' If it is supplied, \code{support.size} would be omit. 
#' @param sigma A numerical vector with length \code{p} specify the standard deviation of each columns.
#' Default \code{sigma = NULL} implies it is determined by \code{snr}. 
#' If it is supplied, \code{support.size} would be omit. 
#'
#' @return A \code{list} object comprising:
#' \item{x}{An \eqn{n}-by-\eqn{p} matrix.}
#' \item{coef}{The sparse loading matrix used to generate x.}
#' \item{support.size}{A vector recording the number of non-zero entries in each .}
#'
#' @details The methods for generating the matrix is detailedly described in the APPENDIX A: Data generation Section in Schipper et al (2021).
#'
#' @references Model selection techniques for sparse weight-based principal component analysis. de Schipper, Niek C and Van Deun, Katrijn. Journal of Chemometrics. 2021. \doi{10.1002/cem.3289}.
#'
#' @export
#'
generate.spc.matrix <- function(n, p, support.size = 3, snr = 20, sigma = NULL, sparse.loading = NULL, seed = 1) {
  set.seed(seed)
  kpc.num <- 1

  stopifnot(length(n) == 1)
  stopifnot(is.numeric(n))
  check_integer_warning(n, "TODO")
  n <- as.integer(n)

  stopifnot(length(p) == 1)
  stopifnot(is.numeric(p))
  check_integer_warning(p, "TODO")
  p <- as.integer(p)

  if (is.null(sparse.loading)) {
    ###### sparse matrix (The result of QR decomposition is not correct) ######
    # row_index <- sample(1:2, support.size - 1, replace = FALSE)
    # col_index <- rep(0, support.size - 1)
    # row_index <- c(row_index, 0:(p - 1))
    # col_index <- c(col_index, 0:(p - 1))
    # value <- stats::runif(p + support.size - 1)
    # sparse.loading <- Matrix::sparseMatrix(
    #   dims = c(p, p),
    #   i = row_index,
    #   j = col_index,
    #   x = value,
    #   index1 = FALSE
    # )
    # sparse.loading <- Matrix::qr.fitted(Matrix::qr(sparse.loading), sparse.loading)

    ###### dense matrix ######
    sparse.loading <- diag(p)
    sparse.loading[, kpc.num] <- c(runif(support.size), rep(0, p - support.size))
    sparse.loading <- qr.Q(qr(sparse.loading))
    sparse.loading <- methods::as(sparse.loading, "dgCMatrix")
  } else {
    stopifnot(class(sparse.loading) == "dgCMatrix")
    stopifnot(ncol(sparse.loading) == p)
  }
  support.size <- Matrix::colSums(sparse.loading != 0)

  if (is.null(sigma)) {
    noise <- stats::rnorm(p - 1)
    signal_scale <- snr * sum(noise^2)
    signal <- stats::rnorm(1, sd = sqrt(signal_scale))
    sigma_mat <- diag(c(signal, noise))
    sigma_mat <- abs(sigma_mat)
  } else {
    sigma_mat <- diag(sigma)
  }
  cov_mat <- sparse.loading %*% sigma_mat %*% t(sparse.loading)
  x <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = cov_mat)

  set.seed(NULL)
  colnames(x) <- paste0("x", 1:p)
  return(list("x" = x, "coef" = sparse.loading, support.size = support.size))
}
