gen_rr_adjmat <- function(n_node,
                          degree,
                          beta,
                          alpha,
                          type = c("ferro", "glass"), 
                          degree_bound = TRUE) 
{
  g <- igraph::sample_k_regular(n_node, degree)
  adj <- as.matrix(igraph::as_adjacency_matrix(g, type = "both"))
  
  ind_nonzero <- which(adj != 0, arr.ind = TRUE)
  ind_nonzero <- ind_nonzero[ind_nonzero[, 1] > ind_nonzero[, 2], ]
  num_nonzero <- nrow(ind_nonzero)
  
  if (type == "ferro") {
    value <- c(alpha, rep(beta, num_nonzero - 1))
  } else if (type == "glass") {
    len_pos <- round(num_nonzero / 2)
    len_neg <- num_nonzero - len_pos
    value_pos <- c(alpha, rep(beta, len_pos - 1))
    value_neg <- c(-alpha, rep(-beta, len_neg - 1))
    value <- c(value_pos, value_neg)
  }
  if (!degree_bound) {
    value <- value / degree
  }
  
  value <- sample(value, size = num_nonzero)
  for (i in 1:num_nonzero) {
    adj[ind_nonzero[i, 1], ind_nonzero[i, 2]] <- value[i]
    adj[ind_nonzero[i, 2], ind_nonzero[i, 1]] <- value[i]
  }
  adj
}


#' A recursive function used to find the path that: 
#' (i) go through all node via the pre-specified edges;
#' (ii) the path have no loop.
#'
#' @param i 
#' @param sub_path 
#' @param adj 
#' @param n_node 
#' @noRd
#'
#'
#' @examples
#' index <- find_path(3, c(3), adj, n_node)
#' length(unique(index)) == n_node
#' alpha_adj_index <- cbind(index[-n_node], index[-1])
#' result <- c()
#' for (i in 1:nrow(alpha_adj_index)) {
#'   result <- c(result, adj[alpha_adj_index[i, 1], alpha_adj_index[i, 2]])
#' }
#' all(result == 1)
#' 
find_path <- function(i, sub_path, adj, n_node) {
  if (length(sub_path) == n_node) {
    return(sub_path)
  }
  i_adj <- which(adj[, i] != 0)
  i_adj <- setdiff(i_adj, sub_path)
  if (length(i_adj) == 0) {
    return(c(sub_path, NA))
  }
  for (j in i_adj) {
    new_sub_path <- c(sub_path, j)
    new_node <- find_path(j, new_sub_path, adj, n_node)
    if (anyNA(new_node)) {
    } else {
      return(new_node)
    }
  }
  return(c(sub_path, NA))
}


gen_rr_adjmat2 <- function(n_node,
                           degree,
                           beta,
                           alpha,
                           type = c("ferro", "glass")) 
{
  g <- igraph::sample_k_regular(n_node, degree)
  adj <- as.matrix(igraph::as_adjacency_matrix(g, type = "both"))
  
  type <- match.arg(type)
  
  ind_nonzero <- which(adj != 0, arr.ind = TRUE)
  ind_nonzero <- ind_nonzero[ind_nonzero[, 1] > ind_nonzero[, 2], ]
  num_nonzero <- nrow(ind_nonzero)
  
  for (i in 1:num_nonzero) {
    value <- beta
    if (type == "glass") {
      value <- sample(c(-1, 1), size = 1) * value
    }
    adj[ind_nonzero[i, 1], ind_nonzero[i, 2]] <- value
    adj[ind_nonzero[i, 2], ind_nonzero[i, 1]] <- value
  }
  
  if (alpha != beta) {
    value <- rep(alpha, n_node - 1)
    if (type == "glass") {
      value <- sample(c(-1, 1), size = (n_node - 1), replace = TRUE) * value
    }
    index <- find_path(1, c(1), adj, n_node)
    alpha_adj_index <- cbind(index[-n_node], index[-1])
    for (i in 1:(n_node - 1)) {
      adj[alpha_adj_index[i, 1], alpha_adj_index[i, 2]] <- value[i]
    }
  }
  adj
}

gen_4nn_cyc <- function(n_node, degree, beta, alpha, type = c("ferro", "glass", "glass_weak")) 
{
  adj <- matrix(0, n_node, n_node)
  index_lr <- (row(adj) - col(adj) == 1 & col(adj) %% sqrt(n_node) != 0)
  index_ud <- (row(adj) - col(adj) == sqrt(n_node))
  cycle_lr <- (col(adj) %% sqrt(n_node) == 0 & col(adj) - row(adj) == sqrt(n_node) - 1)
  cycle_ud <- (col(adj) <= sqrt(n_node) & row(adj) - col(adj) == sqrt(n_node) * (sqrt(n_node) - 1))
  index <- index_lr | index_ud | cycle_lr | cycle_ud
  
  adj[index] <- 1
  adj <- adj + t(adj)
  
  ind_nonzero <- which(adj != 0, arr.ind = TRUE)
  ind_nonzero <- ind_nonzero[ind_nonzero[, 1] > ind_nonzero[, 2],]
  num_nonzero <- nrow(ind_nonzero)
  
  if (type == "ferro") {
    value <- c(alpha, rep(beta, num_nonzero - 1))
  } else if (type == "glass") {
    len_pos <- round(num_nonzero / 2)
    len_neg <- num_nonzero - len_pos
    value_pos <- c(alpha, rep(beta, len_pos - 1))
    value_neg <- c(-alpha, rep(-beta, len_neg - 1))
    value <- c(value_pos, value_neg)
  } else if (type == "glass_weak") {
    value <- c(-alpha, rep(beta, num_nonzero - 1))
  }
  
  value <- sample(value, size = num_nonzero, replace = FALSE)
  for (i in 1:num_nonzero) {
    adj[ind_nonzero[i, 1], ind_nonzero[i, 2]] <- value[i]
    adj[ind_nonzero[i, 2], ind_nonzero[i, 1]] <- value[i]
  }
  adj
}

half_cyc_lattice <- function(half_n_node, lattice_col) {
  adj <- matrix(0, half_n_node, half_n_node)
  lattice_row <- half_n_node / lattice_col
  index_lr <- (row(adj) - col(adj) == 1 & col(adj) %% lattice_col != 0)
  index_ud <- (row(adj) - col(adj) == lattice_col)
  cycle_lr <- (col(adj) %% lattice_col == 0 & col(adj) - row(adj) == lattice_col - 1)
  cycle_ud <- (col(adj) <= lattice_col & row(adj) - col(adj) == lattice_col * (lattice_row - 1))
  index <- index_lr | index_ud | cycle_lr | cycle_ud
  adj[index] <- 1
  adj <- adj + t(adj)
  adj
}

gen_5nn_cyc <- function(n_node, lattice_col, degree, beta, alpha, type = c("ferro", "glass", "glass_weak")) {
  stopifnot(n_node %% 2 == 0)
  stopifnot(lattice_col >= 3)
  lattice_row <- n_node / lattice_col
  stopifnot(lattice_row == round(lattice_row))
  stopifnot(lattice_row >= 3)
  
  half_n_node <- n_node / 2
  half_adj <- half_cyc_lattice(half_n_node, lattice_col)
  adj <- rbind(cbind(half_adj, matrix(0, nrow = half_n_node, ncol = half_n_node)), 
               cbind(matrix(0, nrow = half_n_node, ncol = half_n_node), half_adj))
  adj[abs(row(adj) - col(adj)) == half_n_node] <- 1
  # rowSums(adj)
  
  ind_nonzero <- which(adj != 0, arr.ind = TRUE)
  ind_nonzero <- ind_nonzero[ind_nonzero[, 1] > ind_nonzero[, 2],]
  num_nonzero <- nrow(ind_nonzero)
  
  type <- match.arg(type)
  
  if (type == "ferro") {
    value <- c(alpha, rep(beta, num_nonzero - 1))
  } else if (type == "glass") {
    len_pos <- round(num_nonzero / 2)
    len_neg <- num_nonzero - len_pos
    value_pos <- c(alpha, rep(beta, len_pos - 1))
    value_neg <- c(-alpha, rep(-beta, len_neg - 1))
    value <- c(value_pos, value_neg)
  } else if (type == "glass_weak") {
    value <- c(-alpha, rep(beta, num_nonzero - 1))
  }
  
  value <- sample(value, size = num_nonzero, replace = FALSE)
  for (i in 1:num_nonzero) {
    adj[ind_nonzero[i, 1], ind_nonzero[i, 2]] <- value[i]
    adj[ind_nonzero[i, 2], ind_nonzero[i, 1]] <- value[i]
  }
  adj
}

gen_ld_adjmat <- function(p, degree, alpha) {
  adj <- matrix(0, p, p)
  adj[1:(degree+1), 1:(degree+1)] <- 1
  adj[lower.tri(adj, diag = TRUE)] <- 0
  adj <- alpha * adj
  adj <- t(adj) + adj
  adj
}

# if (type == 13)
#   theta <- gen_5nn_cyc(p, lattice_col, degree, beta, alpha, type = "ferro")
# if (type == 14)
#   theta <- gen_5nn_cyc(p, lattice_col, degree, beta, alpha, type = "glass")
# if (type == 15)
#   theta <- gen_5nn_cyc(p, lattice_col, degree, beta, alpha, type = "glass_weak")
# if (type == 16)
#   theta <- gen_rr_adjmat2(p, degree, beta, alpha, type = "ferro")
# if (type == 17)
#   theta <- gen_rr_adjmat2(p, degree, beta, alpha, type = "glass")
# # chain structure: size = p - 1
# if (type == 1) {
#   theta <- matrix(0, p, p)
#   # for(i in 1: p) {
#   #   theta[i, i] <- sample(c(-0.5, 0, 0.5), 1)
#   # }
#   for (i in 1:(p - 1)) {
#     theta[i, i + 1] <- sample(c(0.5,-0.5), 1)
#     theta[i + 1, i] <- theta[i, i + 1]
#   }
# }
# # random graph
# if (type == 2) {
#   # set.seed(seed)
#   sigma_inv <-
#     fastclime::fastclime.generator(
#       n = 20,
#       d = p,
#       graph = "random",
#       verbose = FALSE
#     )
#   sparse <- as.matrix(sigma_inv$theta)
#   theta <- matrix(0, p, p)
#   # for(i in 1:p) {
#   #   theta[i, i] <- sample(c(-0.5, 0, 0.5), 1)
#   # }
#   for (i in 1:(p - 1)) {
#     for (j in (i + 1):p) {
#       if (sparse[i, j] != 0) {
#         theta[i, j] <- sample(c(0.5,-0.5), 1)
#         theta[j, i] <- theta[i, j]
#       }
#     }
#   }
# }
# # 4 nearest neighbor: size = 2 * p
# if (type == 3) {
#   theta <- matrix(0, p, p)
#   # for(i in 1:p) {
#   #   theta[i, i] <- sample(c(-0.5, 0, 0.5), 1)
#   # }
#   sqr <- sqrt(p)
#   for (i in 0:(sqr - 1)) {
#     for (j in 1:(sqr - 1)) {
#       theta[i * sqr + j, i * sqr + j + 1] <- sample(c(-0.5, 0.5), 1)
#       theta[i * sqr + j + 1, i * sqr + j] <-
#         theta[i * sqr + j, i * sqr + j + 1]
#     }
#   }
#   for (i in 0:(sqr - 2)) {
#     for (j in 1:sqr) {
#       theta[i * sqr + j, i * sqr + j + sqr] <- sample(c(-0.5, 0.5), 1)
#       theta[i * sqr + j + sqr, i * sqr + j] <-
#         theta[i * sqr + j, i * sqr + j + sqr]
#     }
#   }
# }
# # 8 nearest neighbor: size = 4 * p
# if (type == 4) {
#   theta <- matrix(0, p, p)
#   # for(i in 1:p) {
#   #   theta[i, i] <- sample(c(-0.5, 0, 0.5), 1)
#   # }
#   sqr <- sqrt(p)
#   for (i in 0:(sqr - 1)) {
#     for (j in 1:(sqr - 1)) {
#       theta[i * sqr + j, i * sqr + j + 1] <- sample(c(-0.5, 0.5), 1)
#       theta[i * sqr + j + 1, i * sqr + j] <-
#         theta[i * sqr + j, i * sqr + j + 1]
#     }
#   }
#   for (i in 0:(sqr - 2)) {
#     for (j in 1:sqr) {
#       theta[i * sqr + j, i * sqr + j + sqr] <- sample(c(-0.5, 0.5), 1)
#       theta[i * sqr + j + sqr, i * sqr + j] <-
#         theta[i * sqr + j, i * sqr + j + sqr]
#     }
#   }
#   for (i in 0:(sqr - 2)) {
#     for (j in 1:(sqr - 1)) {
#       theta[i * sqr + j, i * sqr + j + sqr + 1] <- sample(c(-0.5, 0.5), 1)
#       theta[i * sqr + j + sqr + 1, i * sqr + j] <-
#         theta[i * sqr + j, i * sqr + j + sqr + 1]
#     }
#   }
#   for (i in 0:(sqr - 2)) {
#     for (j in 2:sqr) {
#       theta[i * sqr + j, i * sqr + j + sqr - 1] <- sample(c(-0.5, 0.5), 1)
#       theta[i * sqr + j + sqr - 1, i * sqr + j] <-
#         theta[i * sqr + j, i * sqr + j + sqr - 1]
#     }
#   }
# }

sim_theta <- function(p, type, graph_seed, beta, degree, alpha) {
  set.seed(graph_seed)  
  if (type == 1)
    theta <- gen_rr_adjmat(p, degree, beta, alpha, type = "ferro")
  if (type == 2)
    theta <- gen_rr_adjmat(p, degree, beta, alpha, type = "glass")
  if (type == 3)
    theta <- gen_4nn_cyc(p, degree, beta, alpha, type = "ferro")
  if (type == 4)
    theta <- gen_4nn_cyc(p, degree, beta, alpha, type = "glass")
  if (type == 5)
    theta <- gen_4nn_cyc(p, degree, beta, alpha, type = "glass_weak")
  if (type == 6) 
    theta <- gen_ld_adjmat(p, degree, alpha)
  if (type == 7)
    theta <- gen_rr_adjmat2(p, degree, beta, alpha, type = "ferro")

  set.seed(NULL)
  return(theta)
}

#' @title Generate a dataset from binary markov network
#' @inheritParams generate.data
#' 
#' @description Generate a dataset with binary variables upon the interaction matrix. If the interaction matrix is not provided, and then, an interaction matrix would be generated. 
#' 
#' @param theta An user specified interaction matrix. It should be a symmetric matrix.
#' @param seed Seed to be used to reproduce dataset
#' @param type a numeric value that indicates the type of an interaction matrix
#' @param degree an integer value that specifies the maximum node degree in the interaction matrix
#' @param alpha a numeric value that specifies the minimum signal of the interaction matrix
#' @param beta a numeric value that specifies the remaining signal of the interaction matrix
#' @param graph.seed Seed used to reproduce the interaction matrix
#' @param method the method used for generating samples. One is the "freq", another is it "gibbs". The former is more suitable when p<18, while the later is more suitable when p > 18.
#' 
#' @return a list that includes three components: data, weight, theta. data is an n-by-p matrix that record samples; weight: an n-dimensional vector that records the frequency of each row of data; theta is the interaction matrix used for generating samples.
#' @export
#'
#' @examples
#' library(abess)
#' p <- 16
#' n <- 1e3
#' train <- generate.bmn.data(n, p, type = 7, graph.seed = 1, seed = 1, beta = 1.3, alpha = 0.2)
#' res <- slide(train[["data"]], train[["weight"]], tune.type = "gic")
generate.bmn.data <- function(n, p, type = 1, seed = NULL, graph.seed = NULL, theta = NULL,  beta = 0.7, degree = 3, alpha = 0.4, method = "freq") {
  if (is.null(graph.seed)) {
    graph_seed <- round(runif(1 , 0, .Machine$integer.max))
  } else {
    graph_seed <- graph.seed
  }
  
  if (is.null(theta)) {
    theta <-
      sim_theta(
        p,
        type = type,
        graph_seed = graph_seed,
        beta = beta,
        degree = degree,
        alpha = alpha
      )
  } else {
    ## check theta (TODO)
  }
  
  if (is.null(seed)) {
    seed <- round(runif(1 , 0, .Machine$integer.max))
  }

  if (method == "freq") {
    data <- sample_by_conf(n = n, theta = theta, seed = seed)
    data <- data[data[, 1] > 0, ]
    weight <- data[, 1]
    data <- data[, -1]
  } else {
    value <- c(-1, 1)
    data <- Ising_Gibbs(theta = theta, n_sample = n, value = value, burn = 1e5, skip = 50, seed = seed)
    weight <- rep(1, n)
  }
  
  set.seed(NULL)
  
  return(list(data = data, weight = weight, theta = theta))
}
