#' Title
#'
#' @param X a covariance/correaltion matrix
#' @param loading a sparse loading matrix learned by algorithms
#'
#' @return
#' @export
#'
#' @examples
variance_explained <- function(X, loading) {
  pc <- X %*% loading
  Z <- qr(pc)
  ev <- sum(abs(diag(qr.R(Z))))
  ev
}

library(elasticnet)
library(abess)
library(data.table)

load("christensen.RData")

community <- christensen[["x"]]

# row missing analysis
sum(apply(community, 1, anyNA))
summary(apply(community, 1, function(x) {
  mean(is.na(x))
}))

# column missing analysis
sum(apply(community, 2, anyNA))
missing_prop <- apply(community, 2, function(x) {
  mean(is.na(x))
})
sum(missing_prop > 0.5)
missing_prop[missing_prop > 0.5]

anyNA(community)
dim(community)

community <- apply(community, 2, as.numeric)
community_cov <- cov(community)
k_list <- c(5, 10, 20)
spca_time_vec <- c()
abess_time_vec <- c()
spca_ev_vec <- c()
abess_ev_vec <- c()
for (k in k_list) {
  t1 <- system.time(spca_fit_list <- spca(community_cov, type = "Gram", K = 1, 
                                          para = k, sparse = "varnum", 
                                          use.corr = FALSE))
  t2 <- system.time(abess_fit <- abesspca(x = community_cov, type = "gram", 
                                          support.size = k))
  print(t1)
  print(t2)
  spca_time_vec <- c(spca_time_vec, t1[3])
  abess_time_vec <- c(abess_time_vec, t2[3])
  spca_ev_vec <- c(spca_ev_vec, 
                   variance_explained(community_cov, 
                                      spca_fit_list[["loadings"]]))
  abess_ev_vec <- c(abess_ev_vec, 
                    variance_explained(community_cov, 
                                       abess_fit[["coef"]][, 1, drop = FALSE]))
}

res <- rbind(rbind(spca_ev_vec, abess_ev_vec), 
             rbind(spca_time_vec, abess_time_vec))

xtable::xtable(res, digits = 3)
