#' @keywords internal
#' @aliases abess-package
"_PACKAGE"

# The following block is used by usethis to automatically manage
# roxygen namespace tags. Modify with care!
## usethis namespace: start
#' @useDynLib abess, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom MASS mvrnorm
#' @importFrom MASS cov.rob
#' @importFrom stats runif
#' @importFrom stats cov
#' @importFrom stats cor
#' @importFrom stats var
#' @importFrom stats rnorm
#' @importFrom stats rbinom
#' @importFrom stats rpois
#' @importFrom stats poisson
#' @importFrom stats binomial
#' @importFrom stats terms
#' @importFrom stats model.matrix
#' @importFrom stats model.frame
#' @importFrom stats model.response
#' @importFrom stats heatmap
#' @importFrom methods rbind2
#' @importFrom methods as
#' @import Matrix
#' @importFrom Matrix Matrix
#' @importFrom Matrix rowSums
#' @importFrom Matrix colSums
#' @importFrom Matrix sparseMatrix
#' @importFrom Matrix as.matrix
#' @importFrom grDevices cm.colors
## usethis namespace: end
NULL


#' @title The Bardet-Biedl syndrome Gene expression data
#' @name trim32
#' @docType data
#' @description Gene expression data (500 gene probes for 120 samples) from the microarray experiments of mammalianeye tissue samples of Scheetz et al. (2006).
#'
#' @details In this study, laboratory rats (Rattus norvegicus) were studied to learn about gene expression and regulation in the mammalian eye.
#' Inbred rat strains were crossed and tissue extracted from the eyes of 120 animals from the F2 generation. Microarrays were used to measure levels of RNA expression in the isolated eye tissues of each subject.
#' Of the 31,000 different probes, 18,976 were detected at a sufficient level to be considered expressed in the mammalian eye.
#' For the purposes of this analysis, we treat one of those genes, Trim32, as the outcome.
#' Trim32 is known to be linked with a genetic disorder called Bardet-Biedl Syndrome (BBS): the mutation (P130S) in Trim32 gives rise to BBS.
#'
#' @note This data set contains 120 samples with 500 predictors. The 500 predictors are features with maximum marginal correlation to Trim32 gene.
#'
#' @format A data frame with 120 rows and 501 variables, where the first variable is the expression level of TRIM32 gene,
#' and the remaining 500 variables are 500 gene probes.
#'
#' @references T. Scheetz, k. Kim, R. Swiderski, A. Philp, T. Braun, K. Knudtson, A. Dorrance, G. DiBona, J. Huang, T. Casavant, V. Sheffield, E. Stone. Regulation of gene expression in the mammalian eye and its relevance to eye disease. Proceedings of the National Academy of Sciences of the United States of America, 2006.
NULL
