#ifndef POISSON_H
#define POISSON_H

#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
#else
#include <Eigen/Eigen>
#include "List.h"
#endif

#include <algorithm>
#include <vector>

double loglik_poisson(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd coef, int n, Eigen::VectorXd weights);

double factorial_poiss(Eigen::VectorXd y, int n);

double loglik_poiss(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd coef, int n, Eigen::VectorXd weights);

Eigen::VectorXd poisson_fit(Eigen::MatrixXd x, Eigen::VectorXd y, int n, int p, Eigen::VectorXd weights);

#endif