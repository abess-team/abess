// #define R_BUILD
#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
#else
#include <Eigen/Eigen>
#include "List.h"
#endif
#include <algorithm>
#include <vector>

using namespace std;

double loglik_cox(Eigen::MatrixXd X, Eigen::VectorXd status, Eigen::VectorXd beta, Eigen::VectorXd weights)
{
  int n = X.rows();
  Eigen::VectorXd eta = X * beta;
  for (int i = 0; i < n; i++)
  {
    if (eta(i) > 30)
    {
      eta(i) = 30;
    }
    else if (eta(i) < -30)
    {
      eta(i) = -30;
    }
  }
  Eigen::VectorXd expeta = eta.array().exp();
  Eigen::VectorXd cum_expeta(n);
  cum_expeta(n - 1) = expeta(n - 1);
  for (int i = n - 2; i >= 0; i--)
  {
    cum_expeta(i) = cum_expeta(i + 1) + expeta(i);
  }
  Eigen::VectorXd ratio = (expeta.cwiseQuotient(cum_expeta)).array().log();
  return (ratio.cwiseProduct(status)).dot(weights);
}

Eigen::VectorXd cox_fit(Eigen::MatrixXd X, Eigen::VectorXd status, int n, int p, Eigen::VectorXd weights)
{
  Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd theta(n);
  Eigen::MatrixXd one = (Eigen::MatrixXd::Ones(n, n)).triangularView<Eigen::Upper>();
  Eigen::MatrixXd x_theta(n, p);
  Eigen::VectorXd xij_theta(n);
  Eigen::VectorXd cum_theta(n);
  Eigen::VectorXd g(p);
  Eigen::MatrixXd h(p, p);
  Eigen::VectorXd d(p);
  double loglik0 = 1e5;
  double loglik1;

  double step;
  int m;
  int l;
  for (l = 1; l <= 30; l++)
  {
    step = 0.5;
    m = 1;
    theta = X * beta0;
    for (int i = 0; i < n; i++)
    {
      if (theta(i) > 50)
        theta(i) = 50;
      else if (theta(i) < -50)
        theta(i) = -50;
    }
    theta = theta.array().exp();
    cum_theta = one * theta;
    x_theta = X.array().colwise() * theta.array();
    x_theta = one * x_theta;
    x_theta = x_theta.array().colwise() / cum_theta.array();
    g = (X - x_theta).transpose() * (weights.cwiseProduct(status));

    for (int k1 = 0; k1 < p; k1++)
    {
      for (int k2 = k1; k2 < p; k2++)
      {
        xij_theta = (theta.cwiseProduct(X.col(k1))).cwiseProduct(X.col(k2));
        for (int j = n - 2; j >= 0; j--)
        {
          xij_theta(j) = xij_theta(j + 1) + xij_theta(j);
        }
        h(k1, k2) = -(xij_theta.cwiseQuotient(cum_theta) - x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(status));
        h(k2, k1) = h(k1, k2);
      }
    }
    d = h.ldlt().solve(g);
    beta1 = beta0 - pow(step, m) * d;
    loglik1 = loglik_cox(X, status, beta1, weights);
    while ((loglik0 > loglik1) && (m < 5))
    {
      m = m + 1;
      beta1 = beta0 - pow(step, m) * d;
      loglik1 = loglik_cox(X, status, beta1, weights);
    }
    if (abs(loglik0 - loglik1) / abs(0.1 + loglik0) < 1e-5)
    {
      break;
    }
    beta0 = beta1;
    loglik0 = loglik1;
  }
  return beta0;
}
