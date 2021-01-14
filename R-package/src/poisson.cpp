// #define R_BUILD
#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
#else
#include <Eigen/Eigen>
#include "List.h"
#endif

#include <algorithm>
#include <vector>

using namespace Eigen;

double loglik_poisson(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd coef, int n, Eigen::VectorXd weights)
{
  int p = x.cols();
  Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
  Eigen::VectorXd temp = Eigen::VectorXd::Zero(n);
  X.rightCols(p) = x;
  Eigen::VectorXd eta = X * coef;
  for (int i = 0; i <= n - 1; i++)
  {
    if (eta(i) < -30.0)
      eta(i) = -30.0;
    if (eta(i) > 30.0)
      eta(i) = 30.0;
  }
  for (int i = 0; i < n; i++)
  {
    if (y(i) == 1)
    {
      temp(i) = 0;
    }
    else
    {
      for (int j = 1; j <= y(i); j++)
      {
        temp(i) = temp(i) + log(j);
      }
    }
  }
  Eigen::VectorXd expeta = eta.array().exp();
  return (y.cwiseProduct(eta) - expeta - temp).dot(weights);
}

double factorial_poiss(Eigen::VectorXd y, int n)
{
  Eigen::VectorXd temp = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; i++)
  {
    if (y(i) == 1)
    {
      temp(i) = 0;
    }
    else
    {
      for (int j = 1; j <= y(i); j++)
      {
        temp(i) = temp(i) + log(j);
      }
    }
  }
  return temp.sum();
}

double loglik_poiss(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd coef, int n, Eigen::VectorXd weights)
{
  int p = x.cols();
  Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
  X.rightCols(p) = x;
  Eigen::VectorXd eta = X * coef;
  for (int i = 0; i <= n - 1; i++)
  {
    if (eta(i) < -30.0)
      eta(i) = -30.0;
    if (eta(i) > 30.0)
      eta(i) = 30.0;
  }
  Eigen::VectorXd expeta = eta.array().exp();
  return (y.cwiseProduct(eta) - expeta).dot(weights);
}

Eigen::VectorXd poisson_fit(Eigen::MatrixXd x, Eigen::VectorXd y, int n, int p, Eigen::VectorXd weights)
{
  Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
  X.rightCols(p) = x;
  Eigen::MatrixXd h(p + 1, p + 1);
  Eigen::VectorXd d = Eigen::VectorXd::Zero(p + 1);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(p + 1);
  Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
  Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p + 1);
  Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(n, p + 1);
  Eigen::VectorXd eta = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd expeta = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd expeta_w = Eigen::VectorXd::Zero(n);
  double loglik0;
  double loglik1;

  int j;
  for (j = 0; j < 100; j++)
  {
    double step = 0.2;
    int m = 0;
    eta = X * beta0;
    for (int i = 0; i <= n - 1; i++)
    {
      if (eta(i) < -30.0)
        eta(i) = -30.0;
      if (eta(i) > 30.0)
        eta(i) = 30.0;
    }
    expeta = eta.array().exp();
    expeta_w = expeta.cwiseProduct(weights);
    for (int i = 0; i < p + 1; i++)
    {
      temp.col(i) = X.col(i) * expeta_w;
    }
    g = X.transpose() * (y - expeta).cwiseProduct(weights);
    h = X.transpose() * temp;
    d = h.ldlt().solve(g);
    beta1 = beta0 - pow(step, m) * d;
    loglik0 = loglik_poiss(x, y, beta0, n, weights);
    loglik1 = loglik_poiss(x, y, beta1, n, weights);
    while ((loglik0 >= loglik1) && (m < 10))
    {
      m = m + 1;
      beta1 = beta0 - pow(step, m) * d;
      loglik1 = loglik_poiss(x, y, beta1, n, weights);
    }
    beta0 = beta1;
    if (abs(loglik0 - loglik1) / abs(loglik0) < 1e-8)
    {
      break;
    }
  }
  return beta0;
}