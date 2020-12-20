// #define R_BUILD
#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Eigen>
#include "List.h"
#endif
#include <algorithm>
#include <vector>

using namespace std;
Eigen::VectorXd pi(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd coef, int n)
{
  int p = coef.size();
  Eigen::VectorXd Pi = Eigen::VectorXd::Zero(n);
  if (X.cols() == p - 1)
  {
    Eigen::VectorXd intercept = Eigen::VectorXd::Ones(n) * coef(0);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd eta = X * (coef.tail(p - 1)) + intercept;
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
    Pi = expeta.array() / (one + expeta).array();
    return Pi;
  }
  else
  {
    Eigen::VectorXd eta = X * coef;
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

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
    Pi = expeta.array() / (one + expeta).array();
    return Pi;
  }
}

Eigen::VectorXd logit_fit(Eigen::MatrixXd x, Eigen::VectorXd y, int n, int p, Eigen::VectorXd weights)
{
  if (n <= p)
  {
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, n);
    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    X.rightCols(n - 1) = x.leftCols(n - 1);
    Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd Pi = pi(X, y, beta0, n);
    Eigen::VectorXd log_Pi = Pi.array().log();
    Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
    double loglik0 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
    Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
    Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);
    W = W.cwiseProduct(weights);
    for (int i = 0; i < n; i++)
    {
      X_new.row(i) = X.row(i) * W(i);
    }
    beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);

    double loglik1;

    int j;
    for (j = 0; j < 30; j++)
    {
      Pi = pi(X, y, beta1, n);
      log_Pi = Pi.array().log();
      log_1_Pi = (one - Pi).array().log();
      loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
      if (abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < 1e-6)
      {
        break;
      }
      beta0 = beta1;
      loglik0 = loglik1;
      W = Pi.cwiseProduct(one - Pi);
      Z = X * beta0 + (y - Pi).cwiseQuotient(W);
      W = W.cwiseProduct(weights);
      for (int i = 0; i < n; i++)
      {
        X_new.row(i) = X.row(i) * W(i);
      }
      beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
    }
    return beta0;
  }

  else
  {
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
    X.rightCols(p) = x;
    Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
    Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p + 1);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd Pi = pi(X, y, beta0, n);
    Eigen::VectorXd log_Pi = Pi.array().log();
    Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
    double loglik0 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
    Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
    Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);
    W = W.cwiseProduct(weights);
    for (int i = 0; i < n; i++)
    {
      X_new.row(i) = X.row(i) * W(i);
    }
    beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
    double loglik1;

    int j;
    for (j = 0; j < 30; j++)
    {
      Pi = pi(X, y, beta1, n);
      log_Pi = Pi.array().log();
      log_1_Pi = (one - Pi).array().log();
      loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
      if (abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < 1e-6)
      {
        break;
      }
      beta0 = beta1;
      loglik0 = loglik1;
      W = Pi.cwiseProduct(one - Pi);
      Z = X * beta0 + (y - Pi).cwiseQuotient(W);
      W = W.cwiseProduct(weights);
      for (int i = 0; i < n; i++)
      {
        X_new.row(i) = X.row(i) * W(i);
      }
      beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
    }
    return beta0;
  }
}

double loglik_logit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd coef, int n, Eigen::VectorXd weights)
{
  Eigen::VectorXd Pi = pi(X, y, coef, n);
  Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd log_Pi = Pi.array().log();
  Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
  return (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
}
