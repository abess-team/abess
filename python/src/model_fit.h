#ifndef MODEL_FIT_H
#define MODEL_FIT_H

#ifdef R_BUILD

#include <Rcpp.h>
#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
#else
#include <Eigen/Eigen>
#include "List.h"
#endif

#include "utilities.h"
#include <cfloat>

template <class T4>
Eigen::VectorXd pi(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &coef)
{
  int p = coef.size();
  int n = X.rows();
  Eigen::VectorXd Pi = Eigen::VectorXd::Zero(n);
  if (X.cols() == p - 1)
  {
    Eigen::VectorXd intercept = Eigen::VectorXd::Ones(n) * coef(0);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd eta = X * (coef.tail(p - 1).eval()) + intercept;
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

template <class T4>
void pi(T4 &X, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, Eigen::MatrixXd &pr)
{
  int n = X.rows();
  Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, 1);
  Eigen::MatrixXd Xbeta = X * beta + one * coef0.transpose();
  pr = Xbeta.array().exp();
  Eigen::VectorXd sumpi = pr.rowwise().sum();
  for (int i = 0; i < n; i++)
  {
    pr.row(i) = pr.row(i) / sumpi(i);
  }

  // return pi;
};

template <class T4>
void pi(T4 &X, Eigen::MatrixXd &y, Eigen::MatrixXd &coef, Eigen::MatrixXd &pr)
{
  int n = X.rows();
  // Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, 1);
  Eigen::MatrixXd Xbeta = X * coef;
  pr = Xbeta.array().exp();
  Eigen::VectorXd sumpi = pr.rowwise().sum();
  for (int i = 0; i < n; i++)
  {
    pr.row(i) = pr.row(i) / sumpi(i);
  }

  // return pi;
};

template <class T4>
bool multinomial_fit(T4 &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{

  // if (X.cols() == 0)
  // {
  //   coef0 = -log(y.colwise().sum().eval() - 1.0);
  //   return;
  // }

  int n = x.rows();
  int p = x.cols();
  int M = y.cols();
  T4 X(n, p + 1);
  X.rightCols(p) = x;
  add_constant_column(X);
  Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p + 1, p + 1);
  Eigen::MatrixXd beta0 = Eigen::MatrixXd::Zero(p + 1, M);

  Eigen::MatrixXd one_vec = Eigen::VectorXd::Ones(n);
  beta0.row(0) = coef0;
  beta0.block(1, 0, p, M) = beta;
  Eigen::MatrixXd Pi;
  pi(X, y, beta0, Pi);
  Eigen::MatrixXd log_Pi = Pi.array().log();
  array_product(log_Pi, weights, 1);
  double loglik1 = DBL_MAX, loglik0 = (log_Pi.array() * y.array()).sum();

  int j;
  if (approximate_Newton)
  {
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, M);
    double t = 2 * (Pi.array() * (one - Pi).array()).maxCoeff();
    Eigen::MatrixXd res = X.transpose() * (y - Pi) / t;
    // ConjugateGradient<MatrixXd, Lower | Upper> cg;
    // cg.compute(X.adjoint() * X);
    Eigen::MatrixXd XTX = X.transpose() * X + lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols());
    // if (check_ill_condition(XTX)) return false;
    Eigen::MatrixXd invXTX = XTX.ldlt().solve(Eigen::MatrixXd::Identity(p + 1, p + 1));

    Eigen::MatrixXd beta1;
    for (j = 0; j < primary_model_fit_max_iter; j++)
    {
      //

      // beta1 = beta0 + cg.solve(res);
      beta1 = beta0 + invXTX * res;

      // double app_loss0, app_loss1, app_loss2;
      // app_loss0 = ((y - Pi) / t).squaredNorm();
      // app_loss1 = (-X * beta0 - (y - Pi) / t).squaredNorm();
      // app_loss2 = (X * (beta1 - beta0) - (y - Pi) / t).squaredNorm();

      pi(X, y, beta1, Pi);
      log_Pi = Pi.array().log();
      array_product(log_Pi, weights, 1);
      loglik1 = (log_Pi.array() * y.array()).sum();

      bool condition1 = -(loglik1 + (primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
      // bool condition1 = false;
      bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < primary_model_fit_epsilon;
      bool condition3 = abs(loglik1) < min(1e-3, tau);
      bool condition4 = loglik1 < loglik0;
      // bool condition4 = false;
      if (condition1 || condition2 || condition3 || condition4)
      {
        break;
      }
      loglik0 = loglik1;

      for (int m1 = 0; m1 < M; m1++)
      {
        beta0.col(m1) = beta1.col(m1) - beta1.col(M - 1);
      }

      // beta0 = beta1;
      t = 2 * (Pi.array() * (one - Pi).array()).maxCoeff();
      res = X.transpose() * (y - Pi) / t;
    }
  }
  else
  {
    Eigen::MatrixXd W(M * n, M * n);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    for (int m1 = 0; m1 < M; m1++)
    {
      for (int m2 = m1; m2 < M; m2++)
      {
        if (m1 == m2)
        {
          W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);

          Eigen::VectorXd PiPj = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array();

          for (int i = 0; i < PiPj.size(); i++)
          {
            if (PiPj(i) < 0.001)
            {
              PiPj(i) = 0.001;
            }
          }
          W.block(m1 * n, m2 * n, n, n).diagonal() = PiPj;
        }
        else
        {
          W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);

          Eigen::VectorXd PiPj = Pi.col(m1).array() * Pi.col(m2).array();

          for (int i = 0; i < PiPj.size(); i++)
          {
            if (PiPj(i) < 0.001)
            {
              PiPj(i) = 0.001;
            }
          }
          W.block(m1 * n, m2 * n, n, n).diagonal() = -PiPj;
          W.block(m2 * n, m1 * n, n, n) = W.block(m1 * n, m2 * n, n, n);
        }
      }
    }

    Eigen::MatrixXd XTWX(M * (p + 1), M * (p + 1));
    Eigen::MatrixXd XTW(M * (p + 1), M * n);
    for (int m1 = 0; m1 < M; m1++)
    {
      for (int m2 = m1; m2 < M; m2++)
      {
        XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) = X.transpose() * W.block(m1 * n, m2 * n, n, n);
        XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X + 2 * lambda * lambdamat;
        XTW.block(m2 * (p + 1), m1 * n, (p + 1), n) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n);
        XTWX.block(m2 * (p + 1), m1 * (p + 1), (p + 1), (p + 1)) = XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1));
      }
    }

    // Eigen::Matrix<Eigen::MatrixXd, -1, -1> res(M, 1);
    Eigen::VectorXd res(M * n);
    for (int m1 = 0; m1 < M; m1++)
    {
      res.segment(m1 * n, n) = y.col(m1).eval() - Pi.col(m1).eval();
    }

    Eigen::VectorXd Xbeta(M * n);
    for (int m1 = 0; m1 < M; m1++)
    {
      Xbeta.segment(m1 * n, n) = X * beta0.col(m1).eval();
    }

    Eigen::VectorXd Z = Xbeta + W.ldlt().solve(res);

    Eigen::MatrixXd beta1;
    Eigen::VectorXd beta0_tmp;
    for (j = 0; j < primary_model_fit_max_iter; j++)
    {

      beta0_tmp = XTWX.ldlt().solve(XTW * Z);
      for (int m1 = 0; m1 < M; m1++)
      {
        beta0.col(m1) = beta0_tmp.segment(m1 * (p + 1), (p + 1)) - beta0_tmp.segment((M - 1) * (p + 1), (p + 1));
      }
      for (int m1 = 0; m1 < M; m1++)
      {
        beta0.col(m1) = beta0_tmp.segment(m1 * (p + 1), (p + 1));
      }

      pi(X, y, beta0, Pi);
      log_Pi = Pi.array().log();
      array_product(log_Pi, weights, 1);
      loglik1 = (log_Pi.array() * y.array()).sum();

      bool condition1 = -(loglik1 + (primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
      // bool condition1 = false;
      bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < primary_model_fit_epsilon;
      bool condition3 = abs(loglik1) < min(1e-3, tau);
      bool condition4 = loglik1 < loglik0;
      if (condition1 || condition2 || condition3 || condition4)
      {

        break;
      }
      loglik0 = loglik1;

      for (int m1 = 0; m1 < M; m1++)
      {
        for (int m2 = m1; m2 < M; m2++)
        {
          if (m1 == m2)
          {
            // W(m1, m2) = Eigen::MatrixXd::Zero(n, n);
            // W(m1, m2).diagonal() = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array();

            W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
            Eigen::VectorXd PiPj = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array();
            for (int i = 0; i < PiPj.size(); i++)
            {
              if (PiPj(i) < 0.001)
              {
                PiPj(i) = 0.001;
              }
            }
            W.block(m1 * n, m2 * n, n, n).diagonal() = PiPj;
          }
          else
          {
            W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
            Eigen::VectorXd PiPj = Pi.col(m1).array() * Pi.col(m2).array();
            for (int i = 0; i < PiPj.size(); i++)
            {
              if (PiPj(i) < 0.001)
              {
                PiPj(i) = 0.001;
              }
            }
            W.block(m1 * n, m2 * n, n, n).diagonal() = -PiPj;
            W.block(m2 * n, m1 * n, n, n) = W.block(m1 * n, m2 * n, n, n);
          }
        }
      }

      for (int m1 = 0; m1 < M; m1++)
      {
        for (int m2 = m1; m2 < M; m2++)
        {
          XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) = X.transpose() * W.block(m1 * n, m2 * n, n, n);
          XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X + 2 * lambda * lambdamat;
          XTW.block(m2 * (p + 1), m1 * n, (p + 1), n) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n);
          XTWX.block(m2 * (p + 1), m1 * (p + 1), (p + 1), (p + 1)) = XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1));
        }
      }

      for (int m1 = 0; m1 < M; m1++)
      {
        res.segment(m1 * n, n) = y.col(m1).eval() - Pi.col(m1).eval();
      }

      for (int m1 = 0; m1 < M; m1++)
      {
        Xbeta.segment(m1 * n, n) = X * beta0.col(m1).eval();
      }

      Z = Xbeta + W.ldlt().solve(res);
    }
  }

  beta = beta0.block(1, 0, p, M);
  coef0 = beta0.row(0).eval();
  return true;
}

template <class T4>
bool multigaussian_fit(T4 &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{
  // beta = (X.adjoint() * X + lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);

  int n = x.rows();
  int p = x.cols();
  int M = y.cols();

  // to ensure
  T4 X(n, p + 1);
  X.rightCols(p) = x;
  add_constant_column(X);
  // beta = (X.adjoint() * X + lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);
  Eigen::MatrixXd XTX = X.adjoint() * X + lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols());
  // if (check_ill_condition(XTX)) return false;
  Eigen::MatrixXd beta0 = XTX.ldlt().solve(X.adjoint() * y);

  beta = beta0.block(1, 0, p, M);
  coef0 = beta0.row(0).eval();
  return true;
  // if (X.cols() == 0)
  // {
  //   // coef0 = y.colwise().sum();
  //   return;
  // }
  //
  // // overload_ldlt(X, X, y, beta);
  // Eigen::MatrixXd XTX = X.transpose() * X;
  // beta = (XTX + lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols())).ldlt().solve(X.transpose() * y);

  // CG
  // ConjugateGradient<T4, Lower | Upper> cg;
  // cg.compute(X.adjoint() * X + lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols()));
  // beta = cg.solveWithGuess(X.adjoint() * y, beta);
}

template <class T4>
double loglik_logit(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &coef, int n, Eigen::VectorXd weights)
{
  Eigen::VectorXd Pi = pi(X, y, coef);
  Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd log_Pi = Pi.array().log();
  Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
  return (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
}

template <class T4>
bool logistic_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{

  if (x.cols() == 0)
  {
    coef0 = -log(1 / y.mean() - 1);
    return true;
  }

  int n = x.rows();
  int p = x.cols();

  // to ensure
  T4 X(n, p + 1);
  X.rightCols(p) = x;
  add_constant_column(X);

  T4 X_new(X);

  Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
  beta0(0) = coef0;
  beta0.tail(p) = beta;
  Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
  Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p + 1, p + 1);
  lambdamat(0, 0) = 0;

  Eigen::VectorXd Pi = pi(X, y, beta0);
  Eigen::VectorXd log_Pi = Pi.array().log();
  Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
  double loglik1 = DBL_MAX, loglik0 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
  Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
  for (int i = 0; i < n; i++)
  {
    if (W(i) < 0.001)
      W(i) = 0.001;
  }
  Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);

  int j;
  double step = 1;
  Eigen::VectorXd g(p + 1);
  Eigen::VectorXd beta1;
  for (j = 0; j < primary_model_fit_max_iter; j++)
  {
    // To do: Approximate Newton method
    if (approximate_Newton)
    {
      Eigen::VectorXd h_diag(p + 1);
      for (int i = 0; i < p + 1; i++)
      {
        h_diag(i) = 1.0 / X.col(i).cwiseProduct(W).cwiseProduct(weights).dot(X.col(i));
      }
      g = X.transpose() * ((y - Pi).cwiseProduct(weights));
      beta1 = beta0 + step * g.cwiseProduct(h_diag);
      Pi = pi(X, y, beta1);
      log_Pi = Pi.array().log();
      log_1_Pi = (one - Pi).array().log();
      loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);

      while (loglik1 < loglik0 && step > primary_model_fit_epsilon)
      {
        step = step / 2;
        beta1 = beta0 + step * g.cwiseProduct(h_diag);
        Pi = pi(X, y, beta1);
        log_Pi = Pi.array().log();
        log_1_Pi = (one - Pi).array().log();
        loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
      }

      bool condition1 = -(loglik1 + (primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
      // bool condition1 = false;
      if (condition1)
        break;

      if (loglik1 > loglik0)
      {
        beta0 = beta1;
        loglik0 = loglik1;
        W = Pi.cwiseProduct(one - Pi);
        for (int i = 0; i < n; i++)
        {
          if (W(i) < 0.001)
            W(i) = 0.001;
        }
      }

      if (step < primary_model_fit_epsilon)
      {
        break;
      }
    }
    else
    {

      for (int i = 0; i < p + 1; i++)
      {
        X_new.col(i) = X.col(i).cwiseProduct(W).cwiseProduct(weights);
      }

      Eigen::MatrixXd XTX = 2 * lambda * lambdamat + X_new.transpose() * X;
      // if (check_ill_condition(XTX)) return false;
      beta0 = XTX.ldlt().solve(X_new.transpose() * Z);

      // overload_ldlt(X_new, X, Z, beta0);

      // CG
      // ConjugateGradient<T4, Lower | Upper> cg;
      // cg.compute(X_new.transpose() * X);
      // beta0 = cg.solve(X_new.transpose() * Z);

      Pi = pi(X, y, beta0);
      log_Pi = Pi.array().log();
      log_1_Pi = (one - Pi).array().log();
      loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);

      bool condition1 = -(loglik1 + (primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
      bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < primary_model_fit_epsilon;
      bool condition3 = abs(loglik1) < min(1e-3, tau);
      if (condition1 || condition2 || condition3)
      {
        break;
      }

      loglik0 = loglik1;
      W = Pi.cwiseProduct(one - Pi);
      for (int i = 0; i < n; i++)
      {
        if (W(i) < 0.001)
          W(i) = 0.001;
      }
      Z = X * beta0 + (y - Pi).cwiseQuotient(W);
    }
  }

  beta = beta0.tail(p).eval();
  coef0 = beta0(0);
  return true;
};

template <class T4>
bool lm_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{
  int n = x.rows();
  int p = x.cols();

  // to ensure
  T4 X(n, p + 1);
  X.rightCols(p) = x;
  add_constant_column(X);
  // beta = (X.adjoint() * X + lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);
  Eigen::MatrixXd XTX = X.adjoint() * X + lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols());
  // if (check_ill_condition(XTX)) return false;
  Eigen::VectorXd beta0 = XTX.ldlt().solve(X.adjoint() * y);

  beta = beta0.tail(p).eval();
  coef0 = beta0(0);
  return true;

  // if (X.cols() == 0)
  // {
  //   coef0 = y.mean();
  //   return;
  // }

  // CG
  // ConjugateGradient<MatrixXd, Lower | Upper> cg;
  // cg.compute(X.adjoint() * X + lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols()));
  // beta = cg.solveWithGuess(X.adjoint() * y, beta);
}

template <class T4>
bool poisson_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{

  int n = x.rows();
  int p = x.cols();
  T4 X(n, p + 1);
  X.rightCols(p) = x;
  add_constant_column(X);

  Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p + 1, p + 1);
  lambdamat(0, 0) = 0;

  // Eigen::MatrixXd X_trans = X.transpose();
  T4 X_new(n, p + 1);
  Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
  beta0.tail(p) = beta;
  beta0(0) = coef0;
  Eigen::VectorXd eta = X * beta0;
  Eigen::VectorXd expeta = eta.array().exp();
  Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
  double loglik0 = (y.cwiseProduct(eta) - expeta).dot(weights);
  double loglik1;

  int j;
  for (j = 0; j < primary_model_fit_max_iter; j++)
  {
    for (int i = 0; i < p + 1; i++)
    {
      // temp.col(i) = X_trans.col(i) * expeta(i) * weights(i);
      X_new.col(i) = X.col(i).cwiseProduct(expeta).cwiseProduct(weights);
    }
    z = eta + (y - expeta).cwiseQuotient(expeta);
    Eigen::MatrixXd XTX = X_new.transpose() * X + 2 * lambda * lambdamat;
    // if (check_ill_condition(XTX)) return false;
    beta0 = (XTX).ldlt().solve(X_new.transpose() * z);
    eta = X * beta0;
    for (int i = 0; i <= n - 1; i++)
    {
      if (eta(i) < -30.0)
        eta(i) = -30.0;
      if (eta(i) > 30.0)
        eta(i) = 30.0;
    }
    expeta = eta.array().exp();
    loglik1 = (y.cwiseProduct(eta) - expeta).dot(weights);
    bool condition1 = -(loglik1 + (primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
    // bool condition1 = false;
    bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < primary_model_fit_epsilon;
    bool condition3 = abs(loglik1) < min(1e-3, tau);
    if (condition1 || condition2 || condition3)
    {

      break;
    }
    loglik0 = loglik1;
  }

  beta = beta0.tail(p).eval();
  coef0 = beta0(0);
  return true;
}

template <class T4>
double loglik_cox(T4 &X, Eigen::VectorXd &status, Eigen::VectorXd &beta, Eigen::VectorXd &weights)
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

template <class T4>
bool cox_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{

  int n = x.rows();
  int p = x.cols();
  Eigen::VectorXd theta(n);
  Eigen::MatrixXd one = (Eigen::MatrixXd::Ones(n, n)).triangularView<Eigen::Upper>();
  Eigen::MatrixXd x_theta(n, p);
  Eigen::VectorXd xij_theta(n);
  Eigen::VectorXd cum_theta(n);
  Eigen::VectorXd g(p);
  Eigen::VectorXd beta0 = beta, beta1;
  Eigen::VectorXd cum_eta(n);
  Eigen::VectorXd cum_eta2(n);
  Eigen::VectorXd cum_eta3(n);
  Eigen::MatrixXd h(p, p);
  Eigen::VectorXd eta;

  Eigen::VectorXd d(p);
  double loglik1, loglik0 = loglik_cox(x, y, beta0, weight);
  // beta = Eigen::VectorXd::Zero(p);

  double step = 1.0;
  int l;
  for (l = 1; l <= primary_model_fit_max_iter; l++)
  {

    eta = x * beta0;
    for (int i = 0; i <= n - 1; i++)
    {
      if (eta(i) < -30.0)
        eta(i) = -30.0;
      if (eta(i) > 30.0)
        eta(i) = 30.0;
    }
    eta = weight.array() * eta.array().exp();
    cum_eta(n - 1) = eta(n - 1);
    for (int k = n - 2; k >= 0; k--)
    {
      cum_eta(k) = cum_eta(k + 1) + eta(k);
    }
    cum_eta2(0) = (y(0) * weight(0)) / cum_eta(0);
    for (int k = 1; k <= n - 1; k++)
    {
      cum_eta2(k) = (y(k) * weight(k)) / cum_eta(k) + cum_eta2(k - 1);
    }
    cum_eta3(0) = (y(0) * weight(0)) / pow(cum_eta(0), 2);
    for (int k = 1; k <= n - 1; k++)
    {
      cum_eta3(k) = (y(k) * weight(k)) / pow(cum_eta(k), 2) + cum_eta3(k - 1);
    }
    h = -cum_eta3.replicate(1, n);
    h = h.cwiseProduct(eta.replicate(1, n));
    h = h.cwiseProduct(eta.replicate(1, n).transpose());
    for (int i = 0; i < n; i++)
    {
      for (int j = i + 1; j < n; j++)
      {
        h(j, i) = h(i, j);
      }
    }
    h.diagonal() = cum_eta2.cwiseProduct(eta) + h.diagonal();
    g = weight.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);

    Eigen::MatrixXd temp = x.transpose() * h * x;
    if (approximate_Newton)
    {
      d = (x.transpose() * g).cwiseQuotient((temp).diagonal());
    }
    else
    {
      // if (check_ill_condition(temp)) return false;
      d = (temp).ldlt().solve(x.transpose() * g);
    }

    // theta = x * beta0;
    // for (int i = 0; i < n; i++)
    // {
    //   if (theta(i) > 30)
    //     theta(i) = 30;
    //   else if (theta(i) < -30)
    //     theta(i) = -30;
    // }
    // theta = theta.array().exp();
    // cum_theta = one * theta;
    // x_theta = x.array().colwise() * theta.array();
    // x_theta = one * x_theta;
    // x_theta = x_theta.array().colwise() / cum_theta.array();
    // g = (x - x_theta).transpose() * (weights.cwiseProduct(y));

    // if (approximate_Newton)
    // {
    //   Eigen::VectorXd h(p);
    //   for (int k1 = 0; k1 < p; k1++)
    //   {
    //     xij_theta = (theta.cwiseProduct(x.col(k1))).cwiseProduct(x.col(k1));
    //     for (int j = n - 2; j >= 0; j--)
    //     {
    //       xij_theta(j) = xij_theta(j + 1) + xij_theta(j);
    //     }
    //     h(k1) = -(xij_theta.cwiseQuotient(cum_theta) - x_theta.col(k1).cwiseProduct(x_theta.col(k1))).dot(weights.cwiseProduct(y));
    //   }
    //   d = g.cwiseQuotient(h);
    // }
    // else
    // {
    //   Eigen::MatrixXd h(p, p);
    //   for (int k1 = 0; k1 < p; k1++)
    //   {
    //     for (int k2 = k1; k2 < p; k2++)
    //     {
    //       xij_theta = (theta.cwiseProduct(x.col(k1))).cwiseProduct(x.col(k2));
    //       for (int j = n - 2; j >= 0; j--)
    //       {
    //         xij_theta(j) = xij_theta(j + 1) + xij_theta(j);
    //       }
    //       h(k1, k2) = -(xij_theta.cwiseQuotient(cum_theta) - x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(y));
    //       h(k2, k1) = h(k1, k2);
    //     }
    //   }
    //   d = h.ldlt().solve(g);
    // }

    beta1 = beta0 + step * d;

    loglik1 = loglik_cox(x, y, beta1, weight);

    while (loglik1 < loglik0 && step > primary_model_fit_epsilon)
    {
      step = step / 2;
      beta1 = beta0 + step * d;
      loglik1 = loglik_cox(x, y, beta1, weight);
    }

    bool condition1 = -(loglik1 + (primary_model_fit_max_iter - l - 1) * (loglik1 - loglik0)) + tau > loss0;
    if (condition1)
    {
      loss0 = -loglik0;
      beta = beta0;

      return true;
    }

    if (loglik1 > loglik0)
    {
      beta0 = beta1;
      loglik0 = loglik1;
    }

    if (step < primary_model_fit_epsilon)
    {
      loss0 = -loglik0;
      beta = beta0;

      return true;
    }
    return true;
  }

  beta = beta0;
}

template <class T4>
double loglik_poiss(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &coef, int n, Eigen::VectorXd &weights)
{
  int p = x.cols();
  T4 X(n, p + 1);
  X.rightCols(p) = x;
  add_constant_column(X);
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

#endif