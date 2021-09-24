#ifndef SRC_ALGORITHMGLM_H
#define SRC_ALGORITHMGLM_H

#include "Algorithm.h"
#include "model_fit.h"

using namespace std;

template <class T4>
class abessLogistic : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4, Eigen::VectorXd>
{
public:
  abessLogistic(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4, Eigen::VectorXd>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessLogistic(){};

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
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
    for (j = 0; j < this->primary_model_fit_max_iter; j++)
    {
      // To do: Approximate Newton method
      if (this->approximate_Newton)
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

        while (loglik1 < loglik0 && step > this->primary_model_fit_epsilon)
        {
          step = step / 2;
          beta1 = beta0 + step * g.cwiseProduct(h_diag);
          Pi = pi(X, y, beta1);
          log_Pi = Pi.array().log();
          log_1_Pi = (one - Pi).array().log();
          loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
        }

        bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + this->tau > loss0;
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

        if (step < this->primary_model_fit_epsilon)
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

        Eigen::MatrixXd XTX = 2 * this->lambda_level * lambdamat + X_new.transpose() * X;
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

        bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + this->tau > loss0;
        bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < this->primary_model_fit_epsilon;
        bool condition3 = abs(loglik1) < min(1e-3, this->tau);
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

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Ones(p + 1);
    coef(0) = coef0;
    coef.tail(p) = beta;
    return -loglik_logit(X, y, coef, n, weights);
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {

    int p = X.cols();
    int n = X.rows();

    Eigen::VectorXd coef(XA.cols() + 1);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    coef << coef0, beta_A;

    Eigen::VectorXd pr = pi(XA, y, coef);
    Eigen::VectorXd res = (y - pr).cwiseProduct(weights);

    Eigen::VectorXd d = X.transpose() * res - 2 * this->lambda_level * beta;
    Eigen::VectorXd h = weights.array() * pr.array() * (one - pr).array();

    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);

    for (int i = 0; i < N; i++)
    {
      T4 XG = X.middleCols(g_index(i), g_size(i));
      T4 XG_new = XG;
      for (int j = 0; j < g_size(i); j++)
      {
        XG_new.col(j) = XG.col(j).cwiseProduct(h);
      }
      Eigen::MatrixXd XGbar;
      XGbar = XG_new.transpose() * XG + 2 * this->lambda_level * Eigen::MatrixXd::Identity(g_size(i), g_size(i));

      //to do
      Eigen::MatrixXd phiG;
      XGbar.sqrt().evalTo(phiG);
      Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
      betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
      dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
    }
    for (int i = 0; i < A_size; i++)
    {
      bd(A(i)) = betabar.segment(g_index(A(i)), g_size(A(i))).squaredNorm() / g_size(A(i));
    }
    for (int i = 0; i < I_size; i++)
    {
      bd(I(i)) = dbar.segment(g_index(I(i)), g_size(I(i))).squaredNorm() / g_size(I(i));
    }

    return;
  }

  double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0)
  {
    if (this->lambda_level == 0.)
    {
      return XA.cols();
    }
    else
    {
      if (XA.cols() == 0)
        return 0.;

      // int p = X.cols();
      int n = X.rows();

      Eigen::VectorXd coef = Eigen::VectorXd::Ones(XA.cols() + 1);
      Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
      coef(0) = coef0;
      coef.tail(XA.cols()) = beta_A;

      Eigen::VectorXd pr = pi(XA, y, coef);
      // Eigen::VectorXd res = (y - pr).cwiseProduct(weights);

      // Eigen::VectorXd d = X.transpose() * res - 2 * this->lambda_level * beta;
      Eigen::VectorXd h = weights.array() * pr.array() * (one - pr).array();

      T4 XA_new = XA;
      for (int j = 0; j < XA.cols(); j++)
      {
        XA_new.col(j) = XA.col(j).cwiseProduct(h);
      }
      Eigen::MatrixXd XGbar;
      XGbar = XA_new.transpose() * XA;

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> adjoint_eigen_solver(XGbar);

      double enp = 0.;
      for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++)
      {
        enp += adjoint_eigen_solver.eigenvalues()(i) / (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
      }

      return enp;
    }
  }
};

template <class T4>
class abessLm : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4, Eigen::VectorXd>
{
public:
  abessLm(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = true, int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4, Eigen::VectorXd>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, covariance_update, splicing_type, sub_search){};

  ~abessLm(){};

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    int n = x.rows();
    int p = x.cols();

    // to ensure
    T4 X(n, p + 1);
    X.rightCols(p) = x;
    add_constant_column(X);
    // beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);
    Eigen::MatrixXd XTX = X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols());
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
    // cg.compute(X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols()));
    // beta = cg.solveWithGuess(X.adjoint() * y, beta);
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    int n = X.rows();
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    return (y - X * beta - coef0 * one).array().square().sum() / n;
  }

  void mapping_U(Eigen::VectorXi &U, Eigen::VectorXi &U_ind)
  {
    int N = U.size(), p = U_ind.size();
    if (this->covariance_update)
      for (int i = 0; i < p; i++)
      {
        this->XTy_U(i) = this->XTy(U_ind(i), 0);
        this->XTone_U(i) = this->XTone(U_ind(i), 0);
      }

    for (int i = 0; i < N; i++)
    {
      this->PhiG_U(i, 0) = this->PhiG(U(i), 0);
      this->invPhiG_U(i, 0) = this->invPhiG(U(i), 0);
    }
    return;
  }

  Eigen::MatrixXd covariance_update_f_U(Eigen::VectorXi &U_ind, Eigen::VectorXi &A_ind_U)
  {
    int k = A_ind_U.size(), p = U_ind.size();
    Eigen::MatrixXd cov_A(p, k);

    for (int i = 0; i < k; i++)
    {
      int Ai = U_ind(A_ind_U(i));
      if (!this->covariance_update_flag[Ai])
      {
        this->covariance[Ai] = new Eigen::VectorXd;
        *this->covariance[Ai] = (*this->x).transpose() * (*this->x).col(Ai);
        this->covariance_update_flag[Ai] = true;
      }
      if (p == this->XTy.rows())
      {
        cov_A.col(i) = *this->covariance[Ai];
      }
      else
      {
        for (int j = 0; j < p; j++)
          cov_A(j, i) = (*this->covariance[Ai])(U_ind(j));
      }
    }

    return cov_A;
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {

    int p = X.cols();
    int n = X.rows();

    if (num == 0)
    {
      if (p == this->XTy.rows())
      {
        this->XTy_U = this->XTy;
        this->XTone_U = this->XTone;
        this->PhiG_U = this->PhiG;
        this->invPhiG_U = this->invPhiG;
      }
      else
      {
        this->XTy_U.resize(p, 1);
        this->XTone_U.resize(p, 1);
        this->mapping_U(U, U_ind);
      }
    }

    Eigen::VectorXd d;
    if (!this->covariance_update)
    {
      Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

      if (beta.size() != 0)
      {
        d = X.adjoint() * (y - XA * beta_A - coef0 * one) / double(n) - 2 * this->lambda_level * beta;
      }
      else
      {
        d = X.adjoint() * (y - coef0 * one) / double(n);
      }
    }
    else
    {
      Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
      if (beta.size() != 0)
      {
        Eigen::VectorXd XTXbeta = this->covariance_update_f_U(U_ind, A_ind) * beta_A;
        d = (this->XTy_U - XTXbeta - this->XTone_U * coef0) / double(n) - 2 * this->lambda_level * beta;
      }
      else
      {
        Eigen::VectorXd XTonecoef0 = this->XTone_U * coef0;
        d = (this->XTy_U - XTonecoef0) / double(n);
      }
    }

    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd phiG, invphiG;

    for (int i = 0; i < N; i++)
    {
      phiG = this->PhiG_U(i, 0);
      invphiG = this->invPhiG_U(i, 0);
      betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
      dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
    }
    for (int i = 0; i < A_size; i++)
    {
      bd(A[i]) = betabar.segment(g_index(A[i]), g_size(A[i])).squaredNorm() / g_size(A[i]);
    }
    for (int i = 0; i < I_size; i++)
    {
      bd(I[i]) = dbar.segment(g_index(I[i]), g_size(I[i])).squaredNorm() / g_size(I[i]);
    }
  }

  double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0)
  {
    if (this->lambda_level == 0.)
    {
      return XA.cols();
    }
    else
    {
      return double(XA.cols()) / (this->lambda_level + 1);
    }
  }
};

template <class T4>
class abessPoisson : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4, Eigen::VectorXd>
{
public:
  abessPoisson(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4, Eigen::VectorXd>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessPoisson(){};

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
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
    for (j = 0; j < this->primary_model_fit_max_iter; j++)
    {
      for (int i = 0; i < p + 1; i++)
      {
        // temp.col(i) = X_trans.col(i) * expeta(i) * weights(i);
        X_new.col(i) = X.col(i).cwiseProduct(expeta).cwiseProduct(weights);
      }
      z = eta + (y - expeta).cwiseQuotient(expeta);
      Eigen::MatrixXd XTX = X_new.transpose() * X + 2 * this->lambda_level * lambdamat;
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
      bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + this->tau > loss0;
      // bool condition1 = false;
      bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < this->primary_model_fit_epsilon;
      bool condition3 = abs(loglik1) < min(1e-3, this->tau);
      if (condition1 || condition2 || condition3)
      {

        break;
      }
      loglik0 = loglik1;
    }

    beta = beta0.tail(p).eval();
    coef0 = beta0(0);
    return true;
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Ones(p + 1);
    coef(0) = coef0;
    coef.tail(p) = beta;
    return -loglik_poiss(X, y, coef, n, weights);
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {

    int p = X.cols();
    int n = X.rows();

    Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
    Eigen::VectorXd xbeta_exp = XA * beta_A + coef;
    for (int i = 0; i <= n - 1; i++)
    {
      if (xbeta_exp(i) > 30.0)
        xbeta_exp(i) = 30.0;
      if (xbeta_exp(i) < -30.0)
        xbeta_exp(i) = -30.0;
    }
    xbeta_exp = xbeta_exp.array().exp();

    Eigen::VectorXd d = X.transpose() * (y - xbeta_exp) - 2 * this->lambda_level * beta;
    Eigen::VectorXd h = xbeta_exp;

    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);

    for (int i = 0; i < N; i++)
    {
      T4 XG = X.middleCols(g_index(i), g_size(i));
      T4 XG_new = XG;
      for (int j = 0; j < g_size(i); j++)
      {
        XG_new.col(j) = XG.col(j).cwiseProduct(h);
      }
      Eigen::MatrixXd XGbar;
      XGbar = XG_new.transpose() * XG + 2 * this->lambda_level * Eigen::MatrixXd::Identity(g_size(i), g_size(i));

      Eigen::MatrixXd phiG;
      XGbar.sqrt().evalTo(phiG);
      Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
      betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
      dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
    }
    for (int i = 0; i < A_size; i++)
    {
      bd(A[i]) = betabar.segment(g_index(A[i]), g_size(A[i])).squaredNorm() / g_size(A[i]);
    }
    for (int i = 0; i < I_size; i++)
    {
      bd(I[i]) = dbar.segment(g_index(I[i]), g_size(I[i])).squaredNorm() / g_size(I[i]);
    }
  }

  double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0)
  {
    if (this->lambda_level == 0.)
    {
      return XA.cols();
    }
    else
    {
      if (XA.cols() == 0)
        return 0.;

      // int p = X.cols();
      int n = X.rows();

      Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
      Eigen::VectorXd xbeta_exp = XA * beta_A + coef;
      for (int i = 0; i <= n - 1; i++)
      {
        if (xbeta_exp(i) > 30.0)
          xbeta_exp(i) = 30.0;
        if (xbeta_exp(i) < -30.0)
          xbeta_exp(i) = -30.0;
      }
      xbeta_exp = xbeta_exp.array().exp();

      // Eigen::VectorXd d = X.transpose() * res - 2 * this->lambda_level * beta;
      Eigen::VectorXd h = xbeta_exp;

      T4 XA_new = XA;
      for (int j = 0; j < XA.cols(); j++)
      {
        XA_new.col(j) = XA.col(j).cwiseProduct(h);
      }
      Eigen::MatrixXd XGbar;
      XGbar = XA_new.transpose() * XA;

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> adjoint_eigen_solver(XGbar);

      double enp = 0.;
      for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++)
      {
        enp += adjoint_eigen_solver.eigenvalues()(i) / (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
      }

      return enp;
    }
  }
};

template <class T4>
class abessCox : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4, Eigen::VectorXd>
{
public:
  abessCox(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4, Eigen::VectorXd>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessCox(){};

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, Eigen::VectorXd &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {

    if (x.cols() == 0)
    {
      coef0 = 0.;
      return true;
    }

    int n = x.rows();
    int p = x.cols();
    Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p, p);
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
    double loglik1, loglik0 = -neg_loglik_loss(x, y, weight, beta0, coef0, A, g_index, g_size);
    // beta = Eigen::VectorXd::Zero(p);

    double step = 1.0;
    int l;
    for (l = 1; l <= this->primary_model_fit_max_iter; l++)
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
      // g = x.transpose() * (weight.cwiseProduct(y) - cum_eta2.cwiseProduct(eta));
      // g = g - 2 * this->lambda_level * beta0;

      g = weight.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);

      Eigen::MatrixXd temp = x.transpose() * h * x;
      if (this->approximate_Newton)
      {
        // d = g.cwiseQuotient((x.transpose() * h * x + 2 * this->lambda_level * lambdamat).diagonal());
        d = (x.transpose() * g - 2 * this->lambda_level * beta0).cwiseQuotient(temp.diagonal());
      }
      else
      {
        // d = (x.transpose() * h * x + 2 * this->lambda_level * lambdamat).ldlt().solve(g);
        // if (check_ill_condition(temp)) return false;
        d = temp.ldlt().solve(x.transpose() * g - 2 * this->lambda_level * beta0);
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

      // if (this->approximate_Newton)
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

      loglik1 = -neg_loglik_loss(x, y, weight, beta1, coef0, A, g_index, g_size);

      while (loglik1 < loglik0 && step > this->primary_model_fit_epsilon)
      {
        step = step / 2;
        beta1 = beta0 + step * d;
        loglik1 = -neg_loglik_loss(x, y, weight, beta1, coef0, A, g_index, g_size);
      }

      bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - l - 1) * (loglik1 - loglik0)) + this->tau > loss0;
      if (condition1)
      {
        loss0 = -loglik0;
        beta = beta0;
        this->cox_hessian = h;
        this->cox_g = g;

        return true;
      }

      if (loglik1 > loglik0)
      {
        beta0 = beta1;
        loglik0 = loglik1;
        this->cox_hessian = h;
        this->cox_g = g;
      }

      if (step < this->primary_model_fit_epsilon)
      {
        loss0 = -loglik0;
        beta = beta0;
        this->cox_hessian = h;
        this->cox_g = g;

        return true;
      }
    }

    beta = beta0;
    return true;
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    return -loglik_cox(X, y, beta, weights);
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {

    int p = X.cols();
    int n = X.rows();

    Eigen::VectorXd d;
    Eigen::MatrixXd h;
    Eigen::VectorXd g;
    if (this->cox_g.size() != 0)
    {
      h = this->cox_hessian;
      g = this->cox_g;
    }
    else
    {
      Eigen::VectorXd cum_eta(n);
      Eigen::VectorXd cum_eta2(n);
      Eigen::VectorXd cum_eta3(n);
      Eigen::VectorXd eta = XA * beta_A;
      for (int i = 0; i <= n - 1; i++)
      {
        if (eta(i) < -30.0)
          eta(i) = -30.0;
        if (eta(i) > 30.0)
          eta(i) = 30.0;
      }
      eta = weights.array() * eta.array().exp();
      cum_eta(n - 1) = eta(n - 1);
      for (int k = n - 2; k >= 0; k--)
      {
        cum_eta(k) = cum_eta(k + 1) + eta(k);
      }
      cum_eta2(0) = (y(0) * weights(0)) / cum_eta(0);
      for (int k = 1; k <= n - 1; k++)
      {
        cum_eta2(k) = (y(k) * weights(k)) / cum_eta(k) + cum_eta2(k - 1);
      }
      cum_eta3(0) = (y(0) * weights(0)) / pow(cum_eta(0), 2);
      for (int k = 1; k <= n - 1; k++)
      {
        cum_eta3(k) = (y(k) * weights(k)) / pow(cum_eta(k), 2) + cum_eta3(k - 1);
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
      g = weights.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);
    }

    d = X.transpose() * g - 2 * this->lambda_level * beta;

    // Eigen::VectorXd d = X.transpose() * res;
    // Eigen::VectorXd h = weights.array() * pr.array() * (one - pr).array();

    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);

    for (int i = 0; i < N; i++)
    {
      T4 XG = X.middleCols(g_index(i), g_size(i));
      Eigen::MatrixXd XGbar = XG.transpose() * h * XG + 2 * this->lambda_level * Eigen::MatrixXd::Identity(g_size(i), g_size(i));

      Eigen::MatrixXd phiG;
      XGbar.sqrt().evalTo(phiG);
      Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
      betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
      dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
    }
    for (int i = 0; i < A_size; i++)
    {
      bd(A[i]) = betabar.segment(g_index(A[i]), g_size(A[i])).squaredNorm() / g_size(A[i]);
    }
    for (int i = 0; i < I_size; i++)
    {
      bd(I[i]) = dbar.segment(g_index(I[i]), g_size(I[i])).squaredNorm() / g_size(I[i]);
    }
  }

  double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0)
  {
    if (this->lambda_level == 0.)
    {
      return XA.cols();
    }
    else
    {
      if (XA.cols() == 0)
        return 0.;

      // int p = X.cols();
      int n = X.rows();

      Eigen::VectorXd d;
      Eigen::MatrixXd h;
      Eigen::VectorXd g;
      if (this->cox_g.size() != 0)
      {
        h = this->cox_hessian;
        // g = this->cox_g;
      }

      // Eigen::VectorXd d = X.transpose() * res - 2 * this->lambda_level * beta;
      // Eigen::VectorXd h = weights.array() * pr.array() * (one - pr).array();

      Eigen::MatrixXd XGbar = XA.transpose() * h * XA;

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> adjoint_eigen_solver(XGbar);

      double enp = 0.;
      for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++)
      {
        enp += adjoint_eigen_solver.eigenvalues()(i) / (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
      }

      return enp;
    }
  }
};

template <class T4>
class abessMLm : public Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4, Eigen::VectorXd>
{
public:
  abessMLm(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = true, int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4, Eigen::VectorXd>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, covariance_update, splicing_type, sub_search){};

  ~abessMLm(){};

  bool primary_model_fit(T4 &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    // beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);

    int n = x.rows();
    int p = x.cols();
    int M = y.cols();

    // to ensure
    T4 X(n, p + 1);
    X.rightCols(p) = x;
    add_constant_column(X);
    // beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);
    Eigen::MatrixXd XTX = X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols());
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
    // beta = (XTX + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).ldlt().solve(X.transpose() * y);

    // CG
    // ConjugateGradient<T4, Lower | Upper> cg;
    // cg.compute(X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols()));
    // beta = cg.solveWithGuess(X.adjoint() * y, beta);
  };

  double neg_loglik_loss(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    int n = X.rows();
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, y.cols());
    return (y - X * beta - array_product(one, coef0)).array().square().sum() / n / 2.0;
  }

  void mapping_U(Eigen::VectorXi &U, Eigen::VectorXi &U_ind)
  {
    int N = U.size(), p = U_ind.size(), M = this->XTy.cols();
    if (this->covariance_update)
      for (int i = 0; i < p; i++)
        for (int j = 0; j < M; j++)
        {
          this->XTy_U(i, j) = this->XTy(U_ind(i), j);
          this->XTone_U(i, j) = this->XTone(U_ind(i), j);
        }

    for (int i = 0; i < N; i++)
    {
      this->PhiG_U(i, 0) = this->PhiG(U(i), 0);
      this->invPhiG_U(i, 0) = this->invPhiG(U(i), 0);
    }
    return;
  }

  Eigen::MatrixXd covariance_update_f_U(Eigen::VectorXi &U_ind, Eigen::VectorXi &A_ind_U)
  {
    int k = A_ind_U.size(), p = U_ind.size();
    Eigen::MatrixXd cov_A(p, k);

    for (int i = 0; i < k; i++)
    {
      int Ai = U_ind(A_ind_U(i));
      if (!this->covariance_update_flag[Ai])
      {
        this->covariance[Ai] = new Eigen::VectorXd;
        *this->covariance[Ai] = (*this->x).transpose() * (*this->x).col(Ai);
        this->covariance_update_flag[Ai] = true;
      }
      if (p == this->XTy.rows())
      {
        cov_A.col(i) = *this->covariance[Ai];
      }
      else
      {
        for (int j = 0; j < p; j++)
          cov_A(j, i) = (*this->covariance[Ai])(U_ind(j));
      }
    }

    return cov_A;
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {

    int p = X.cols();
    int n = X.rows();
    int M = y.cols();

    if (num == 0)
      if (p == this->XTy.rows())
      {
        this->XTy_U = this->XTy;
        this->XTone_U = this->XTone;
        this->PhiG_U = this->PhiG;
        this->invPhiG_U = this->invPhiG;
      }
      else
      {
        this->XTy_U.resize(p, M);
        this->XTone_U.resize(p, M);
        this->mapping_U(U, U_ind);
      }

    Eigen::MatrixXd d;
    if (!this->covariance_update)
    {
      Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, y.cols());
      if (beta.size() != 0)
      {
        d = X.adjoint() * (y - XA * beta_A - array_product(one, coef0)) / double(n) - 2 * this->lambda_level * beta;
      }
      else
      {
        d = X.adjoint() * (y - array_product(one, coef0)) / double(n);
      }
    }
    else
    {
      if (beta.size() != 0)
      {

        Eigen::MatrixXd XTXbeta = this->covariance_update_f_U(U_ind, A_ind) * beta_A;

        d = (this->XTy_U - XTXbeta - array_product(this->XTone_U, coef0)) / double(n) - 2 * this->lambda_level * beta;
      }
      else
      {
        Eigen::MatrixXd XTonecoef0 = array_product(this->XTone_U, coef0);
        d = (this->XTy_U - XTonecoef0) / double(n);
      }
    }

    int A_size = A.size();
    int I_size = I.size();

    Eigen::MatrixXd betabar = Eigen::MatrixXd::Zero(p, M);
    Eigen::MatrixXd dbar = Eigen::MatrixXd::Zero(p, M);
    Eigen::MatrixXd phiG, invphiG;

    for (int i = 0; i < N; i++)
    {
      phiG = this->PhiG_U(i, 0);
      invphiG = this->invPhiG_U(i, 0);
      betabar.block(g_index(i), 0, g_size(i), M) = phiG * beta.block(g_index(i), 0, g_size(i), M);
      dbar.block(g_index(i), 0, g_size(i), M) = invphiG * d.block(g_index(i), 0, g_size(i), M);
    }
    for (int i = 0; i < A_size; i++)
    {
      bd(A[i]) = betabar.block(g_index(A[i]), 0, g_size(A[i]), M).squaredNorm() / g_size(A[i]);
    }
    for (int i = 0; i < I_size; i++)
    {
      bd(I[i]) = dbar.block(g_index(I[i]), 0, g_size(I[i]), M).squaredNorm() / g_size(I[i]);
    }
  }

  double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0)
  {
    if (this->lambda_level == 0.)
    {
      return XA.cols();
    }
    else
    {

      return double(XA.cols()) / (this->lambda_level + 1.0);
    }
  }
};

template <class T4>
class abessMultinomial : public Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4, Eigen::VectorXd>
{
public:
  abessMultinomial(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = true, int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4, Eigen::VectorXd>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, covariance_update, splicing_type, sub_search){};

  ~abessMultinomial(){};

  bool primary_model_fit(T4 &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
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
    if (this->approximate_Newton)
    {
      Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, M);
      double t = 2 * (Pi.array() * (one - Pi).array()).maxCoeff();
      Eigen::MatrixXd res = X.transpose() * (y - Pi) / t;
      // ConjugateGradient<MatrixXd, Lower | Upper> cg;
      // cg.compute(X.adjoint() * X);
      Eigen::MatrixXd XTX = X.transpose() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols());
      // if (check_ill_condition(XTX)) return false;
      Eigen::MatrixXd invXTX = XTX.ldlt().solve(Eigen::MatrixXd::Identity(p + 1, p + 1));

      Eigen::MatrixXd beta1;
      for (j = 0; j < this->primary_model_fit_max_iter; j++)
      {
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

        bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + this->tau > loss0;
        // bool condition1 = false;
        bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < this->primary_model_fit_epsilon;
        bool condition3 = abs(loglik1) < min(1e-3, this->tau);
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
          XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X + 2 * this->lambda_level * lambdamat;
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
      for (j = 0; j < this->primary_model_fit_max_iter; j++)
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

        bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + this->tau > loss0;
        // bool condition1 = false;
        bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < this->primary_model_fit_epsilon;
        bool condition3 = abs(loglik1) < min(1e-3, this->tau);
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
            XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X + 2 * this->lambda_level * lambdamat;
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
  };

  double neg_loglik_loss(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    // weight
    Eigen::MatrixXd pr;
    pi(X, y, beta, coef0, pr);
    Eigen::MatrixXd log_pr = pr.array().log();
    // Eigen::VectorXd one_vec = Eigen::VectorXd::Ones(X.rows());

    array_product(log_pr, weights, 1);

    return -((log_pr.array() * y.array()).sum());
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {

    int p = X.cols();
    int n = X.rows();
    int M = y.cols();

    Eigen::MatrixXd d;
    Eigen::MatrixXd h;
    Eigen::MatrixXd pr;
    pi(XA, y, beta_A, coef0, pr);
    Eigen::MatrixXd Pi = pr.leftCols(M - 1);
    Eigen::MatrixXd res = (y.leftCols(M - 1) - Pi);
    for (int i = 0; i < n; i++)
    {
      res.row(i) = res.row(i) * weights(i);
    }
    d = X.transpose() * res - 2 * this->lambda_level * beta;
    h = Pi;

    // int A_size = A.size();
    // int I_size = I.size();

    Eigen::MatrixXd betabar = Eigen::MatrixXd::Zero(p, M);
    Eigen::MatrixXd dbar = Eigen::MatrixXd::Zero(p, M);
    Eigen::MatrixXd phiG, invphiG;

    for (int i = 0; i < N; i++)
    {
      T4 XG = X.middleCols(g_index(i), g_size(i));
      T4 XG_new(h.rows(), h.cols());
      for (int m = 0; m < M - 1; m++)
      {
        XG_new.col(m) = h.col(m).cwiseProduct(XG);
      }
      Eigen::MatrixXd XGbar = -XG_new.transpose() * XG_new;

      XGbar.diagonal() = Eigen::VectorXd(XG_new.transpose() * XG) + XGbar.diagonal();

      XGbar = XGbar + 2 * this->lambda_level * Eigen::MatrixXd::Identity(M - 1, M - 1);

      // Eigen::MatrixXd phiG;
      // XGbar.sqrt().evalTo(phiG);
      // Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(M, M));
      // betabar.block(g_index(i), 0, g_size(i), M) = phiG * beta.block(g_index(i), 0, g_size(i), M);
      // dbar.block(g_index(i), 0, g_size(i), M) = invphiG * d.block(g_index(i), 0, g_size(i), M);

      Eigen::MatrixXd invXGbar = XGbar.ldlt().solve(Eigen::MatrixXd::Identity(M - 1, M - 1));
      Eigen::MatrixXd temp = d.block(g_index(i), 0, g_size(i), M - 1) * invXGbar + beta.block(g_index(i), 0, g_size(i), M - 1);
      bd(i) = (temp * XGbar * temp.transpose()).squaredNorm() / g_size(i);
    }
    // for (int i = 0; i < A_size; i++)
    // {
    //   bd(A[i]) = betabar.block(g_index(A[i]), 0, g_size(A[i]), M).squaredNorm() / g_size(A[i]);
    // }
    // for (int i = 0; i < I_size; i++)
    // {
    //   bd(I[i]) = dbar.block(g_index(I[i]), 0, g_size(I[i]), M).squaredNorm() / g_size(I[i]);
    // }
  }

  double effective_number_of_parameter(T4 &x, T4 &XA, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0)
  {
    if (this->lambda_level == 0.)
    {
      return XA.cols();
    }
    else
    {
      if (XA.cols() == 0)
        return 0.;

      int n = XA.rows();
      int p = XA.cols();
      int M = y.cols();
      T4 X(n, p + 1);
      X.rightCols(p) = XA;
      add_constant_column(X);
      Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p + 1, p + 1);
      Eigen::MatrixXd beta0 = Eigen::MatrixXd::Zero(p + 1, M);

      Eigen::MatrixXd one_vec = Eigen::VectorXd::Ones(n);
      beta0.row(0) = coef0;
      beta0.block(1, 0, p, M) = beta;
      Eigen::MatrixXd Pi;
      pi(X, y, beta0, Pi);
      // Eigen::MatrixXd log_Pi = Pi.array().log();
      // array_product(log_Pi, weights, 1);
      // double loglik1 = DBL_MAX, loglik0 = (log_Pi.array() * y.array()).sum();

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
          XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X + 2 * this->lambda_level * lambdamat;
          XTW.block(m2 * (p + 1), m1 * n, (p + 1), n) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n);
          XTWX.block(m2 * (p + 1), m1 * (p + 1), (p + 1), (p + 1)) = XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1));
        }
      }

      // Eigen::VectorXd d = X.transpose() * res - 2 * this->lambda_level * beta;
      // Eigen::VectorXd h = weights.array() * pr.array() * (one - pr).array();

      // T4 XA_new = XA;
      // for (int j = 0; j < XA.cols(); j++)
      // {
      //   XA_new.col(j) = XA.col(j).cwiseProduct(h);
      // }
      // Eigen::MatrixXd XGbar;
      // XGbar = XA_new.transpose() * XA;

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> adjoint_eigen_solver(XTWX);

      double enp = 0.;
      for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++)
      {
        enp += adjoint_eigen_solver.eigenvalues()(i) / (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
      }

      return enp;
    }
  }
};

#endif // SRC_ALGORITHMGLM_H