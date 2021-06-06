//
// Created by jk on 2020/3/18.
//
// #define TEST

#ifndef SRC_ALGORITHM_H
#define SRC_ALGORITHM_H

#ifndef R_BUILD
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Eigen>
#include <Spectra/SymEigsSolver.h>
#endif

#include "Data.h"
#include "utilities.h"
#include "model_fit.h"
#include <iostream>

#include <time.h>
#include <cfloat>

using namespace std;
using namespace Spectra;

bool quick_sort_pair_max(std::pair<int, double> x, std::pair<int, double> y);

//  T1 for y, XTy, XTone
//  T2 for beta
//  T3 for coef0
//  T4 for X
//  <Eigen::VectorXd, Eigen::VectorXd, double, Eigen::MatrixXd> for Univariate Dense
//  <Eigen::VectorXd, Eigen::VectorXd, double, Eigen::SparseMatrix<double> > for Univariate Sparse
//  <Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> for Multivariable Dense
//  <Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::SparseMatrix<double> > for Multivariable Sparse
template <class T1, class T2, class T3, class T4>
class Algorithm
{
public:
  int l;
  int model_fit_max;
  int model_type;
  int algorithm_type;

  int group_df = 0;
  int sparsity_level = 0;
  double lambda_level = 0;
  Eigen::VectorXi train_mask;
  int max_iter;
  int exchange_num;
  bool warm_start;
  T2 beta;
  Eigen::VectorXd bd;
  T3 coef0;
  double train_loss = 0.;
  T2 beta_init;
  T3 coef0_init;
  Eigen::VectorXi A_init;
  Eigen::VectorXi I_init;
  Eigen::VectorXd bd_init;

  Eigen::VectorXi A_out;
  Eigen::VectorXi I_out;

  Eigen::Matrix<Eigen::MatrixXd, -1, -1> PhiG;
  Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhiG;
  Eigen::Matrix<T4, -1, -1> group_XTX;

  Eigen::VectorXi always_select;
  double tau;
  int primary_model_fit_max_iter;
  double primary_model_fit_epsilon;
  bool approximate_Newton;

  T2 beta_warmstart;
  T3 coef0_warmstart;

  Eigen::VectorXi status;

  Eigen::MatrixXd cox_hessian;
  Eigen::VectorXd cox_g;

  bool covariance_update;

  // to ensure
  Eigen::MatrixXd covariance;
  Eigen::VectorXi covariance_update_flag;
  T1 XTy;
  T1 XTone;

  Eigen::VectorXi U1;

  Eigen::MatrixXd Sigma;

  int splicing_type;

  Algorithm() = default;

  virtual ~Algorithm(){};

  Algorithm(int algorithm_type, int model_type, int max_iter = 100, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = false, int splicing_type = 0)
  {
    this->max_iter = max_iter;
    this->model_type = model_type;
    // this->coef0_init = 0.0;
    this->warm_start = warm_start;
    this->exchange_num = exchange_num;
    this->approximate_Newton = approximate_Newton;
    this->always_select = always_select;
    this->algorithm_type = algorithm_type;
    this->primary_model_fit_max_iter = primary_model_fit_max_iter;
    this->primary_model_fit_epsilon = primary_model_fit_epsilon;

    this->covariance_update = covariance_update;

    this->splicing_type = splicing_type;
  };

  void update_PhiG(Eigen::Matrix<Eigen::MatrixXd, -1, -1> &PhiG) { this->PhiG = PhiG; }

  void update_invPhiG(Eigen::Matrix<Eigen::MatrixXd, -1, -1> &invPhiG) { this->invPhiG = invPhiG; }

  void set_warm_start(bool warm_start) { this->warm_start = warm_start; }

  void update_beta_init(T2 &beta_init) { this->beta_init = beta_init; }

  void update_A_init(Eigen::VectorXi &A_init, int g_num)
  {
    this->A_init = A_init;
    this->I_init = Ac(A_init, g_num);
  }

  void update_bd_init(Eigen::VectorXd &bd_init) { this->bd_init = bd_init; }

  void update_coef0_init(T3 coef0) { this->coef0_init = coef0; }

  void update_group_df(int group_df) { this->group_df = group_df; }

  void update_sparsity_level(int sparsity_level) { this->sparsity_level = sparsity_level; }

  void update_lambda_level(double lambda_level) { this->lambda_level = lambda_level; }

  void update_train_mask(Eigen::VectorXi &train_mask) { this->train_mask = train_mask; }

  void update_exchange_num(int exchange_num) { this->exchange_num = exchange_num; }

  void update_group_XTX(Eigen::Matrix<T4, -1, -1> &group_XTX) { this->group_XTX = group_XTX; }

  bool get_warm_start() { return this->warm_start; }

  double get_train_loss() { return this->train_loss; }

  int get_group_df() { return this->group_df; }

  int get_sparsity_level() { return this->sparsity_level; }

  T2 get_beta() { return this->beta; }

  T3 get_coef0() { return this->coef0; }

  Eigen::VectorXi get_A_out() { return this->A_out; };

  Eigen::VectorXi get_I_out() { return this->I_out; };

  Eigen::VectorXd get_bd() { return this->bd; }

  int get_l() { return this->l; }

  void fit(T4 &train_x, T1 &train_y, Eigen::VectorXd &train_weight, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int train_n, int p, int N, Eigen::VectorXi &status)
  {
    // std::cout << "fit" << endl;
    int T0 = this->sparsity_level;
    this->status = status;
    this->cox_g = Eigen::VectorXd::Zero(0);

    this->tau = 0.01 * (double)this->sparsity_level * log((double)N) * log(log((double)train_n)) / (double)train_n;

    this->beta = this->beta_init;
    this->coef0 = this->coef0_init;
    this->bd = this->bd_init;

    if (N == T0)
    {
      this->primary_model_fit(train_x, train_y, train_weight, this->beta, this->coef0, DBL_MAX);
      this->train_loss = neg_loglik_loss(train_x, train_y, train_weight, this->beta, this->coef0);
      this->A_out = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      return;
    }

#ifdef TEST
    clock_t t1, t2;
    t1 = clock();
#endif

    if (this->model_type == 1 || this->model_type == 5)
    {
      if (this->algorithm_type == 6 && this->PhiG.rows() == 0)
      {
        this->PhiG = Phi(train_x, g_index, g_size, train_n, p, N, this->lambda_level, this->group_XTX);
        this->invPhiG = invPhi(PhiG, N);
      }
    }

#ifdef TEST
    t2 = clock();
    std::cout << "PhiG invPhiG time" << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif
    // cout << "this->beta: " << this->beta << endl;
    // cout << "this->coef0_init" << this->coef0_init << endl;
    // cout << "this->A_init: " << this->A_init << endl;
    // cout << "this->I_init: " << this->I_init << endl;

    // input: this->beta_init, this->coef0_init, this->A_init, this->I_init
    // for splicing get A;for the others 0;
    // std::cout << "fit 2" << endl;
    Eigen::VectorXi A = inital_screening(train_x, train_y, this->beta, this->coef0, this->A_init, this->I_init, this->bd, train_weight, g_index, g_size, N);
#ifdef TEST
    t2 = clock();
    std::cout << "init screening time" << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif

    // std::cout << "fit 3" << endl;
    Eigen::VectorXi I = Ac(A, N);
    Eigen::MatrixXi A_list(T0, max_iter + 2);
    A_list.col(0) = A;

    T4 X_A;
    T2 beta_A;
    Eigen::VectorXi A_ind;

    // std::cout << "fit 5" << endl;
#ifdef TEST
    t1 = clock();
#endif
    if (this->algorithm_type == 6)
    {
      A_ind = find_ind(A, g_index, g_size, p, N);
      X_A = X_seg(train_x, train_n, A_ind);
      slice(this->beta, A_ind, beta_A);
      this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0, DBL_MAX);
      slice_restore(beta_A, A_ind, this->beta);
    }

    this->beta_warmstart = this->beta;
    this->coef0_warmstart = this->coef0;

#ifdef TEST
    t2 = clock();
    std::cout << "primary fit" << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif
    // std::cout << "fit 6" << endl;
    int C_max = min(min(T0, N - T0), this->exchange_num);

    for (this->l = 1; this->l <= this->max_iter; l++)
    {
#ifdef TEST
      std::cout << "fit 7" << endl;
#endif
      this->get_A(train_x, train_y, A, I, C_max, this->beta, this->coef0, this->bd, T0, train_weight, g_index, g_size, N, this->tau, this->train_loss);
#ifdef TEST
      t2 = clock();
      std::cout << "get A" << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif

      A_list.col(this->l) = A;
      // std::cout << "fit 8" << endl;

      if (this->algorithm_type != 6)
      {
        A_ind = find_ind(A, g_index, g_size, p, N);
        X_A = X_seg(train_x, train_n, A_ind);
        slice(this->beta, A_ind, beta_A);
        this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0, DBL_MAX);
        slice_restore(beta_A, A_ind, this->beta);
        for (int ll = 0; ll < this->l; ll++)
        {
          if (A == A_list.col(ll))
          {
            this->group_df = 0;
            for (unsigned int i = 0; i < A.size(); i++)
            {
              this->group_df = this->group_df + g_size(A(i));
            }
            return;
          }
        }
      }
      else
      {
        if (A == A_list.col(this->l - 1))
        {
#ifdef TEST
          std::cout << "------------iter time: ----------" << this->l << endl;
          t2 = clock();
#endif
#ifdef TEST
          std::cout << "fit get A" << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
          t1 = clock();
#endif
          this->A_out = A;

          A_ind = find_ind(A, g_index, g_size, p, N);
          X_A = X_seg(train_x, train_n, A_ind);
          slice(this->beta, A_ind, beta_A);
          // cout << "A: " << endl;
          // cout << A << endl;
          // cout << "beta" << endl;
          // cout << beta_A << endl;

          this->group_df = 0;
          for (unsigned int i = 0; i < A.size(); i++)
          {
            this->group_df = this->group_df + g_size(A(i));
          }
#ifdef TEST
          t2 = clock();
          std::cout << "group_df time " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
          return;
        }
      }
    }
  };

  void get_A(T4 &X, T1 &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, T2 &beta, T3 &coef0, Eigen::VectorXd &bd, int T0, Eigen::VectorXd &weights,
             Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau, double &train_loss)
  {
    int p = X.cols();
    int n = X.rows();
#ifdef TEST
    clock_t t0, t1, t2;
    t1 = clock();
#endif

    // calculate beta & d & h
    Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
    T4 X_A = X_seg(X, n, A_ind);
    T2 beta_A;
    slice(beta, A_ind, beta_A);

#ifdef TEST
    t2 = clock();
    std::cout << "XA beatA ind time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    double L1, L0 = neg_loglik_loss(X_A, y, weights, beta_A, coef0);
    train_loss = L0;

#ifdef TEST
    t2 = clock();
    std::cout << "loss time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    bd = Eigen::VectorXd::Zero(N);
    this->sacrifice(X, X_A, y, beta, beta_A, coef0, A, I, weights, g_index, g_size, N, A_ind, bd);

#ifdef TEST
    t2 = clock();
    std::cout << "bd time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    int A_size = A.size();
    int I_size = I.size();
    Eigen::VectorXd beta_A_group = Eigen::VectorXd::Zero(A_size);
    Eigen::VectorXd d_I_group = Eigen::VectorXd::Zero(I_size);
    for (int i = 0; i < A_size; i++)
    {
      beta_A_group(i) = bd(A[i]);
    }

    for (int i = 0; i < I_size; i++)
    {
      d_I_group(i) = bd(I[i]);
    }

    Eigen::VectorXi A_min_k = min_k(beta_A_group, C_max, true);
    Eigen::VectorXi I_max_k = max_k(d_I_group, C_max, true);
    Eigen::VectorXi s1 = vector_slice(A, A_min_k);
    Eigen::VectorXi s2 = vector_slice(I, I_max_k);
#ifdef TEST
    t2 = clock();
    std::cout << "s1 s2 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t0 = clock();
#endif

    // cout << "get A 5" << endl;
    Eigen::VectorXi A_exchange(A_size);
    Eigen::VectorXi A_ind_exchage;
    T4 X_A_exchage;
    T2 beta_A_exchange;
    T3 coef0_A_exchange;

    for (int k = C_max; k >= 1;)
    {
      A_exchange = diff_union(A, s1, s2);
      A_ind_exchage = find_ind(A_exchange, g_index, g_size, p, N);
      X_A_exchage = X_seg(X, n, A_ind_exchage);
      slice(this->beta_warmstart, A_ind_exchage, beta_A_exchange);
      coef0_A_exchange = this->coef0_warmstart;

      primary_model_fit(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange, L0);

      L1 = neg_loglik_loss(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange);

      // cout << "L0: " << L0 << " L1: " << L1 << endl;
      if (L0 - L1 > tau)
      {
        train_loss = L1;
        A = A_exchange;
        I = Ac(A_exchange, N);
        slice_restore(beta_A_exchange, A_ind_exchage, beta);
        coef0 = coef0_A_exchange;
#ifdef TEST
        std::cout << "C_max: " << C_max << " k: " << k << endl;
#endif
        C_max = k;
#ifdef TEST
        t2 = clock();
        std::cout << "splicing time: " << ((double)(t2 - t0) / CLOCKS_PER_SEC) << endl;
#endif
        return;
      }
      else
      {
        if (this->splicing_type == 1){
          k = k - 1;
        }else{
          k = k / 2;
        }
        s1 = s1.head(k).eval();
        s2 = s2.head(k).eval();
      }
    }
#ifdef TEST
    t2 = clock();
    std::cout << "splicing time: " << ((double)(t2 - t0) / CLOCKS_PER_SEC) << endl;
#endif
  };

  Eigen::VectorXi inital_screening(T4 &X, T1 &y, T2 &beta, T3 &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &bd, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {
    // cout << "inital_screening: " << endl;
#ifdef TEST
    clock_t t3, t4;
    t3 = clock();
#endif

    if (bd.size() == 0)
    {
      // variable initialization
      int n = X.rows();
      int p = X.cols();
      bd = Eigen::VectorXd::Zero(N);

      // calculate beta & d & h
      Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
      T4 X_A = X_seg(X, n, A_ind);
      T2 beta_A;
      slice(beta, A_ind, beta_A);

      this->sacrifice(X, X_A, y, beta, beta_A, coef0, A, I, weights, g_index, g_size, N, A_ind, bd);
    }

#ifdef TEST
    t4 = clock();
    std::cout << "inital_screening bd: " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
    t3 = clock();
#endif

    // get Active-set A according to max_k bd
    Eigen::VectorXi A_new = max_k_2(bd, this->sparsity_level);
    //int p = X.cols();

    this->U1 = max_k(bd, min(this->sparsity_level + 100, N));

#ifdef TEST
    t4 = clock();
    std::cout << "inital_screening max_k: " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
#endif
    return A_new;
  }

  virtual double neg_loglik_loss(T4 &X, T1 &y, Eigen::VectorXd &weights, T2 &beta, T3 &coef0) = 0;

  virtual void sacrifice(T4 &X, T4 &XA, T1 &y, T2 &beta, T2 &beta_A, T3 &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd) = 0;

  virtual void primary_model_fit(T4 &X, T1 &y, Eigen::VectorXd &weights, T2 &beta, T3 &coef0, double loss0) = 0;
};

template <class T4>
class abessLogistic : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessLogistic(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, splicing_type){};

  ~abessLogistic(){};

  void primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0)
  {
#ifdef TEST
    clock_t t1 = clock();
#endif
    // cout << "primary_fit-----------" << endl;
    if (x.cols() == 0)
    {
      coef0 = -log(1 / y.mean() - 1);
      return;
    }

    int n = x.rows();
    int p = x.cols();

    // to ensure
    T4 X(n, p + 1);
    // set_nonzeros(X, x);
    X.rightCols(p) = x;

    // to do !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    add_constant_column(X);

    T4 X_new(X);
    // T4 X_new_transpose(X.transpose());

#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit init time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif

    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
    beta0(0) = coef0;
    beta0.tail(p) = beta;
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

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

    // cout << "l0 loglik: " << loglik0 << endl;

    int j;
    // double step = 1;
    Eigen::VectorXd g(p + 1);
    Eigen::VectorXd beta1;
    for (j = 0; j < this->primary_model_fit_max_iter; j++)
    {
      // // To do: Approximate Newton method
      // if (this->approximate_Newton)
      // {
      //   Eigen::VectorXd h_diag(p + 1);
      //   for (int i = 0; i < p + 1; i++)
      //   {
      //     h_diag(i) = 1.0 / X.col(i).cwiseProduct(W).cwiseProduct(weights).dot(X.col(i));
      //   }
      //   g = X.transpose() * ((y - Pi).cwiseProduct(weights));
      //   beta1 = beta0 + step * g.cwiseProduct(h_diag);
      //   Pi = pi(X, y, beta1);
      //   log_Pi = Pi.array().log();
      //   log_1_Pi = (one - Pi).array().log();
      //   loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);

      //   while (loglik1 < loglik0 && step > this->primary_model_fit_epsilon)
      //   {
      //     step = step / 2;
      //     beta1 = beta0 + step * g.cwiseProduct(h_diag);
      //     Pi = pi(X, y, beta1);
      //     log_Pi = Pi.array().log();
      //     log_1_Pi = (one - Pi).array().log();
      //     loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
      //   }

      //   // cout << "j=" << j << " loglik: " << loglik1 << endl;
      //   // cout << "j=" << j << " loglik diff: " << loglik1 - loglik0 << endl;
      //   bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + this->tau > loss0;
      //   // bool condition1 = false;
      //   if (condition1)
      //     break;

      //   if (loglik1 > loglik0)
      //   {
      //     beta0 = beta1;
      //     loglik0 = loglik1;
      //     W = Pi.cwiseProduct(one - Pi);
      //     for (int i = 0; i < n; i++)
      //     {
      //       if (W(i) < 0.001)
      //         W(i) = 0.001;
      //     }
      //   }

      //   if (step < this->primary_model_fit_epsilon)
      //   {
      //     break;
      //   }
      // }
      // else
      // {
      // #ifdef TEST
      //         t1 = clock();
      // #endif
      for (int i = 0; i < p + 1; i++)
      {
        X_new.col(i) = X.col(i).cwiseProduct(W).cwiseProduct(weights);
      }

      // X_new_transpose = X_new.transpose();
      // #ifdef TEST
      //         t2 = clock();
      //         std::cout << "primary fit iter 1 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
      //         t1 = clock();
      // #endif

      // to ensure
      // beta0 = (X_new_transpose * X).llt().solve(X_new_transpose * Z);

      overload_ldlt(X_new, X, Z, beta0);

      // CG
      // ConjugateGradient<T4, Lower | Upper> cg;
      // cg.compute(X_new.transpose() * X);
      // beta0 = cg.solve(X_new.transpose() * Z);

      // #ifdef TEST
      //         t2 = clock();
      //         std::cout << "primary fit iter 2 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
      //         t1 = clock();
      // #endif

      Pi = pi(X, y, beta0);
      log_Pi = Pi.array().log();
      log_1_Pi = (one - Pi).array().log();
      loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
      // cout << "j=" << j << " loglik: " << loglik1 << endl;
      // cout << "j=" << j << " loglik diff: " << loglik0 - loglik1 << endl;
      bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + this->tau > loss0;
      // bool condition1 = false;
      bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < this->primary_model_fit_epsilon;
      bool condition3 = abs(loglik1) < min(1e-3, this->tau);
      if (condition1 || condition2 || condition3)
      {
        // cout << "condition1:" << condition1 << endl;
        // cout << "condition2:" << condition2 << endl;
        // cout << "condition3:" << condition3 << endl;
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
    // }
#ifdef TEST
    t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << j << endl;
#endif
    beta = beta0.tail(p).eval();
    coef0 = beta0(0);
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
  {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Ones(p + 1);
    coef(0) = coef0;
    coef.tail(p) = beta;
    return -loglik_logit(X, y, coef, n, weights);
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd)
  {
#ifdef TEST
    clock_t t1 = clock(), t2;
#endif
    int p = X.cols();
    int n = X.rows();

    Eigen::VectorXd coef = Eigen::VectorXd::Ones(XA.cols() + 1);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    coef(0) = coef0;
    coef.tail(XA.cols()) = beta_A;

    Eigen::VectorXd pr = pi(XA, y, coef);
    Eigen::VectorXd res = (y - pr).cwiseProduct(weights);

#ifdef TEST
    t2 = clock();
    std::cout << "d1 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    Eigen::VectorXd d = X.transpose() * res;
    Eigen::VectorXd h = weights.array() * pr.array() * (one - pr).array();

#ifdef TEST
    t2 = clock();
    std::cout << "d2 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

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
      XGbar = XG_new.transpose() * XG;

      //to do
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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
  }
};

template <class T4>
class abessLm : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessLm(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = true, int splicing_type = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, covariance_update, splicing_type){};

  ~abessLm(){};

  void primary_model_fit(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0)
  {
    if (X.cols() == 0)
    {
      coef0 = y.mean();
      return;
    }
    // beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);
    Eigen::MatrixXd XTX = X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols());
    beta = XTX.ldlt().solve(X.adjoint() * y);

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

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
  {
    int n = X.rows();
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    return (y - X * beta - coef0 * one).array().square().sum() / n;
  }

  void covariance_update_f(T4 &X, Eigen::VectorXi &A_ind)
  {
    if (this->covariance.rows() == 0)
    {
      this->covariance = Eigen::MatrixXd::Zero(X.cols(), X.cols());
    }
    for (int i = 0; i < A_ind.size(); i++)
    {
      if (this->covariance_update_flag(A_ind(i)) == 0)
      {
        this->covariance.col(A_ind(i)) = X.transpose() * (X.col(A_ind(i)).eval());
        this->covariance_update_flag(A_ind(i)) = 1;
      }
    }
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd)
  {
#ifdef TEST
    clock_t t1 = clock(), t2;
#endif
    int p = X.cols();
    int n = X.rows();

    Eigen::VectorXd d;
    if (!this->covariance_update)
    {
      Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
      if (beta.size() != 0)
      {
        d = X.adjoint() * (y - XA * beta_A - coef0 * one) / double(n);
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
        this->covariance_update_f(X, A_ind);
        Eigen::VectorXd XTXbeta = X_seg(this->covariance, this->covariance.rows(), A_ind) * beta_A;
        d = (this->XTy - XTXbeta - this->XTone * coef0) / double(n);
      }
      else
      {
        Eigen::VectorXd XTonecoef0 = this->XTone * coef0;
        d = (this->XTy - XTonecoef0) / double(n);
      }
    }

#ifdef TEST
    t2 = clock();
    std::cout << "d1 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

#ifdef TEST
    t2 = clock();
    std::cout << "d2 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd phiG, invphiG;

    for (int i = 0; i < N; i++)
    {
      phiG = this->PhiG(i, 0);
      invphiG = this->invPhiG(i, 0);
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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
  }
};

template <class T4>
class abessPoisson : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessPoisson(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, splicing_type){};

  ~abessPoisson(){};

  void primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0)
  {
#ifdef TEST
    clock_t t1 = clock();
#endif
    // cout << "primary_fit-----------" << endl;
    int n = x.rows();
    int p = x.cols();
    T4 X(n, p + 1);
    X.rightCols(p) = x;
    add_constant_column(X);

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
      Eigen::MatrixXd XTX = X_new.transpose() * X;
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
        // cout << "condition1:" << condition1 << endl;
        // cout << "condition2:" << condition2 << endl;
        // cout << "condition3:" << condition3 << endl;
        break;
      }
      loglik0 = loglik1;
    }
#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << j << endl;
#endif
    beta = beta0.tail(p).eval();
    coef0 = beta0(0);
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
  {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Ones(p + 1);
    coef(0) = coef0;
    coef.tail(p) = beta;
    return -loglik_poiss(X, y, coef, n, weights);
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd)
  {
#ifdef TEST
    clock_t t1 = clock(), t2;
#endif
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

#ifdef TEST
    t2 = clock();
    std::cout << "d1 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    Eigen::VectorXd d = -X.transpose() * (y - xbeta_exp);
    Eigen::VectorXd h = xbeta_exp;

#ifdef TEST
    t2 = clock();
    std::cout << "d2 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);

    for (int i = 0; i < N; i++)
    {
      // T4 XG = X.middleCols(g_index(i), g_size(i));
      // T4 XG_new(g_index(i), g_size(i));
      // for (int j = 0; j < g_size(i); j++)
      // {
      //   XG_new.col(j) = XG.col(j).cwiseProduct(h);
      // }

      T4 XG = X.middleCols(g_index(i), g_size(i));
      T4 XG_new = XG;
      for (int j = 0; j < g_size(i); j++)
      {
        XG_new.col(j) = XG.col(j).cwiseProduct(h);
      }
      Eigen::MatrixXd XGbar;
      XGbar = XG_new.transpose() * XG;

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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
  }
};

template <class T4>
class abessCox : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessCox(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, splicing_type){};

  ~abessCox(){};

  void primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, Eigen::VectorXd &beta, double &coef0, double loss0)
  {
#ifdef TEST
    clock_t t1 = clock();
#endif
    if (x.cols() == 0)
    {
      coef0 = 0.;
      return;
    }

    // cout << "primary_fit-----------" << endl;
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
    double loglik1, loglik0 = -neg_loglik_loss(x, y, weight, beta0, coef0);
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
      g = weight.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);

      if (this->approximate_Newton)
      {
        d = (x.transpose() * g).cwiseQuotient((x.transpose() * h * x).diagonal());
      }
      else
      {
        d = (x.transpose() * h * x).ldlt().solve(x.transpose() * g);
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

      loglik1 = -neg_loglik_loss(x, y, weight, beta1, coef0);

      while (loglik1 < loglik0 && step > this->primary_model_fit_epsilon)
      {
        step = step / 2;
        beta1 = beta0 + step * d;
        loglik1 = -neg_loglik_loss(x, y, weight, beta1, coef0);
      }

      bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - l - 1) * (loglik1 - loglik0)) + this->tau > loss0;
      if (condition1)
      {
        loss0 = -loglik0;
        beta = beta0;
        this->cox_hessian = h;
        this->cox_g = g;
        // cout << "condition1" << endl;
        return;
      }

      if (loglik1 > loglik0)
      {
        beta0 = beta1;
        loglik0 = loglik1;
        this->cox_hessian = h;
        this->cox_g = g;
        // cout << "condition1" << endl;
      }

      if (step < this->primary_model_fit_epsilon)
      {
        loss0 = -loglik0;
        beta = beta0;
        this->cox_hessian = h;
        this->cox_g = g;
        // cout << "condition2" << endl;
        return;
      }
    }
#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << l << endl;
#endif

    beta = beta0;
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
  {
    return -loglik_cox(X, y, beta, weights);
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd)
  {
#ifdef TEST
    clock_t t1 = clock(), t2;
#endif
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

    d = X.transpose() * g;

#ifdef TEST
    t2 = clock();
    std::cout << "d1 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    // Eigen::VectorXd d = X.transpose() * res;
    // Eigen::VectorXd h = weights.array() * pr.array() * (one - pr).array();

#ifdef TEST
    t2 = clock();
    std::cout << "d2 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);

    for (int i = 0; i < N; i++)
      for (int i = 0; i < N; i++)
      {
        T4 XG = X.middleCols(g_index(i), g_size(i));
        Eigen::MatrixXd XGbar = XG.transpose() * h * XG;

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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
  }
};

template <class T4>
class abessMLm : public Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>
{
public:
  abessMLm(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = true, int splicing_type = 0) : Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, covariance_update, splicing_type){};

  ~abessMLm(){};

  void primary_model_fit(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0)
  {
    // beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);

    if (X.cols() == 0)
    {
      // coef0 = y.colwise().sum();
      return;
    }
    // cout << "primary_fit 1" << endl;
    overload_ldlt(X, X, y, beta);
    // Eigen::MatrixXd XTX = X.transpose() * X;
    // beta = (XTX + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).ldlt().solve(X.transpose() * y);
    // cout << "primary_fit 2" << endl;

    // CG
    // ConjugateGradient<T4, Lower | Upper> cg;
    // cg.compute(X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols()));
    // beta = cg.solveWithGuess(X.adjoint() * y, beta);
  };

  double neg_loglik_loss(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0)
  {
    int n = X.rows();
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, y.cols());
    return (y - X * beta - array_product(one, coef0)).array().square().sum() / n / 2.0;
  }

  void covariance_update_f(T4 &X, Eigen::VectorXi &A_ind)
  {
    if (this->covariance.rows() == 0)
    {
      this->covariance = Eigen::MatrixXd::Zero(X.cols(), X.cols());
    }
    for (int i = 0; i < A_ind.size(); i++)
    {
      if (this->covariance_update_flag(A_ind(i)) == 0)
      {
        this->covariance.col(A_ind(i)) = X.transpose() * X.col(A_ind(i));
        this->covariance_update_flag(A_ind(i)) = 1;
      }
    }
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd)
  {
#ifdef TEST
    clock_t t1 = clock(), t2;
#endif
    int p = X.cols();
    int n = X.rows();
    int M = y.cols();

    Eigen::MatrixXd d;
    if (!this->covariance_update)
    {
      Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, y.cols());
      if (beta.size() != 0)
      {
        d = X.adjoint() * (y - XA * beta_A - array_product(one, coef0)) / double(n);
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
#ifdef TEST
        clock_t t1 = clock();
#endif

        this->covariance_update_f(X, A_ind);
#ifdef TEST
        clock_t t2 = clock();
        std::cout << "covariance_update_f: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
        std::cout << "this->covariance_update_flag sum: " << this->covariance_update_flag.sum() << endl;
        t1 = clock();
#endif

        Eigen::MatrixXd XTXbeta = X_seg(this->covariance, this->covariance.rows(), A_ind) * beta_A;
        d = (this->XTy - XTXbeta - array_product(this->XTone, coef0)) / double(n);

#ifdef TEST
        t2 = clock();
        std::cout << "X beta time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
        std::cout << "d 1" << endl;
#endif
      }
      else
      {
        Eigen::MatrixXd XTonecoef0 = array_product(this->XTone, coef0);
        d = (this->XTy - XTonecoef0) / double(n);
      }
    }

#ifdef TEST
    t2 = clock();
    std::cout << "d2 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    int A_size = A.size();
    int I_size = I.size();

    Eigen::MatrixXd betabar = Eigen::MatrixXd::Zero(p, M);
    Eigen::MatrixXd dbar = Eigen::MatrixXd::Zero(p, M);
    Eigen::MatrixXd phiG, invphiG;

    for (int i = 0; i < N; i++)
    {
      phiG = this->PhiG(i, 0);
      invphiG = this->invPhiG(i, 0);
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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
  }
};

template <class T4>
class abessMultinomial : public Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>
{
public:
  abessMultinomial(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = true, int splicing_type = 0) : Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, covariance_update, splicing_type){};

  ~abessMultinomial(){};

  void primary_model_fit(T4 &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0)
  {
#ifdef TEST
    clock_t t1 = clock();
#endif
    // cout << "primary_fit-----------" << endl;
    // if (X.cols() == 0)
    // {
    //   coef0 = -log(y.colwise().sum().eval() - 1.0);
    //   return;
    // }

#ifdef TEST
    std::cout << "primary_model_fit 1" << endl;
#endif
    int n = x.rows();
    int p = x.cols();
    int M = y.cols();
    T4 X(n, p + 1);
    X.rightCols(p) = x;
    add_constant_column(X);
    // Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
    // Eigen::MatrixXd X_new_transpose = Eigen::MatrixXd::Zero(p + 1, n);
    Eigen::MatrixXd beta0 = Eigen::MatrixXd::Zero(p + 1, M);

    Eigen::MatrixXd one_vec = Eigen::VectorXd::Ones(n);
    // #ifdef TEST
    //     std::cout << "primary_model_fit 2" << endl;
    // #endif
    beta0.row(0) = coef0;
    beta0.block(1, 0, p, M) = beta;
    // #ifdef TEST
    //     std::cout << "primary_model_fit 3" << endl;
    // #endif
    // Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::MatrixXd Pi;
    pi(X, y, beta0, Pi);
    Eigen::MatrixXd log_Pi = Pi.array().log();
    array_product(log_Pi, weights, 1);
    double loglik1 = DBL_MAX, loglik0 = (log_Pi.array() * y.array()).sum();
    // #ifdef TEST
    //     std::cout << "primary_model_fit 4" << endl;
    // #endif

    int j;
    if (this->approximate_Newton)
    {
      Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, M);
      double t = 2 * (Pi.array() * (one - Pi).array()).maxCoeff();
      Eigen::MatrixXd res = X.transpose() * (y - Pi) / t;
      // ConjugateGradient<MatrixXd, Lower | Upper> cg;
      // cg.compute(X.adjoint() * X);
      Eigen::MatrixXd XTX = X.transpose() * X;
      Eigen::MatrixXd invXTX = XTX.ldlt().solve(Eigen::MatrixXd::Identity(p + 1, p + 1));

      // cout << "y: " << y.rows() << " " << y.cols() << endl;
      // cout << "Pi: " << Pi.rows() << " " << Pi.cols() << endl;

      // cout << "Pi: " << Pi << endl;
      // cout << "t: " << t << endl;
      // cout << "invXTX: " << invXTX << endl;
      // cout << "one: " << invXTX * XTX << endl;

      Eigen::MatrixXd beta1;
      for (j = 0; j < this->primary_model_fit_max_iter; j++)
      {
        // #ifdef TEST
        //         std::cout << "primary_model_fit 3: " << j << endl;
        // #endif

        // beta1 = beta0 + cg.solve(res);
        beta1 = beta0 + invXTX * res;
        // cout << "beta1: " << beta1 << endl;

        // double app_loss0, app_loss1, app_loss2;
        // app_loss0 = ((y - Pi) / t).squaredNorm();
        // app_loss1 = (-X * beta0 - (y - Pi) / t).squaredNorm();
        // app_loss2 = (X * (beta1 - beta0) - (y - Pi) / t).squaredNorm();
        // cout << "app_loss0: " << app_loss0 << endl;
        // cout << "app_loss1: " << app_loss1 << endl;
        // cout << "app_loss2: " << app_loss2 << endl;

        pi(X, y, beta1, Pi);
        log_Pi = Pi.array().log();
        array_product(log_Pi, weights, 1);
        loglik1 = (log_Pi.array() * y.array()).sum();
        // cout << "loglik1: " << loglik1 << endl;
        // cout << "loglik0: " << loglik0 << endl;

        // cout << "j=" << j << " loglik: " << loglik1 << endl;
        // cout << "j=" << j << " loglik diff: " << loglik0 - loglik1 << endl;
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
#ifdef TEST
            std::cout << "primary_model_fit 5" << endl;

#endif
            Eigen::VectorXd PiPj = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array();
            // cout << "PiPj: " << PiPj << endl;
            for (int i = 0; i < PiPj.size(); i++)
            {
              if (PiPj(i) < 0.001)
              {
                PiPj(i) = 0.001;
              }
            }
            W.block(m1 * n, m2 * n, n, n).diagonal() = PiPj;

#ifdef TEST
            std::cout << "primary_model_fit 6" << endl;
            cout << "W m1 m2: " << W.block(m1 * n, m2 * n, n, n) << endl;
#endif
          }
          else
          {
            W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
#ifdef TEST
            std::cout << "primary_model_fit 5" << endl;

#endif
            Eigen::VectorXd PiPj = Pi.col(m1).array() * Pi.col(m2).array();
            // cout << "PiPj: " << PiPj << endl;
            for (int i = 0; i < PiPj.size(); i++)
            {
              if (PiPj(i) < 0.001)
              {
                PiPj(i) = 0.001;
              }
            }
            W.block(m1 * n, m2 * n, n, n).diagonal() = -PiPj;
            W.block(m2 * n, m1 * n, n, n) = W.block(m1 * n, m2 * n, n, n);

            // cout << "W m1 m2: " << W.block(m1 * n, m2 * n, n, n) << endl;
          }
        }
      }

#ifdef TEST
      std::cout << "primary_model_fit 7" << endl;
#endif
      // cout << "W: " << W << endl;

      Eigen::MatrixXd XTWX(M * (p + 1), M * (p + 1));
      Eigen::MatrixXd XTW(M * (p + 1), M * n);
      for (int m1 = 0; m1 < M; m1++)
      {
        for (int m2 = m1; m2 < M; m2++)
        {
          XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) = X.transpose() * W.block(m1 * n, m2 * n, n, n);
          XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X;
          XTW.block(m2 * (p + 1), m1 * n, (p + 1), n) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n);
          XTWX.block(m2 * (p + 1), m1 * (p + 1), (p + 1), (p + 1)) = XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1));
        }
      }

#ifdef TEST
      std::cout << "primary_model_fit 8" << endl;
#endif

      // Eigen::Matrix<Eigen::MatrixXd, -1, -1> res(M, 1);
      Eigen::VectorXd res(M * n);
      for (int m1 = 0; m1 < M; m1++)
      {
        res.segment(m1 * n, n) = y.col(m1).eval() - Pi.col(m1).eval();
      }

#ifdef TEST
      std::cout << "primary_model_fit 9" << endl;
      cout << "res: " << res << endl;
#endif

      Eigen::VectorXd Xbeta(M * n);
      for (int m1 = 0; m1 < M; m1++)
      {
        Xbeta.segment(m1 * n, n) = X * beta0.col(m1).eval();
      }

#ifdef TEST
      std::cout << "primary_model_fit 10" << endl;
      cout << "Xbeta: " << Xbeta << endl;
#endif

      Eigen::VectorXd Z = Xbeta + W.ldlt().solve(res);
#ifdef TEST
      std::cout << "primary_model_fit 11" << endl;
#endif

#ifdef TEST
      std::cout << "primary_model_fit 2" << endl;
#endif

      Eigen::MatrixXd beta1;
      Eigen::VectorXd beta0_tmp;
      for (j = 0; j < this->primary_model_fit_max_iter; j++)
      {
#ifdef TEST
        std::cout << "primary_model_fit 3: " << j << endl;
#endif
        beta0_tmp = XTWX.ldlt().solve(XTW * Z);
        for (int m1 = 0; m1 < M; m1++)
        {
          beta0.col(m1) = beta0_tmp.segment(m1 * (p + 1), (p + 1)) - beta0_tmp.segment((M - 1) * (p + 1), (p + 1));
        }
        for (int m1 = 0; m1 < M; m1++)
        {
          beta0.col(m1) = beta0_tmp.segment(m1 * (p + 1), (p + 1));
        }
        // cout << "beta0" << beta0 << endl;

        pi(X, y, beta0, Pi);
        log_Pi = Pi.array().log();
        array_product(log_Pi, weights, 1);
        loglik1 = (log_Pi.array() * y.array()).sum();
        // cout << "loss" << loglik1 << endl;
        // cout << "j=" << j << " loglik: " << loglik1 << endl;
        // cout << "j=" << j << " loglik diff: " << loglik0 - loglik1 << endl;
        bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + this->tau > loss0;
        // bool condition1 = false;
        bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < this->primary_model_fit_epsilon;
        bool condition3 = abs(loglik1) < min(1e-3, this->tau);
        bool condition4 = loglik1 < loglik0;
        if (condition1 || condition2 || condition3 || condition4)
        {
          // cout << "condition1:" << condition1 << endl;
          // cout << "condition2:" << condition2 << endl;
          // cout << "condition3:" << condition3 << endl;
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
            XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X;
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

#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << j << endl;
#endif

    beta = beta0.block(1, 0, p, M);
    coef0 = beta0.row(0).eval();
  };

  double neg_loglik_loss(T4 &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0)
  {
    // weight
    Eigen::MatrixXd pr;
    pi(X, y, beta, coef0, pr);
    Eigen::MatrixXd log_pr = pr.array().log();
    // Eigen::VectorXd one_vec = Eigen::VectorXd::Ones(X.rows());
    // cout << "loss 0" << endl;
    array_product(log_pr, weights, 1);
    // cout << "loss 1" << endl;
    return -((log_pr.array() * y.array()).sum());
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd)
  {
#ifdef TEST
    clock_t t1 = clock(), t2;
#endif
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
    d = X.transpose() * res;
    h = Pi;

#ifdef TEST
    t2 = clock();
    std::cout << "d2 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

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
      // cout << "h: " << h << endl;
      XGbar.diagonal() = Eigen::VectorXd(XG_new.transpose() * XG) + XGbar.diagonal();

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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
  }
};

template <class T4>
class abessSPCA : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
  public:
  abessSPCA(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 1) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, splicing_type){};
  
  ~abessSPCA(){};

  void primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0)
  {
#ifdef TEST
    cout << "<< SPCA primary_model_fit >>"<< endl;
#endif
    int p = x.cols();
    if (p == 0) return;
    if (p == 1){
      beta << 1;
      return;
    } 
    
    MatrixXd Y = x.transpose() * x ;
    DenseSymMatProd<double> op(Y);
    SymEigsSolver< double, LARGEST_ALGE, DenseSymMatProd<double> > eig(&op, 1, 2);
    eig.init();
    eig.compute();
    MatrixXd temp;
    if (eig.info() == SUCCESSFUL)
    {
      temp = eig.eigenvectors(1);
    }
    
    beta = temp.col(0);
    return;
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
  {
#ifdef TEST
    cout << "<< SPCA Loss >>" <<endl;
#endif
    return - ( X * beta ).squaredNorm();  // lower, better
  };

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd)
  {
#ifdef TEST
    cout << "<< SPCA sacrifice >>"<< endl;
#endif
    // // 2021/05/27 -- nogroup -- [n]
    // MatrixXd Sigma = - X.transpose() * X;

    // for (int i = 0; i < A.size(); i++){
    //   double h = 0;
    //   h = 2 * Sigma(A(i), A(i));
    //   bd(A(i)) = beta(A(i)) * beta(A(i)) * h / 2;
    // }
    // for (int i = 0; i < I.size(); i++){
    //   double g = 0, h = 0;
    //   g = 2 * Sigma.row(I(i)) * beta;
    //   h = 2 * Sigma(I(i), I(i));
    //   bd(I(i)) = g * g / h / 2;  // gamma * h * gamma / 2 ; (gamma = h\g) 
    // }

    // // 2021/05/27 -- nogroup -- [y]
    // VectorXd D = - X.transpose() * X * beta + (X * beta).squaredNorm() * beta;
    // for (int i = 0; i < A.size(); i++){
    //   bd(A(i)) = abs(beta(A(i)));
    // }
    // for (int i = 0; i < I.size(); i++){
    //   bd(I(i)) = abs(D(I(i)));
    // }

    // 2021/05/31 -- group 
    VectorXd D = - X.transpose() * X * beta + (X * beta).squaredNorm() * beta;
    for (int i = 0; i < A.size(); i++){
      VectorXd temp = beta.segment(g_index(A(i)), g_size(A(i)));
      bd(A(i)) = temp.squaredNorm();
    }
    for (int i = 0; i < I.size(); i++){
      VectorXd temp = D.segment(g_index(I(i)), g_size(I(i)));
      bd(I(i)) = temp.squaredNorm();
  }

#ifdef TEST
    cout << "  --> A : " << endl << A << endl;
    cout << "  --> I : " << endl << I << endl;
    cout << "  --> bd : " << endl << bd << endl;
#endif
  };
};

#endif //SRC_ALGORITHM_H