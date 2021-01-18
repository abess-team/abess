//
// Created by jk on 2020/3/18.
//
// #define TEST

#ifndef SRC_ALGORITHM_H
#define SRC_ALGORITHM_H

#ifndef R_BUILD
#include <unsupported/Eigen/MatrixFunctions>
#endif

#include "Data.h"
#include "utilities.h"
#include "logistic.h"
#include "poisson.h"
#include "coxph.h"
#include <iostream>

#include <time.h>
#include <cfloat>

using namespace std;

bool quick_sort_pair_max(std::pair<int, double> x, std::pair<int, double> y);

class Algorithm
{
public:
  // Data data;
  vector<Eigen::MatrixXd> PhiG;
  vector<Eigen::MatrixXd> invPhiG;
  Eigen::VectorXd beta_init;
  int group_df = 0;
  int sparsity_level = 0;
  double lambda_level = 0;
  Eigen::VectorXi train_mask;
  int max_iter;
  int exchange_num;
  bool warm_start;
  Eigen::VectorXd beta;
  double coef0_init;
  Eigen::VectorXi A_init;
  Eigen::VectorXi I_init;
  Eigen::VectorXd bd_init;
  Eigen::VectorXd bd;
  double coef0 = 0.;
  double train_loss = 0.;
  Eigen::VectorXi A_out;
  Eigen::VectorXi I_out;
  int l;
  int model_fit_max;
  int model_type;
  int algorithm_type;
  std::vector<Eigen::MatrixXd> group_XTX;
  Eigen::VectorXi always_select;
  double tau;
  int primary_model_fit_max_iter;
  double primary_model_fit_epsilon;
  bool approximate_Newton;
  Eigen::VectorXd beta_warmstart;
  double coef0_warmstart;

  Algorithm() = default;

  virtual ~Algorithm(){};

  Algorithm(int algorithm_type, int model_type, int max_iter = 100, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8)
  {
    // this->data = data;
    this->max_iter = max_iter;
    // this->A_out = Eigen::VectorXi::Zero(data.get_p());
    this->model_type = model_type;
    // this->coef0 = 0.0;
    // this->beta = Eigen::VectorXd::Zero(data.get_p());
    this->coef0_init = 0.0;
    // this->beta_init = Eigen::VectorXd::Zero(data.get_p());
    this->warm_start = true;
    this->exchange_num = 5;
    this->algorithm_type = algorithm_type;
    this->primary_model_fit_max_iter = primary_model_fit_max_iter;
    this->primary_model_fit_epsilon = primary_model_fit_epsilon;
  };

  void update_PhiG(vector<Eigen::MatrixXd> &PhiG) { this->PhiG = PhiG; }

  void update_invPhiG(vector<Eigen::MatrixXd> &invPhiG) { this->invPhiG = invPhiG; }

  void set_warm_start(bool warm_start) { this->warm_start = warm_start; }

  void update_beta_init(Eigen::VectorXd &beta_init) { this->beta_init = beta_init; }

  void update_A_init(Eigen::VectorXi &A_init, int g_num)
  {
    this->A_init = A_init;
    this->I_init = Ac(A_init, g_num);
  }

  // void update_I_init(Eigen::VectorXi I_init) { this->I_init = I_init; }

  void update_bd_init(Eigen::VectorXd &bd_init) { this->bd_init = bd_init; }

  void update_coef0_init(double coef0) { this->coef0_init = coef0; }

  void update_group_df(int group_df) { this->group_df = group_df; }

  void update_sparsity_level(int sparsity_level) { this->sparsity_level = sparsity_level; }

  void update_lambda_level(double lambda_level) { this->lambda_level = lambda_level; }

  void update_train_mask(Eigen::VectorXi &train_mask) { this->train_mask = train_mask; }

  void update_exchange_num(int exchange_num) { this->exchange_num = exchange_num; }

  void update_group_XTX(std::vector<Eigen::MatrixXd> &group_XTX)
  {
    this->group_XTX = group_XTX;
  }

  bool get_warm_start() { return this->warm_start; }

  double get_train_loss() { return this->train_loss; }

  int get_group_df() { return this->group_df; }

  int get_sparsity_level() { return this->sparsity_level; }

  Eigen::VectorXd get_beta() { return this->beta; }

  double get_coef0() { return this->coef0; }

  Eigen::VectorXi get_A_out() { return this->A_out; };

  Eigen::VectorXi get_I_out() { return this->I_out; };

  Eigen::VectorXd get_bd() { return this->bd; }

  int get_l() { return this->l; }

  void fit(Eigen::MatrixXd &train_x, Eigen::VectorXd &train_y, Eigen::VectorXd &train_weight, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int train_n, int p, int N)
  {
    // std::cout << "fit" << endl;
    int T0 = this->sparsity_level;

    this->tau = 0.01 * (double)this->sparsity_level * log((double)N) * log(log((double)train_n)) / (double)train_n;

    // std::cout << "fit 1" << endl;
    if (N == T0)
    {
      this->primary_model_fit(train_x, train_y, train_weight, this->beta, this->coef0, DBL_MAX);
      this->A_out = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      return;
    }

    this->beta = this->beta_init;
    this->coef0 = this->coef0_init;
    this->bd = this->bd_init;

#ifdef TEST
    clock_t t1, t2;
    t1 = clock();
#endif

    if (this->model_type == 1)
    {
      if (this->algorithm_type == 7 || this->PhiG.size() == 0)
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

    Eigen::MatrixXd X_A;
    Eigen::VectorXd beta_A;
    Eigen::VectorXi A_ind;

    // std::cout << "fit 5" << endl;
#ifdef TEST
    t1 = clock();
#endif
    if (this->algorithm_type == 6)
    {
      A_ind = find_ind(A, g_index, g_size, p, N);
      X_A = X_seg(train_x, train_n, A_ind);
      beta_A = Eigen::VectorXd::Zero(A_ind.size());
      for (int k = 0; k < A_ind.size(); k++)
      {
        beta_A(k) = this->beta(A_ind(k));
      }
      this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0, DBL_MAX);
      this->beta = Eigen::VectorXd::Zero(p);
      for (int k = 0; k < A_ind.size(); k++)
      {
        this->beta(A_ind(k)) = beta_A(k);
      }
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
      // std::cout << "fit 7" << endl;
      // t1 = clock();
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
        beta_A = Eigen::VectorXd::Zero(A_ind.size());
        this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0, DBL_MAX);
        this->beta = Eigen::VectorXd::Zero(p);
        for (int mm = 0; mm < A_ind.size(); mm++)
        {
          this->beta(A_ind(mm)) = beta_A(mm);
        }
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
          // this->I_out = I;
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
    // std::cout << "------------iter time: ----------" << this->l << endl;
  };

  virtual void primary_model_fit(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0) = 0;

  virtual void get_A(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &bd, int T0, Eigen::VectorXd &weights,
                     Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau, double &train_loss) = 0;

  virtual Eigen::VectorXi inital_screening(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &bd, Eigen::VectorXd &weights,
                                           Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N) = 0;

  virtual double neg_loglik_loss(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0) = 0;
  virtual Eigen::VectorXd dual(Eigen::MatrixXd &XI, Eigen::MatrixXd &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weights, int n, Eigen::VectorXd &h) = 0;
};

class abessLogistic : public Algorithm
{
public:
  abessLogistic(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8) : Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon){};

  ~abessLogistic(){};

  Eigen::VectorXi inital_screening(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &bd, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {
    // cout << "inital_screening: " << endl;
#ifdef TEST
    clock_t t3, t4;
    t3 = clock();
#endif
    if (bd.size() == 0)
    {
#ifdef TEST
      clock_t t1, t2;
      t1 = clock();
#endif
      // variable initialization
      int n = X.rows();
      int p = X.cols();
      Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
      bd = Eigen::VectorXd::Zero(N);

      // calculate beta & d & h
      // cout << "inital_screening 1" << endl;
      Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
      Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
      Eigen::VectorXd beta_A(A_ind.size());
      for (int k = 0; k < A_ind.size(); k++)
      {
        beta_A(k) = beta(A_ind(k));
      }

#ifdef TEST
      t2 = clock();
      std::cout << "inital_screening beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
      cout << "2" << endl;
#endif
      Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd h;

      Eigen::VectorXi I_ind = find_ind(I, g_index, g_size, p, N);
      Eigen::MatrixXd X_I = X_seg(X, n, I_ind);
      Eigen::VectorXd d_I = this->dual(X_I, X_A, y, beta_A, coef0, weights, n, h);

      // cout << "d_I: " << d_I << endl;
      for (int k = 0; k < I_ind.size(); k++)
      {
        d(I_ind(k)) = d_I(k);
      }

      // calculate group bd
#ifdef TEST
      t1 = clock();
#endif
      for (int i = 0; i < N; i++)
      {
        Eigen::MatrixXd XG = X.middleCols(g_index(i), g_size(i));
        Eigen::MatrixXd XG_new = XG;
        for (int j = 0; j < g_size(i); j++)
        {
          XG_new.col(j) = XG.col(j).cwiseProduct(h);
        }
        Eigen::MatrixXd XGbar = XG_new.transpose() * XG;
        Eigen::MatrixXd phiG;
        XGbar.sqrt().evalTo(phiG);
        Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
        // cout << "phiG: " << phiG << endl;
        // cout << "invphiG: " << invphiG << endl;
        betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
        dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
      }
      Eigen::VectorXd temp = betabar + dbar;
      for (int i = 0; i < N; i++)
      {
        bd(i) = (temp.segment(g_index(i), g_size(i))).squaredNorm() / g_size(i);
      }
    }

#ifdef TEST
    t4 = clock();
    std::cout << "inital_screening bd: " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
    t3 = clock();
#endif

    // get Active-set A according to max_k bd

    Eigen::VectorXi A_new = max_k(bd, this->get_sparsity_level());
#ifdef TEST
    t4 = clock();
    std::cout << "inital_screening max_k: " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
#endif
    return A_new;
  }

  void primary_model_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0)
  {
#ifdef TEST
    clock_t t1 = clock();
#endif
    // cout << "primary_fit-----------" << endl;
    int n = x.rows();
    int p = x.cols();
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
    X.rightCols(p) = x;
    Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
    Eigen::MatrixXd X_new_transpose = Eigen::MatrixXd::Zero(p + 1, n);
    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
    beta0(0) = coef0;
    beta0.tail(p) = beta;
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd Pi = pi(X, y, beta0, n);
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
        Pi = pi(X, y, beta1, n);
        log_Pi = Pi.array().log();
        log_1_Pi = (one - Pi).array().log();
        loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);

        while (loglik1 < loglik0 && step > this->primary_model_fit_epsilon)
        {
          step = step / 2;
          beta1 = beta0 + step * g.cwiseProduct(h_diag);
          Pi = pi(X, y, beta1, n);
          log_Pi = Pi.array().log();
          log_1_Pi = (one - Pi).array().log();
          loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
        }

        // cout << "j=" << j << " loglik: " << loglik1 << endl;
        // cout << "j=" << j << " loglik diff: " << loglik1 - loglik0 << endl;
        bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
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
        X_new_transpose = X_new.transpose();

        beta0 = (X_new_transpose * X).ldlt().solve(X_new_transpose * Z);

        // CG
        // ConjugateGradient<MatrixXd, Lower | Upper> cg;
        // cg.compute(X_new_transpose * X);
        // beta0 = cg.solveWithGuess(X_new_transpose * Z, beta0);

        Pi = pi(X, y, beta0, n);
        log_Pi = Pi.array().log();
        log_1_Pi = (one - Pi).array().log();
        loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
        // cout << "j=" << j << " loglik: " << loglik1 << endl;
        // cout << "j=" << j << " loglik diff: " << loglik0 - loglik1 << endl;
        bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
        // bool condition1 = false;
        bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < this->primary_model_fit_epsilon;
        bool condition3 = abs(loglik1) < min(1e-3, tau);
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
    }
#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << j << endl;
#endif
    beta = beta0.tail(p).eval();
    coef0 = beta0(0);
  };

  double neg_loglik_loss(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
  {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Ones(p + 1);
    coef(0) = coef0;
    coef.tail(p) = beta;
    return -loglik_logit(X, y, coef, n, weights);
  }

  Eigen::VectorXd dual(Eigen::MatrixXd &XI, Eigen::MatrixXd &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weights, int n, Eigen::VectorXd &h)
  {
    int p = XA.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Ones(p + 1);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    coef(0) = coef0;
    coef.tail(p) = beta;
    // for (int i = 0; i < XA.cols(); i++)
    //   coef(i + 1) = beta(i);

    Eigen::VectorXd pr = pi(XA, y, coef, n);
    // cout << "pr: " << pr;
    Eigen::VectorXd res = (y - pr).cwiseProduct(weights);
    Eigen::VectorXd d_I = XI.transpose() * res;

    h = weights.array() * pr.array() * (one - pr).array();

    return d_I;
  }

  void get_A(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &bd, int T0, Eigen::VectorXd &weights,
             Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau, double &train_loss)
  {
    // cout << "get A 1" << endl;
    int n = X.rows();
    int p = X.cols();
#ifdef TEST
    clock_t t0, t1, t2;
    t1 = clock();
#endif
    Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
    Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
    Eigen::VectorXd beta_A(A_ind.size());
    for (int k = 0; k < A_ind.size(); k++)
    {
      beta_A(k) = beta(A_ind(k));
    }
#ifdef TEST
    t2 = clock();
    std::cout << "A ind time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    double L1, L0 = neg_loglik_loss(X_A, y, weights, beta_A, coef0);
    train_loss = L0;
#ifdef TEST
    t2 = clock();
    std::cout << "loss time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd h;
    Eigen::VectorXi I_ind = find_ind(I, g_index, g_size, p, N);
    Eigen::MatrixXd X_I = X_seg(X, n, I_ind);
    Eigen::VectorXd d_I = this->dual(X_I, X_A, y, beta_A, coef0, weights, n, h);
    for (int k = 0; k < I_ind.size(); k++)
    {
      d(I_ind(k)) = d_I(k);
    }

    Eigen::VectorXd beta_A_group = Eigen::VectorXd::Zero(A_size);
    Eigen::VectorXd d_I_group = Eigen::VectorXd::Zero(I_size);
    for (int i = 0; i < N; i++)
    {
      Eigen::MatrixXd XG = X.middleCols(g_index(i), g_size(i));
      Eigen::MatrixXd XG_new = XG;
      for (int j = 0; j < g_size(i); j++)
      {
        XG_new.col(j) = XG.col(j).cwiseProduct(h);
      }
      Eigen::MatrixXd XGbar;
      XGbar.noalias() = XG_new.transpose() * XG;
      Eigen::MatrixXd phiG;
      XGbar.sqrt().evalTo(phiG);
      Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
      betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
      dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
    }
    for (int i = 0; i < A_size; i++)
    {
      beta_A_group(i) = betabar.segment(g_index(A[i]), g_size(A[i])).squaredNorm() / g_size(A[i]);
      bd(A[i]) = beta_A_group(i);
    }
    for (int i = 0; i < I_size; i++)
    {
      d_I_group(i) = dbar.segment(g_index(I[i]), g_size(I[i])).squaredNorm() / g_size(I[i]);
      bd(I[i]) = d_I_group(i);
    }
    // Eigen::VectorXd temp = betabar + dbar;
    // for (int i = 0; i < N; i++)
    // {
    //   bd(i) = (temp.segment(g_index(i), g_size(i))).squaredNorm() / g_size(i);
    // }
    // cout << "get A 4" << endl;
#ifdef TEST
    t2 = clock();
    std::cout << "get A beta d: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    // std::cout << "A: " << A << endl;
    // std::cout << "I: " << I << endl;
    // std::cout << "beta_A_group: " << beta_A_group << endl;
    // std::cout << "d_I_group: " << d_I_group << endl;

    Eigen::VectorXi A_min_k = min_k(beta_A_group, C_max);
    Eigen::VectorXi I_max_k = max_k(d_I_group, C_max);
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
    Eigen::MatrixXd X_A_exchage;
    Eigen::VectorXd beta_A_exchange;
    double coef0_A_exchange;

    // Eigen::VectorXd beta_Ac = beta_A;
    // double coef0_Ac = coef0;
    for (int k = C_max; k >= 1;)
    {
      // std::cout << "s1: " << s1 << endl;
      // std::cout << "s2: " << s2 << endl;
      // t1 = clock();
      A_exchange = diff_union(A, s1, s2);
      // cout << "get A 6" << endl;
      // std::cout << "A_exchange: " << A_exchange << endl;
      A_ind_exchage = find_ind(A_exchange, g_index, g_size, p, N);
      X_A_exchage = X_seg(X, n, A_ind_exchage);
      beta_A_exchange = Eigen::VectorXd::Zero(A_ind_exchage.size());
      for (int i = 0; i < A_ind_exchage.size(); i++)
      {
        beta_A_exchange(i) = this->beta_warmstart(A_ind_exchage(i));
      }
      coef0_A_exchange = this->coef0_warmstart;

      primary_model_fit(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange, L0);

      L1 = neg_loglik_loss(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange);

      // cout << "L0: " << L0 << " L1: " << L1 << endl;
      if (L0 - L1 > tau)
      {
        // update A & I & beta & coef0
        train_loss = L1;
        A = A_exchange;
        I = Ac(A_exchange, N);
        beta = Eigen::VectorXd::Zero(p);
        for (int i = 0; i < A_ind_exchage.size(); i++)
        {
          beta(A_ind_exchage[i]) = beta_A_exchange(i);
        }
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
        k = k / 2;
        s1 = s1.head(k).eval();
        s2 = s2.head(k).eval();
      }
    }
#ifdef TEST
    t2 = clock();
    std::cout << "splicing time: " << ((double)(t2 - t0) / CLOCKS_PER_SEC) << endl;
#endif
  };
};

class abessLm : public Algorithm
{
public:
  abessLm(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8) : Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon){};

  ~abessLm(){};

  Eigen::VectorXi inital_screening(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &bd, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {
    // cout << "inital_screening: " << endl;
#ifdef TEST
    clock_t t3, t4;
    t3 = clock();
#endif

    if (bd.size() == 0)
    {
#ifdef TEST
      clock_t t1, t2;
      t1 = clock();
#endif
      // variable initialization
      int n = X.rows();
      int p = X.cols();
      Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
      bd = Eigen::VectorXd::Zero(N);

      // calculate beta & d & h
      // cout << "inital_screening 1" << endl;
      Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
      Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
      Eigen::VectorXd beta_A(A_ind.size());
      for (int k = 0; k < A_ind.size(); k++)
      {
        beta_A(k) = beta(A_ind(k));
      }

#ifdef TEST
      t2 = clock();
      std::cout << "inital_screening beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
      cout << "2" << endl;
#endif

      Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd h;

      Eigen::VectorXi I_ind = find_ind(I, g_index, g_size, p, N);
      Eigen::MatrixXd X_I = X_seg(X, n, I_ind);
      Eigen::VectorXd d_I = this->dual(X_I, X_A, y, beta_A, coef0, weights, n, h);

      for (int k = 0; k < I_ind.size(); k++)
      {
        d(I_ind(k)) = d_I(k);
      }

      // calculate group bd
#ifdef TEST
      t1 = clock();
#endif
      for (int i = 0; i < N; i++)
      {
        Eigen::MatrixXd phiG = PhiG[i];
        Eigen::MatrixXd invphiG = invPhiG[i];
        betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
        dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
      }
      Eigen::VectorXd temp = betabar + dbar;
      for (int i = 0; i < N; i++)
      {
        bd(i) = (temp.segment(g_index(i), g_size(i))).squaredNorm() / g_size(i);
      }
    }

#ifdef TEST
    t4 = clock();
    std::cout << "inital_screening bd: " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
    t3 = clock();
#endif

    // cout << "g_index: " << g_index << endl;
    // cout << "g_size: " << g_size << endl;
    // cout << "betabar: " << betabar << endl;
    // cout << "dbar: " << dbar << endl;

    // get Active-set A according to max_k bd

    Eigen::VectorXi A_new = max_k_2(bd, this->get_sparsity_level());

#ifdef TEST
    t4 = clock();
    std::cout << "inital_screening max_k: " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
#endif
    return A_new;
  }

  void primary_model_fit(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0)
  {
    // beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);

    // beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).ldlt().solve(X.adjoint() * y);

    // CG
    ConjugateGradient<MatrixXd, Lower | Upper> cg;
    cg.compute(X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols()));
    beta = cg.solveWithGuess(X.adjoint() * y, beta);
  };

  double neg_loglik_loss(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
  {
    int n = X.rows();
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    return (y - X * beta - coef0 * one).array().square().sum() / n;
  }

  // double neg_loglik_loss(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
  // {
  //   int n = X.rows();
  //   Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
  //   return n * log((y - X * beta - coef0 * one).array().square().sum() / n) / 2.0;
  // }

  Eigen::VectorXd dual(Eigen::MatrixXd &XI, Eigen::MatrixXd &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weights, int n, Eigen::VectorXd &h)
  {
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd d_I;
    if (XI.cols() != 0)
    {
      if (beta.size() != 0)
      {
        d_I = XI.adjoint() * (y - XA * beta - coef0 * one) / double(n);
      }
      else
      {
        d_I = XI.adjoint() * (y - coef0 * one) / double(n);
      }
    }

    // h = weights.array() * pr.array() * (one - pr).array();

    return d_I;
  }

  void get_A(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &bd, int T0, Eigen::VectorXd &weights,
             Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau, double &train_loss)
  {
    // cout << "get A 1" << endl;
    int n = X.rows();
    int p = X.cols();
#ifdef TEST
    clock_t t0, t1, t2;
    t1 = clock();
#endif
    Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
    Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
    Eigen::VectorXd beta_A(A_ind.size());
    for (int k = 0; k < A_ind.size(); k++)
    {
      beta_A(k) = beta(A_ind(k));
    }
#ifdef TEST
    t2 = clock();
    std::cout << "A ind time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    double L1, L0 = neg_loglik_loss(X_A, y, weights, beta_A, coef0);
    train_loss = L0;
#ifdef TEST
    t2 = clock();
    std::cout << "loss time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd h;
    Eigen::VectorXi I_ind = find_ind(I, g_index, g_size, p, N);
    Eigen::MatrixXd X_I = X_seg(X, n, I_ind);
    Eigen::VectorXd d_I = this->dual(X_I, X_A, y, beta_A, coef0, weights, n, h);
    for (int k = 0; k < I_ind.size(); k++)
    {
      d(I_ind(k)) = d_I(k);
    }

    Eigen::VectorXd beta_A_group = Eigen::VectorXd::Zero(A_size);
    Eigen::VectorXd d_I_group = Eigen::VectorXd::Zero(I_size);
    Eigen::MatrixXd phiG, invphiG;
    for (int i = 0; i < N; i++)
    {
      phiG = PhiG[i];
      invphiG = invPhiG[i];
      betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
      dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
    }
    for (int i = 0; i < A_size; i++)
    {
      beta_A_group(i) = betabar.segment(g_index(A[i]), g_size(A[i])).squaredNorm() / g_size(A[i]);
      bd(A[i]) = beta_A_group(i);
    }
    for (int i = 0; i < I_size; i++)
    {
      d_I_group(i) = dbar.segment(g_index(I[i]), g_size(I[i])).squaredNorm() / g_size(I[i]);
      bd(I[i]) = d_I_group(i);
    }
    // Eigen::VectorXd temp = betabar + dbar;
    // for (int i = 0; i < N; i++)
    // {
    //   bd(i) = (temp.segment(g_index(i), g_size(i))).squaredNorm() / g_size(i);
    // }
    // cout << "get A 4" << endl;
#ifdef TEST
    t2 = clock();
    std::cout << "get A beta d: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif
    // std::cout << "A: " << A << endl;
    // std::cout << "I: " << I << endl;
    // std::cout << "beta_A_group: " << beta_A_group << endl;
    // std::cout << "d_I_group: " << d_I_group << endl;

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
    Eigen::MatrixXd X_A_exchage;
    Eigen::VectorXd beta_A_exchange;
    double coef0_A_exchange;

    // Eigen::VectorXd beta_Ac = beta_A;
    // double coef0_Ac = coef0;
    for (int k = C_max; k >= 1;)
    {
      // std::cout << "s1: " << s1 << endl;
      // std::cout << "s2: " << s2 << endl;
      // t1 = clock();
      A_exchange = diff_union(A, s1, s2);
      // cout << "get A 6" << endl;
      // std::cout << "A_exchange: " << A_exchange << endl;
      A_ind_exchage = find_ind(A_exchange, g_index, g_size, p, N);
      X_A_exchage = X_seg(X, n, A_ind_exchage);
      beta_A_exchange = Eigen::VectorXd::Zero(A_ind_exchage.size());
      for (int i = 0; i < A_ind_exchage.size(); i++)
      {
        beta_A_exchange(i) = this->beta_warmstart(A_ind_exchage(i));
      }
      coef0_A_exchange = this->coef0_warmstart;

      primary_model_fit(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange, L0);

      L1 = neg_loglik_loss(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange);

      // cout << "L0: " << L0 << " L1: " << L1 << endl;
      if (L0 - L1 > tau)
      {
        // update A & I & beta & coef0
        train_loss = L1;
        A = A_exchange;
        I = Ac(A_exchange, N);
        beta = Eigen::VectorXd::Zero(p);
        for (int i = 0; i < A_ind_exchage.size(); i++)
        {
          beta(A_ind_exchage[i]) = beta_A_exchange(i);
        }
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
        k = k / 2;
        s1 = s1.head(k).eval();
        s2 = s2.head(k).eval();
      }
    }
#ifdef TEST
    t2 = clock();
    std::cout << "splicing time: " << ((double)(t2 - t0) / CLOCKS_PER_SEC) << endl;
#endif
  };
};

class abessCox : public Algorithm
{
public:
  abessCox(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 30, double primary_model_fit_epsilon = 1e-8) : Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon){};

  ~abessCox(){};

  Eigen::VectorXi inital_screening(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &bd, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {
    // cout << "inital_screening: " << endl;
#ifdef TEST
    clock_t t3, t4;
    t3 = clock();
#endif
    if (bd.size() == 0)
    {
#ifdef TEST
      clock_t t1, t2;
      t1 = clock();
#endif
      // variable initialization
      int n = X.rows();
      int p = X.cols();
      Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
      bd = Eigen::VectorXd::Zero(N);

      // calculate beta & d & h
      // cout << "inital_screening 1" << endl;
      Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
      Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
      Eigen::VectorXd beta_A(A_ind.size());
      for (int k = 0; k < A_ind.size(); k++)
      {
        beta_A(k) = beta(A_ind(k));
      }

#ifdef TEST
      t2 = clock();
      std::cout << "inital_screening beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
      cout << "2" << endl;
#endif
      Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd h;

      Eigen::VectorXi I_ind = find_ind(I, g_index, g_size, p, N);
      Eigen::MatrixXd X_I = X_seg(X, n, I_ind);
      Eigen::VectorXd d_I = this->dual(X_I, X_A, y, beta_A, coef0, weights, n, h);

      // cout << "d_I: " << d_I << endl;
      for (int k = 0; k < I_ind.size(); k++)
      {
        d(I_ind(k)) = d_I(k);
      }

      // calculate group bd
#ifdef TEST
      t1 = clock();
#endif
      for (int i = 0; i < N; i++)
      {
        Eigen::MatrixXd XG = X.middleCols(g_index(i), g_size(i));
        Eigen::MatrixXd XG_new = XG;
        for (int j = 0; j < g_size(i); j++)
        {
          XG_new.col(j) = XG.col(j).cwiseProduct(h);
        }
        Eigen::MatrixXd XGbar = XG_new.transpose() * XG;
        Eigen::MatrixXd phiG;
        XGbar.sqrt().evalTo(phiG);
        Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
        // cout << "phiG: " << phiG << endl;
        // cout << "invphiG: " << invphiG << endl;
        betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
        dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
      }
      Eigen::VectorXd temp = betabar + dbar;
      for (int i = 0; i < N; i++)
      {
        bd(i) = (temp.segment(g_index(i), g_size(i))).squaredNorm() / g_size(i);
      }
    }

#ifdef TEST
    t4 = clock();
    std::cout << "inital_screening bd: " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
    t3 = clock();
#endif

    // cout << "g_index: " << g_index << endl;
    // cout << "g_size: " << g_size << endl;
    // cout << "betabar: " << betabar << endl;
    // cout << "dbar: " << dbar << endl;

    // get Active-set A according to max_k bd

    Eigen::VectorXi A_new = max_k(bd, this->get_sparsity_level());
#ifdef TEST
    t4 = clock();
    std::cout << "inital_screening max_k: " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
#endif

    // cout << "bd: " << bd << endl;
    // cout << "A_new: " << A_new << endl;
    return A_new;
  }

  void primary_model_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0)
  {
#ifdef TEST
    clock_t t1 = clock();
#endif
    // cout << "primary_fit-----------" << endl;
    int n = x.rows();
    int p = x.cols();
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
    X.rightCols(p) = x;
    Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
    Eigen::MatrixXd X_new_transpose = Eigen::MatrixXd::Zero(p + 1, n);
    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
    beta0(0) = coef0;
    beta0.tail(p) = beta;
    // cout << "beta: " << beta << endl;
    // cout << "beta0: " << beta0 << endl;
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd Pi = pi(X, y, beta0, n);
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

        // beta1 = beta0 + step * g.cwiseQuotient(h_diag);
        beta1 = beta0 + step * g.cwiseProduct(h_diag);
        Pi = pi(X, y, beta1, n);
        log_Pi = Pi.array().log();
        log_1_Pi = (one - Pi).array().log();
        loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);

        while (loglik1 < loglik0 && step > this->primary_model_fit_epsilon)
        {
          step = step / 2;
          // beta1 = beta0 + step * g.cwiseQuotient(h_diag);
          beta1 = beta0 + step * g.cwiseProduct(h_diag);
          Pi = pi(X, y, beta1, n);
          log_Pi = Pi.array().log();
          log_1_Pi = (one - Pi).array().log();
          loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
        }

        // cout << "j=" << j << " loglik: " << loglik1 << endl;
        // cout << "j=" << j << " loglik diff: " << loglik1 - loglik0 << endl;
        bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
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
        X_new_transpose = X_new.transpose();

        beta0 = (X_new_transpose * X).ldlt().solve(X_new_transpose * Z);

        // CG
        // ConjugateGradient<MatrixXd, Lower | Upper> cg;
        // cg.compute(X_new_transpose * X);
        // beta0 = cg.solveWithGuess(X_new_transpose * Z, beta0);

        Pi = pi(X, y, beta0, n);
        log_Pi = Pi.array().log();
        log_1_Pi = (one - Pi).array().log();
        loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
        // cout << "j=" << j << " loglik: " << loglik1 << endl;
        // cout << "j=" << j << " loglik diff: " << loglik0 - loglik1 << endl;
        bool condition1 = -(loglik1 + (this->primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
        // bool condition1 = false;
        bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < this->primary_model_fit_epsilon;
        bool condition3 = abs(loglik1) < min(1e-3, tau);
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
    }
#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << j << endl;
#endif

    beta = beta0.tail(p).eval();
    coef0 = beta0(0);
  };

  double neg_loglik_loss(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
  {
    return -loglik_cox(X, y, beta, weights);
  }

  Eigen::VectorXd dual(Eigen::MatrixXd &XI, Eigen::MatrixXd &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weight, int n, Eigen::VectorXd &h)
  {
    Eigen::VectorXd cum_eta(n);
    Eigen::VectorXd cum_eta2(n);
    Eigen::VectorXd cum_eta3(n);
    Eigen::VectorXd eta = XA * beta;
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
    Eigen::VectorXd g = weight.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);
    return XI.transpose() * g;
  }

  void get_A(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &bd, int T0, Eigen::VectorXd &weights,
             Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau, double &train_loss)
  {
    // cout << "get A 1" << endl;
    int n = X.rows();
    int p = X.cols();

#ifdef TEST
    clock_t t0, t1, t2;
    t1 = clock();
#endif
    Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
    Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
    Eigen::VectorXd beta_A(A_ind.size());
    for (int k = 0; k < A_ind.size(); k++)
    {
      beta_A(k) = beta(A_ind(k));
    }
#ifdef TEST
    t2 = clock();
    std::cout << "A ind time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif
    double L1, L0 = neg_loglik_loss(X_A, y, weights, beta_A, coef0);
    train_loss = L0;
#ifdef TEST
    t2 = clock();
    std::cout << "loss time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif
    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd h;
    Eigen::VectorXi I_ind = find_ind(I, g_index, g_size, p, N);
    Eigen::MatrixXd X_I = X_seg(X, n, I_ind);
    Eigen::VectorXd d_I = this->dual(X_I, X_A, y, beta_A, coef0, weights, n, h);
    for (int k = 0; k < I_ind.size(); k++)
    {
      d(I_ind(k)) = d_I(k);
    }

    Eigen::VectorXd beta_A_group = Eigen::VectorXd::Zero(A_size);
    Eigen::VectorXd d_I_group = Eigen::VectorXd::Zero(I_size);
    for (int i = 0; i < N; i++)
    {
      Eigen::MatrixXd XG = X.middleCols(g_index(i), g_size(i));
      Eigen::MatrixXd XG_new = XG;
      for (int j = 0; j < g_size(i); j++)
      {
        XG_new.col(j) = XG.col(j).cwiseProduct(h);
      }
      Eigen::MatrixXd XGbar;
      XGbar.noalias() = XG_new.transpose() * XG;
      Eigen::MatrixXd phiG;
      XGbar.sqrt().evalTo(phiG);
      Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
      betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
      dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
    }
    for (int i = 0; i < A_size; i++)
    {
      beta_A_group(i) = betabar.segment(g_index(A[i]), g_size(A[i])).squaredNorm() / g_size(A[i]);
      bd(A[i]) = beta_A_group(i);
    }
    for (int i = 0; i < I_size; i++)
    {
      d_I_group(i) = dbar.segment(g_index(I[i]), g_size(I[i])).squaredNorm() / g_size(I[i]);
      bd(I[i]) = d_I_group(i);
    }
    // Eigen::VectorXd temp = betabar + dbar;
    // for (int i = 0; i < N; i++)
    // {
    //   bd(i) = (temp.segment(g_index(i), g_size(i))).squaredNorm() / g_size(i);
    // }
    // cout << "get A 4" << endl;
#ifdef TEST
    t2 = clock();
    std::cout << "get A beta d: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif
    // std::cout << "A: " << A << endl;
    // std::cout << "I: " << I << endl;
    // std::cout << "beta_A_group: " << beta_A_group << endl;
    // std::cout << "d_I_group: " << d_I_group << endl;

    Eigen::VectorXi A_min_k = min_k(beta_A_group, C_max);
    Eigen::VectorXi I_max_k = max_k(d_I_group, C_max);
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
    Eigen::MatrixXd X_A_exchage;
    Eigen::VectorXd beta_A_exchange;
    double coef0_A_exchange;

    // Eigen::VectorXd beta_Ac = beta_A;
    // double coef0_Ac = coef0;
    for (int k = C_max; k >= 1;)
    {
      // std::cout << "s1: " << s1 << endl;
      // std::cout << "s2: " << s2 << endl;
      // t1 = clock();
      A_exchange = diff_union(A, s1, s2);
      // cout << "get A 6" << endl;
      // std::cout << "A_exchange: " << A_exchange << endl;
      A_ind_exchage = find_ind(A_exchange, g_index, g_size, p, N);
      X_A_exchage = X_seg(X, n, A_ind_exchage);
      beta_A_exchange = Eigen::VectorXd::Zero(A_ind_exchage.size());
      for (int i = 0; i < A_ind_exchage.size(); i++)
      {
        beta_A_exchange(i) = this->beta_warmstart(A_ind_exchage(i));
      }
      coef0_A_exchange = this->coef0_warmstart;

      primary_model_fit(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange, L0);

      L1 = neg_loglik_loss(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange);

      // cout << "L0: " << L0 << " L1: " << L1 << endl;
      if (L0 - L1 > tau)
      {
        // update A & I & beta & coef0
        train_loss = L1;
        A = A_exchange;
        I = Ac(A_exchange, N);
        beta = Eigen::VectorXd::Zero(p);
        for (int i = 0; i < A_ind_exchage.size(); i++)
        {
          beta(A_ind_exchage[i]) = beta_A_exchange(i);
        }
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
        k = k / 2;
        s1 = s1.head(k).eval();
        s2 = s2.head(k).eval();
      }
    }
#ifdef TEST
    t2 = clock();
    std::cout << "splicing time: " << ((double)(t2 - t0) / CLOCKS_PER_SEC) << endl;
#endif
  };
};

#endif //SRC_ALGORITHM_H