//
// Created by jk on 2020/3/18.
//
#ifndef SRC_ALGORITHM_H
#define SRC_ALGORITHM_H

#include "Data.h"
#include "utilities.h"
#include "logistic.h"
#include "poisson.h"
#include "coxph.h"
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>
#include <time.h>
#include <cfloat>

using namespace std;

bool quick_sort_pair_max(std::pair<int, double> x, std::pair<int, double> y);

class Algorithm
{
public:
  Data data;
  vector<Eigen::MatrixXd> PhiG;
  vector<Eigen::MatrixXd> invPhiG;
  Eigen::VectorXd beta_init;
  int group_df;
  int sparsity_level;
  double lambda_level = 0;
  Eigen::VectorXi train_mask;
  int max_iter;
  int exchange_num;
  bool warm_start;
  Eigen::VectorXd beta;
  double coef0_init;
  Eigen::VectorXi A_init;
  Eigen::VectorXi I_init;
  double coef0;
  double loss;
  Eigen::VectorXi A_out;
  int l;
  int model_fit_max;
  int model_type;
  int algorithm_type;
  std::vector<Eigen::MatrixXd> group_XTX;
  Eigen::VectorXi always_select;
  double tau;

  Algorithm() = default;

  Algorithm(Data &data, int model_type, int algorithm_type, int max_iter = 100)
  {
    this->data = data;
    this->max_iter = max_iter;
    this->A_out = Eigen::VectorXi::Zero(data.get_p());
    this->model_type = model_type;
    this->coef0 = 0.0;
    this->beta = Eigen::VectorXd::Zero(data.get_p());
    this->coef0_init = 0.0;
    this->beta_init = Eigen::VectorXd::Zero(data.get_p());
    this->warm_start = true;
    this->exchange_num = 5;
    this->algorithm_type = algorithm_type;
  };

  void update_PhiG(vector<Eigen::MatrixXd> &PhiG) { this->PhiG = PhiG; }

  void update_invPhiG(vector<Eigen::MatrixXd> &invPhiG) { this->invPhiG = invPhiG; }

  void set_warm_start(bool warm_start) { this->warm_start = warm_start; }

  void update_beta_init(Eigen::VectorXd beta_init) { this->beta_init = beta_init; }

  void update_coef0_init(double coef0) { this->coef0_init = coef0; }

  void update_group_df(int group_df) { this->group_df = group_df; }

  void update_sparsity_level(int sparsity_level)
  {
    this->sparsity_level = sparsity_level;
    this->group_df = sparsity_level;
  }

  void update_lambda_level(double lambda_level) { this->lambda_level = lambda_level; }

  void update_train_mask(Eigen::VectorXi train_mask) { this->train_mask = train_mask; }

  void update_exchange_num(int exchange_num) { this->exchange_num = exchange_num; }

  void update_group_XTX(std::vector<Eigen::MatrixXd> group_XTX) { this->group_XTX = group_XTX; }

  bool get_warm_start() { return this->warm_start; }

  double get_loss() { return this->loss; }

  int get_group_df() { return this->group_df; }

  int get_sparsity_level() { return this->sparsity_level; }

  Eigen::VectorXd get_beta() { return this->beta; }

  double get_coef0() { return this->coef0; }

  Eigen::VectorXi get_A_out() { return this->A_out; };

  int get_l() { return this->l; }

  void fit()
  {
    cout << "fit" << endl;
    int N = data.get_g_num();
    int train_n = this->train_mask.size();
    int p = data.get_p();
    Eigen::MatrixXd train_x(train_n, p);
    Eigen::VectorXd train_y(train_n);
    Eigen::VectorXd train_weight(train_n);
    int T0 = this->sparsity_level;

    Eigen::VectorXi g_index = data.get_g_index();
    Eigen::VectorXi g_size = data.get_g_size();
    if (train_n == data.get_n())
    {
      train_x = data.x;
      train_y = data.y;
      train_weight = data.weight;
    }
    else
    {
      for (int i = 0; i < train_n; i++)
      {
        train_x.row(i) = data.x.row(this->train_mask(i)).eval();
        train_y(i) = data.y(this->train_mask(i));
        train_weight(i) = data.weight(this->train_mask(i));
      }
    }

    cout << "fit 1" << endl;
    if (N == this->get_sparsity_level())
    {
      this->primary_model_fit(train_x, train_y, train_weight, this->beta, this->coef0);
      this->A_out = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      return;
    }

    this->beta = this->beta_init;
    this->coef0 = this->coef0_init;

    // input: this->beta_init, this->coef0_init, this->A_init, this->I_init
    // for splicing get A;for the others 0;
    cout << "fit 2" << endl;
    Eigen::VectorXi A = inital_screening(train_x, train_y, this->beta_init, this->coef0_init, this->A_init, this->I_init, train_weight, g_index, g_size, N);
    cout << "fit 3" << endl;
    Eigen::VectorXi I = Ac(A, N);
    cout << "fit 4" << endl;
    Eigen::MatrixXi A_list(T0, max_iter + 2);
    A_list.col(0) = A;

    Eigen::MatrixXd X_A;
    Eigen::VectorXd beta_A;
    Eigen::VectorXi A_ind;

    cout << "fit 5" << endl;
    if (this->algorithm_type == 6)
    {
      A_ind = find_ind(A, g_index, g_size, p, N);
      X_A = X_seg(train_x, train_n, A_ind);
      beta_A = Eigen::VectorXd::Zero(A_ind.size());
      this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0);
      this->beta = Eigen::VectorXd::Zero(p);
      for (int k = 0; k < A_ind.size(); k++)
      {
        this->beta(A_ind(k)) = beta_A(k);
      }
    }
    cout << "fit 6" << endl;
    for (this->l = 1; this->l <= this->max_iter; l++)
    {
      cout << "fit 7" << endl;
      this->get_A(train_x, train_y, A, I, min(min(T0, N - T0), this->exchange_num), beta, coef0, T0, train_weight, g_index, g_size, N, tau);
      A_list.col(this->l) = A;
      cout << "fit 8" << endl;

      if (this->algorithm_type != 6)
      {
        A_ind = find_ind(A, g_index, g_size, p, N);
        X_A = X_seg(train_x, train_n, A_ind);
        beta_A = Eigen::VectorXd::Zero(A_ind.size());
        this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0);
        this->beta = Eigen::VectorXd::Zero(p);
        for (int mm = 0; mm < A_ind.size(); mm++)
        {
          this->beta(A_ind(mm)) = beta_A(mm);
        }
        for (int ll = 0; ll < this->l; ll++)
        {
          if (A == A_list.col(ll))
          {
            return;
          }
        }
      }
      else
      {
        if (A == A_list.col(this->l - 1))
        {
          // l++;    ?????
          break;
        }
      }
    }

    /// ### keep
    // if (this->algorithm_type == 1 || this->algorithm_type == 5)
    // {
    //   for (this->l = 1; this->l <= this->max_iter; l++)
    //   {
    //     this->get_A(train_x, train_y, this->beta, this->coef0, T0, train_weight, g_index, g_size, N, A);
    //     A_list.col(this->l) = A;
    //     for (int mm = 0; mm < T0; mm++)
    //     {
    //       X_A.col(mm) = train_x.col(A[mm]);
    //     }
    //     this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0);
    //     this->beta = Eigen::VectorXd::Zero(p);
    //     for (int mm = 0; mm < T0; mm++)
    //     {
    //       this->beta(A[mm]) = beta_A(mm);
    //     }
    //     for (int ll = 0; ll < this->l; ll++)
    //     {
    //       if (A == A_list.col(ll))
    //       {
    //         return;
    //       }
    //     }
    //   }
    // }
    // else
    // {
    //   Eigen::VectorXi ind;
    //   for (this->l = 1; this->l <= this->max_iter; l++)
    //   {
    //     this->get_A(train_x, train_y, this->beta, this->coef0, T0, train_weight, g_index, g_size, N, A);
    //     A_list.col(this->l) = A;
    //     ind = find_ind(A, g_index, g_size, p, N);
    //     X_A = X_seg(train_x, train_n, ind);
    //     beta_A = Eigen::VectorXd::Zero(ind.size());
    //     this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0);
    //     this->beta = Eigen::VectorXd::Zero(p);
    //     for (int mm = 0; mm < ind.size(); mm++)
    //     {
    //       this->beta(ind(mm)) = beta_A(mm);
    //     }
    //     for (int ll = 0; ll < this->l; ll++)
    //     {
    //       if (A == A_list.col(ll))
    //       {
    //         return;
    //       }
    //     }
    //   }
    // }
  };

  virtual void primary_model_fit(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0) = 0;

  virtual void get_A(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int C_max, Eigen::VectorXd &beta, double &coef0, int T0, Eigen::VectorXd &weights,
                     Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau) = 0;

  virtual Eigen::VectorXi inital_screening(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights,
                                           Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N) = 0;

  virtual double neg_loglik_loss(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0) = 0;
  virtual Eigen::VectorXd dual(Eigen::MatrixXd &XI, Eigen::MatrixXd &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weights, int n, Eigen::VectorXd &h) = 0;
};

class GSplicingLogistic : public Algorithm
{
public:
  GSplicingLogistic(Data &data, int max_iter, int exchange_num) : Algorithm(data, max_iter, exchange_num){};

  Eigen::VectorXi inital_screening(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {
    // variable initialization
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd bd = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(p);

    // calculate beta & d & h
    Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
    Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
    Eigen::VectorXd beta_A(A_ind.size());
    for (int k = 0; k < A_ind.size(); k++)
    {
      beta_A(k) = beta(A_ind(k));
    }

    Eigen::VectorXd h;
    Eigen::VectorXi I_ind = find_ind(I, g_index, g_size, p, N);
    Eigen::MatrixXd X_I = X_seg(X, n, I_ind);
    Eigen::VectorXd d_I = this->dual(X_I, X_A, y, beta_A, coef0, weights, n, h);
    for (int k = 0; k < I_ind.size(); k++)
    {
      d(I_ind(k)) = d_I(k);
    }

    // calculate group bd
    for (int i = 0; i < N; i++)
    {
      Eigen::MatrixXd XG = X.middleCols(g_index(i), g_size(i));
      Eigen::MatrixXd XG_new = XG;
      for (int j = 0; j < n; j++)
      {
        XG_new.row(j) = XG.row(j) * h(j);
      }
      Eigen::MatrixXd XGbar = XG_new.transpose() * XG;
      Eigen::MatrixXd phiG = XGbar.sqrt();
      Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
      Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
      betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
      dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
    }
    Eigen::VectorXd temp = betabar + dbar;
    for (int i = 0; i < N; i++)
    {
      bd(i) = (temp.segment(g_index(i), g_size(i))).squaredNorm() / g_size(i);
    }

    // get Active-set A according to max_k bd
    Eigen::VectorXi A_new;
    max_k(bd, this->get_sparsity_level(), A_new);

    return A_new;
  }

  void primary_model_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
  {
    int n = x.rows();
    int p = x.cols();
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
    X.rightCols(p) = x;
    Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd Pi = pi(X, y, beta0, n);
    Eigen::VectorXd log_Pi = Pi.array().log();
    Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
    double loglik1, loglik0 = 1e5;
    Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
    for (int i = 0; i < n; i++)
    {
      if (W(i) < 0.001)
        W(i) = 0.001;
    }
    Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);

    int j;
    for (j = 0; j < 30; j++)
    {
      for (int i = 0; i < n; i++)
      {
        X_new.row(i) = X.row(i) * W(i) * weights(i);
      }
      beta0 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
      Pi = pi(X, y, beta0, n);
      log_Pi = Pi.array().log();
      log_1_Pi = (one - Pi).array().log();
      loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
      if (abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < 1e-6 || abs(loglik1) < 1e-3)
        break;
      loglik0 = loglik1;
      W = Pi.cwiseProduct(one - Pi);
      for (int i = 0; i < n; i++)
      {
        if (W(i) < 0.001)
          W(i) = 0.001;
      }
      Z = X * beta0 + (y - Pi).cwiseQuotient(W);
    }

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
    Eigen::VectorXd coef = Eigen::VectorXd::Ones(XA.cols() + 1);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    coef(0) = coef0;
    coef.tail(XA.cols()) = beta;
    Eigen::VectorXd pr = pi(XA, y, coef, n);
    Eigen::VectorXd d_I = XI.transpose() * ((y - pi(XA, y, coef, n)).cwiseProduct(weights));

    h = weights.array() * pr.array() * (one - pr).array();

    return d_I;
  }

  void get_A(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int C_max, Eigen::VectorXd &beta, double &coef0, int T0, Eigen::VectorXd &weights,
             Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau)
  {
    int n = X.rows();
    int p = X.cols();

    double L0 = neg_loglik_loss(X, y, weights, beta, coef0);
    double L1;

    int A_size = A.size();
    int I_size = I.size();
    Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);

    // calculate d & h
    Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
    Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
    Eigen::VectorXd beta_A(A_ind.size());
    for (int k = 0; k < A_ind.size(); k++)
    {
      beta_A(k) = beta(A_ind(k));
    }

    Eigen::VectorXd h;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
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
      for (int j = 0; j < n; j++)
      {
        XG_new.row(j) = XG.row(j) * h(j);
      }
      Eigen::MatrixXd XGbar = XG_new.transpose() * XG;
      Eigen::MatrixXd phiG = XGbar.sqrt();
      Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(g_size(i), g_size(i)));
      betabar.segment(g_index(i), g_size(i)) = phiG * beta.segment(g_index(i), g_size(i));
      dbar.segment(g_index(i), g_size(i)) = invphiG * d.segment(g_index(i), g_size(i));
    }

    for (int i = 0; i < A_size; i++)
    {
      beta_A_group(i) = betabar.segment(g_index(A[i]), g_size(A[i])).squaredNorm() / g_size(A[i]);
    }
    for (int i = 0; i < I_size; i++)
    {
      d_I_group(i) = dbar.segment(g_index(I[i]), g_size(I[i])).squaredNorm() / g_size(I[i]);
    }

    Eigen::VectorXi s1 = slice(A, min_k(beta_A, C_max));
    Eigen::VectorXi s2 = slice(I, max_k(d_I, C_max));
    Eigen::VectorXi A_c(A_size);

    for (int k = C_max; k >= 1; k--)
    {
      A_c = diff_union(A, s1, s2);
      Eigen::VectorXi A_ind = find_ind(A_c, g_index, g_size, p, N);
      Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
      Eigen::VectorXd beta_Ac;
      double coef0_Ac;

      primary_model_fit(X_A, y, weights, beta_Ac, coef0_Ac);

      L1 = neg_loglik_loss(X_A, y, weights, beta_Ac, coef0_Ac);
      if (L0 - L1 > tau)
      {
        // update A & I & beta & coef0
        A = A_c;
        I = Ac(A_c, N);
        beta = Eigen::VectorXd::Zero(p);
        for (int i = 0; i < A_ind.size(); i++)
        {
          beta(A_ind[i]) = beta_Ac(i);
        }
        coef0 = coef0_Ac;

        // // update d & h
        // d = Eigen::VectorXd::zeros(p);
        // Eigen::VectorXi I_ind = find_ind(I, g_index, g_size, p, N);
        // Eigen::MatrixXd X_I = X_seg(X, n, I_ind);
        // Eigen::VectorXd d_I = this->dual(X_I, X_A, y, beta_Ac, coef0_Ac, weights, n, h);
        // for (int k = 0; k < I_ind.size(); k++)
        // {
        //   d(I_ind(k)) = d_I(k);
        // }

        break;
      }
      else
      {
        s1 = s1.head(k - 1).eval();
        s2 = s2.head(k - 1).eval();
      }
    }
  };
};

// class GSplicingLm : public Algorithm
// {
// public:
//   GSplicingLm(Data &data, int max_iter, int exchange_num) : Algorithm(data, max_iter, exchange_num){};

//   Eigen::VectorXd inital_screening(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weights,
//                                    Eigen::VectorXi &index, Eigen::VectorXi &gsize, int &N)
//   {
//     Eigen::VectorXd res = y - X * beta;
//     Eigen::VectorXd Xy = X.transpose() * res;
//     Eigen::VectorXd inital = Eigen::VectorXd::Zero(N);
//     for (int i = 0; i < N; i++)
//     {
//       inital(i) = (Xy.segment(index(i), gsize(i)).squaredNorm()) / gsize(i);
//     }
//     return inital;
//   }

//   Eigen::VectorXd dual(Eigen::MatrixXd &XI, Eigen::MatrixXd &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weight, int n)
//   {
//     return XI.transpose() * (y - XA * beta) / n;
//   }

//   void get_A(Eigen::MatrixXd &X, Eigen::VectorXd &y, vector<int> &A, vector<int> &I, int C_max, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &d, Eigen::VectorXd &weights,
//              Eigen::VectorXi &index, Eigen::VectorXi &gsize, int n, int p, int N, double tau)
//   {
//     double L0 = (y - X * beta).squaredNorm();
//     double L1;

//     int A_size = A.size();
//     int I_size = I.size();

//     Eigen::VectorXd beta_A_group = Eigen::VectorXd::Zero(A_size);
//     Eigen::VectorXd d_I_group = Eigen::VectorXd::Zero(I_size);
//     for (int i = 0; i < A_size; i++)
//     {
//       beta_A_group(i) = beta.segment(index(A[i]), gsize(A[i])).squaredNorm() / gsize(A[i]);
//     }
//     for (int i = 0; i < I_size; i++)
//     {
//       d_I_group(i) = d.segment(index(I[i]), gsize(I[i])).squaredNorm() / gsize(I[i]);
//     }

//     vector<int> s1(vec_seg(A, min_k(beta_A_group, C_max)));
//     vector<int> s2(vec_seg(I, max_k(d_I_group, C_max)));
//     vector<int> A_c(A_size);
//     for (int k = C_max; k >= 1; k--)
//     {
//       A_c = diff_union(A, s1, s2);
//       Eigen::VectorXi A_ind = find_ind(A_c, index, gsize, p, N);
//       Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
//       Eigen::VectorXd beta_Ac = Eigen::VectorXd::Zero(A_ind.size());
//       primary_model_fit(X_A, y, weights, beta_Ac, coef0);
//       L1 = (y - X_A * beta_Ac).squaredNorm();
//       if (L0 - L1 > tau)
//       {
//         A = A_c;
//         I = Ac(A, N);
//         beta = Eigen::VectorXd::Zero(p);
//         Eigen::VectorXi I_ind = find_ind(I, index, gsize, p, N);
//         Eigen::MatrixXd X_I = X_seg(x, n, I_ind);
//         Eigen::VectorXd d_I = this->dual(X_I, X_A, y, beta_A, this->coef, weight, n);
//         d = Eigen::VectorXd::Zero(p);
//         for (int i = 0; i < A_ind.size(); i++)
//         {
//           beta(A_ind[i]) = beta_Ac(i);
//         }
//         for (int i = 0; i < I_ind.size(); i++)
//         {
//           d(I_ind[i]) = d_I(i);
//         }
//         break;
//       }
//       else
//       {
//         s1.pop_back();
//         s2.pop_back();
//       }
//     }
//   };

//   void primary_model_fit(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     beta = X.colPivHouseholderQr().solve(y);
//   };
// };

// class GSplicingPoisson : public Algorithm
// {
// public:
//   GSplicingPoisson(Data &data, int max_iter, int exchange_num) : Algorithm(data, max_iter, exchange_num){};

//   Eigen::VectorXd inital_screening(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weights,
//                                    Eigen::VectorXi &index, Eigen::VectorXi &gsize, int &N)
//   {
//     int n = X.rows();
//     Eigen::VectorXd eta = X * beta + Eigen::VectorXd::Ones(n) * coef0;
//     for (int i = 0; i <= n - 1; i++)
//     {
//       if (eta(i) < -30.0)
//         eta(i) = -30.0;
//       if (eta(i) > 30.0)
//         eta(i) = 30.0;
//     }
//     Eigen::VectorXd expeta = eta.array().exp();
//     Eigen::VectorXd res = (y - expeta).cwiseProduct(weights);
//     Eigen::VectorXd Xy = (X.transpose() * res).array().square();
//     Eigen::VectorXd inital = Eigen::VectorXd::Zero(N);
//     for (int i = 0; i < N; i++)
//     {
//       Eigen::MatrixXd XG = X.middleCols(index(i), gsize(i));
//       Eigen::MatrixXd XG_new = XG;
//       for (int j = 0; j < n; j++)
//       {
//         XG_new.row(j) = XG.row(j) * expeta(j) * weights(j);
//       }
//       Eigen::MatrixXd XGbar = XG_new.transpose() * XG;
//       Eigen::MatrixXd phiG = XGbar.sqrt();
//       Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(gsize(i), gsize(i)));
//       Xy.segment(index(i), gsize(i)) = invphiG * Xy.segment(index(i), gsize(i));
//       inital(i) = (Xy.segment(index(i), gsize(i)).squaredNorm()) / gsize(i);
//     }
//     return inital;
//   }

//   void primary_model_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     int n = x.rows();
//     int p = x.cols();
//     Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
//     X.rightCols(p) = x;
//     Eigen::MatrixXd X_trans = X.transpose();
//     Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(p + 1, n);
//     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
//     Eigen::VectorXd eta = X * beta0;
//     Eigen::VectorXd expeta = eta.array().exp();
//     Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
//     double loglik0 = 1e5;
//     double loglik1;

//     for (int j = 0; j < 30; j++)
//     {
//       for (int i = 0; i < n; i++)
//       {
//         temp.col(i) = X_trans.col(i) * expeta(i) * weights(i);
//       }
//       z = eta + (y - expeta).cwiseQuotient(expeta);
//       beta0 = (temp * X).ldlt().solve(temp * z);
//       eta = X * beta0;
//       for (int i = 0; i <= n - 1; i++)
//       {
//         if (eta(i) < -30.0)
//           eta(i) = -30.0;
//         if (eta(i) > 30.0)
//           eta(i) = 30.0;
//       }
//       expeta = eta.array().exp();
//       loglik1 = (y.cwiseProduct(eta) - expeta).dot(weights);
//       if (abs(loglik0 - loglik1) / abs(0.1 + loglik0) < 1e-6)
//         break;
//       loglik0 = loglik1;
//     }
//     for (int i = 0; i < p; i++)
//       beta(i) = beta0(i + 1);
//     coef0 = beta0(0);
//   };

//   Eigen::VectorXd dual(Eigen::MatrixXd &XI, Eigen::MatrixXd &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weight, int n)
//   {
//     Eigen::VectorXd coef = Eigen::VectorXd::Ones(XA.cols() + 1);
//     coef(0) = coef0;
//     coef.tail(XA.cols()) = beta;
//     Eigen::VectorXd eta = XA * coef;
//     for (int i = 0; i <= n - 1; i++)
//     {
//       if (eta(i) < -30.0)
//         eta(i) = -30.0;
//       if (eta(i) > 30.0)
//         eta(i) = 30.0;
//     }
//     Eigen::VectorXd expeta = eta.array().exp();
//     return XI.transpose() * ((y - expeta).cwiseProduct(weight));
//   }

//   void get_A(Eigen::MatrixXd &X, Eigen::VectorXd &y, vector<int> &A, vector<int> &I, int C_max, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &d, Eigen::VectorXd &weights,
//              Eigen::VectorXi &index, Eigen::VectorXi &gsize, int n, int p, int N, double tau)
//   {
//     Eigen::VectorXd coef = Eigen::VectorXd::Ones(p + 1);
//     coef(0) = coef0;
//     coef.tail(p) = beta;
//     double L0 = -loglik_poiss(X, y, coef, n, weights);
//     double L1;
//     int A_size = A.size();
//     int I_size = I.size();
//     Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
//     Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd beta_A = Eigen::VectorXd::Zero(A_size);
//     Eigen::VectorXd d_I = Eigen::VectorXd::Zero(I_size);
//     Eigen::VectorXd eta = X * beta + Eigen::VectorXd::Ones(n) * coef0;
//     for (int i = 0; i <= n - 1; i++)
//     {
//       if (eta(i) < -30.0)
//         eta(i) = -30.0;
//       if (eta(i) > 30.0)
//         eta(i) = 30.0;
//     }
//     Eigen::VectorXd expeta = eta.array().exp();
//     for (int i = 0; i < N; i++)
//     {
//       Eigen::MatrixXd XG = X.middleCols(index(i), gsize(i));
//       Eigen::MatrixXd XG_new = XG;
//       for (int j = 0; j < n; j++)
//       {
//         XG_new.row(j) = XG.row(j) * expeta(j) * weights(j);
//       }
//       Eigen::MatrixXd XGbar = XG_new.transpose() * XG;
//       Eigen::MatrixXd phiG = XGbar.sqrt();
//       Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(gsize(i), gsize(i)));
//       betabar.segment(index(i), gsize(i)) = phiG * beta.segment(index(i), gsize(i));
//       dbar.segment(index(i), gsize(i)) = invphiG * d.segment(index(i), gsize(i));
//     }
//     for (int i = 0; i < A_size; i++)
//     {
//       beta_A(i) = betabar.segment(index(A[i]), gsize(A[i])).squaredNorm() / gsize(A[i]);
//     }
//     for (int i = 0; i < I_size; i++)
//     {
//       d_I(i) = dbar.segment(index(I[i]), gsize(I[i])).squaredNorm() / gsize(I[i]);
//     }
//     vector<int> s1(vec_seg(A, min_k(beta_A, C_max)));
//     vector<int> s2(vec_seg(I, max_k(d_I, C_max)));
//     vector<int> A_c(A_size);
//     for (int k = C_max; k >= 1; k--)
//     {
//       A_c = diff_union(A, s1, s2);
//       Eigen::VectorXi A_ind = find_ind(A_c, index, gsize, p, N);
//       Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
//       Eigen::VectorXd beta_Ac = poisson_fit(X_A, y, n, A_ind.size(), weights);
//       L1 = -loglik_poiss(X_A, y, beta_Ac, n, weights);
//       if (L0 - L1 > tau)
//       {
//         A = A_c;
//         I = Ac(A_c, N);
//         beta = Eigen::VectorXd::Zero(p);
//         coef0 = beta_Ac(0);
//         eta = X_A * beta_Ac.tail(A_ind.size()) + Eigen::VectorXd::Ones(n) * coef0;
//         expeta = eta.array().exp();
//         d = X.transpose() * ((y - expeta).cwiseProduct(weights));
//         for (int i = 0; i < A_ind.size(); i++)
//         {
//           beta(A_ind[i]) = beta_Ac(i + 1);
//         }
//         break;
//       }
//       else
//       {
//         s1.pop_back();
//         s2.pop_back();
//       }
//     }
//   };
// };

// class GSplicingCox : public Algorithm
// {
// public:
//   GSplicingCox(Data &data, int max_iter, int exchange_num) : Algorithm(data, max_iter, exchange_num){};

//   Eigen::VectorXd inital_screening(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weights,
//                                    Eigen::VectorXi &index, Eigen::VectorXi &gsize, int &N)
//   {
//     int n = X.rows();
//     Eigen::MatrixXd h(n, n);
//     Eigen::VectorXd cum_eta(n);
//     Eigen::VectorXd cum_eta2(n);
//     Eigen::VectorXd cum_eta3(n);
//     Eigen::VectorXd eta = X * beta;
//     for (int i = 0; i <= n - 1; i++)
//     {
//       if (eta(i) < -30.0)
//         eta(i) = -30.0;
//       if (eta(i) > 30.0)
//         eta(i) = 30.0;
//     }
//     eta = weights.array() * eta.array().exp();
//     cum_eta(n - 1) = eta(n - 1);
//     for (int k = n - 2; k >= 0; k--)
//     {
//       cum_eta(k) = cum_eta(k + 1) + eta(k);
//     }
//     cum_eta2(0) = (y(0) * weights(0)) / cum_eta(0);
//     for (int k = 1; k <= n - 1; k++)
//     {
//       cum_eta2(k) = (y(k) * weights(k)) / cum_eta(k) + cum_eta2(k - 1);
//     }
//     cum_eta3(0) = (y(0) * weights(0)) / pow(cum_eta(0), 2);
//     for (int k = 1; k <= n - 1; k++)
//     {
//       cum_eta3(k) = (y(k) * weights(k)) / pow(cum_eta(k), 2) + cum_eta3(k - 1);
//     }
//     h = -cum_eta3.replicate(1, n);
//     h = h.cwiseProduct(eta.replicate(1, n));
//     h = h.cwiseProduct(eta.replicate(1, n).transpose());
//     for (int i = 0; i < n; i++)
//     {
//       for (int j = i + 1; j < n; j++)
//       {
//         h(j, i) = h(i, j);
//       }
//     }
//     h.diagonal() = cum_eta2.cwiseProduct(eta) + h.diagonal();
//     Eigen::VectorXd g = weights.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);
//     Eigen::VectorXd Xy = X.transpose() * g;
//     Eigen::VectorXd inital = Eigen::VectorXd::Zero(N);
//     for (int i = 0; i < N; i++)
//     {
//       Eigen::MatrixXd XG = X.middleCols(index(i), gsize(i));
//       Eigen::MatrixXd XGbar = XG.transpose() * h * XG;
//       Eigen::MatrixXd invphiG = (XGbar.sqrt()).ldlt().solve(Eigen::MatrixXd::Identity(gsize(i), gsize(i)));
//       Xy.segment(index(i), gsize(i)) = invphiG * Xy.segment(index(i), gsize(i));
//       inital(i) = (Xy.segment(index(i), gsize(i)).squaredNorm()) / gsize(i);
//     }
//     return inital;
//   }

//   void primary_model_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     int n = x.rows();
//     int p = x.cols();
//     Eigen::VectorXd theta(n);
//     Eigen::MatrixXd one = (Eigen::MatrixXd::Ones(n, n)).triangularView<Eigen::Upper>();
//     Eigen::MatrixXd x_theta(n, p);
//     Eigen::VectorXd xij_theta(n);
//     Eigen::VectorXd cum_theta(n);
//     Eigen::VectorXd g(p);
//     Eigen::MatrixXd h(p, p);
//     Eigen::VectorXd d(p);
//     double loglik1, loglik0 = 1e5;
//     beta = Eigen::VectorXd::Zero(p);

//     double step = 0.5;
//     int l, m = 1;
//     for (l = 1; l <= 30; l++)
//     {
//       theta = x * beta;
//       for (int i = 0; i < n; i++)
//       {
//         if (theta(i) > 50)
//           theta(i) = 50;
//         else if (theta(i) < -50)
//           theta(i) = -50;
//       }
//       theta = theta.array().exp();
//       cum_theta = one * theta;
//       x_theta = x.array().colwise() * theta.array();
//       x_theta = one * x_theta;
//       x_theta = x_theta.array().colwise() / cum_theta.array();
//       g = (x - x_theta).transpose() * (weights.cwiseProduct(y));

//       for (int k1 = 0; k1 < p; k1++)
//       {
//         for (int k2 = k1; k2 < p; k2++)
//         {
//           xij_theta = (theta.cwiseProduct(x.col(k1))).cwiseProduct(x.col(k2));
//           for (int j = n - 2; j >= 0; j--)
//           {
//             xij_theta(j) = xij_theta(j + 1) + xij_theta(j);
//           }
//           h(k1, k2) = -(xij_theta.cwiseQuotient(cum_theta) - x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(y));
//           h(k2, k1) = h(k1, k2);
//         }
//       }
//       d = h.ldlt().solve(g);
//       beta = beta - pow(step, m) * d;
//       loglik1 = loglik_cox(x, y, beta, weights);
//       while ((loglik0 > loglik1) && (m < 5))
//       {
//         m = m + 1;
//         beta = beta - pow(step, m) * d;
//         loglik1 = loglik_cox(x, y, beta, weights);
//       }
//       if (abs(loglik0 - loglik1) / abs(0.1 + loglik0) < 1e-6)
//       {
//         break;
//       }
//       loglik0 = loglik1;
//     }
//   };

//   Eigen::VectorXd dual(Eigen::MatrixXd &XI, Eigen::MatrixXd &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &weight, int n)
//   {
//     Eigen::VectorXd cum_eta(n);
//     Eigen::VectorXd cum_eta2(n);
//     Eigen::VectorXd eta = XA * beta;
//     for (int i = 0; i <= n - 1; i++)
//     {
//       if (eta(i) < -30.0)
//         eta(i) = -30.0;
//       if (eta(i) > 30.0)
//         eta(i) = 30.0;
//     }
//     eta = weight.array() * eta.array().exp();
//     cum_eta(n - 1) = eta(n - 1);
//     for (int k = n - 2; k >= 0; k--)
//     {
//       cum_eta(k) = cum_eta(k + 1) + eta(k);
//     }
//     cum_eta2(0) = (y(0) * weight(0)) / cum_eta(0);
//     for (int k = 1; k <= n - 1; k++)
//     {
//       cum_eta2(k) = (y(k) * weight(k)) / cum_eta(k) + cum_eta2(k - 1);
//     }
//     Eigen::VectorXd g = weight.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);
//     return XI.transpose() * g;
//   }

//   void get_A(Eigen::MatrixXd &X, Eigen::VectorXd &y, vector<int> &A, vector<int> &I, int C_max, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &d, Eigen::VectorXd &weights,
//              Eigen::VectorXi &index, Eigen::VectorXi &gsize, int n, int p, int N, double tau)
//   {
//     double L0 = -loglik_cox(X, y, beta, weights);
//     double L1;
//     int A_size = A.size();
//     int I_size = I.size();
//     Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd beta_A = Eigen::VectorXd::Zero(A_size);
//     Eigen::VectorXd d_I = Eigen::VectorXd::Zero(I_size);
//     Eigen::MatrixXd h(n, n);
//     Eigen::VectorXd cum_eta(n);
//     Eigen::VectorXd cum_eta2(n);
//     Eigen::VectorXd cum_eta3(n);
//     Eigen::VectorXd eta = X * beta;
//     for (int i = 0; i <= n - 1; i++)
//     {
//       if (eta(i) < -30.0)
//         eta(i) = -30.0;
//       if (eta(i) > 30.0)
//         eta(i) = 30.0;
//     }
//     eta = weights.array() * eta.array().exp();
//     cum_eta(n - 1) = eta(n - 1);
//     for (int k = n - 2; k >= 0; k--)
//     {
//       cum_eta(k) = cum_eta(k + 1) + eta(k);
//     }
//     cum_eta2(0) = (y(0) * weights(0)) / cum_eta(0);
//     for (int k = 1; k <= n - 1; k++)
//     {
//       cum_eta2(k) = (y(k) * weights(k)) / cum_eta(k) + cum_eta2(k - 1);
//     }
//     cum_eta3(0) = (y(0) * weights(0)) / pow(cum_eta(0), 2);
//     for (int k = 1; k <= n - 1; k++)
//     {
//       cum_eta3(k) = (y(k) * weights(k)) / pow(cum_eta(k), 2) + cum_eta3(k - 1);
//     }
//     h = -cum_eta3.replicate(1, n);
//     h = h.cwiseProduct(eta.replicate(1, n));
//     h = h.cwiseProduct(eta.replicate(1, n).transpose());
//     for (int i = 0; i < n; i++)
//     {
//       for (int j = i + 1; j < n; j++)
//       {
//         h(j, i) = h(i, j);
//       }
//     }
//     h.diagonal() = cum_eta2.cwiseProduct(eta) + h.diagonal();
//     Eigen::VectorXd g = weights.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);
//     Eigen::VectorXd Xy = X.transpose() * g;
//     Eigen::VectorXd inital = Eigen::VectorXd::Zero(N);
//     for (int i = 0; i < N; i++)
//     {
//       Eigen::MatrixXd XG = X.middleCols(index(i), gsize(i));
//       Eigen::MatrixXd XGbar = XG.transpose() * h * XG;
//       Eigen::MatrixXd phiG = XGbar.sqrt();
//       Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(gsize(i), gsize(i)));
//       betabar.segment(index(i), gsize(i)) = phiG * beta.segment(index(i), gsize(i));
//       dbar.segment(index(i), gsize(i)) = invphiG * d.segment(index(i), gsize(i));
//     }
//     for (int i = 0; i < A_size; i++)
//     {
//       beta_A(i) = betabar.segment(index(A[i]), gsize(A[i])).squaredNorm() / gsize(A[i]);
//     }
//     for (int i = 0; i < I_size; i++)
//     {
//       d_I(i) = dbar.segment(index(I[i]), gsize(I[i])).squaredNorm() / gsize(I[i]);
//     }
//     vector<int> s1(vec_seg(A, min_k(beta_A, C_max)));
//     vector<int> s2(vec_seg(I, max_k(d_I, C_max)));
//     vector<int> A_c(A_size);
//     for (int k = C_max; k >= 1; k--)
//     {
//       A_c = diff_union(A, s1, s2);
//       Eigen::VectorXi A_ind = find_ind(A_c, index, gsize, p, N);
//       Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
//       Eigen::VectorXd beta_Ac = cox_fit(X_A, y, n, A_ind.size(), weights);
//       L1 = -loglik_cox(X_A, y, beta_Ac, weights);
//       if (L0 - L1 > tau)
//       {
//         A = A_c;
//         I = Ac(A_c, N);
//         beta = Eigen::VectorXd::Zero(p);
//         Eigen::VectorXd eta = X_A * beta_Ac;
//         for (int i = 0; i <= n - 1; i++)
//         {
//           if (eta(i) < -30.0)
//             eta(i) = -30.0;
//           if (eta(i) > 30.0)
//             eta(i) = 30.0;
//         }
//         eta = weights.array() * eta.array().exp();
//         cum_eta(n - 1) = eta(n - 1);
//         for (int k = n - 2; k >= 0; k--)
//         {
//           cum_eta(k) = cum_eta(k + 1) + eta(k);
//         }
//         cum_eta2(0) = (y(0) * weights(0)) / cum_eta(0);
//         for (int k = 1; k <= n - 1; k++)
//         {
//           cum_eta2(k) = (y(k) * weights(k)) / cum_eta(k) + cum_eta2(k - 1);
//         }
//         d = X.transpose() * (weights.cwiseProduct(y) - cum_eta2.cwiseProduct(eta));
//         for (int i = 0; i < A_ind.size(); i++)
//         {
//           beta(A_ind[i]) = beta_Ac(i);
//         }
//         break;
//       }
//       else
//       {
//         s1.pop_back();
//         s2.pop_back();
//       }
//     }
//   };
// };

// class PdasLm : public Algorithm
// {
// public:
//   PdasLm(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 1, algorithm_type, max_iter)
//   {
//     this->algorithm_type = algorithm_type;
//   };

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     vector<int> A(T0);

//     Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
//     Eigen::VectorXd d = (X.transpose() * (y - X * beta - coef)) / double(n);
//     Eigen::VectorXd bd = beta + d;
//     bd = bd.cwiseAbs();
//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);
//     max_k(bd, T0, A_out);
//   };

//   void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);
//   };
// };

// class PdasLogistic : public Algorithm
// {
// public:
//   PdasLogistic(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 2, algorithm_type, max_iter){};

//   void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {

//     int n = x.rows();
//     int p = x.cols();
//     if (n <= p)
//     {
//       Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, n);
//       Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(n);
//       Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(n);
//       Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
//       X.rightCols(n - 1) = x.leftCols(n - 1);
//       Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, n);
//       Eigen::VectorXd Pi = pi(X, y, beta0, n);
//       Eigen::VectorXd log_Pi = Pi.array().log();
//       Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
//       double loglik0 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
//       Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
//       Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);
//       W = W.cwiseProduct(weights);
//       for (int i = 0; i < n; i++)
//       {
//         X_new.row(i) = X.row(i) * W(i);
//       }
//       beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);

//       double loglik1;

//       int j;
//       for (j = 0; j < 30; j++)
//       {
//         Pi = pi(X, y, beta1, n);
//         log_Pi = Pi.array().log();
//         log_1_Pi = (one - Pi).array().log();
//         loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
//         if (abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < 1e-6)
//         {
//           break;
//         }
//         beta0 = beta1;
//         loglik0 = loglik1;
//         W = Pi.cwiseProduct(one - Pi);
//         for (int i = 0; i < n; i++)
//         {
//           if (W(i) < 0.001)
//             W(i) = 0.001;
//         }
//         Z = X * beta0 + (y - Pi).cwiseQuotient(W);
//         W = W.cwiseProduct(weights);
//         for (int i = 0; i < n; i++)
//         {
//           X_new.row(i) = X.row(i) * W(i);
//         }
//         beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//       }
//       for (int i = 0; i < p; i++)
//       {
//         if (i < n)
//           beta(i) = beta0(i + 1);
//         else
//           beta(i) = 0;
//       }
//       coef0 = beta0(0);
//     }
//     else
//     {
//       Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
//       X.rightCols(p) = x;
//       Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
//       Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
//       Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p + 1);
//       Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
//       Eigen::VectorXd Pi = pi(X, y, beta0, n);
//       Eigen::VectorXd log_Pi = Pi.array().log();
//       Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
//       double loglik0 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
//       Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
//       Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);
//       W = W.cwiseProduct(weights);
//       for (int i = 0; i < n; i++)
//       {
//         X_new.row(i) = X.row(i) * W(i);
//       }
//       beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//       double loglik1;

//       int j;
//       for (j = 0; j < 30; j++)
//       {
//         Pi = pi(X, y, beta1, n);
//         log_Pi = Pi.array().log();
//         log_1_Pi = (one - Pi).array().log();
//         loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
//         if (abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < 1e-6)
//         {
//           break;
//         }
//         beta0 = beta1;
//         loglik0 = loglik1;
//         W = Pi.cwiseProduct(one - Pi);
//         for (int i = 0; i < n; i++)
//         {
//           if (W(i) < 0.001)
//             W(i) = 0.001;
//         }
//         Z = X * beta0 + (y - Pi).cwiseQuotient(W);
//         W = W.cwiseProduct(weights);
//         for (int i = 0; i < n; i++)
//         {
//           X_new.row(i) = X.row(i) * W(i);
//         }
//         beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//       }
//       for (int i = 0; i < p; i++)
//         beta(i) = beta0(i + 1);
//       coef0 = beta0(0);
//     }
//   }

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     int p = X.cols();
//     Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
//     vector<int> A(T0);
//     Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
//     ;
//     Eigen::VectorXd xbeta_exp = X * beta + coef;
//     //
//     for (int i = 0; i <= n - 1; i++)
//     {
//       if (xbeta_exp(i) > 30.0)
//         xbeta_exp(i) = 30.0;
//       if (xbeta_exp(i) < -30.0)
//         xbeta_exp(i) = -30.0;
//     }
//     xbeta_exp = xbeta_exp.array().exp();
//     Eigen::VectorXd pr = xbeta_exp.array() / (xbeta_exp + one).array();
//     Eigen::VectorXd l1 = -X.adjoint() * ((y - pr).cwiseProduct(weights));
//     X = X.array().square();
//     Eigen::VectorXd l2 = (X.adjoint()) * ((pr.cwiseProduct(one - pr)).cwiseProduct(weights));
//     Eigen::VectorXd d = -l1.cwiseQuotient(l2);
//     bd = (beta + d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);
//     max_k(bd, T0, A_out);
//   };
// };

// class PdasPoisson : public Algorithm
// {
// public:
//   PdasPoisson(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 3, algorithm_type, max_iter)
//   {
//     this->algorithm_type = algorithm_type;
//   };

//   void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     int n = x.rows();
//     int p = x.cols();
//     Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
//     X.rightCols(p) = x;
//     Eigen::MatrixXd X_trans = X.transpose();
//     Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(p + 1, n);
//     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
//     beta0.tail(p) = beta;
//     beta0(0) = coef0;
//     Eigen::VectorXd eta = X * beta0;
//     Eigen::VectorXd expeta = eta.array().exp();
//     Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
//     double loglik0 = 1e5;
//     double loglik1;

//     for (int j = 0; j < 50; j++)
//     {
//       for (int i = 0; i < n; i++)
//       {
//         temp.col(i) = X_trans.col(i) * expeta(i) * weights(i);
//       }
//       z = eta + (y - expeta).cwiseQuotient(expeta);
//       beta0 = (temp * X).ldlt().solve(temp * z);
//       eta = X * beta0;
//       for (int i = 0; i <= n - 1; i++)
//       {
//         if (eta(i) < -30.0)
//           eta(i) = -30.0;
//         if (eta(i) > 30.0)
//           eta(i) = 30.0;
//       }
//       expeta = eta.array().exp();
//       loglik1 = (y.cwiseProduct(eta) - expeta).dot(weights);
//       if (abs(loglik0 - loglik1) / abs(0.1 + loglik0) < 1e-6)
//         break;
//       loglik0 = loglik1;
//     }
//     for (int i = 0; i < p; i++)
//       beta(i) = beta0(i + 1);
//     coef0 = beta0(0);
//   }

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     int p = X.cols();
//     int i;
//     vector<int> A(T0);

//     Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
//     Eigen::VectorXd xbeta_exp = X * beta + coef;
//     for (i = 0; i <= n - 1; i++)
//     {
//       if (xbeta_exp(i) > 30.0)
//         xbeta_exp(i) = 30.0;
//       if (xbeta_exp(i) < -30.0)
//         xbeta_exp(i) = -30.0;
//     }
//     xbeta_exp = xbeta_exp.array().exp();

//     Eigen::VectorXd res = y - xbeta_exp;
//     Eigen::VectorXd g(p);
//     Eigen::VectorXd bd;
//     Eigen::MatrixXd Xsquare;
//     for (i = 0; i < p; i++)
//     {
//       g(i) = -res.dot(X.col(i));
//     }
//     Xsquare = X.array().square();
//     Eigen::VectorXd h(p);
//     for (i = 0; i < p; i++)
//     {
//       h(i) = xbeta_exp.dot(Xsquare.col(i));
//     }
//     bd = h.cwiseProduct((beta - g.cwiseQuotient(h)).cwiseAbs2());

//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);
//     max_k(bd, T0, A_out);
//   }
// };

// class PdasCox : public Algorithm
// {
// public:
//   PdasCox(Data &data, int algorithm_type, int max_iter) : Algorithm(data, 4, algorithm_type, max_iter)
//   {
//     this->algorithm_type = algorithm_type;
//   };

//   void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd status, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     int n = X.rows();
//     int p = X.cols();

//     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd theta(n);
//     Eigen::MatrixXd one = (Eigen::MatrixXd::Ones(n, n)).triangularView<Eigen::Upper>();
//     Eigen::MatrixXd x_theta(n, p);
//     Eigen::VectorXd xij_theta(n);
//     Eigen::VectorXd cum_theta(n);
//     Eigen::VectorXd g(p);
//     Eigen::MatrixXd h(p, p);
//     Eigen::VectorXd d(p);
//     double loglik0 = 1e5;
//     double loglik1;

//     double step;
//     int m;
//     int l;
//     for (l = 1; l <= 30; l++)
//     {
//       step = 0.5;
//       m = 1;
//       theta = X * beta0;
//       for (int i = 0; i < n; i++)
//       {
//         if (theta(i) > 30)
//           theta(i) = 30;
//         else if (theta(i) < -30)
//           theta(i) = -30;
//       }
//       theta = theta.array().exp();
//       cum_theta = one * theta;
//       x_theta = X.array().colwise() * theta.array();
//       x_theta = one * x_theta;
//       x_theta = x_theta.array().colwise() / cum_theta.array();
//       g = (X - x_theta).transpose() * (weights.cwiseProduct(status));

//       for (int k1 = 0; k1 < p; k1++)
//       {
//         for (int k2 = k1; k2 < p; k2++)
//         {
//           xij_theta = (theta.cwiseProduct(X.col(k1))).cwiseProduct(X.col(k2));
//           for (int j = n - 2; j >= 0; j--)
//           {
//             xij_theta(j) = xij_theta(j + 1) + xij_theta(j);
//           }
//           h(k1, k2) = -(xij_theta.cwiseQuotient(cum_theta) - x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(status));
//           h(k2, k1) = h(k1, k2);
//         }
//       }
//       d = h.ldlt().solve(g);
//       beta1 = beta0 - pow(step, m) * d;
//       loglik1 = loglik_cox(X, status, beta1, weights);
//       while ((loglik0 > loglik1) && (m < 5))
//       {
//         m = m + 1;
//         beta1 = beta0 - pow(step, m) * d;
//         loglik1 = loglik_cox(X, status, beta1, weights);
//       }
//       if (abs(loglik0 - loglik1) / abs(0.1 + loglik0) < 1e-5)
//       {
//         break;
//       }
//       beta0 = beta1;
//       loglik0 = loglik1;
//     }
//     beta = beta0;
//   }

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     int p = X.cols();
//     Eigen::VectorXd l1 = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd l2 = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd cum_theta = Eigen::VectorXd::Zero(n);
//     Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
//     Eigen::MatrixXd xtheta(n, p);
//     Eigen::MatrixXd x2theta(n, p);
//     vector<int> A(T0);
//     Eigen::VectorXd theta = X * beta;

//     theta = weights.array() * theta.array().exp();
//     cum_theta(n - 1) = theta(n - 1);
//     for (int k = n - 2; k >= 0; k--)
//     {
//       cum_theta(k) = cum_theta(k + 1) + theta(k);
//     }
//     for (int k = 0; k <= p - 1; k++)
//     {
//       xtheta.col(k) = theta.cwiseProduct(X.col(k));
//     }
//     for (int k = 0; k <= p - 1; k++)
//     {
//       x2theta.col(k) = X.col(k).cwiseProduct(xtheta.col(k));
//     }
//     for (int k = n - 2; k >= 0; k--)
//     {
//       xtheta.row(k) = xtheta.row(k + 1) + xtheta.row(k);
//     }
//     for (int k = n - 2; k >= 0; k--)
//     {
//       x2theta.row(k) = x2theta.row(k + 1) + x2theta.row(k);
//     }
//     for (int k = 0; k <= p - 1; k++)
//     {
//       xtheta.col(k) = xtheta.col(k).cwiseQuotient(cum_theta);
//     }
//     for (int k = 0; k <= p - 1; k++)
//     {
//       x2theta.col(k) = x2theta.col(k).cwiseQuotient(cum_theta);
//     }
//     x2theta = x2theta.array() - xtheta.array().square().array();
//     xtheta = X.array() - xtheta.array();
//     for (int k = 0; k < y.size(); k++)
//     {
//       xtheta.row(k) = xtheta.row(k) * y[k];
//       x2theta.row(k) = x2theta.row(k) * y[k];
//     }
//     l1 = -xtheta.adjoint() * weights;

//     l2 = x2theta.adjoint() * weights;

//     d = -l1.cwiseQuotient(l2);
//     bd = beta + d;
//     bd = bd.cwiseAbs();
//     bd = bd.cwiseProduct(l2.cwiseSqrt());
//     bd = bd.array().square();

//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);

//     max_k(bd, T0, A_out);
//   }
// };

// class L0L2Lm : public Algorithm
// {
// public:
//   L0L2Lm(Data &data, int algorithm_type, int max_iter) : Algorithm(data, 1, algorithm_type, max_iter)
//   {
//   }

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     vector<int> A(T0);
//     Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;

//     Eigen::VectorXd d = (X.adjoint() * (y - X * beta - coef) / double(n) - 2 * this->lambda_level * beta) / sqrt(1 + 2 * this->lambda_level);
//     Eigen::VectorXd bd = sqrt(1 + 2 * this->lambda_level) * beta + d;
//     bd = bd.cwiseAbs2();

//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);

//     max_k(bd, T0, A_out);
//   };

//   void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);
//   }
// };

// class L0L2Logistic : public Algorithm
// {
// public:
//   L0L2Logistic(Data &data, int algorithm_type, int max_iter) : Algorithm(data, 2, algorithm_type, max_iter){};

//   void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {

//     int n = x.rows();
//     int p = x.cols();
//     Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
//     X.rightCols(p) = x;
//     Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
//     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
//     Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p + 1);
//     Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
//     Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p + 1, p + 1);
//     lambdamat(0, 0) = 0;
//     Eigen::VectorXd Pi = pi(X, y, beta0, n);
//     Eigen::VectorXd log_Pi = Pi.array().log();
//     Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
//     double loglik0 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
//     Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
//     for (int i = 0; i < n; i++)
//     {
//       if (W(i) < 0.001)
//         W(i) = 0.001;
//     }
//     Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);
//     W = W.cwiseProduct(weights);
//     for (int i = 0; i < n; i++)
//     {
//       X_new.row(i) = X.row(i) * W(i);
//     }
//     beta1 = (2 * this->lambda_level * lambdamat + X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//     double loglik1;

//     int j;
//     for (j = 0; j < 30; j++)
//     {
//       Pi = pi(X, y, beta1, n);
//       log_Pi = Pi.array().log();
//       log_1_Pi = (one - Pi).array().log();
//       loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
//       if (abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < 1e-6)
//       {
//         break;
//       }
//       beta0 = beta1;
//       loglik0 = loglik1;
//       W = Pi.cwiseProduct(one - Pi);
//       for (int i = 0; i < n; i++)
//       {
//         if (W(i) < 0.001)
//           W(i) = 0.001;
//       }
//       Z = X * beta0 + (y - Pi).cwiseQuotient(W);
//       W = W.cwiseProduct(weights);
//       for (int i = 0; i < n; i++)
//       {
//         X_new.row(i) = X.row(i) * W(i);
//       }
//       beta1 = (2 * this->lambda_level * lambdamat + X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//     }
//     for (int i = 0; i < p; i++)
//       beta(i) = beta0(i + 1);
//     coef0 = beta0(0);
//   }

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     int p = X.cols();
//     Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
//     vector<int> A(T0);
//     Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
//     Eigen::VectorXd xbeta_exp = X * beta + coef;
//     for (int i = 0; i <= n - 1; i++)
//     {
//       if (xbeta_exp(i) > 25.0)
//         xbeta_exp(i) = 25.0;
//       if (xbeta_exp(i) < -25.0)
//         xbeta_exp(i) = -25.0;
//     }
//     xbeta_exp = xbeta_exp.array().exp();
//     Eigen::VectorXd pr = xbeta_exp.array() / (xbeta_exp + one).array();
//     Eigen::VectorXd l1 = -X.adjoint() * ((y - pr).cwiseProduct(weights)) + 2 * this->lambda_level * beta;
//     X = X.array().square();
//     Eigen::VectorXd l2 = (X.adjoint()) * ((pr.cwiseProduct(one - pr)).cwiseProduct(weights)) + 2 * this->lambda_level * Eigen::MatrixXd::Ones(p, 1);
//     Eigen::VectorXd d = -l1.cwiseQuotient(l2);
//     bd = (beta + d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);
//     max_k(bd, T0, A_out);
//   }
// };

// class L0L2Poisson : public Algorithm
// {
// public:
//   L0L2Poisson(Data &data, int algorithm_type, unsigned int max_iter = 20) : Algorithm(data, 3, algorithm_type, max_iter)
//   {
//     this->model_fit_max = model_fit_max;
//   };

//   void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     int n = x.rows();
//     int p = x.cols();
//     Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
//     X.rightCols(p) = x;
//     Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p + 1, p + 1);
//     lambdamat(0, 0) = 0;
//     Eigen::MatrixXd X_trans = X.transpose();
//     Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(p + 1, n);
//     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
//     beta0.tail(p) = beta;
//     beta0(0) = coef0;
//     Eigen::VectorXd eta = X * beta0;
//     Eigen::VectorXd expeta = eta.array().exp();
//     Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
//     double loglik0 = 1e5;
//     double loglik1;

//     for (int j = 0; j < 50; j++)
//     {
//       for (int i = 0; i < n; i++)
//       {
//         temp.col(i) = X_trans.col(i) * expeta(i) * weights(i);
//       }
//       z = eta + (y - expeta).cwiseQuotient(expeta);
//       beta0 = (2 * this->lambda_level * lambdamat + temp * X).ldlt().solve(temp * z);
//       eta = X * beta0;
//       for (int i = 0; i <= n - 1; i++)
//       {
//         if (eta(i) < -30.0)
//           eta(i) = -30.0;
//         if (eta(i) > 30.0)
//           eta(i) = 30.0;
//       }
//       expeta = eta.array().exp();
//       loglik1 = (y.cwiseProduct(eta) - expeta).dot(weights);
//       if (abs(loglik0 - loglik1) / abs(0.1 + loglik0) < 1e-6)
//         break;
//       loglik0 = loglik1;
//     }
//     for (int i = 0; i < p; i++)
//       beta(i) = beta0(i + 1);
//     coef0 = beta0(0);
//   }

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {

//     int n = X.rows();
//     int p = X.cols();
//     int i;

//     vector<int> A(T0);

//     Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
//     Eigen::VectorXd xbeta_exp = X * beta + coef;
//     for (i = 0; i <= n - 1; i++)
//     {
//       if (xbeta_exp(i) > 30.0)
//         xbeta_exp(i) = 30.0;
//       if (xbeta_exp(i) < -30.0)
//         xbeta_exp(i) = -30.0;
//     }
//     xbeta_exp = xbeta_exp.array().exp();

//     Eigen::VectorXd res = y - xbeta_exp;
//     Eigen::VectorXd l1(p);
//     Eigen::VectorXd bd;
//     Eigen::MatrixXd Xsquare;
//     for (i = 0; i < p; i++)
//     {
//       l1(i) = -res.dot(X.col(i)) + 2 * this->lambda_level * beta(i);
//     }

//     Xsquare = X.array().square();

//     Eigen::VectorXd l2(p);
//     for (i = 0; i < p; i++)
//     {
//       l2(i) = xbeta_exp.dot(Xsquare.col(i)) + 2 * this->lambda_level;
//     }

//     Eigen::VectorXd d = -l1.cwiseQuotient(l2);

//     bd = (beta + d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());

//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);

//     max_k(bd, T0, A_out);
//   }
// };

// class L0L2Cox : public Algorithm
// {
// public:
//   L0L2Cox(Data &data, int algorithm_type, unsigned int max_iter = 20) : Algorithm(data, 4, algorithm_type, max_iter)
//   {
//     this->model_fit_max = model_fit_max;
//   };

//   void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd status, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     int n = X.rows();
//     int p = X.cols();

//     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p);

//     Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p, p);
//     Eigen::VectorXd eta(n);
//     Eigen::VectorXd expeta(n);
//     Eigen::VectorXd cum_expeta(n);
//     Eigen::MatrixXd x_theta(n, p);
//     Eigen::VectorXd xij_theta(n);
//     Eigen::VectorXd g(p);
//     Eigen::MatrixXd h(p, p);
//     Eigen::VectorXd d(p);
//     double loglik0;
//     double loglik1;

//     double step;
//     int m;
//     int l;
//     for (l = 1; l <= 50; l++)
//     {
//       step = 0.5;
//       m = 1;
//       eta = X * beta0;
//       for (int i = 0; i < n; i++)
//       {
//         if (eta(i) > 30)
//         {
//           eta(i) = 30;
//         }
//         else if (eta(i) < -30)
//         {
//           eta(i) = -30;
//         }
//       }
//       expeta = eta.array().exp();
//       cum_expeta(n - 1) = expeta(n - 1);
//       for (int i = n - 2; i >= 0; i--)
//       {
//         cum_expeta(i) = cum_expeta(i + 1) + expeta(i);
//       }
//       for (int i = 0; i < p; i++)
//       {
//         x_theta.col(i) = X.col(i).cwiseProduct(expeta);
//       }
//       for (int i = n - 2; i >= 0; i--)
//       {
//         x_theta.row(i) = x_theta.row(i) + x_theta.row(i + 1);
//       }
//       for (int i = 0; i < p; i++)
//       {
//         x_theta.col(i) = x_theta.col(i).cwiseQuotient(cum_expeta);
//       }
//       g = (X - x_theta).transpose() * (weights.cwiseProduct(status)) + 2 * this->lambda_level * beta;
//       for (int k1 = 0; k1 < p; k1++)
//       {
//         for (int k2 = k1; k2 < p; k2++)
//         {
//           xij_theta = (expeta.cwiseProduct(X.col(k1))).cwiseProduct(X.col(k2));
//           for (int j = n - 2; j >= 0; j--)
//           {
//             xij_theta(j) = xij_theta(j + 1) + xij_theta(j);
//           }
//           h(k1, k2) = -(xij_theta.cwiseQuotient(cum_expeta) - x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(status));
//           h(k2, k1) = h(k1, k2);
//         }
//       }
//       h = h + this->lambda_level * lambdamat;
//       d = h.ldlt().solve(g);
//       beta1 = beta0 - pow(step, m) * d;
//       loglik0 = loglik_cox(X, status, beta0, weights);
//       loglik1 = loglik_cox(X, status, beta1, weights);
//       while ((loglik0 >= loglik1) && (m < 10))
//       {
//         m = m + 1;
//         beta1 = beta0 - pow(step, m) * d;
//         loglik1 = loglik_cox(X, status, beta1, weights);
//       }

//       beta0 = beta1;
//       if (abs(loglik0 - loglik1) / abs(loglik0) < 1e-8)
//       {
//         break;
//       }
//     }

//     beta = beta0;
//   }

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     int p = X.cols();

//     Eigen::VectorXd l1 = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd l2 = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd cum_theta = Eigen::VectorXd::Zero(n);
//     Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
//     Eigen::MatrixXd xtheta(n, p);
//     Eigen::MatrixXd x2theta(n, p);

//     vector<int> A(T0);

//     Eigen::VectorXd theta = X * beta;
//     for (int i = 0; i <= n - 1; i++)
//     {
//       if (theta(i) > 25.0)
//         theta(i) = 25.0;
//       if (theta(i) < -25.0)
//         theta(i) = -25.0;
//     }
//     theta = weights.array() * theta.array().exp();
//     cum_theta(n - 1) = theta(n - 1);
//     for (int k = n - 2; k >= 0; k--)
//     {
//       cum_theta(k) = cum_theta(k + 1) + theta(k);
//     }
//     for (int k = 0; k <= p - 1; k++)
//     {
//       xtheta.col(k) = theta.cwiseProduct(X.col(k));
//     }
//     for (int k = 0; k <= p - 1; k++)
//     {
//       x2theta.col(k) = X.col(k).cwiseProduct(xtheta.col(k));
//     }
//     for (int k = n - 2; k >= 0; k--)
//     {
//       xtheta.row(k) = xtheta.row(k + 1) + xtheta.row(k);
//     }
//     for (int k = n - 2; k >= 0; k--)
//     {
//       x2theta.row(k) = x2theta.row(k + 1) + x2theta.row(k);
//     }
//     for (int k = 0; k <= p - 1; k++)
//     {
//       xtheta.col(k) = xtheta.col(k).cwiseQuotient(cum_theta);
//     }
//     for (int k = 0; k <= p - 1; k++)
//     {
//       x2theta.col(k) = x2theta.col(k).cwiseQuotient(cum_theta);
//     }
//     x2theta = x2theta.array() - xtheta.array().square().array();

//     xtheta = X.array() - xtheta.array();

//     for (unsigned int k = 0; k < y.size(); k++)
//     {
//       if (y[k] == 0.0)
//       {
//         xtheta.row(k) = Eigen::VectorXd::Zero(p);
//         x2theta.row(k) = Eigen::VectorXd::Zero(p);
//       }
//     }
//     l1 = -xtheta.adjoint() * weights + 2 * this->lambda_level * beta;
//     l2 = x2theta.adjoint() * weights + 2 * this->lambda_level * Eigen::MatrixXd::Ones(p, 1);
//     d = -l1.cwiseQuotient(l2);

//     bd = beta + d;

//     bd = bd.cwiseAbs();
//     bd = bd.cwiseProduct(l2.cwiseSqrt());

//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);

//     max_k(bd, T0, A_out);
//   }
// };

// class GroupPdasLm : public Algorithm
// {
// public:
//   GroupPdasLm(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 1, algorithm_type, max_iter){};

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     int p = X.cols();
//     vector<Eigen::MatrixXd> PhiG = Phi(X, index, gsize, n, p, N, this->lambda_level, this->group_XTX);
//     vector<Eigen::MatrixXd> invPhiG = invPhi(PhiG, N);

//     Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd bd = Eigen::VectorXd::Zero(N);
//     Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
//     Eigen::VectorXd d = X.adjoint() * (y - X * beta - coef) / double(n) - 2 * this->lambda_level * beta;
//     vector<int> A(T0, -1);

//     for (int i = 0; i < N; i++)
//     {
//       Eigen::MatrixXd phiG = PhiG[i];
//       Eigen::MatrixXd invphiG = invPhiG[i];
//       betabar.segment(index(i), gsize(i)) = phiG * beta.segment(index(i), gsize(i));
//       dbar.segment(index(i), gsize(i)) = invphiG * d.segment(index(i), gsize(i));
//     }
//     Eigen::VectorXd temp = betabar + dbar;
//     for (int i = 0; i < N; i++)
//     {
//       bd(i) = (temp.segment(index(i), gsize(i))).squaredNorm() / gsize(i);
//     }

//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);

//     max_k(bd, T0, A_out);
//   };

//   void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {

//     beta = (X.adjoint() * X + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);
//   };
// };

// class GroupPdasLogistic : public Algorithm
// {
// public:
//   GroupPdasLogistic(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 2, algorithm_type, max_iter)
//   {
//     this->algorithm_type = algorithm_type;
//   };

//   void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     int n = x.rows();
//     int p = x.cols();
//     Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
//     X.rightCols(p) = x;
//     Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
//     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
//     Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p + 1);
//     Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
//     Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p + 1, p + 1);
//     lambdamat(0, 0) = 0;
//     Eigen::VectorXd Pi = pi(X, y, beta0, n);
//     Eigen::VectorXd log_Pi = Pi.array().log();
//     Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
//     double loglik0 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
//     Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
//     Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);
//     W = W.cwiseProduct(weights);
//     for (int i = 0; i < n; i++)
//     {
//       X_new.row(i) = X.row(i) * W(i);
//     }
//     beta1 = (2 * this->lambda_level * lambdamat + X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//     double loglik1;

//     int j;
//     for (j = 0; j < 30; j++)
//     {
//       Pi = pi(X, y, beta1, n);
//       log_Pi = Pi.array().log();
//       log_1_Pi = (one - Pi).array().log();
//       loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
//       if (abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < 1e-6)
//       {
//         break;
//       }
//       beta0 = beta1;
//       loglik0 = loglik1;
//       W = Pi.cwiseProduct(one - Pi);
//       for (int i = 0; i < n; i++)
//       {
//         if (W(i) < 0.001)
//           W(i) = 0.001;
//       }
//       Z = X * beta0 + (y - Pi).cwiseQuotient(W);
//       W = W.cwiseProduct(weights);
//       for (int i = 0; i < n; i++)
//       {
//         X_new.row(i) = X.row(i) * W(i);
//       }
//       beta1 = (2 * this->lambda_level * lambdamat + X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//     }
//     for (int i = 0; i < p; i++)
//       beta(i) = beta0(i + 1);
//     coef0 = beta0(0);
//   }

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     int p = X.cols();
//     Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
//     Eigen::VectorXd bd = Eigen::VectorXd::Zero(N);
//     Eigen::VectorXd eta(n);
//     Eigen::VectorXd expeta(n);
//     Eigen::VectorXd pr(n);
//     Eigen::VectorXd g(n);
//     Eigen::VectorXd h(n);
//     Eigen::VectorXd d(n);
//     vector<int> A(T0, -1);

//     eta = X * beta + one * coef0;
//     for (int i = 0; i < n; i++)
//     {

//       if (eta(i) > 30)
//         eta(i) = 30;
//       else if (eta(i) < -30)
//         eta(i) = -30;
//     }
//     expeta = eta.array().exp();
//     pr = expeta.array() / (expeta + one).array();
//     g = weights.array() * (y - pr).array();
//     h = weights.array() * pr.array() * (one - pr).array();
//     d = X.transpose() * g - 2 * this->lambda_level * beta;

//     for (int i = 0; i < N; i++)
//     {
//       Eigen::MatrixXd XG = X.middleCols(index(i), gsize(i));
//       Eigen::MatrixXd XG_new = XG;
//       for (int j = 0; j < n; j++)
//       {
//         XG_new.row(j) = XG.row(j) * h(j);
//       }
//       Eigen::MatrixXd XGbar = XG_new.transpose() * XG + 2 * this->lambda_level * Eigen::MatrixXd::Identity(gsize(i), gsize(i));
//       Eigen::MatrixXd phiG(gsize(i), gsize(i));
//       XGbar.sqrt().evalTo(phiG);
//       Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(gsize(i), gsize(i)));
//       betabar.segment(index(i), gsize(i)) = phiG * beta.segment(index(i), gsize(i));
//       dbar.segment(index(i), gsize(i)) = invphiG * d.segment(index(i), gsize(i));
//     }
//     Eigen::VectorXd temp = betabar + dbar;
//     for (int i = 0; i < N; i++)
//     {
//       bd(i) = (temp.segment(index(i), gsize(i))).squaredNorm() / gsize(i);
//     }

//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);

//     max_k(bd, T0, A_out);
//   };
// };

// class GroupPdasPoisson : public Algorithm
// {
// public:
//   GroupPdasPoisson(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 3, algorithm_type, max_iter){};

//   void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     int n = x.rows();
//     int p = x.cols();
//     Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
//     X.rightCols(p) = x;
//     Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p + 1, p + 1);
//     lambdamat(0, 0) = 0;
//     Eigen::MatrixXd X_trans = X.transpose();
//     Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(p + 1, n);
//     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
//     beta0.tail(p) = beta;
//     beta0(0) = coef0;
//     Eigen::VectorXd eta = X * beta0;
//     Eigen::VectorXd expeta = eta.array().exp();
//     Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
//     double loglik0 = 1e5;
//     double loglik1;

//     for (int j = 0; j < 50; j++)
//     {
//       for (int i = 0; i < n; i++)
//       {
//         temp.col(i) = X_trans.col(i) * expeta(i) * weights(i);
//       }
//       z = eta + (y - expeta).cwiseQuotient(expeta);
//       beta0 = (2 * this->lambda_level * lambdamat + temp * X).ldlt().solve(temp * z);
//       eta = X * beta0;
//       for (int i = 0; i <= n - 1; i++)
//       {
//         if (eta(i) < -30.0)
//           eta(i) = -30.0;
//         else if (eta(i) > 30.0)
//           eta(i) = 30.0;
//       }
//       expeta = eta.array().exp();
//       for (int i = 0; i < n; i++)
//       {
//         if (expeta(i) < 0.001)
//           expeta(i) = 0.001;
//       }
//       loglik1 = (y.cwiseProduct(eta) - expeta).dot(weights);
//       if (abs(loglik0 - loglik1) / abs(0.1 + loglik0) < 1e-6)
//         break;
//       loglik0 = loglik1;
//     }
//     for (int i = 0; i < p; i++)
//       beta(i) = beta0(i + 1);
//     coef0 = beta0(0);
//   }

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     int p = X.cols();
//     Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd bd = Eigen::VectorXd::Zero(N);
//     Eigen::VectorXd eta(n);
//     Eigen::VectorXd expeta(n);
//     Eigen::VectorXd g(n);
//     Eigen::VectorXd d(n);
//     vector<int> A(T0, -1);

//     eta = X * beta + Eigen::VectorXd::Ones(n) * coef0;
//     expeta = eta.array().exp();
//     g = (y - expeta).cwiseProduct(weights);
//     d = X.transpose() * g - 2 * this->lambda_level * beta;
//     for (int i = 0; i < N; i++)
//     {
//       Eigen::MatrixXd XG = X.middleCols(index(i), gsize(i));
//       Eigen::MatrixXd XG_new = XG;
//       for (int j = 0; j < n; j++)
//       {
//         XG_new.row(j) = XG.row(j) * expeta(j) * weights(j);
//       }
//       Eigen::MatrixXd XGbar = XG_new.transpose() * XG + 2 * this->lambda_level * Eigen::MatrixXd::Identity(gsize(i), gsize(i));
//       Eigen::MatrixXd phiG(gsize(i), gsize(i));
//       XGbar.sqrt().evalTo(phiG);
//       Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(gsize(i), gsize(i)));
//       betabar.segment(index(i), gsize(i)) = phiG * beta.segment(index(i), gsize(i));
//       dbar.segment(index(i), gsize(i)) = invphiG * d.segment(index(i), gsize(i));
//     }
//     Eigen::VectorXd temp = betabar + dbar;
//     for (int i = 0; i < N; i++)
//     {
//       bd(i) = (temp.segment(index(i), gsize(i)).squaredNorm()) / gsize(i);
//     }

//     // keep always_select in active_set
//     slice_assignment(bd, always_select, DBL_MAX);

//     max_k(bd, T0, A_out);
//   };
// };

// class GroupPdasCox : public Algorithm
// {
// public:
//   GroupPdasCox(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 4, algorithm_type, max_iter){};

//   void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd status, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
//   {
//     // to be ensure
//     int n = X.rows();
//     int p = X.cols();
//     Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p, p);
//     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p);
//     Eigen::VectorXd theta(n);
//     Eigen::MatrixXd one = (Eigen::MatrixXd::Ones(n, n)).triangularView<Eigen::Upper>();
//     Eigen::MatrixXd x_theta(n, p);
//     Eigen::VectorXd xij_theta(n);
//     Eigen::VectorXd cum_theta(n);
//     Eigen::VectorXd g(p);
//     Eigen::MatrixXd h(p, p);
//     Eigen::VectorXd d(p);
//     double loglik0 = 1e5;
//     double loglik1;

//     // New version:
//     // vector<vector<Eigen::VectorXd> > XiXj(p);
//     // vector<Eigen::VectorXd> XiXj_tmp(p);
//     // for(int k1=0;k1<p;k1++)
//     // {
//     //   for(int k2=k1;k2<p;k2++)
//     //   {
//     //     XiXj_tmp[k2] = X.col(k1).cwiseProduct(X.col(k2));
//     //   }
//     //   XiXj[k1] = XiXj_tmp;
//     // }

//     double step;
//     int m;
//     int l;
//     for (l = 1; l <= 30; l++)
//     {
//       step = 0.5;
//       m = 1;
//       theta = X * beta0;
//       for (int i = 0; i < n; i++)
//       {
//         if (theta(i) > 30)
//           theta(i) = 30;
//         else if (theta(i) < -30)
//           theta(i) = -30;
//       }
//       theta = theta.array().exp();

//       cum_theta = one * theta;
//       x_theta = X.array().colwise() * theta.array();
//       x_theta = one * x_theta; // O(np^2) flow, can be improved
//       x_theta = x_theta.array().colwise() / cum_theta.array();
//       g = (X - x_theta).transpose() * (weights.cwiseProduct(status)) + 2 * this->lambda_level * beta0;

//       // // New version:
//       // int sub_n = sub_ind.size();
//       // Eigen::VectorXd sub_x_theta = row_slice(x_theta, sub_ind);
//       // // Eigen::VectorXd sub_theta = slice(theta, sub_ind);
//       // Eigen::VectorXd sub_cum_theta = slice(cum_theta, sub_ind);
//       // Eigen::VectorXd sub_weights = slice(weights, sub_ind);

//       // clock_t t1 = clock();
//       // for (int k1 = 0; k1 < p; k1++)
//       // {
//       //   for (int k2 = k1; k2 < p; k2++)
//       //   {
//       //     xij_theta = theta.cwiseProduct(XiXj[k1][k2]);
//       //     for (int j = n - 2; j >= 0; j--)
//       //     {
//       //       xij_theta(j) = xij_theta(j + 1) + xij_theta(j);
//       //     }
//       //     // Eigen::VectorXd sub_xij_theta = slice(xij_theta, sub_ind);
//       //     // h(k1, k2) = -(sub_xij_theta.cwiseQuotient(sub_cum_theta) - sub_x_theta.col(k1).cwiseProduct(sub_x_theta.col(k2))).dot(sub_weights);
//       //     h(k1, k2) = -(xij_theta.cwiseQuotient(cum_theta) - x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(status));
//       //     h(k2, k1) = h(k1, k2);
//       //   }
//       // }
//       // clock_t t2 = clock();
//       // printf("new npp time=%f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

//       // Old version:
//       for (int k1 = 0; k1 < p; k1++)
//       {
//         for (int k2 = k1; k2 < p; k2++)
//         {
//           xij_theta = (theta.cwiseProduct(X.col(k1))).cwiseProduct(X.col(k2));
//           for (int j = n - 2; j >= 0; j--)
//           {
//             xij_theta(j) = xij_theta(j + 1) + xij_theta(j);
//           }
//           h(k1, k2) = -(xij_theta.cwiseQuotient(cum_theta) - x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(status));
//           h(k2, k1) = h(k1, k2);
//         }
//       }

//       h = h + 2 * this->lambda_level * lambdamat;
//       d = h.ldlt().solve(g);
//       beta1 = beta0 - pow(step, m) * d;
//       loglik1 = loglik_cox(X, status, beta1, weights);
//       while ((loglik0 > loglik1) && (m < 5))
//       {
//         m = m + 1;
//         beta1 = beta0 - pow(step, m) * d;
//         loglik1 = loglik_cox(X, status, beta1, weights);
//       }
//       if (abs(loglik0 - loglik1) / abs(0.1 + loglik0) < 1e-5)
//       {
//         break;
//       }
//       beta0 = beta1;
//       loglik0 = loglik1;
//     }
//     beta = beta0;
//   }

//   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
//              Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
//   {
//     int n = X.rows();
//     int p = X.cols();
//     if (this->algorithm_type == 2 || this->algorithm_type == 3)
//     {
//       Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
//       Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
//       Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
//       Eigen::MatrixXd h(n, n);
//       Eigen::VectorXd cum_theta(n);
//       Eigen::VectorXd cum_theta2(n);
//       Eigen::VectorXd cum_theta3(n);
//       Eigen::VectorXd theta(n);
//       Eigen::VectorXd d(p);
//       Eigen::VectorXd g(n);
//       vector<int> A(T0, -1);

//       theta = X * beta;
//       for (int i = 0; i <= n - 1; i++)
//       {
//         if (theta(i) > 30.0)
//           theta(i) = 30.0;
//         if (theta(i) < -30.0)
//           theta(i) = -30.0;
//       }
//       theta = weights.array() * theta.array().exp();
//       cum_theta(n - 1) = theta(n - 1);
//       for (int k = n - 2; k >= 0; k--)
//       {
//         cum_theta(k) = cum_theta(k + 1) + theta(k);
//       }
//       cum_theta2(0) = (y(0) * weights(0)) / cum_theta(0);
//       for (int k = 1; k <= n - 1; k++)
//       {
//         cum_theta2(k) = (y(k) * weights(k)) / cum_theta(k) + cum_theta2(k - 1);
//       }
//       cum_theta3(0) = (y(0) * weights(0)) / pow(cum_theta(0), 2);
//       for (int k = 1; k <= n - 1; k++)
//       {
//         cum_theta3(k) = (y(k) * weights(k)) / pow(cum_theta(k), 2) + cum_theta3(k - 1);
//       }
//       h = -cum_theta3.replicate(1, n);
//       h = h.cwiseProduct(theta.replicate(1, n));
//       h = h.cwiseProduct(theta.replicate(1, n).transpose());
//       for (int i = 0; i < n; i++)
//       {
//         for (int j = i + 1; j < n; j++)
//         {
//           h(j, i) = h(i, j);
//         }
//       }
//       h.diagonal() = cum_theta2.cwiseProduct(theta) + h.diagonal();
//       g = weights.cwiseProduct(y) - cum_theta2.cwiseProduct(theta);
//       d = X.transpose() * g - 2 * this->lambda_level * beta;
//       for (int i = 0; i < N; i++)
//       {
//         Eigen::MatrixXd XG = X.middleCols(index(i), gsize(i));
//         Eigen::MatrixXd XGbar = XG.transpose() * h * XG + 2 * this->lambda_level * Eigen::MatrixXd::Identity(gsize(i), gsize(i));
//         Eigen::MatrixXd phiG(gsize(i), gsize(i));
//         XGbar.sqrt().evalTo(phiG);
//         Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(gsize(i), gsize(i)));
//         betabar.segment(index(i), gsize(i)) = phiG * beta.segment(index(i), gsize(i));
//         dbar.segment(index(i), gsize(i)) = invphiG * d.segment(index(i), gsize(i));
//       }
//       Eigen::VectorXd temp = betabar + dbar;
//       for (int i = 0; i < N; i++)
//       {
//         bd(i) = (temp.segment(index(i), gsize(i))).squaredNorm() / (gsize(i));
//       }

//       // keep always_select in active_set
//       slice_assignment(bd, always_select, DBL_MAX);

//       max_k(bd, T0, A_out);
//     }
//     else if (this->algorithm_type == 1 || this->algorithm_type == 5)
//     {
//       Eigen::VectorXd l1 = Eigen::VectorXd::Zero(p);
//       Eigen::VectorXd l2 = Eigen::VectorXd::Zero(p);
//       Eigen::VectorXd cum_theta = Eigen::VectorXd::Zero(n);
//       Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
//       Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
//       Eigen::MatrixXd xtheta(n, p);
//       Eigen::MatrixXd x2theta(n, p);
//       vector<int> A(T0);
//       Eigen::VectorXd theta = X * beta;
//       for (int i = 0; i <= n - 1; i++)
//       {
//         if (theta(i) > 30.0)
//           theta(i) = 30.0;
//         if (theta(i) < -30.0)
//           theta(i) = -30.0;
//       }
//       theta = weights.array() * theta.array().exp();
//       cum_theta(n - 1) = theta(n - 1);
//       for (int k = n - 2; k >= 0; k--)
//       {
//         cum_theta(k) = cum_theta(k + 1) + theta(k);
//       }
//       for (int k = 0; k <= p - 1; k++)
//       {
//         xtheta.col(k) = theta.cwiseProduct(X.col(k));
//       }
//       for (int k = 0; k <= p - 1; k++)
//       {
//         x2theta.col(k) = X.col(k).cwiseProduct(xtheta.col(k));
//       }
//       for (int k = n - 2; k >= 0; k--)
//       {
//         xtheta.row(k) = xtheta.row(k + 1) + xtheta.row(k);
//       }
//       for (int k = n - 2; k >= 0; k--)
//       {
//         x2theta.row(k) = x2theta.row(k + 1) + x2theta.row(k);
//       }
//       for (int k = 0; k <= p - 1; k++)
//       {
//         xtheta.col(k) = xtheta.col(k).cwiseQuotient(cum_theta);
//       }
//       for (int k = 0; k <= p - 1; k++)
//       {
//         x2theta.col(k) = x2theta.col(k).cwiseQuotient(cum_theta);
//       }
//       x2theta = x2theta.array() - xtheta.array().square().array();

//       xtheta = X.array() - xtheta.array();

//       for (unsigned int k = 0; k < y.size(); k++)
//       {
//         if (y[k] == 0.0)
//         {
//           xtheta.row(k) = Eigen::VectorXd::Zero(p);
//           x2theta.row(k) = Eigen::VectorXd::Zero(p);
//         }
//       }
//       l1 = -xtheta.adjoint() * weights + 2 * this->lambda_level * beta;
//       l2 = x2theta.adjoint() * weights + 2 * this->lambda_level * Eigen::MatrixXd::Ones(p, 1);
//       d = -l1.cwiseQuotient(l2);
//       bd = beta + d;
//       bd = bd.cwiseAbs();
//       bd = bd.cwiseProduct(l2.cwiseSqrt());

//       // keep always_select in active_set
//       slice_assignment(bd, always_select, DBL_MAX);

//       max_k(bd, T0, A_out);
//     }
//     else
//     {
// #ifdef R_BUILD
//       Rcpp::Rcout << "algorithm can not be " << this->algorithm_type << endl;
// #else
//       cout << "algorithm can not be " << this->algorithm_type << endl;
// #endif
//     }
//   };
// };

#endif //SRC_ALGORITHM_H
