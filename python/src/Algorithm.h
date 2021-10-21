/*****************************************************************************
*  OpenST Basic tool library                                                 *
*  Copyright (C) 2021 Kangkang Jiang  jiangkk3@mail2.sysu.edu.cn                         *
*                                                                            *
*  This file is part of OST.                                                 *
*                                                                            *
*  This program is free software; you can redistribute it and/or modify      *
*  it under the terms of the GNU General Public License version 3 as         *
*  published by the Free Software Foundation.                                *
*                                                                            *
*  You should have received a copy of the GNU General Public License         *
*  along with OST. If not, see <http://www.gnu.org/licenses/>.               *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*  @file     Algorithm.h                                                     *
*  @brief    the algorithm for fitting.                                      *
*                                                                            *
*                                                                            *
*  @author   Jin Zhu, Kangkang Jiang, Junhao Huang                           *
*  @email    zhuj37@mail2.sysu.edu.cn, jiangkk3@mail2.sysu.edu.cn            *
*  @version  0.0.1                                                           *
*  @date     2021-07-31                                                      *
*  @license  GNU General Public License (GPL)                                *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         : Description                                              *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2021/07/31 | 0.0.1     | Kangkang Jiang | First version                   *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/

#ifndef SRC_ALGORITHM_H
#define SRC_ALGORITHM_H

#ifndef R_BUILD
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Eigen>
#endif

#include "Data.h"
#include "utilities.h"
#include <iostream>
#include <cfloat>

using namespace std;

bool quick_sort_pair_max(std::pair<int, double> x, std::pair<int, double> y);

/**
 * @brief Variable select based on splicing algorithm.
 * T1 for y, XTy, XTone
 * T2 for beta
 * T3 for coef0
 * T4 for X
 * <Eigen::VectorXd, Eigen::VectorXd, double, Eigen::MatrixXd> for Univariate Dense
 * <Eigen::VectorXd, Eigen::VectorXd, double, Eigen::SparseMatrix<double> > for Univariate Sparse
 * <Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> for Multivariable Dense
 * <Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::SparseMatrix<double> > for Multivariable Sparse
 */

template <class T1, class T2, class T3, class T4, class T5>
class Algorithm
{
public:
  int model_fit_max;  /* Maximum number of iterations taken for the primary model fitting. */
  int model_type;     /* primary model type. */
  int algorithm_type; /* algorithm type. */

  int group_df = 0;        /* freedom */
  int sparsity_level = 0;  /* Number of non-zero coefficients. */
  double lambda_level = 0; /* l2 normalization coefficients. */
  // Eigen::VectorXi train_mask;
  int max_iter;     /* Maximum number of iterations taken for the splicing algorithm to converge.  */
  int exchange_num; /* Max exchange variable num. */
  bool warm_start;  /* When tuning the optimal parameter combination, whether to use the last solution as a warm start to accelerate the iterative convergence of the splicing algorithm.*/
  T4 *x = NULL;
  T1 *y = NULL;
  T2 beta;                 /* coefficients. */
  T5 bd;      /* sacrifices. */
  T3 coef0;                /* intercept. */
  double train_loss = 0.;  /* train loss. */
  T2 beta_init;            /* initialization coefficients. */
  T3 coef0_init;           /* initialization intercept. */
  Eigen::VectorXi A_init;  /* initialization active set. */
  Eigen::VectorXi I_init;  /* initialization inactive set. */
  T5 bd_init; /* initialization bd vector. */

  Eigen::VectorXi A_out; /* final active set. */
  Eigen::VectorXi I_out; /* final active set. */

  Eigen::Matrix<Eigen::MatrixXd, -1, -1> PhiG;    /* PhiG for linear model. */
  Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhiG; /* invPhiG for linear model. */
  Eigen::Matrix<T4, -1, -1> group_XTX;            /* XTX. */
  bool lambda_change;                             /* lambda_change or not. */

  Eigen::VectorXi always_select;    /* always select variable. */
  double tau;                       /* algorithm stop threshold */
  int primary_model_fit_max_iter;   /* The maximal number of iteration for primaty model fit*/
  double primary_model_fit_epsilon; /* The epsilon (threshold) of iteration for primaty model fit*/
  bool approximate_Newton;          /* use approximate Newton method or not. */

  T2 beta_warmstart;  /*warmstart beta.*/
  T3 coef0_warmstart; /*warmstart intercept.*/

  Eigen::VectorXi status;

  Eigen::MatrixXd cox_hessian; /* hessian matrix for cox model. */
  Eigen::VectorXd cox_g;       /* score function for cox model. */

  bool covariance_update;              /* use covairance update mathod or not. */
  Eigen::VectorXd **covariance = NULL; /* covairance matrix. */
  bool *covariance_update_flag = NULL; /* each variable have updated in covairance matirx. */
  T1 XTy;                              /*X.transpose() * y */
  T1 XTone;                            /* X.transpose() * Eigen::MatrixXd::one() */

  double effective_number; /* effective number of parameter. */

  int splicing_type;     /* exchange number update mathod. */
  Eigen::MatrixXd Sigma; /* covariance matrix for pca*/

  int sub_search; /* size of sub_searching in splicing */
  int U_size;

  int pca_n = -1;
  long int ising_n = 0;

  Eigen::MatrixXi map1; /* single ind -> full ind */
  Eigen::MatrixXi map2; /* full ind -> single ind */

  T1 XTy_U;
  T1 XTone_U;
  Eigen::Matrix<Eigen::MatrixXd, -1, -1> PhiG_U;
  Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhiG_U;

  Algorithm() = default;

  virtual ~Algorithm(){};

  Algorithm(int algorithm_type, int model_type, int max_iter = 100, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = false, int splicing_type = 0, int sub_search = 0)
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
    this->sub_search = sub_search;
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

  void update_bd_init(T5 &bd_init) { this->bd_init = bd_init; }

  void update_coef0_init(T3 coef0) { this->coef0_init = coef0; }

  void update_group_df(int group_df) { this->group_df = group_df; }

  void update_sparsity_level(int sparsity_level) { this->sparsity_level = sparsity_level; }

  void update_lambda_level(double lambda_level)
  {
    this->lambda_change = this->lambda_level != lambda_level;
    this->lambda_level = lambda_level;
  }

  void update_train_mask(Eigen::VectorXi &train_mask) { this->train_mask = train_mask; }

  void update_exchange_num(int exchange_num) { this->exchange_num = exchange_num; }

  void update_group_XTX(Eigen::Matrix<T4, -1, -1> &group_XTX) { this->group_XTX = group_XTX; }

  virtual void update_tau(int train_n, int N)
  {
    if (train_n == 1)
    {
      this->tau = 0.0;
    }
    else
    {
      this->tau = 0.01 * (double)this->sparsity_level * log((double)N) * log(log((double)train_n)) / (double)train_n;
    }
  }

  bool get_warm_start() { return this->warm_start; }

  double get_train_loss() { return this->train_loss; }

  int get_group_df() { return this->group_df; }

  double get_effective_number() { return this->effective_number; }

  int get_sparsity_level() { return this->sparsity_level; }

  T2 get_beta() { return this->beta; }

  T3 get_coef0() { return this->coef0; }

  Eigen::VectorXi get_A_out() { return this->A_out; };

  Eigen::VectorXi get_I_out() { return this->I_out; };

  T5 get_bd() { return this->bd; }

  void fit(T4 &train_x, T1 &train_y, Eigen::VectorXd &train_weight, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int train_n, int p, int N, Eigen::VectorXi &status, Eigen::MatrixXd sigma)
  {
    // cout<<" Fit for sparsity = "<<this->sparsity_level<<endl;///
    int T0 = this->sparsity_level;
    // this->status = status;
    this->cox_g = Eigen::VectorXd::Zero(0);
    this->x = &train_x;
    this->y = &train_y;
    this->beta = this->beta_init;
    this->coef0 = this->coef0_init;
    this->bd = this->bd_init;

    if (this->sub_search == 0 || this->sparsity_level + this->sub_search > N){
      this->U_size = N;
    }else{
      this->U_size = this->sparsity_level + this->sub_search;
    }

    // special initial
    switch (this->model_type){
      case 1: // Lm
      case 5: // GLm
        if ((this->algorithm_type == 6 && this->PhiG.rows() == 0) || this->lambda_change)
        { 
          this->PhiG = Phi(train_x, g_index, g_size, train_n, p, N, this->lambda_level, this->group_XTX);
          this->invPhiG = invPhi(PhiG, N);
          this->PhiG_U.resize(N, 1);
          this->invPhiG_U.resize(N, 1);
        }
        break;
      case 7: // PCA
        this->Sigma = sigma;
        break;
      case 8: // Ising (also Graph)
        this->ising_n = (long int) train_weight.sum();
      case 9: // Graph
        this->map1 = Eigen::MatrixXi::Zero(N, 2);
        this->map2 = Eigen::MatrixXi::Zero(train_x.cols(), train_x.cols());
        int i = 0, j = 0;
        for (int k = 0; k < N; k++){
          this->map1(k, 0) = i;
          this->map1(k, 1) = j;
          this->map2(i, j) = k;
          this->map2(j, i) = k;
          i++;
          if (i > j) {
            i = 0; j++;
          }
        }
        break;
    }

    this->update_tau(train_n, N);

    if (N <= T0)
    {
      this->A_out = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      
      T2 beta_old = this->beta;
      T3 coef0_old = this->coef0;
      bool success = this->primary_model_fit(train_x, train_y, train_weight, this->beta, this->coef0, DBL_MAX, this->A_out, g_index, g_size);
      if (!success){
        this->beta = beta_old;
        this->coef0 = coef0_old;
      }
      this->train_loss = this->neg_loglik_loss(train_x, train_y, train_weight, this->beta, this->coef0, this->A_out, g_index, g_size);
      this->effective_number = this->effective_number_of_parameter(train_x, train_x, train_y, train_weight, this->beta, this->beta, this->coef0);
      return;
    }


    // input: this->beta_init, this->coef0_init, this->A_init, this->I_init
    // for splicing get A;for the others 0;

    // cout<<" initial screen "<<endl;///
    Eigen::VectorXi I, A = this->inital_screening(train_x, train_y, this->beta, this->coef0, this->A_init, this->I_init, this->bd, train_weight, g_index, g_size, N);
    
    // cout<<" A_init = ";///
    // for (int i=0;i<A.size();i++) cout<<"("<<this->map1(A(i), 0)<<","<<this->map1(A(i), 1)<<") ";cout<<endl;///
    I = Ac(A, N);
    
    Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N, this->model_type);
    T4 X_A;
    if (this->model_type == 8 || this->model_type == 9){
      // Eigen::VectorXi XA_ind = find_ind_graph(A_ind, this->map1, p);
      // X_A = X_seg(train_x, train_n, XA_ind);
      X_A = train_x;
    }else{
      X_A = X_seg(train_x, train_n, A_ind);
    }
    T2 beta_A;
    slice(this->beta, A_ind, beta_A);

    // if (this->algorithm_type == 6)
    // {

    T3 coef0_old = this->coef0;
    bool success = this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0, DBL_MAX, A, g_index, g_size);
    if (!success){
      this->coef0 = coef0_old;
    }else{
      slice_restore(beta_A, A_ind, this->beta);
      this->train_loss = this->neg_loglik_loss(X_A, train_y, train_weight, beta_A, this->coef0, A, g_index, g_size);
    }

    // cout<<"train_loss = "<<this->train_loss<<endl;///

    // for (int i=0;i<A.size();i++) cout<<A(i)<<" ";cout<<endl<<"init loss = "<<this->train_loss<<endl;
    // }

    this->beta_warmstart = this->beta;
    this->coef0_warmstart = this->coef0;

    int always_select_size = this->always_select.size();
    int C_max = min(min(T0 - always_select_size, this->U_size - T0 - always_select_size), this->exchange_num);

    // cout<<" get_A"<<endl;///
    this->get_A(train_x, train_y, A, I, C_max, this->beta, this->coef0, this->bd, T0, train_weight, g_index, g_size, N, this->tau, this->train_loss);

    // final fit
    if (this->model_type < 7){
      // cout<<"==> Final fit"<<endl;///
      A_ind = find_ind(A, g_index, g_size, p, N, this->model_type);
      X_A = X_seg(train_x, train_n, A_ind);
      slice(this->beta, A_ind, beta_A);

      this->primary_model_fit_max_iter += 20;
      coef0_old = this->coef0;
      success = this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0, DBL_MAX, A, g_index, g_size);
      if (!success){
        this->coef0 = coef0_old;
      }else{
        slice_restore(beta_A, A_ind, this->beta);
        this->train_loss = this->neg_loglik_loss(X_A, train_y, train_weight, beta_A, this->coef0, A, g_index, g_size);
      }
      this->primary_model_fit_max_iter -= 20;
    }

    this->A_out = A;
    this->effective_number = this->effective_number_of_parameter(train_x, X_A, train_y, train_weight, this->beta, beta_A, this->coef0);
    this->group_df = A_ind.size();

    // cout<<" End Fit"<<endl;///
    return;
  };

  virtual void get_A(T4 &X, T1 &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, T2 &beta, T3 &coef0, T5 &bd, int T0, Eigen::VectorXd &weights,
             Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau, double &train_loss)
  {
    Eigen::VectorXi U(this->U_size);
    Eigen::VectorXi U_ind;
    Eigen::VectorXi g_index_U(this->U_size);
    Eigen::VectorXi g_size_U(this->U_size);
    T4 *X_U = new T4;
    T2 beta_U;
    Eigen::VectorXi A_U(T0);
    Eigen::VectorXi I_U(this->U_size - T0);
    Eigen::VectorXi always_select_U(this->always_select.size());

    if (this->U_size == N)
    {
      U = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
    }
    else
    {
      U = max_k(bd, this->U_size, true);
    }

    int p = X.cols();
    int n = X.rows();
    int C = C_max;
    int iter = 0;
    while (iter++ < this->max_iter)
    {
      // cout<<"==> get_A | iter = "<<iter<<" | T0 = "<<T0<<" | Usize = "<<this->U_size<<" | N = "<<N<<" | p = "<<p<<endl;///
      // mapping
      if (this->U_size == N)
      {
        delete X_U;
        X_U = &X;
        if (this->model_type == 8 || this->model_type == 9){
          U_ind = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
        }else{
          U_ind = Eigen::VectorXi::LinSpaced(p, 0, p - 1);
        }
        beta_U = beta;
        g_size_U = g_size;
        g_index_U = g_index;
        A_U = A;
        I_U = I;
        always_select_U = this->always_select;
      }
      else
      {
        U_ind = find_ind(U, g_index, g_size, p, N, this->model_type);
        if (this->model_type == 8 || this->model_type == 9){ /// todo
          // Eigen::VectorXi XU_ind = find_ind_graph(U_ind, this->map1, p);
          // *X_U = X_seg(X, n, XU_ind);
          *X_U = X;
        }else{
          *X_U = X_seg(X, n, U_ind);
        }
        slice(beta, U_ind, beta_U);

        if (this->model_type != 8 || this->model_type != 9){ // group is not supported in Ising
          int pos = 0;
          for (int i = 0; i < this->U_size; i++)
          {
            g_size_U(i) = g_size(U(i));
            g_index_U(i) = pos;
            pos += g_size_U(i);
          }
        }

        A_U = Eigen::VectorXi::LinSpaced(T0, 0, T0 - 1);
        I_U = Eigen::VectorXi::LinSpaced(this->U_size - T0, T0, this->U_size - 1);

        int *temp = new int[N], as = this->always_select.size();
        memset(temp, 0, N);
        for (int i = 0; i < as; i++)
          temp[this->always_select(i)] = 1;
        for (int i = 0; i < this->U_size; i++)
        {
          if (as <= 0)
            break;
          if (temp[U(i)] == 1)
          {
            always_select_U(this->always_select.size() - as) = i;
            as--;
          }
        }
        delete[] temp;
      }

      // cout<<"beta_U:\n";
      // for (int i=0;i<beta_U.size();i++){
      //   int mi=this->map1(i, 0);
      //   int mj=this->map1(i, 1);
      //   cout<<" ("<<mi<<","<<mj<<") -> "<<beta_U(i)<<endl;
      // }

      int num = -1;
      while (true)
      {
        num++;
        // cout<<" A_U = ";///
        // for (int i=0;i<A_U.size();i++) cout<<A_U(i)<<"("<<this->map1(A_U(i), 0)<<","<<this->map1(A_U(i), 1)<<") ";cout<<endl;

        Eigen::VectorXi A_ind = find_ind(A_U, g_index_U, g_size_U, U_ind.size(), this->U_size, this->model_type);
        T4 X_A;
        if (this->model_type == 8 || this->model_type == 9){ 
          // Eigen::VectorXi temp = Eigen::VectorXi::Zero(A_ind.size());
          // for (int i = 0; i < A_ind.size(); i++)
          //   temp(i) = U_ind(A_ind(i));
          // Eigen::VectorXi XA_ind = find_ind_graph(temp, this->map1, p); 
          // X_A = X_seg(X, n, XA_ind);
          X_A = *X_U;
        }else{
          X_A = X_seg(*X_U, n, A_ind);
        }
        T2 beta_A;
        slice(beta_U, A_ind, beta_A);

        T5 bd_U = T5::Zero(this->U_size);
        this->sacrifice(*X_U, X_A, y, beta_U, beta_A, coef0, A_U, I_U, weights, g_index_U, g_size_U, this->U_size, A_ind, bd_U, U, U_ind, num);
        
        for (int i = 0; i < always_select_U.size(); i++)
        {
          bd_U(always_select_U(i)) = DBL_MAX;
        }

        double l0 = train_loss;
        bool exchange = this->splicing(*X_U, y, A_U, I_U, C_max, beta_U, coef0, bd_U, weights,
                                       g_index_U, g_size_U, this->U_size, tau, l0);

        // cout << "exchange A: ";
        // for (int i=0;i<A_U.size();i++) cout<<"("<<this->map1(A_U(i), 0)<<","<<this->map1(A_U(i), 1)<<") ";cout<<endl;///
        // cout<<"  --> splicing num = "<<num<<endl;///
        // break;

        if (exchange)
          train_loss = l0;
        else
          break; // A_U is stable
      }

      if (A_U.size() == 0 || A_U.maxCoeff() == T0 - 1){
        break; // if A_U not change, stop
      }

      // store beta, A, I
      slice_restore(beta_U, U_ind, beta);

      // int *temp = new int[N];
      // memset(temp, 0, N);
      // for (int i = 0; i < T0; i++)
      //   temp[U(A_U(i))] = 1;

      // int tempA = 0, tempI = 0;
      // for (int i = 0; i < N; i++)
      //   if (temp[i] == 0)
      //     I(tempI++) = i;
      //   else
      //     A(tempA++) = i;
      
      // delete[] temp;

      Eigen::VectorXi ind = Eigen::VectorXi::Zero(N);
      for (int i = 0; i < T0; i++)
        ind(U(A_U(i))) = 1;

      int tempA = 0, tempI = 0;
      for (int i = 0; i < N; i++)
        if (ind(i) == 0)
          I(tempI++) = i;
        else
          A(tempA++) = i;

      // bd in full set
      Eigen::VectorXi A_ind0 = find_ind(A, g_index, g_size, p, N, this->model_type);
      T4 X_A0;
      if (this->model_type == 8 || this->model_type == 9){
        // Eigen::VectorXi XA_ind0 = find_ind_graph(A_ind0, this->map1, p);
        // X_A0 = X_seg(X, n, XA_ind0);
        X_A0 = X;
      }else{
        X_A0 = X_seg(X, n, A_ind0);
      }
      T2 beta_A0;
      slice(beta, A_ind0, beta_A0);
      Eigen::VectorXi U_ind0, U0 = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      if (model_type == 8 || model_type == 9){
        U_ind0 = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      }else{
        U_ind0 = Eigen::VectorXi::LinSpaced(p, 0, p - 1);
      }
      this->sacrifice(X, X_A0, y, beta, beta_A0, coef0, A, I, weights, g_index, g_size, N, A_ind0, bd, U0, U_ind0, 0);

      if (this->U_size == N)
      {

        for (int i = 0; i < this->always_select.size(); i++)
          bd(this->always_select(i)) = DBL_MAX;

        break;
      }
      else
      {

        // keep A in U_new
        for (int i = 0; i < T0; i++)
          bd(A(i)) = DBL_MAX;

        //update U
        Eigen::VectorXi U_new = max_k(bd, this->U_size, true);

        U = U_new;
        C_max = C;
      }
    }

    if (this->U_size != N)
      delete X_U;

    return;
  };

  virtual bool splicing(T4 &X, T1 &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, T2 &beta, T3 &coef0, T5 &bd, Eigen::VectorXd &weights,
                Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau, double &train_loss)
  {
    // cout<<" --> splicing | A = ";///
    // for (int i = 0;i<A.size();i++) {cout<<A(i)<<" ";} cout<<endl;///
    if (C_max <= 0)
      return false;

    // init
    int p = X.cols();
    int n = X.rows();

    int A_size = A.size();
    int I_size = I.size();

    T5 beta_A_group(A_size);
    T5 d_I_group(I_size);
    for (int i = 0; i < A_size; i++)
    {
      beta_A_group(i) = bd(A(i));
    }

    for (int i = 0; i < I_size; i++)
    {
      d_I_group(i) = bd(I(i));
    }

    Eigen::VectorXi A_min_k = min_k(beta_A_group, C_max, true);
    Eigen::VectorXi I_max_k = max_k(d_I_group, C_max, true);
    Eigen::VectorXi s1 = vector_slice(A, A_min_k);
    Eigen::VectorXi s2 = vector_slice(I, I_max_k);

    Eigen::VectorXi A_exchange(A_size);
    Eigen::VectorXi A_ind_exchange;
    T4 X_A_exchange;
    T2 beta_A_exchange;
    T3 coef0_A_exchange;

    // cout << " | A_min_k : \n";///
    // for (int i=0;i<A_min_k.size();i++) cout<<" | ("<<this->map1(A(A_min_k(i)), 0)<<","<<this->map1(A(A_min_k(i)), 1)<<") -> "<<bd(A(A_min_k(i)))<<endl;
    // cout << " | I_max_k : \n";
    // for (int i=0;i<I_max_k.size();i++) cout<<" | ("<<this->map1(I(I_max_k(i)), 0)<<","<<this->map1(I(I_max_k(i)), 1)<<") -> "<<bd(I(I_max_k(i)))<<endl;

    double L;
    for (int k = C_max; k >= 1;)
    {
      A_exchange = diff_union(A, s1, s2);
      A_ind_exchange = find_ind(A_exchange, g_index, g_size, p, N, this->model_type);
      if (this->model_type == 8 || this->model_type == 9){
        // Eigen::VectorXi XA_ind_exchange = find_ind_graph(A_ind_exchange, this->map1, p);
        // X_A_exchange = X_seg(X, n, XA_ind_exchange);
        X_A_exchange = X;
      }else{
        X_A_exchange = X_seg(X, n, A_ind_exchange);
      }
      slice(beta, A_ind_exchange, beta_A_exchange);
      coef0_A_exchange = coef0;

      bool success = primary_model_fit(X_A_exchange, y, weights, beta_A_exchange, coef0_A_exchange, train_loss, A_exchange, g_index, g_size);
      if (success){
        L = neg_loglik_loss(X_A_exchange, y, weights, beta_A_exchange, coef0_A_exchange, A_exchange, g_index, g_size);
      }else{
        L = train_loss + 1;
      }

      if (train_loss - L > tau)
      {
        // cout<<"   ~~> exchange k = "<<k<<endl;///
        train_loss = L;
        A = A_exchange;
        I = Ac(A_exchange, N);
        slice_restore(beta_A_exchange, A_ind_exchange, beta);
        coef0 = coef0_A_exchange;
        C_max = k;
        return true;
      }
      else
      {
        if (this->splicing_type == 1)
          k = k - 1;
        else
          k = k / 2;
        s1 = s1.head(k).eval();
        s2 = s2.head(k).eval();
      }
    }

    return false;
  };

  virtual Eigen::VectorXi inital_screening(T4 &X, T1 &y, T2 &beta, T3 &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, T5 &bd, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {

    if (bd.size() == 0)
    {
      // variable initialization
      int n = X.rows();
      int p = X.cols();
      bd = T5::Zero(N);

      // calculate beta & d & h
      Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N, this->model_type);
      T4 X_A = X_seg(X, n, A_ind);
      T2 beta_A;
      slice(beta, A_ind, beta_A);

      Eigen::VectorXi U_ind, U = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      U_ind = Eigen::VectorXi::LinSpaced(p, 0, p - 1);
      this->sacrifice(X, X_A, y, beta, beta_A, coef0, A, I, weights, g_index, g_size, N, A_ind, bd, U, U_ind, 0);
      for (int i = 0; i < this->always_select.size(); i++)
      {
        bd(this->always_select(i)) = DBL_MAX;
      }
    }

    // get Active-set A according to max_k bd
    Eigen::VectorXi A_new = max_k(bd, this->sparsity_level);

    return A_new;
  }

  virtual double neg_loglik_loss(T4 &X, T1 &y, Eigen::VectorXd &weights, T2 &beta, T3 &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size){ return 1; };

  virtual void sacrifice(T4 &X, T4 &XA, T1 &y, T2 &beta, T2 &beta_A, T3 &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, T5 &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num){};

  virtual bool primary_model_fit(T4 &X, T1 &y, Eigen::VectorXd &weights, T2 &beta, T3 &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size){ return true; };

  virtual double effective_number_of_parameter(T4 &X, T4 &XA, T1 &y, Eigen::VectorXd &weights, T2 &beta, T2 &beta_A, T3 &coef0){ return beta_A.rows(); };
};

#endif //SRC_ALGORITHM_H