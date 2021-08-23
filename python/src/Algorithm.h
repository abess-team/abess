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
*  @file     algorithm.h                                                         *
*  @brief    the algorithm for fitting.                            *
*                                                                            *
*                                                                            *
*  @author   Kangkang Jiang                                                  *
*  @email    jiangkk3@mail2.sysu.edu.cn                                      *
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

// #define TEST

#ifndef SRC_ALGORITHM_H
#define SRC_ALGORITHM_H

#ifndef R_BUILD
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Eigen>
#endif

#include <Spectra/SymEigsSolver.h>
#include "Data.h"
#include "utilities.h"
#include "model_fit.h"
#include <iostream>

#include <time.h>
#include <cfloat>

using namespace std;
using namespace Spectra;

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

template <class T1, class T2, class T3, class T4>
class Algorithm
{
public:
  // int l;              /* the final itertation time when the splicing algorithm converge. */
  int model_fit_max;  /* Maximum number of iterations taken for the primary model fitting. */
  int model_type;     /* primary model type. */
  int algorithm_type; /* algorithm type. */

  int group_df = 0;        /* freedom */
  int sparsity_level = 0;  /* Number of non-zero coefficients. */
  double lambda_level = 0; /* l2 normalization coefficients. */
  // Eigen::VectorXi train_mask;
  int max_iter;            /* Maximum number of iterations taken for the splicing algorithm to converge.  */
  int exchange_num;        /* Max exchange variable num. */
  bool warm_start;         /* When tuning the optimal parameter combination, whether to use the last solution as a warm start to accelerate the iterative convergence of the splicing algorithm.*/
  T2 beta;                 /* coefficients. */
  Eigen::VectorXd bd;      /* */
  T3 coef0;                /* intercept. */
  double train_loss = 0.;  /* train loss. */
  T2 beta_init;            /* initialization coefficients. */
  T3 coef0_init;           /* initialization intercept. */
  Eigen::VectorXi A_init;  /* initialization active set. */
  Eigen::VectorXi I_init;  /* initialization inactive set. */
  Eigen::VectorXd bd_init; /* initialization bd vector. */

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

  bool covariance_update;                 /* use covairance update mathod or not. */
  Eigen::MatrixXd covariance;             /* covairance matrix. */
  Eigen::MatrixXi covariance_update_flag; /* each variable have updated in covairance matirx. */
  T1 XTy;                                 /*X.transpose() * y */
  T1 XTone;                               /* X.transpose() * Eigen::MatrixXd::one() */

  double effective_number; /* effective number of parameter. */

  int splicing_type;     /* exchange number update mathod. */
  Eigen::MatrixXd Sigma; /* covariance matrix for pca. */
  
  int sub_search; /* size of sub_searching in splicing */ 
  int U_size;

  T1 XTy_U; 
  T1 XTone_U;
  Eigen::Matrix<Eigen::MatrixXd, -1, -1> PhiG_U;
  Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhiG_U;

  Algorithm() = default;

  ~Algorithm(){};

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

  void update_bd_init(Eigen::VectorXd &bd_init) { this->bd_init = bd_init; }

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

  bool get_warm_start() { return this->warm_start; }

  double get_train_loss() { return this->train_loss; }

  int get_group_df() { return this->group_df; }

  double get_effective_number() { return this->effective_number; }

  int get_sparsity_level() { return this->sparsity_level; }

  T2 get_beta() { return this->beta; }

  T3 get_coef0() { return this->coef0; }

  Eigen::VectorXi get_A_out() { return this->A_out; };

  Eigen::VectorXi get_I_out() { return this->I_out; };

  Eigen::VectorXd get_bd() { return this->bd; }

  int get_l() { return this->l; }

  void fit(T4 &train_x, T1 &train_y, Eigen::VectorXd &train_weight, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int train_n, int p, int N, Eigen::VectorXi &status, Eigen::MatrixXd sigma)
  {
    // std::cout<<"cpp fit enter. | sparsity = "<<this->sparsity_level<<endl;///
    // std::cout << "fit" << endl;

    int T0 = this->sparsity_level;
    // this->status = status;
    this->cox_g = Eigen::VectorXd::Zero(0);

    this->tau = 0.01 * (double)this->sparsity_level * log((double)N) * log(log((double)train_n)) / (double)train_n;

    this->beta = this->beta_init;
    this->coef0 = this->coef0_init;
    this->bd = this->bd_init;

    if (this->sub_search == 0 || this->sparsity_level + this->sub_search > N)
      this->U_size = N;
    else
      this->U_size = this->sparsity_level + this->sub_search;

    if (this->model_type == 7)
    {
      if (sigma.cols() == 1 && sigma(0, 0) == -1)
        this->Sigma = train_x.transpose() * train_x;
      else
        this->Sigma = sigma;
    }

    if (N == T0)
    {
      this->A_out = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      T2 beta_old = this->beta;
      T3 coef0_old = this->coef0;
      bool success = this->primary_model_fit(train_x, train_y, train_weight, this->beta, this->coef0, DBL_MAX, this->A_out, g_index, g_size);
      if (!success){
        this->beta = beta_old;
        this->coef0 = coef0_old;
      }
      this->train_loss = neg_loglik_loss(train_x, train_y, train_weight, this->beta, this->coef0, this->A_out, g_index, g_size);
      this->effective_number = effective_number_of_parameter(train_x, train_x, train_y, train_weight, this->beta, this->beta, this->coef0);
      return;
    }

#ifdef TEST
    clock_t t1, t2;///
    t1 = clock();
#endif

    if (this->model_type == 1 || this->model_type == 5)
    {
      // this->covariance = Eigen::MatrixXd::Zero(train_x.cols(), train_x.cols());
      if ((this->algorithm_type == 6 && this->PhiG.rows() == 0) || this->lambda_change)
      {
        this->PhiG = Phi(train_x, g_index, g_size, train_n, p, N, this->lambda_level, this->group_XTX);
        this->invPhiG = invPhi(PhiG, N);
        this->PhiG_U.resize(N, 1);
        this->invPhiG_U.resize(N, 1);
      }
    }

#ifdef TEST
    t2 = clock();
    std::cout << "PhiG invPhiG time" << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();///
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
    t2 = clock();///
    std::cout << "init screening time" << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;///
#endif

    Eigen::VectorXi I = Ac(A, N);
    // Eigen::MatrixXi A_list(T0, max_iter + 2);
    // A_list.col(0) = A;

    Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
    T4 X_A = X_seg(train_x, train_n, A_ind);
    T2 beta_A;
    slice(this->beta, A_ind, beta_A);

#ifdef TEST
    t1 = clock();///
#endif
    // if (this->algorithm_type == 6)
    // {
    
    T3 coef0_old = this->coef0;
    bool success = this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0, DBL_MAX, A, g_index, g_size);
    if (!success){
      this->coef0 = coef0_old;
    }else{
      slice_restore(beta_A, A_ind, this->beta);
      this->train_loss = neg_loglik_loss(X_A, train_y, train_weight, beta_A, this->coef0, A, g_index, g_size);
    }
    
    // for (int i=0;i<A.size();i++) cout<<A(i)<<" ";cout<<endl<<"init loss = "<<this->train_loss<<endl; 
    // }

    this->beta_warmstart = this->beta;
    this->coef0_warmstart = this->coef0;

#ifdef TEST
    t2 = clock();///
    std::cout << "primary fit" << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;///
#endif

    // std::cout << "fit 6" << endl;
    int always_select_size = this->always_select.size();
    int C_max = min(min(T0 - always_select_size, this->U_size - T0 - always_select_size), this->exchange_num);

#ifdef TEST
    // std::cout << "fit 7" << endl;
    t1 = clock();///
#endif

    this->get_A(train_x, train_y, A, I, C_max, this->beta, this->coef0, this->bd, T0, train_weight, g_index, g_size, N, this->tau, this->train_loss);
  
#ifdef TEST
    t2 = clock();///
    std::cout << "get A " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;///
    t1 = clock();
#endif
      
    // final fit
    this->A_out = A;

    A_ind = find_ind(A, g_index, g_size, p, N);
    X_A = X_seg(train_x, train_n, A_ind);
    slice(this->beta, A_ind, beta_A);

    this->primary_model_fit_max_iter += 20;
    coef0_old = this->coef0;
    success = this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0, DBL_MAX, A, g_index, g_size);
    if (!success){
      this->coef0 = coef0_old;
    }else{
      slice_restore(beta_A, A_ind, this->beta);
      this->train_loss = neg_loglik_loss(X_A, train_y, train_weight, beta_A, this->coef0, A, g_index, g_size);
    }
    this->primary_model_fit_max_iter -= 20;

#ifdef TEST
    t2 = clock();
    std::cout << "final fit time " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    // cout << "A: " << endl;
    // cout << A << endl;
    // cout << "beta" << endl;
    // cout << beta_A << endl;

    this->effective_number = effective_number_of_parameter(train_x, X_A, train_y, train_weight, this->beta, beta_A, this->coef0);
    this->group_df = A_ind.size();

#ifdef TEST
    t2 = clock();
    std::cout << "group_df time " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
    return;
  };

  void get_A(T4 &X, T1 &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, T2 &beta, T3 &coef0, Eigen::VectorXd &bd, int T0, Eigen::VectorXd &weights,
             Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau, double &train_loss)
  {
#ifdef TEST
    clock_t t1, t2, t3, t4;///
    std::cout << "get_A 1 | T0 = " << T0  << " | U_size = " << this->U_size << " | N = " << N << endl;
    t1 = clock();
#endif

    Eigen::VectorXi U(this->U_size);
    Eigen::VectorXi U_ind;
    Eigen::VectorXi g_index_U(this->U_size);
    Eigen::VectorXi g_size_U(this->U_size); 
    T4 *X_U = new T4;
    T2 beta_U;
    Eigen::VectorXi A_U(T0);
    Eigen::VectorXi I_U(this->U_size - T0);
    Eigen::VectorXi always_select_U(this->always_select.size());

    if (this->U_size == N){
      U = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
    }else{
      U = max_k(bd, this->U_size, true);
    }
#ifdef TEST
    t2 = clock();
    std::cout << "U time " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif    

    int p = X.cols();
    int n = X.rows();
    int C = C_max;
    int iter = 0;
    while (iter++ < this->max_iter){
#ifdef TEST
      std::cout << "get_A 2 | iter = " << iter << endl;///
      // for (int i=0;i<A.size();i++) cout<<A(i)<<" ";cout<<endl<<"loss = "<<train_loss<<endl; ///
      // for (int i=0;i<U.size();i++) cout<<U(i)<<" ";cout<<endl;///
      t3 = clock();
#endif
      // mapping 
      if (this->U_size == N) {
        delete X_U;
        X_U = &X;
        U_ind = Eigen::VectorXi::LinSpaced(p, 0, p - 1);
        beta_U = beta;
        g_size_U = g_size;
        g_index_U = g_index;
        A_U = A;
        I_U = I;
        always_select_U = this->always_select;
      }else{
        U_ind = find_ind(U, g_index, g_size, p, N);
        *X_U = X_seg(X, n, U_ind);
        slice(beta, U_ind, beta_U);
        
        int pos = 0;
        for (int i = 0; i < U.size(); i++){
            g_size_U(i) = g_size(U(i));
            g_index_U(i) = pos;
            pos += g_size_U(i);
        }

        A_U = Eigen::VectorXi::LinSpaced(T0, 0, T0 - 1);
        I_U = Eigen::VectorXi::LinSpaced(this->U_size - T0, T0, this->U_size - 1);

        int temp[N], s = this->always_select.size();
        for (int i = 0; i < s; i++) temp[this->always_select(i)] = 1;
        for (int i = 0; i < this->U_size; i++){
          if (s <= 0) break;
          if (temp[U(i)] == 1)
            always_select_U(this->always_select.size() - s) = i;
          s--;
        }
      }

#ifdef TEST
      std::cout << "get_A 2.5" << endl;
      t4 = clock();
      std::cout << "mapping U time = " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
      t3 = clock();///
#endif

      int num = 0;
      while (true){
        num ++;
#ifdef TEST
        t1 = clock();
        std::cout << "get_A 3 | num  = " << num << endl;///
#endif      

        Eigen::VectorXi A_ind = find_ind(A_U, g_index_U, g_size_U, U_ind.size(), this->U_size); 
        T4 X_A = X_seg(*X_U, n, A_ind);
        T2 beta_A;
        slice(beta_U, A_ind, beta_A);

        // cout<<"AUsize = "<<A_U.size()<<" | IU_size = "<<I_U.size()<<endl;
        
        Eigen::VectorXd bd_U = Eigen::VectorXd::Zero(this->U_size);
        this->sacrifice(*X_U, X_A, y, beta_U, beta_A, coef0, A_U, I_U, weights, g_index_U, g_size_U, this->U_size, A_ind, bd_U, U, U_ind, num);

        for (int i = 0; i < always_select_U.size(); i++)
        {
          bd_U(always_select_U(i)) = DBL_MAX;
        }

        double l0 = train_loss;
        this->splicing(*X_U, y, A_U, I_U, C_max, beta_U, coef0, bd_U, weights,
                      g_index_U, g_size_U, this->U_size, tau, l0);
#ifdef TEST        
        cout << train_loss << " >" << l0<<endl;///
        t2 = clock();
        std::cout << "splicing time " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
        if (l0 < train_loss) 
          train_loss = l0; 
        else 
          break; // if loss not decrease, A_U is stable
      }
      
#ifdef TEST
      t4 = clock();
      std::cout << "total splicing time = " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << " | num = " << num << endl;///
      std::cout << "get_A 4 " << endl;///
      t1 = clock();
#endif

      // store beta, A, I
      slice_restore(beta_U, U_ind, beta);
      // A = A_U;
      // std::sort(A.data(), A.data() + T0);
      // I = Ac(A, N);

      Eigen::VectorXi ind = Eigen::VectorXi::Zero(N);
      for (int i = 0; i < T0; i++) ind(U(A_U(i))) = 1;

      int tempA = 0, tempI = 0;
      for (int i = 0; i < N; i++) 
          if (ind(i)==0) I(tempI++) = i; else A(tempA++) = i;
          
#ifdef TEST      
      t2 = clock();
      std::cout << "restore time " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
      t1 = clock();
#endif      

#ifdef TEST
        t2 = clock();
        std::cout << "full bd time " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
     
      // bd in full set        
      Eigen::VectorXi A_ind0 = find_ind(A, g_index, g_size, p, N);
      T4 X_A0 = X_seg(X, n, A_ind0);
      T2 beta_A0;
      slice(beta, A_ind0, beta_A0);
      Eigen::VectorXi U0 = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      Eigen::VectorXi U_ind0 = Eigen::VectorXi::LinSpaced(p, 0, p - 1);
      this->sacrifice(X, X_A0, y, beta, beta_A0, coef0, A, I, weights, g_index, g_size, N, A_ind0, bd, U0, U_ind0, 0);

      if (this->U_size == N){
        for (int i = 0; i < this->always_select.size(); i++) 
          bd(this->always_select(i)) = DBL_MAX;

        break;
      }else{
        
        // keep A in U_new
        for (int i = 0; i < T0; i++) bd(A(i)) = DBL_MAX; 
        
        //update U
        Eigen::VectorXi U_new = max_k(bd, this->U_size, true);
        if (check_same_vector(U_new, U)) break; // if U not change, stop

        U = U_new;  
        C_max = C;
      } 
    }

    if (this->U_size != N) delete X_U;

#ifdef TEST
    std::cout << "get_A 5" << endl;///
    std::cout << "get_A iter = " << iter << endl;///
#endif
    return;
  };
  
  void splicing(T4 &X, T1 &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, T2 &beta, T3 &coef0, Eigen::VectorXd &bd, Eigen::VectorXd &weights,
             Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau, double &train_loss)
  {
    if (C_max <= 0) return;

#ifdef TEST
    clock_t t0, t1, t2;
    t1 = clock();
#endif

    // init
    int p = X.cols();
    int n = X.rows();
    
#ifdef TEST
    t2 = clock();
    std::cout << "Splicing init time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    int A_size = A.size();
    int I_size = I.size();

    Eigen::VectorXd beta_A_group(A_size);
    Eigen::VectorXd d_I_group(I_size);
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

#ifdef TEST
    t2 = clock();
    std::cout << "Splicing s1 s2 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t0 = clock();
#endif

    // cout << "get A 5" << endl;

    Eigen::VectorXi A_exchange(A_size);
    Eigen::VectorXi A_ind_exchage;
    T4 X_A_exchage;
    T2 beta_A_exchange;
    T3 coef0_A_exchange;

    double L;
    for (int k = C_max; k >= 1;)
    {
      A_exchange = diff_union(A, s1, s2);
      A_ind_exchage = find_ind(A_exchange, g_index, g_size, p, N);
      X_A_exchage = X_seg(X, n, A_ind_exchage);
      slice(beta, A_ind_exchage, beta_A_exchange);
      coef0_A_exchange = coef0;

      bool success = primary_model_fit(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange, train_loss, A_exchange, g_index, g_size);
      if (success){
        L = neg_loglik_loss(X_A_exchage, y, weights, beta_A_exchange, coef0_A_exchange, A_exchange, g_index, g_size);
      }else{
        L = train_loss + 1;
      }

      // cout << "L0: " << L0 << " L1: " << L1 << endl;
      if (train_loss - L > tau)
      {
        train_loss = L;
        A = A_exchange;
        I = Ac(A_exchange, N);
        slice_restore(beta_A_exchange, A_ind_exchage, beta);
        coef0 = coef0_A_exchange;
        C_max = k;
        
#ifdef TEST
        std::cout << "C_max: " << C_max << " k: " << k << endl;
        t2 = clock();
        std::cout << "splicing time: " << ((double)(t2 - t0) / CLOCKS_PER_SEC) << endl;
#endif
        return;
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
#ifdef TEST
    t2 = clock();
    std::cout << "splicing time: " << ((double)(t2 - t0) / CLOCKS_PER_SEC) << endl;
#endif
    return;
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
  
      Eigen::VectorXi U = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      Eigen::VectorXi U_ind = Eigen::VectorXi::LinSpaced(p, 0, p - 1);
      this->sacrifice(X, X_A, y, beta, beta_A, coef0, A, I, weights, g_index, g_size, N, A_ind, bd, U, U_ind, 0);
      for (int i = 0; i < this->always_select.size(); i++)
      {
        bd(this->always_select(i)) = DBL_MAX;
      }
    }

#ifdef TEST
    t4 = clock();
    std::cout << "inital_screening bd: " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
    t3 = clock();
#endif

    // get Active-set A according to max_k bd
    Eigen::VectorXi A_new = max_k(bd, this->sparsity_level);

#ifdef TEST
    t4 = clock();
    std::cout << "inital_screening max_k: " << ((double)(t4 - t3) / CLOCKS_PER_SEC) << endl;
#endif
    return A_new;
  }

  virtual double neg_loglik_loss(T4 &X, T1 &y, Eigen::VectorXd &weights, T2 &beta, T3 &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) = 0;

  virtual void sacrifice(T4 &X, T4 &XA, T1 &y, T2 &beta, T2 &beta_A, T3 &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num) = 0;

  virtual bool primary_model_fit(T4 &X, T1 &y, Eigen::VectorXd &weights, T2 &beta, T3 &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size) = 0;

  virtual double effective_number_of_parameter(T4 &X, T4 &XA, T1 &y, Eigen::VectorXd &weights, T2 &beta, T2 &beta_A, T3 &coef0) = 0;
};

template <class T4>
class abessLogistic : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessLogistic(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessLogistic(){};

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
#ifdef TEST
    clock_t t1 = clock();
#endif
    // cout << "primary_fit-----------" << endl;
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

#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit init time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif

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

        // cout << "j=" << j << " loglik: " << loglik1 << endl;
        // cout << "j=" << j << " loglik diff: " << loglik1 - loglik0 << endl;
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
#ifdef TEST
        t1 = clock();
#endif
        for (int i = 0; i < p + 1; i++)
        {
          X_new.col(i) = X.col(i).cwiseProduct(W).cwiseProduct(weights);
        }

        Eigen::MatrixXd XTX = 2 * this->lambda_level * lambdamat + X_new.transpose() * X;
        if (check_ill_condition(XTX)) return false;
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
        // cout << "j=" << j << " loglik: " << loglik1 << endl;
        // cout << "j=" << j << " loglik diff: " << loglik0 - loglik1 << endl;
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
#ifdef TEST
    t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << j << endl;
#endif
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
#ifdef TEST
    clock_t t1 = clock(), t2;
#endif
    int p = X.cols();
    int n = X.rows();

    Eigen::VectorXd coef(XA.cols() + 1);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    coef << coef0, beta_A;

    Eigen::VectorXd pr = pi(XA, y, coef);
    Eigen::VectorXd res = (y - pr).cwiseProduct(weights);

#ifdef TEST
    t2 = clock();
    std::cout << "d1 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    t1 = clock();
#endif

    Eigen::VectorXd d = X.transpose() * res - 2 * this->lambda_level * beta;
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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
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

#ifdef TEST
      cout << "effective_number_of_parameter" << endl;
      clock_t t1 = clock(), t2;
#endif
      // int p = X.cols();
      int n = X.rows();

      Eigen::VectorXd coef = Eigen::VectorXd::Ones(XA.cols() + 1);
      Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
      coef(0) = coef0;
      coef.tail(XA.cols()) = beta_A;

      Eigen::VectorXd pr = pi(XA, y, coef);
      // Eigen::VectorXd res = (y - pr).cwiseProduct(weights);

#ifdef TEST
      t2 = clock();
      std::cout << "d1 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
      t1 = clock();
#endif

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
class abessLm : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessLm(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = true, int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, covariance_update, splicing_type, sub_search){};

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
    if (check_ill_condition(XTX)) return false;
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
      for (int i = 0; i < p; i++){
        this->XTy_U(i) = this->XTy(U_ind(i), 0);
        this->XTone_U(i) = this->XTone(U_ind(i), 0);
      }

    for (int i = 0; i < N; i++){
      this->PhiG_U(i, 0) = this->PhiG(U(i), 0);
      this->invPhiG_U(i, 0) = this->invPhiG(U(i), 0);
    }
    return;
  }

  Eigen::MatrixXd covariance_update_f_U(T4 &X, Eigen::VectorXi &U_ind, Eigen::VectorXi &A_ind_U)
  {
    int k = A_ind_U.size(), p = U_ind.size();
    Eigen::MatrixXd cov_A(p, k);
    Eigen::VectorXi A_ind(k);
    for (int i = 0; i < k; i++) A_ind(i) = U_ind(A_ind_U(i));

    for (int i = 0; i < p; i++)
      for (int j = 0; j < k; j++){
        if (this->covariance_update_flag(U_ind(i), A_ind(j)) == 0)
        {
          Eigen::MatrixXd temp = X.col(i).transpose() * X.col(A_ind_U(j));
          this->covariance(U_ind(i), A_ind(j)) = temp(0, 0);
          this->covariance_update_flag(U_ind(i), A_ind(j)) = 1;
        }
        cov_A(i, j) = this->covariance(U_ind(i), A_ind(j));
      }
    return cov_A;
  }

  void covariance_update_f(T4 &X, Eigen::VectorXi &A_ind)
  {
    for (int i = 0; i < A_ind.size(); i++)
    {
      if (this->covariance_update_flag(A_ind(i), 0) == 0)
      {
        this->covariance.col(A_ind(i)) = X.transpose() * (X.col(A_ind(i)).eval());
        this->covariance_update_flag(A_ind(i), 0) = 1;
      }
    }
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {
#ifdef TEST
    clock_t t1 = clock(), t2;
#endif
    int p = X.cols();
    int n = X.rows();

    if (p == this->XTy.rows()){
      this->XTy_U = this->XTy;
      this->XTone_U = this->XTone;
      this->PhiG_U = this->PhiG;
      this->invPhiG_U = this->invPhiG;
    }else if (num == 0){
      this->XTy_U.resize(p, 1); 
      this->XTone_U.resize(p, 1);
      this->mapping_U(U, U_ind);
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
        Eigen::VectorXd XTXbeta;
        if (p == this->XTy.rows()){
          this->covariance_update_f(X, A_ind);
          XTXbeta = X_seg(this->covariance, this->covariance.rows(), A_ind) * beta_A;
        }else{
          Eigen::MatrixXd cov_A = this->covariance_update_f_U(X, U_ind, A_ind);
          XTXbeta = cov_A * beta_A;
        }
        d = (this->XTy_U - XTXbeta - this->XTone_U * coef0) / double(n) - 2 * this->lambda_level * beta;
      }
      else
      {
        Eigen::VectorXd XTonecoef0 = this->XTone_U * coef0;
        d = (this->XTy_U - XTonecoef0) / double(n);
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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
  }

  double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0)
  {
    if (this->lambda_level == 0.)
    {
      return XA.cols();
    }
    else
    {
#ifdef TEST
      cout << "effective_number_of_parameter" << endl;
#endif

      return double(XA.cols()) / (this->lambda_level + 1);
    }
  }
};

template <class T4>
class abessPoisson : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessPoisson(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessPoisson(){};

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
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
      if (check_ill_condition(XTX)) return false;
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

    Eigen::VectorXd d = X.transpose() * (y - xbeta_exp) - 2 * this->lambda_level * beta;
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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
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
#ifdef TEST
      cout << "effective_number_of_parameter" << endl;
      clock_t t1 = clock(), t2;
#endif

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

#ifdef TEST
      t2 = clock();
      std::cout << "d1 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
      t1 = clock();
#endif

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
class abessCox : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessCox(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessCox(){};

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, Eigen::VectorXd &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
#ifdef TEST
    clock_t t1 = clock();
#endif
    if (x.cols() == 0)
    {
      coef0 = 0.;
      return true;
    }

    // cout << "primary_fit-----------" << endl;
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

#ifdef TEST
      std::cout << "g: " << g << endl;
      std::cout << "this->lambda_level: " << this->lambda_level << endl;
      std::cout << "g.rows(): " << g.rows() << endl;
      std::cout << "g.cols(): " << g.cols() << endl;
      std::cout << "beta0.rows(): " << beta0.rows() << endl;
      std::cout << "beta0.cols(): " << beta0.cols() << endl;
#endif
      Eigen::MatrixXd temp = x.transpose() * h * x;
      if (this->approximate_Newton)
      {
        // d = g.cwiseQuotient((x.transpose() * h * x + 2 * this->lambda_level * lambdamat).diagonal());
        d = (x.transpose() * g - 2 * this->lambda_level * beta0).cwiseQuotient(temp.diagonal());
      }
      else
      {
        // d = (x.transpose() * h * x + 2 * this->lambda_level * lambdamat).ldlt().solve(g);
        if (check_ill_condition(temp)) return false;
        d = temp.ldlt().solve(x.transpose() * g - 2 * this->lambda_level * beta0);
      }

#ifdef TEST
      cout << "d: " << d << endl;
#endif

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
        // cout << "condition1" << endl;
        return true;
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
        return true;
      }
    }
#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << l << endl;
#endif

    beta = beta0;
    return true;
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    return -loglik_cox(X, y, beta, weights);
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
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

    d = X.transpose() * g - 2 * this->lambda_level * beta;

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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
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
#ifdef TEST
      cout << "effective_number_of_parameter" << endl;
      clock_t t1 = clock(), t2;
#endif
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
#ifdef TEST
      t2 = clock();
      std::cout << "d1 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
      t1 = clock();
#endif

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
class abessMLm : public Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>
{
public:
  abessMLm(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = true, int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, covariance_update, splicing_type, sub_search){};

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
    if (check_ill_condition(XTX)) return false;
    Eigen::MatrixXd beta0 = XTX.ldlt().solve(X.adjoint() * y);
    
    beta = beta0.block(1, 0, p, M);
    coef0 = beta0.row(0).eval();
    return true;
    // if (X.cols() == 0)
    // {
    //   // coef0 = y.colwise().sum();
    //   return;
    // }
    // // cout << "primary_fit 1" << endl;
    // // overload_ldlt(X, X, y, beta);
    // Eigen::MatrixXd XTX = X.transpose() * X;
    // beta = (XTX + this->lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).ldlt().solve(X.transpose() * y);
    // cout << "primary_fit 2" << endl;

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
      for (int j = 0; j < M; j++){
        this->XTy_U(i, j) = this->XTy(U_ind(i), j);
        this->XTone_U(i, j) = this->XTone(U_ind(i), j);
      }

    for (int i = 0; i < N; i++){
      this->PhiG_U(i, 0) = this->PhiG(U(i), 0);
      this->invPhiG_U(i, 0) = this->invPhiG(U(i), 0);
    }
    return;
  }

  Eigen::MatrixXd covariance_update_f_U(T4 &X, Eigen::VectorXi &U_ind, Eigen::VectorXi &A_ind_U)
  {
    int k = A_ind_U.size(), p = U_ind.size();
    Eigen::MatrixXd cov_A(p, k);
    Eigen::VectorXi A_ind(k);
    for (int i = 0; i < k; i++) A_ind(i) = U_ind(A_ind_U(i));

    for (int i = 0; i < p; i++)
      for (int j = 0; j < k; j++){
        if (this->covariance_update_flag(U_ind(i), A_ind(j)) == 0)
        {
          Eigen::MatrixXd temp = X.col(i).transpose() * X.col(A_ind_U(j));
          this->covariance(U_ind(i), A_ind(j)) = temp(0, 0);
          this->covariance_update_flag(U_ind(i), A_ind(j)) = 1;
        }
        cov_A(i, j) = this->covariance(U_ind(i), A_ind(j));
      }
    return cov_A;
  }

  void covariance_update_f(T4 &X, Eigen::VectorXi &A_ind)
  {
    for (int i = 0; i < A_ind.size(); i++)
    {
      if (this->covariance_update_flag(A_ind(i), 0) == 0)
      {
        this->covariance.col(A_ind(i)) = X.transpose() * X.col(A_ind(i));
        this->covariance_update_flag(A_ind(i), 0) = 1;
      }
    }
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {
#ifdef TEST
    clock_t t1 = clock(), t2;
#endif
    int p = X.cols();
    int n = X.rows();
    int M = y.cols();

    if (p == this->XTy.rows()){
      this->XTy_U = this->XTy;
      this->XTone_U = this->XTone;
      this->PhiG_U = this->PhiG;
      this->invPhiG_U = this->invPhiG;
    }else if (num == 0){
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
#ifdef TEST
        clock_t t1 = clock();
#endif
        Eigen::MatrixXd XTXbeta;
        if (p == this->XTy.rows()){
          this->covariance_update_f(X, A_ind);
          XTXbeta = X_seg(this->covariance, this->covariance.rows(), A_ind) * beta_A;
        }else{
          Eigen::MatrixXd cov_A = this->covariance_update_f_U(X, U_ind, A_ind);
          XTXbeta = cov_A * beta_A;
        }
#ifdef TEST
        clock_t t2 = clock();
        std::cout << "covariance_update_f: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
        std::cout << "this->covariance_update_flag sum: " << this->covariance_update_flag.sum() << endl;
        t1 = clock();
#endif

        d = (this->XTy_U - XTXbeta - array_product(this->XTone_U, coef0)) / double(n) - 2 * this->lambda_level * beta;

#ifdef TEST
        t2 = clock();
        std::cout << "X beta time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
        std::cout << "d 1" << endl;
#endif
      }
      else
      {
        Eigen::MatrixXd XTonecoef0 = array_product(this->XTone_U, coef0);
        d = (this->XTy_U - XTonecoef0) / double(n);
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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
  }

  double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0)
  {
    if (this->lambda_level == 0.)
    {
      return XA.cols();
    }
    else
    {
#ifdef TEST
      cout << "effective_number_of_parameter" << endl;
#endif

      return double(XA.cols()) / (this->lambda_level + 1.0);
    }
  }
};

template <class T4>
class abessMultinomial : public Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>
{
public:
  abessMultinomial(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), bool covariance_update = true, int splicing_type = 0, int sub_search = 0) : Algorithm<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, covariance_update, splicing_type, sub_search){};

  ~abessMultinomial(){};

  bool primary_model_fit(T4 &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
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
      if (check_ill_condition(XTX)) return false;
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
          XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X + 2 * this->lambda_level * lambdamat;
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

#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << j << endl;
#endif

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
    // cout << "loss 0" << endl;
    array_product(log_pr, weights, 1);
    // cout << "loss 1" << endl;
    return -((log_pr.array() * y.array()).sum());
  }

  void sacrifice(T4 &X, T4 &XA, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::MatrixXd &beta_A, Eigen::VectorXd &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
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
    d = X.transpose() * res - 2 * this->lambda_level * beta;
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
#ifdef TEST
    t2 = clock();
    std::cout << "group bd time beta: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
#endif
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
#ifdef TEST
      cout << "effective_number_of_parameter" << endl;
      clock_t t1 = clock(), t2;
#endif
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
          XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X + 2 * this->lambda_level * lambdamat;
          XTW.block(m2 * (p + 1), m1 * n, (p + 1), n) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n);
          XTWX.block(m2 * (p + 1), m1 * (p + 1), (p + 1), (p + 1)) = XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1));
        }
      }

#ifdef TEST
      t2 = clock();
      std::cout << "d1 time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
      t1 = clock();
#endif

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

template <class T4>
class abessPCA : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessPCA(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 1, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessPCA(){};

  MatrixXd SigmaA(Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    int len = 0;
    for (int i = 0; i < A.size(); i++)
    {
      len += g_size(A(i));
    }
    int k = 0;
    VectorXd ind(len);
    for (int i = 0; i < A.size(); i++)
      for (int j = 0; j < g_size(A(i)); j++)
        ind(k++) = g_index(A(i)) + j;

    MatrixXd SA(len, len);
    for (int i = 0; i < len; i++)
      for (int j = 0; j < i + 1; j++)
      {
        int di = ind(i), dj = ind(j);
        SA(i, j) = this->Sigma(di, dj);
        SA(j, i) = this->Sigma(dj, di);
      }

    return SA;
  }

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
#ifdef TEST
    cout << "<< SPCA primary_model_fit >>" << endl;
#endif
    int p = x.cols();
    if (p == 0)
      return true;
    if (p == 1)
    {
      beta << 1;
      return true;
    }

    MatrixXd Y = SigmaA(A, g_index, g_size);
#ifdef TEST
    cout << "<< SPCA primary_model_fit 1 >>" << endl;
#endif

    DenseSymMatProd<double> op(Y);
    SymEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>> eig(&op, 1, 2);
    eig.init();
    eig.compute();
    MatrixXd temp;
    if (eig.info() == SUCCESSFUL)
    {
      temp = eig.eigenvectors(1);
    }else{
      return false;
    }

    beta = temp.col(0);
#ifdef TEST
    cout << "<< SPCA primary_model_fit end>>" << endl;
#endif
    return true;
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
#ifdef TEST
    cout << "<< SPCA Loss >>" << endl;
#endif
    MatrixXd Y = SigmaA(A, g_index, g_size);

#ifdef TEST
    cout << "<< SPCA Loss end>>" << endl;
#endif
    return -beta.transpose() * Y * beta;
  };

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {
#ifdef TEST
    cout << "<< SPCA sacrifice >>" << endl;
#endif
    VectorXd D = -this->Sigma * beta + beta.transpose() * this->Sigma * beta * beta;

    for (int i = 0; i < A.size(); i++)
    {
      VectorXd temp = beta.segment(g_index(A(i)), g_size(A(i)));
      bd(A(i)) = temp.squaredNorm();
    }
    for (int i = 0; i < I.size(); i++)
    {
      VectorXd temp = D.segment(g_index(I(i)), g_size(I(i)));
      bd(I(i)) = temp.squaredNorm();
    }

#ifdef TEST
    // cout << "  --> A : " << endl
    //      << A << endl;
    // cout << "  --> I : " << endl
    //      << I << endl;
    // cout << "  --> bd : " << endl << bd << endl;
#endif
  };

  double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0)
  {
    //  to be added
    return XA.cols();
  }
};

#endif //SRC_ALGORITHM_H