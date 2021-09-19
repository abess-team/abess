#ifndef SRC_ALGORITHMISING_H
#define SRC_ALGORITHMISING_H

#include "Algorithm.h"

template <class T4>
class abessIsing : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessIsing(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 1, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessIsing(){};

  Eigen::MatrixXd theta, theta_A;

  void ind_transfer(Eigen::VectorXi &ind1, Eigen::MatrixXi &ind2, int N, int type = 1){
    // type == 1: from single to full
    // type == 2: from full to single
    if (type == 1){
      ind2 = Eigen::MatrixXi::Zero(ind1.rows(), 2);
      ind2.col(1) = ind1 / N;
      ind2.col(0) = ind1 - ind2.col(0) * N;
    }else{
      ind1 = Eigen::MatrixXi::Zero(ind2.rows(), 1);
      ind1 = ind2.col(0) + ind2.col(1) * N;
    }
  }

  void get_A(T4 &X, Eigen::VectorXd &y, Eigen::VectorXi &A, Eigen::VectorXi &I, int &C_max, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXd &bd, int T0, Eigen::VectorXd &weights,
             Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, double tau, double &train_loss)
  {
    cout<<"==> get_A | T0 = "<<T0<<endl;///

    Eigen::VectorXi U(this->U_size);
    Eigen::VectorXi U_ind;
    // Eigen::VectorXi g_index_U(this->U_size);
    // Eigen::VectorXi g_size_U(this->U_size);
    T4 *X_U = new T4;
    Eigen::VectorXd beta_U;
    Eigen::VectorXi A_U(T0);
    Eigen::VectorXi I_U(this->U_size - T0);
    Eigen::VectorXi always_select_U(this->always_select.size());

    if (this->U_size == N*N)
    {
      U = Eigen::VectorXi::LinSpaced(N*N, 0, N*N - 1);
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
      // mapping
      if (this->U_size == N*N)
      {
        delete X_U;
        X_U = &X;
        U_ind = Eigen::VectorXi::LinSpaced(N*N, 0, N*N - 1);
        beta_U = beta;
        // g_size_U = g_size;
        // g_index_U = g_index;
        A_U = A;
        I_U = I;
        always_select_U = this->always_select;
      }
      else
      {
        U_ind = U;
        *X_U = X_seg_graph(X, n, U_ind);
        slice(beta, U_ind, beta_U);

        // int pos = 0;
        // for (int i = 0; i < U.size(); i++)
        // {
        //   g_size_U(i) = g_size(U(i));
        //   g_index_U(i) = pos;
        //   pos += g_size_U(i);
        // }

        A_U = Eigen::VectorXi::LinSpaced(T0, 0, T0 - 1);
        I_U = Eigen::VectorXi::LinSpaced(this->U_size - T0, T0, this->U_size - 1);

        int *temp = new int[N*N], s = this->always_select.size();
        memset(temp, 0, N*N);
        for (int i = 0; i < s; i++)
          temp[this->always_select(i)] = 1;
        for (int i = 0; i < this->U_size; i++)
        {
          if (s <= 0)
            break;
          if (temp[U(i)] == 1)
          {
            always_select_U(this->always_select.size() - s) = i;
            s--;
          }
        }
        delete[] temp;
      }

      int num = -1;
      while (true)
      {
        num++;

        Eigen::VectorXi A_ind = A_U;
        T4 X_A = X_seg_graph(*X_U, n, A_ind);
        Eigen::VectorXd beta_A;
        slice(beta_U, A_ind, beta_A);

        Eigen::VectorXd bd_U = Eigen::VectorXd::Zero(this->U_size);
        this->sacrifice(*X_U, X_A, y, beta_U, beta_A, coef0, A_U, I_U, weights, g_index, g_size, this->U_size, A_ind, bd_U, U, U_ind, num);

        for (int i = 0; i < always_select_U.size(); i++)
        {
          bd_U(always_select_U(i)) = DBL_MAX;
        }

        double l0 = train_loss;
        bool exchange = this->splicing(*X_U, y, A_U, I_U, C_max, beta_U, coef0, bd_U, weights,
                                       g_index, g_size, this->U_size, tau, l0);

        if (exchange)
          train_loss = l0;
        else
          break; // A_U is stable
      }

      if (A_U.size() == 0 || A_U.maxCoeff() == T0 - 1)
        break; // if A_U not change, stop

      // store beta, A, I
      slice_restore(beta_U, U_ind, beta);

      int *temp = new int[N*N];
      memset(temp, 0, N*N);
      for (int i = 0; i < T0; i++)
        temp[U(A_U(i))] = 1;

      int tempA = 0, tempI = 0;
      for (int i = 0; i < N*N; i++)
        if (temp[i] == 0)
          I(tempI++) = i;
        else
          A(tempA++) = i;
      
      delete[] temp;

      // bd in full set
      Eigen::VectorXi A_ind0 = A;
      T4 X_A0 = X_seg_graph(X, n, A_ind0);
      Eigen::VectorXd beta_A0;
      slice(beta, A_ind0, beta_A0);
      Eigen::VectorXi U0 = Eigen::VectorXi::LinSpaced(N*N, 0, N*N - 1);
      Eigen::VectorXi U_ind0 = Eigen::VectorXi::LinSpaced(p*p, 0, p*p - 1);
      this->sacrifice(X, X_A0, y, beta, beta_A0, coef0, A, I, weights, g_index, g_size, N, A_ind0, bd, U0, U_ind0, 0);

      if (this->U_size == N*N)
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

    if (this->U_size != N*N)
      delete X_U;

    return;
  };

  Eigen::VectorXi inital_screening(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &bd, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {
    cout<<"==> init"<<endl;///
    this->theta = beta;
    this->theta.resize(N, N);

    if (bd.size() == 0)
    {
      // variable initialization
      int n = X.rows();
      int p = X.cols();
      bd = Eigen::VectorXd::Zero(N*N);
    }

    // get Active-set A according to max_k bd
    Eigen::VectorXi A_new = max_k(bd, this->sparsity_level);

    return A_new;
  }

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    cout<<" --> primary fit"<<endl;///
    return true;
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    cout<<" --> loss"<<endl;///
    return 1;
  };

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {
    cout<<" --> sacrifice"<<endl;///
  };

  double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0)
  {
    //  to be added
    return XA.cols();
  }
};

#endif // SRC_ALGORITHMISING_H