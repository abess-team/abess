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

  Eigen::VectorXi inital_screening(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &bd, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {
    cout<<"==> init"<<endl;///

    if (bd.size() == 0)
    {
      // variable initialization
      int n = X.rows();
      int p = X.cols();
      bd = Eigen::VectorXd::Zero(N);
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