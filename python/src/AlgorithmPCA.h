#ifndef SRC_ALGORITHMPCA_H
#define SRC_ALGORITHMPCA_H

#include "Algorithm.h"
#include <Spectra/SymEigsSolver.h>

using namespace Spectra;

template <class T4>
class abessPCA : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessPCA(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 1, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessPCA(){};

  void updata_tau(int train_n, int N)
  {
    if (this->pca_n > 0)
      train_n = this->pca_n;
    if (train_n == 1)
    {
      this->tau = 0.0;
    }
    else
    {
      this->tau = 0.01 * (double)this->sparsity_level * log((double)N) * log(log((double)train_n)) / (double)train_n;
    }
  }

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

    int p = x.cols();
    if (p == 0)
      return true;
    if (p == 1)
    {
      beta << 1;
      return true;
    }

    MatrixXd Y = SigmaA(A, g_index, g_size);

    DenseSymMatProd<double> op(Y);
    SymEigsSolver<DenseSymMatProd<double>> eig(op, 1, 2);
    eig.init();
    eig.compute();
    MatrixXd temp;
    if (eig.info() == CompInfo::Successful)
    {
      temp = eig.eigenvectors(1);
    }
    else
    {
      return false;
    }

    beta = temp.col(0);

    return true;
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, double lambda)
  {

    MatrixXd Y = SigmaA(A, g_index, g_size);

    return -beta.transpose() * Y * beta;
  };

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {

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
  };

  double effective_number_of_parameter(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0)
  {
    //  to be added
    return XA.cols();
  }
};

#endif // SRC_ALGORITHMPCA_H