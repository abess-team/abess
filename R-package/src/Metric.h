//
// Created by Mamba on 2020/2/18.
//
// #define R_BUILD
#ifndef SRC_METRICS_H
#define SRC_METRICS_H

#include "Data.h"
#include "Algorithm.h"
#include "model_fit.h"
#include <vector>
#include <random>
#include <algorithm>
#include "utilities.h"

template <class T1, class T2, class T3, class T4>
// To do: calculate loss && all to one && lm poisson cox
class Metric
{
public:
  bool is_cv;
  int Kfold;
  int ic_type;
  // Eigen::Matrix<T2, Dynamic, 1> cv_initial_model_param;
  // Eigen::Matrix<T3, Dynamic, 1> cv_initial_coef0;

  std::vector<Eigen::VectorXi> cv_initial_A;
  std::vector<Eigen::VectorXi> cv_initial_I;

  std::vector<Eigen::VectorXi> train_mask_list;
  std::vector<Eigen::VectorXi> test_mask_list;

  // std::vector<std::vector<T4>> group_XTX_list;

  double ic_coef;

  Metric() = default;

  Metric(int ic_type, double ic_coef = 1.0, bool is_cv = false, int Kfold = 5)
  {
    this->is_cv = is_cv;
    this->ic_type = ic_type;
    this->Kfold = Kfold;
    this->ic_coef = ic_coef;
    // cv_initial_model_param.resize(Kfold, 1);
    // cv_coef0_model_param.resize(Kfold, 1);
  };

  // void set_cv_initial_model_param(int Kfold, int p)
  // {
  //   this->cv_initial_model_param = Eigen::MatrixXd::Zero(p, Kfold);
  // };

  void set_cv_initial_A(int Kfold, int p)
  {
    vector<Eigen::VectorXi> tmp(Kfold);
    this->cv_initial_A = tmp;
  };

  // void set_cv_initial_coef0(int Kfold, int p)
  // {
  //   vector<double> tmp(Kfold);
  //   for (int i = 0; i < Kfold; i++)
  //     tmp[i] = 0;
  //   this->cv_initial_coef0 = tmp;
  // };

  void update_cv_initial_model_param(Eigen::VectorXd model_param, int k)
  {
    this->cv_initial_model_param.col(k) = model_param;
  }

  void update_cv_initial_A(Eigen::VectorXi A, int k)
  {
    this->cv_initial_A[k] = A;
  }

  // void update_cv_initial_coef0(double coef0, int k)
  // {
  //   this->cv_initial_coef0[k] = coef0;
  // }

  void set_cv_train_test_mask(int n)
  {
    Eigen::VectorXi index_list(n);
    std::vector<int> index_vec((unsigned int)n);
    for (int i = 0; i < n; i++)
    {
      index_vec[i] = i;
    }
    // std::random_device rd;
    std::mt19937 g(123);
    std::shuffle(index_vec.begin(), index_vec.end(), g);

    for (int i = 0; i < n; i++)
    {
      index_list(i) = index_vec[i];
    }

    Eigen::VectorXd loss_list(this->Kfold);
    std::vector<Eigen::VectorXi> group_list((unsigned int)this->Kfold);
    int group_size = int(n / this->Kfold);
    for (int k = 0; k < (this->Kfold - 1); k++)
    {
      group_list[k] = index_list.segment(int(k * group_size), group_size);
    }
    group_list[this->Kfold - 1] = index_list.segment(int((this->Kfold - 1) * group_size),
                                                     n - int(int(this->Kfold - 1) * group_size));
    for (int k = 0; k < this->Kfold; k++)
    {
      std::sort(group_list[k].data(), group_list[k].data() + group_list[k].size());
    }

    // cv train-test partition:
    std::vector<Eigen::VectorXi> train_mask_list_tmp((unsigned int)this->Kfold);
    std::vector<Eigen::VectorXi> test_mask_list_tmp((unsigned int)this->Kfold);
    for (int k = 0; k < this->Kfold; k++)
    {
      int train_x_size = n - group_list[k].size();
      // get train_mask
      Eigen::VectorXi train_mask(train_x_size);
      int i = 0;
      for (int j = 0; j < this->Kfold; j++)
      {
        if (j != k)
        {
          for (int s = 0; s < group_list[j].size(); s++)
          {
            train_mask(i) = group_list[j](s);
            i++;
          }
        }
      }
      std::sort(train_mask.data(), train_mask.data() + train_mask.size());
      train_mask_list_tmp[k] = train_mask;
      test_mask_list_tmp[k] = group_list[k];
    }
    // cout << "train_mask[0]: " << train_mask_list_tmp[0] << endl;
    this->train_mask_list = train_mask_list_tmp;
    this->test_mask_list = test_mask_list_tmp;
  };

  // void cal_cv_group_XTX(Data<T1, T2, T3> &data)
  // {
  //   int p = data.p;
  //   Eigen::VectorXi index = data.g_index;
  //   Eigen::VectorXi gsize = data.g_size;
  //   int N = data.g_num;

  //   std::vector<std::vector<Eigen::MatrixXd>> group_XTX_list_tmp(this->Kfold);

  //   for (int k = 0; k < this->Kfold; k++)
  //   {
  //     int train_size = this->train_mask_list[k].size();
  //     Eigen::MatrixXd train_x(train_size, p);

  //     for (int i = 0; i < train_size; i++)
  //     {
  //       train_x.row(i) = data.x.row(this->train_mask_list[k](i));
  //     };
  //     group_XTX_list_tmp[k] = group_XTX(train_x, index, gsize, train_size, p, N, 1);
  //   }
  //   this->group_XTX_list = group_XTX_list_tmp;
  // }

  double ic(int train_n, int M, int N, Algorithm<T1, T2, T3, T4> *algorithm)
  {
    double loss;
    if (algorithm->model_type == 1 || algorithm->model_type == 5)
    {
      loss = train_n * log(algorithm->get_train_loss());
    }
    else
    {
      loss = 2 * algorithm->get_train_loss();
    }

    if (ic_type == 1)
    {
      return loss + 2.0 * algorithm->get_group_df();
    }
    else if (ic_type == 2)
    {
      return loss + this->ic_coef * (double(train_n)) * algorithm->get_group_df();
    }
    else if (ic_type == 3)
    {
      return loss + this->ic_coef * log(double(N)) * log(log(double(train_n))) * algorithm->get_group_df();
    }
    else if (ic_type == 4)
    {
      return loss + this->ic_coef * (log(double(train_n)) + 2 * log(double(N))) * algorithm->get_group_df();
    }
    else
      return 0;
  };

  double neg_loglik_loss(T4 &train_x, T1 &train_y, Eigen::VectorXd &train_weight, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int train_n, int p, int N, Algorithm<T1, T2, T3, T4> *algorithm)
  {
    // clock_t t1 = clock();
    Eigen::VectorXi A = algorithm->get_A_out();
    T2 beta = algorithm->get_beta();
    T3 coef0 = algorithm->get_coef0();

    Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N);
    T4 X_A = X_seg(train_x, train_n, A_ind);

    T2 beta_A;
    slice(beta, A_ind, beta_A);

    // Eigen::VectorXd beta_A(A_ind.size());
    // for (int k = 0; k < A_ind.size(); k++)
    // {
    //   beta_A(k) = beta(A_ind(k));
    // }
    double L0 = algorithm->neg_loglik_loss(X_A, train_y, train_weight, beta_A, coef0);
    // clock_t t2 = clock();
    // std::cout << "ic loss time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;

    return L0;
  }

  // // to do
  //   double fit_and_evaluate_in_metric(Algorithm<T1, T2, T3> *algorithm, Data &data)
  //   {
  //     if (!this->is_cv)
  //     {
  //       algorithm->fit(data.x, data.y, data.weight, data.g_index, data.g_size, data.n, data.p, data.g_num, data.status);
  //       return this->ic(data.n, data.g_num, algorithm);
  //     }
  //     else
  //     {
  //       int k;
  //       Eigen::VectorXi g_index = data.g_index;
  //       Eigen::VectorXi g_size = data.g_size;
  //       int p = data.p;
  //       int N = data.g_num;

  //       Eigen::VectorXd loss_list(this->Kfold);
  //       ///////////////////////parallel/////////////////////////
  //       for (k = 0; k < this->Kfold; k++)
  //       {
  //         //get test_x, test_y
  //         int test_n = this->test_mask_list[k].size();
  //         int train_n = this->train_mask_list[k].size();
  //         // train & test data
  //         Eigen::MatrixXd train_x = matrix_slice(data.x, this->train_mask_list[k], 0);
  //         Eigen::MatrixXd test_x = matrix_slice(data.x, this->test_mask_list[k], 0);
  //         Eigen::VectorXd train_y = vector_slice(data.y, this->train_mask_list[k]);
  //         Eigen::VectorXd test_y = vector_slice(data.y, this->test_mask_list[k]);
  //         Eigen::VectorXd train_weight = vector_slice(data.weight, this->train_mask_list[k]);
  //         Eigen::VectorXd test_weight = vector_slice(data.weight, this->test_mask_list[k]);
  //         Eigen::VectorXd beta_init;

  //         if (algorithm->get_warm_start())
  //         {
  //           beta_init = this->cv_initial_model_param.col(k).eval();
  //           algorithm->update_beta_init(beta_init);
  //           algorithm->update_coef0_init(this->cv_initial_coef0[k]);
  //           algorithm->update_A_init(this->cv_initial_A[k], N);
  //         }
  //         // algorithm->update_train_mask(this->train_mask_list[k]);
  //         /// ??????????????????????????????????????????????????????????????
  //         algorithm->fit(train_x, train_y, train_weight, g_index, g_size, train_n, p, N, data.status);
  //         if (algorithm->get_warm_start())
  //         {
  //           this->update_cv_initial_model_param(algorithm->get_beta(), k);
  //           this->update_cv_initial_A(algorithm->get_A_out(), k);
  //           this->update_cv_initial_coef0(algorithm->get_coef0(), k);
  //         }

  //         loss_list(k) = this->neg_loglik_loss(test_x, test_y, test_weight, g_index, g_size, test_n, p, N, algorithm);
  //       }
  //     }
  //   };
};

#endif //SRC_METRICS_H