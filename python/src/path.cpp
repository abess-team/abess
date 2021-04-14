//
// Created by Mamba on 2020/2/18.
//
// #define R_BUILD
#define TEST

#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppEigen)]]
#else

#include <Eigen/Eigen>
#include "List.h"

#endif

#include <iostream>
#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"
#include "abess.h"

using namespace Eigen;
using namespace std;

// template <class T1, class T2, class T3>
// void sequential_path_cv(Data<T1, T2, T3> &data, Algorithm<T1, T2, T3> *algorithm, Metric<T1, T2, T3> *metric, Eigen::VectorXi &sequence, Eigen::VectorXd &lambda_seq, bool early_stop, int k, Result<T2, T3> &result)
// {
// #ifdef TEST
//   clock_t t0, t1, t2;
// #endif
//   int p = data.get_p();
//   int N = data.g_num;
//   int M = data.y.cols();
//   Eigen::VectorXi g_index = data.g_index;
//   Eigen::VectorXi g_size = data.g_size;
//   Eigen::VectorXi status = data.status;
//   int sequence_size = sequence.size();
//   int lambda_size = lambda_seq.size();
//   // int early_stop_s = sequence_size;

//   Eigen::VectorXi train_mask, test_mask;
//   T1 train_y, test_y;
//   Eigen::VectorXd train_weight, test_weight;
//   Eigen::MatrixXd train_x, test_x;
//   int train_n = 0, test_n = 0;

//   // train & test data
//   if (!metric->is_cv)
//   {
//     train_x = data.x;
//     train_y = data.y;
//     train_weight = data.weight;
//     train_n = data.n;
//   }
//   else
//   {
//     train_mask = metric->train_mask_list[k];
//     test_mask = metric->test_mask_list[k];
//     slice(data.x, train_mask, train_x);
//     slice(data.x, test_mask, test_x);
//     slice(data.y, train_mask, train_y);
//     slice(data.y, test_mask, test_y);
//     slice(data.weight, train_mask, train_weight);
//     slice(data.weight, test_mask, test_weight);

//     train_n = train_mask.size();
//     test_n = test_mask.size();
//   }

//   vector<Eigen::MatrixXd> train_group_XTX = group_XTX(train_x, g_index, g_size, train_n, p, N, algorithm->model_type);
//   if (algorithm->covariance_update)
//   {
//     algorithm->covariance_update_flag = Eigen::VectorXi::Zero(p);
//     algorithm->XTy = train_x.transpose() * train_y;

//     // to do : add ifelse
//     algorithm->XTone = train_x.transpose() * Eigen::MatrixXd::Ones(train_n, M);
//   }

//   Eigen::Matrix<T2, Dynamic, Dynamic> beta_matrix(sequence_size, lambda_size);
//   Eigen::Matrix<T3, Dynamic, Dynamic> coef0_matrix(sequence_size, lambda_size);
//   Eigen::MatrixXd train_loss_matrix(sequence_size, lambda_size);
//   Eigen::MatrixXd ic_matrix(sequence_size, lambda_size);
//   Eigen::MatrixXd test_loss_matrix(sequence_size, lambda_size);
//   Eigen::Matrix<VectorXd, Dynamic, Dynamic> bd_matrix(sequence_size, lambda_size);
//   // Eigen::Matrix<VectorXi, Dynamic, Dynamic> A_matrix(sequence_size, lambda_size);

//   // to ensure
//   T2 beta_init;
//   T3 coef0_init;
//   coef_set_zero(p, M, beta_init, coef0_init);
//   Eigen::VectorXi A_init;
//   Eigen::VectorXd bd_init;

//   for (int i = 0; i < sequence_size; i++)
//   {
// #ifdef TEST
//     std::cout << "\n sequence= " << sequence(i);
// #endif
//     for (int j = (1 - pow(-1, i)) * (lambda_size - 1) / 2; j < lambda_size && j >= 0; j = j + pow(-1, i))
//     {
// #ifdef TEST
//       std::cout << " =========j: " << j << ", lambda= " << lambda_seq(j) << ", T: " << sequence(i) << endl;
//       t0 = clock();
//       t1 = clock();
// #endif

//       algorithm->update_sparsity_level(sequence(i));
//       algorithm->update_lambda_level(lambda_seq(j));
//       algorithm->update_beta_init(beta_init);
//       algorithm->update_bd_init(bd_init);
//       algorithm->update_coef0_init(coef0_init);
//       algorithm->update_A_init(A_init, N);
//       algorithm->update_group_XTX(train_group_XTX);

//       algorithm->fit(train_x, train_y, train_weight, g_index, g_size, train_n, p, N, status);
// #ifdef TEST
//       t2 = clock();
//       std::cout << "fit time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
//       t1 = clock();
// #endif

//       if (algorithm->warm_start)
//       {
//         beta_init = algorithm->get_beta();
//         coef0_init = algorithm->get_coef0();
//         // A_init = algorithm->get_A_out();
//         bd_init = algorithm->get_bd();
//       }
// #ifdef TEST
//       t2 = clock();
//       std::cout << "warm start time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
//       t1 = clock();
// #endif

//       // evaluate the beta
//       if (metric->is_cv)
//       {
//         test_loss_matrix(i, j) = metric->neg_loglik_loss(test_x, test_y, test_weight, g_index, g_size, test_n, p, N, algorithm);
//       }
//       else
//       {
//         ic_matrix(i, j) = metric->ic(train_n, N, algorithm);
//       }
// #ifdef TEST
//       t2 = clock();
//       std::cout << "ic time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
//       t1 = clock();
// #endif

//       // save for best_model fit
//       beta_matrix(i, j) = algorithm->beta;
//       coef0_matrix(i, j) = algorithm->coef0;
//       train_loss_matrix(i, j) = algorithm->get_train_loss();
//       // A_matrix(i, j) = algorithm->A_out;
//       bd_matrix(i, j) = algorithm->bd;

// #ifdef TEST
//       t2 = clock();
//       std::cout << "save time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
//       std::cout << "path i= " << i << " j= " << j << " time = " << ((double)(t2 - t0) / CLOCKS_PER_SEC) << endl;
// #endif
//     }

//     // To be ensured
//     // if (early_stop && lambda_size <= 1 && i >= 3)
//     // {
//     //     bool condition1 = ic_sequence(i, 0) > ic_sequence(i - 1, 0);
//     //     bool condition2 = ic_sequence(i - 1, 0) > ic_sequence(i - 2, 0);
//     //     bool condition3 = ic_sequence(i - 2, 0) > ic_sequence(i - 3, 0);
//     //     if (condition1 && condition2 && condition3)
//     //     {
//     //         early_stop_s = i + 1;
//     //         break;
//     //     }
//     // }
//   }

//   // if (early_stop)
//   // {
//   //     ic_sequence = ic_sequence.block(0, 0, early_stop_s, lambda_size).eval();
//   // }

//   result.beta_matrix = beta_matrix;
//   result.coef0_matrix = coef0_matrix;
//   result.train_loss_matrix = train_loss_matrix;
//   result.bd_matrix = bd_matrix;
//   // result.A_matrix = A_matrix;
//   result.ic_matrix = ic_matrix;
//   result.test_loss_matrix = test_loss_matrix;
// }

// List gs_path(Data &data, Algorithm *algorithm, Metric *metric, int s_min, int s_max, int K_max, double epsilon)
// {
//     int p = data.get_p();
//     int n = data.get_n();
//     int i;
//     vector<Eigen::MatrixXd> full_group_XTX = group_XTX(data.x, data.g_index, data.g_size, data.n, data.p, data.g_num, algorithm->model_type);

//     Eigen::VectorXi full_mask(n);
//     for (i = 0; i < n; i++)
//     {
//         full_mask(i) = int(i);
//     }
//     Eigen::MatrixXd beta_matrix = Eigen::MatrixXd::Zero(p, 4);
//     Eigen::VectorXd coef0_sequence = Eigen::VectorXd::Zero(4);
//     Eigen::VectorXd train_loss_sequence = Eigen::VectorXd::Zero(4);
//     Eigen::VectorXd ic_sequence = Eigen::VectorXd::Zero(4);

//     Eigen::MatrixXd beta_all = Eigen::MatrixXd::Zero(p, 100);
//     Eigen::VectorXd coef0_all = Eigen::VectorXd::Zero(100);
//     Eigen::VectorXd train_loss_all = Eigen::VectorXd::Zero(100);
//     Eigen::VectorXd ic_all = Eigen::VectorXd::Zero(100);

//     Eigen::VectorXd beta_init = Eigen::VectorXd::Zero(p);
//     double coef0_init = 0.0;

//     int Tmin = s_min;
//     int Tmax = s_max;
//     int T1 = round(0.618 * Tmin + 0.382 * Tmax);
//     int T2 = round(0.382 * Tmin + 0.618 * Tmax);
//     double icT1;
//     double icT2;

//     algorithm->update_train_mask(full_mask);
//     algorithm->update_sparsity_level(T1);
//     algorithm->update_beta_init(beta_init);
//     algorithm->update_coef0_init(coef0_init);
//     algorithm->update_group_XTX(full_group_XTX);

//     // algorithm->fit();

//     ic_sequence(1) = metric->fit_and_evaluate_in_metric(algorithm, data);

//     if (algorithm->warm_start)
//     {
//         beta_init = algorithm->get_beta();
//         coef0_init = algorithm->get_coef0();
//     }

//     beta_matrix.col(1) = algorithm->get_beta();
//     coef0_sequence(1) = algorithm->get_coef0();
//     // train_loss_sequence(1) = metric->train_loss(algorithm, data);
//     // ic_sequence(1) = metric->ic(algorithm, data);
//     beta_all.col(0) = beta_matrix.col(1);
//     coef0_all(0) = coef0_sequence(1);
//     train_loss_all(0) = train_loss_sequence(1);
//     ic_all(0) = ic_sequence(1);
//     icT1 = ic_sequence(1);

//     algorithm->update_train_mask(full_mask);
//     algorithm->update_sparsity_level(T2);
//     algorithm->update_beta_init(beta_init);
//     algorithm->update_coef0_init(coef0_init);
//     algorithm->update_group_XTX(full_group_XTX);
//     algorithm->fit();
//     if (algorithm->warm_start)
//     {
//         beta_init = algorithm->get_beta();
//         coef0_init = algorithm->get_coef0();
//     }

//     beta_matrix.col(2) = algorithm->get_beta();
//     coef0_sequence(2) = algorithm->get_coef0();
//     train_loss_sequence(2) = metric->train_loss(algorithm, data);
//     ic_sequence(2) = metric->ic(algorithm, data);
//     beta_all.col(1) = beta_matrix.col(2);
//     coef0_all(1) = coef0_sequence(2);
//     train_loss_all(1) = train_loss_sequence(2);
//     ic_all(1) = ic_sequence(2);

//     icT2 = metric->ic(algorithm, data);

//     int iter = 2;
//     while (T1 != T2)
//     {
//         if (icT1 < icT2)
//         {
//             Tmax = T2;
//             beta_matrix.col(3) = beta_matrix.col(2);
//             coef0_sequence(3) = coef0_sequence(2);
//             train_loss_sequence(3) = train_loss_sequence(2);
//             ic_sequence(3) = ic_sequence(2);

//             T2 = T1;
//             beta_matrix.col(2) = beta_matrix.col(1);
//             coef0_sequence(2) = coef0_sequence(1);
//             train_loss_sequence(2) = train_loss_sequence(1);
//             ic_sequence(2) = ic_sequence(1);
//             icT2 = ic_sequence(1);

//             T1 = round(0.618 * Tmin + 0.382 * Tmax);
//             algorithm->update_train_mask(full_mask);
//             algorithm->update_sparsity_level(T1);
//             algorithm->update_beta_init(beta_init);
//             algorithm->update_coef0_init(coef0_init);
//             algorithm->update_group_XTX(full_group_XTX);
//             algorithm->fit();
//             if (algorithm->warm_start)
//             {
//                 beta_init = algorithm->get_beta();
//                 coef0_init = algorithm->get_coef0();
//             }
//             beta_matrix.col(1) = algorithm->get_beta();
//             coef0_sequence(1) = algorithm->get_coef0();
//             train_loss_sequence(1) = metric->train_loss(algorithm, data);
//             ic_sequence(1) = metric->ic(algorithm, data);

//             beta_all.col(iter) = beta_matrix.col(1);
//             coef0_all(iter) = coef0_sequence(1);
//             train_loss_all(iter) = train_loss_sequence(1);
//             ic_all(iter) = ic_sequence(1);
//             iter++;

//             icT1 = metric->ic(algorithm, data);
//         }
//         else
//         {
//             Tmin = T1;
//             beta_matrix.col(0) = beta_matrix.col(1);
//             coef0_sequence(0) = coef0_sequence(1);
//             train_loss_sequence(0) = train_loss_sequence(1);
//             ic_sequence(0) = ic_sequence(1);

//             T1 = T2;
//             beta_matrix.col(1) = beta_matrix.col(2);
//             coef0_sequence(1) = coef0_sequence(2);
//             train_loss_sequence(1) = train_loss_sequence(2);
//             ic_sequence(1) = ic_sequence(2);
//             icT1 = ic_sequence(2);

//             T2 = round(0.382 * Tmin + 0.618 * Tmax);
//             algorithm->update_train_mask(full_mask);
//             algorithm->update_sparsity_level(T2);
//             algorithm->update_beta_init(beta_init);
//             algorithm->update_coef0_init(coef0_init);
//             algorithm->update_group_XTX(full_group_XTX);
//             algorithm->fit();
//             if (algorithm->warm_start)
//             {
//                 beta_init = algorithm->get_beta();
//                 coef0_init = algorithm->get_coef0();
//             }

//             beta_matrix.col(2) = algorithm->get_beta();
//             coef0_sequence(2) = algorithm->get_coef0();
//             train_loss_sequence(2) = metric->train_loss(algorithm, data);
//             ic_sequence(2) = metric->ic(algorithm, data);

//             beta_all.col(iter) = beta_matrix.col(2);
//             coef0_all(iter) = coef0_sequence(2);
//             train_loss_all(iter) = train_loss_sequence(2);
//             ic_all(iter) = ic_sequence(2);
//             iter++;

//             icT2 = metric->ic(algorithm, data);
//         };
//     }
//     Eigen::VectorXd best_beta = Eigen::VectorXd::Zero(p);
//     double best_coef0 = 0;
//     double best_train_loss = 0;
//     double best_ic = DBL_MAX;
//     for (int T_tmp = Tmin; T_tmp <= Tmax; T_tmp++)
//     {
//         algorithm->update_train_mask(full_mask);
//         algorithm->update_sparsity_level(T_tmp);
//         algorithm->update_beta_init(beta_init);
//         algorithm->update_coef0_init(coef0_init);
//         algorithm->update_group_XTX(full_group_XTX);
//         algorithm->fit();
//         if (algorithm->warm_start)
//         {
//             beta_init = algorithm->get_beta();
//             coef0_init = algorithm->get_coef0();
//         }
//         double ic_tmp = metric->ic(algorithm, data);
//         if (ic_tmp < best_ic)
//         {
//             best_beta = algorithm->get_beta();
//             best_coef0 = algorithm->get_coef0();
//             best_train_loss = metric->train_loss(algorithm, data);
//             best_ic = ic_tmp;

//             beta_all.col(iter) = best_beta;
//             coef0_all(iter) = best_coef0;
//             train_loss_all(iter) = best_train_loss;
//             ic_all(iter) = best_ic;
//             iter++;
//         }
//     }

//     if (data.is_normal)
//     {
//         if (algorithm->model_type == 1)
//         {
//             best_beta = sqrt(double(n)) * best_beta.cwiseQuotient(data.x_norm);
//             best_coef0 = data.y_mean - best_beta.dot(data.x_mean);
//         }
//         else
//         {
//             best_beta = sqrt(double(n)) * best_beta.cwiseQuotient(data.x_norm);
//             best_coef0 = best_coef0 - best_beta.dot(data.x_mean);
//         }
//     }
//     beta_all = beta_all.leftCols(iter).eval();
//     coef0_all = coef0_all.head(iter).eval();
//     train_loss_all = train_loss_all.head(iter).eval();
//     ic_all = ic_all.head(iter).eval();

//     if (data.is_normal)
//     {
//         if (algorithm->model_type == 1)
//         {
//             for (int k = 0; k < iter; k++)
//             {
//                 beta_all.col(k) = sqrt(double(n)) * beta_all.col(k).cwiseQuotient(data.x_norm);
//                 coef0_all(k) = data.y_mean - beta_all.col(k).dot(data.x_mean);
//             }
//         }
//         else if (data.data_type == 2)
//         {
//             for (int k = 0; k < iter; k++)
//             {
//                 beta_all.col(k) = sqrt(double(n)) * beta_all.col(k).cwiseQuotient(data.x_norm);
//                 coef0_all(k) = coef0_all(k) - beta_all.col(k).dot(data.x_mean);
//             }
//         }
//         else
//         {
//             for (int k = 0; k < iter; k++)
//             {
//                 beta_all.col(k) = sqrt(double(n)) * beta_all.col(k).cwiseQuotient(data.x_norm);
//             }
//         }
//     }

// #ifdef R_BUILD
//     return List::create(Named("beta") = best_beta, Named("coef0") = best_coef0, Named("train_loss") = best_train_loss, Named("ic") = best_ic,
//                         Named("beta_all") = beta_all,
//                         Named("coef0_all") = coef0_all,
//                         Named("train_loss_all") = train_loss_all,
//                         Named("ic_all") = ic_all);
// #else
//     List mylist;
//     mylist.add("beta", best_beta);
//     mylist.add("coef0", best_coef0);
//     mylist.add("train_loss", best_train_loss);
//     mylist.add("ic", best_ic);
//     return mylist;
// #endif
// }

int sign(double a)
{
  if (a > 0)
  {
    return 1;
  }
  else if (a < 0)
  {
    return -1;
  }
  else
  {
    return 0;
  }
}

double det(double a[], double b[])
{
  return a[0] * b[1] - a[1] * b[0];
}

// calculate the intersection of two lines
// if parallal, need_flag = false.
void line_intersection(double line1[2][2], double line2[2][2], double intersection[], bool &need_flag)
{
  double xdiff[2], ydiff[2], d[2];
  double div;

  xdiff[0] = line1[0][0] - line1[1][0];
  xdiff[1] = line2[0][0] - line2[1][0];
  ydiff[0] = line1[0][1] - line1[1][1];
  ydiff[1] = line2[0][1] - line2[1][1];

  div = det(xdiff, ydiff);
  if (div == 0)
  {
    need_flag = false;
    return;
  }
  else
  {
    d[0] = det(line1[0], line1[1]);
    d[1] = det(line2[0], line2[1]);

    intersection[0] = det(d, xdiff) / div;
    intersection[1] = det(d, ydiff) / div;
    need_flag = true;
    return;
  }
}

// boundary: s=smin, s=max, lambda=lambda_min, lambda_max
// line: crosses p and is parallal to u
// calculate the intersections between boundary and line
void cal_intersections(double p[], double u[], int s_min, int s_max, double lambda_min, double lambda_max, double a[], double b[])
{
  double line0[2][2], line_set[4][2][2], intersections[4][2];
  bool need_flag[4];
  int i, j;

  line0[0][0] = double(p[0]);
  line0[0][1] = double(p[1]);
  line0[1][0] = double(p[0] + u[0]);
  line0[1][1] = double(p[1] + u[1]);

  line_set[0][0][0] = double(s_min);
  line_set[0][0][1] = double(lambda_min);
  line_set[0][1][0] = double(s_min);
  line_set[0][1][1] = double(lambda_max);

  line_set[1][0][0] = double(s_max);
  line_set[1][0][1] = double(lambda_min);
  line_set[1][1][0] = double(s_max);
  line_set[1][1][1] = double(lambda_max);

  line_set[2][0][0] = double(s_min);
  line_set[2][0][1] = double(lambda_min);
  line_set[2][1][0] = double(s_max);
  line_set[2][1][1] = double(lambda_min);

  line_set[3][0][0] = double(s_min);
  line_set[3][0][1] = double(lambda_max);
  line_set[3][1][0] = double(s_max);
  line_set[3][1][1] = double(lambda_max);

  for (i = 0; i < 4; i++)
  {
    line_intersection(line0, line_set[i], intersections[i], need_flag[i]);
  }

  // delete intersections beyond boundary
  for (i = 0; i < 4; i++)
  {
    if (need_flag[i])
    {
      if ((intersections[i][0] < s_min - 0.0001) | (intersections[i][0] > s_max + 0.0001) | (intersections[i][1] < lambda_min - 0.001) | (intersections[i][1] > lambda_max + 0.001))
      {
        need_flag[i] = false;
      }
    }
  }

  // delecte repetitive intersections
  for (i = 0; i < 4; i++)
  {
    if (need_flag[i])
    {
      for (j = i + 1; j < 4; j++)
      {
        if (need_flag[j])
        {
          if (abs(intersections[i][0] - intersections[j][0]) < 0.0001 && abs(intersections[i][1] - intersections[j][1]) < 0.0001)
          {
            need_flag[j] = false;
          }
        }
      }
    }
  }

  j = 0;
  for (i = 0; i < 4; i++)
  {
    if (need_flag[i])
    {
      if (j == 2)
      {
        j += 1;
      }
      if (j == 1)
      {
        b[0] = intersections[i][0];
        b[1] = intersections[i][1];
        j += 1;
      }
      if (j == 0)
      {
        a[0] = intersections[i][0];
        a[1] = intersections[i][1];
        j += 1;
      }
    }
  }

  if (j != 2)
  {
#ifdef R_BUILD
    Rcpp::Rcout << "---------------------------" << endl;
    Rcpp::Rcout << "j: " << j << endl;
    Rcpp::Rcout << "inetrsection numbers wrong" << j << endl;
    Rcpp::Rcout << "p" << p[0] << "," << p[1] << endl;
    Rcpp::Rcout << "u" << u[0] << "," << u[1] << endl;
    Rcpp::Rcout << "s_min" << s_min << endl;
    Rcpp::Rcout << "s_max" << s_max << endl;
    Rcpp::Rcout << "lambda_min" << lambda_min << endl;
    Rcpp::Rcout << "lambda_max" << lambda_max << endl;
    Rcpp::Rcout << "intersections[0]" << intersections[0][0] << "," << intersections[0][1] << endl;
    Rcpp::Rcout << "intersections[1]" << intersections[1][0] << "," << intersections[1][1] << endl;
    Rcpp::Rcout << "intersections[2]" << intersections[2][0] << "," << intersections[2][1] << endl;
    Rcpp::Rcout << "intersections[3]" << intersections[3][0] << "," << intersections[3][1] << endl;
    Rcpp::Rcout << "need_flag[0]" << need_flag[0] << endl;
    Rcpp::Rcout << "need_flag[1]" << need_flag[1] << endl;
    Rcpp::Rcout << "need_flag[2]" << need_flag[2] << endl;
    Rcpp::Rcout << "need_flag[3]" << need_flag[3] << endl;
#else
    cout << "---------------------------" << endl;
    cout << "j: " << j << endl;
    cout << "inetrsection numbers wrong" << j << endl;
    cout << "p" << p[0] << "," << p[1] << endl;
    cout << "u" << u[0] << "," << u[1] << endl;
    cout << "s_min" << s_min << endl;
    cout << "s_max" << s_max << endl;
    cout << "lambda_min" << lambda_min << endl;
    cout << "lambda_max" << lambda_max << endl;
    cout << "intersections[0]" << intersections[0][0] << "," << intersections[0][1] << endl;
    cout << "intersections[1]" << intersections[1][0] << "," << intersections[1][1] << endl;
    cout << "intersections[2]" << intersections[2][0] << "," << intersections[2][1] << endl;
    cout << "intersections[3]" << intersections[3][0] << "," << intersections[3][1] << endl;
    cout << "need_flag[0]" << need_flag[0] << endl;
    cout << "need_flag[1]" << need_flag[1] << endl;
    cout << "need_flag[2]" << need_flag[2] << endl;
    cout << "need_flag[3]" << need_flag[3] << endl;
#endif
  }

  return;
}

// void golden_section_search(Data &data, Algorithm *algorithm, Metric *metric, double p[], double u[], int s_min, int s_max, double log_lambda_min, double log_lambda_max, double best_arg[],
//                            Eigen::VectorXd &beta1, double &coef01, double &train_loss1, double &ic1, Eigen::MatrixXd &ic_sequence)
// {
//     int n = data.get_n();
//     Eigen::VectorXi full_mask(n);
//     for (int i = 0; i < n; i++)
//     {
//         full_mask(i) = int(i);
//     }
//     vector<Eigen::MatrixXd> full_group_XTX = group_XTX(data.x, data.g_index, data.g_size, data.n, data.p, data.g_num, algorithm->model_type);

//     Eigen::VectorXd beta_init = Eigen::VectorXd::Zero(data.get_p());
//     Eigen::VectorXd beta_temp1 = Eigen::VectorXd::Zero(data.get_p());
//     Eigen::VectorXd beta_temp2 = Eigen::VectorXd::Zero(data.get_p());
//     double train_loss_temp1, train_loss_temp2;
//     double coef0_temp1, coef0_temp2;
//     double coef0_init = 0.0;
//     int ic_row, ic_col;
//     double d_lambda = (log_lambda_max - log_lambda_min) / 99;

//     double invphi, invphi2, closs, dloss;
//     double a[2], b[2], c[2], d[2], h[2];

//     // stop condiction
//     double s_tol = 2;
//     double log_lambda_tol = (log_lambda_max - log_lambda_min) / 200;

//     invphi = (pow(5, 0.5) - 1.0) / 2.0;
//     invphi2 = (3.0 - pow(5, 0.5)) / 2.0;
//     cal_intersections(p, u, s_min, s_max, log_lambda_min, log_lambda_max, a, b);

//     h[0] = b[0] - a[0];
//     h[1] = b[1] - a[1];

//     c[0] = a[0] + invphi2 * h[0];
//     c[1] = a[1] + invphi2 * h[1];
//     d[0] = a[0] + invphi * h[0];
//     d[1] = a[1] + invphi * h[1];

//     if (h[0] > 0.0001)
//     {
//         c[0] = int(c[0]);
//         d[0] = ceil(d[0]);
//     }
//     else if (h[0] < -0.0001)
//     {
//         c[0] = ceil(c[0]);
//         d[0] = int(d[0]);
//     }
//     else
//     {
//         c[0] = round(c[0]);
//         d[0] = round(d[0]);
//     }

//     algorithm->update_train_mask(full_mask);
//     algorithm->update_sparsity_level(int(c[0]));
//     algorithm->update_lambda_level(exp(c[1]));
//     algorithm->update_beta_init(beta_init);
//     algorithm->update_coef0_init(coef0_init);
//     algorithm->update_group_XTX(full_group_XTX);
//     algorithm->fit();
//     if (algorithm->warm_start)
//     {
//         beta_init = algorithm->get_beta();
//         coef0_init = algorithm->get_coef0();
//     }
//     closs = metric->ic(algorithm, data);
//     coef0_temp1 = algorithm->get_coef0();
//     beta_temp1 = algorithm->get_beta();
//     train_loss_temp1 = metric->train_loss(algorithm, data);

//     ic_row = int(c[0]);
//     ic_col = floor((c[1] - log_lambda_min) / d_lambda);
//     if (std::isnan(ic_sequence(ic_row, ic_col)))
//     {
//         ic_sequence(ic_row, ic_col) = closs;
//     }
//     else
//     {
//         ic_sequence(ic_row, ic_col) = (ic_sequence(ic_row, ic_col) > closs) ? closs : ic_sequence(ic_row, ic_col);
//     }

//     algorithm->update_train_mask(full_mask);
//     algorithm->update_sparsity_level(int(d[0]));
//     algorithm->update_lambda_level(exp(d[1]));
//     algorithm->update_beta_init(beta_init);
//     algorithm->update_coef0_init(coef0_init);
//     algorithm->update_group_XTX(full_group_XTX);
//     algorithm->fit();
//     if (algorithm->warm_start)
//     {
//         beta_init = algorithm->get_beta();
//         coef0_init = algorithm->get_coef0();
//     }

//     dloss = metric->ic(algorithm, data);
//     coef0_temp2 = algorithm->get_coef0();
//     beta_temp2 = algorithm->get_beta();
//     train_loss_temp2 = metric->train_loss(algorithm, data);
//     ic_row = int(d[0]);
//     ic_col = floor((d[1] - log_lambda_min) / d_lambda);
//     if (std::isnan(ic_sequence(ic_row, ic_col)))
//     {
//         ic_sequence(ic_row, ic_col) = dloss;
//     }
//     else
//     {
//         ic_sequence(ic_row, ic_col) = (ic_sequence(ic_row, ic_col) > dloss) ? dloss : ic_sequence(ic_row, ic_col);
//     }

//     if (abs((invphi2 - invphi) * h[0]) <= s_tol && abs((invphi2 - invphi) * h[1]) < log_lambda_tol)
//     {
//         double min_loss;
//         double tmp_loss;
//         if (closs < dloss)
//         {
//             best_arg[0] = c[0];
//             best_arg[1] = c[1];
//             min_loss = closs;

//             beta1 = beta_temp1;
//             coef01 = coef0_temp1;
//             ic1 = closs;
//             train_loss1 = train_loss_temp1;
//         }
//         else
//         {
//             best_arg[0] = d[0];
//             best_arg[1] = d[1];
//             min_loss = dloss;

//             beta1 = beta_temp2;
//             coef01 = coef0_temp2;
//             ic1 = dloss;
//             train_loss1 = train_loss_temp2;
//         }
//         for (int i = 1; i < abs((invphi2 - invphi) * h[0]); i++)
//         {
//             algorithm->update_train_mask(full_mask);
//             algorithm->update_sparsity_level(int(c[0] + sign(h[0]) * i));
//             algorithm->update_lambda_level(exp(c[1]));
//             algorithm->update_beta_init(beta_init);
//             algorithm->update_coef0_init(coef0_init);
//             algorithm->update_group_XTX(full_group_XTX);
//             algorithm->fit();
//             if (algorithm->warm_start)
//             {
//                 beta_init = algorithm->get_beta();
//                 coef0_init = algorithm->get_coef0();
//             }

//             tmp_loss = metric->ic(algorithm, data);
//             ic_row = int(c[0]);
//             ic_col = floor((c[1] - log_lambda_min) / d_lambda);
//             if (std::isnan(ic_sequence(ic_row, ic_col)))
//             {
//                 ic_sequence(ic_row, ic_col) = tmp_loss;
//             }
//             else
//             {
//                 ic_sequence(ic_row, ic_col) = (ic_sequence(ic_row, ic_col) > tmp_loss) ? tmp_loss : ic_sequence(ic_row, ic_col);
//             }
//             if (tmp_loss < min_loss)
//             {
//                 best_arg[0] = c[0] + sign(h[0]) * i;
//                 best_arg[1] = c[1];
//                 min_loss = tmp_loss;

//                 beta1 = algorithm->get_beta();
//                 coef01 = algorithm->get_coef0();
//                 train_loss1 = metric->train_loss(algorithm, data);
//                 ic1 = min_loss;
//             }
//         }
//         return;
//     }
//     int tt = 0;
//     while (tt < 100)
//     {
//         tt++;
//         if (closs < dloss)
//         {
//             b[0] = d[0];
//             b[1] = d[1];
//             d[0] = c[0];
//             d[1] = c[1];
//             dloss = closs;
//             h[0] = b[0] - a[0];
//             h[1] = b[1] - a[1];

//             c[0] = a[0] + invphi2 * h[0];
//             c[1] = a[1] + invphi2 * h[1];
//             if (h[0] > 0.0001)
//             {
//                 c[0] = int(c[0]);
//             }
//             else if (h[0] < -0.0001)
//             {
//                 c[0] = ceil(c[0]);
//             }
//             else
//             {
//                 c[0] = round(c[0]);
//             }

//             algorithm->update_train_mask(full_mask);
//             algorithm->update_sparsity_level(int(c[0]));
//             algorithm->update_lambda_level(exp(c[1]));
//             algorithm->update_beta_init(beta_init);
//             algorithm->update_coef0_init(coef0_init);
//             algorithm->update_group_XTX(full_group_XTX);
//             algorithm->fit();
//             if (algorithm->warm_start)
//             {
//                 beta_init = algorithm->get_beta();
//                 coef0_init = algorithm->get_coef0();
//             }
//             closs = metric->ic(algorithm, data);
//             coef0_temp1 = algorithm->get_coef0();
//             beta_temp1 = algorithm->get_beta();
//             train_loss_temp1 = metric->train_loss(algorithm, data);
//             ic_row = int(c[0]);
//             ic_col = floor((c[1] - log_lambda_min) / d_lambda);
//             if (std::isnan(ic_sequence(ic_row, ic_col)))
//             {
//                 ic_sequence(ic_row, ic_col) = closs;
//             }
//             else
//             {
//                 ic_sequence(ic_row, ic_col) = (ic_sequence(ic_row, ic_col) > closs) ? closs : ic_sequence(ic_row, ic_col);
//             }
//         }

//         else
//         {
//             a[0] = c[0];
//             a[1] = c[1];
//             c[0] = d[0];
//             c[1] = d[1];
//             closs = dloss;
//             h[0] = b[0] - a[0];
//             h[1] = b[1] - a[1];

//             d[0] = a[0] + invphi * h[0];
//             d[1] = a[1] + invphi * h[1];

//             if (h[0] > 0.0001)
//             {
//                 d[0] = ceil(d[0]);
//             }
//             else if (h[0] < -0.0001)
//             {
//                 d[0] = int(d[0]);
//             }
//             else
//             {
//                 d[0] = round(d[0]);
//             }

//             algorithm->update_train_mask(full_mask);
//             algorithm->update_sparsity_level(int(d[0]));
//             algorithm->update_lambda_level(exp(d[1]));
//             algorithm->update_beta_init(beta_init);
//             algorithm->update_coef0_init(coef0_init);
//             algorithm->update_group_XTX(full_group_XTX);
//             algorithm->fit();
//             if (algorithm->warm_start)
//             {
//                 beta_init = algorithm->get_beta();
//                 coef0_init = algorithm->get_coef0();
//             }

//             dloss = metric->ic(algorithm, data);
//             coef0_temp2 = algorithm->get_coef0();
//             beta_temp2 = algorithm->get_beta();
//             train_loss_temp2 = metric->train_loss(algorithm, data);
//             ic_row = int(d[0]);
//             ic_col = floor((d[1] - log_lambda_min) / d_lambda);
//             if (std::isnan(ic_sequence(ic_row, ic_col)))
//             {
//                 ic_sequence(ic_row, ic_col) = dloss;
//             }
//             else
//             {
//                 ic_sequence(ic_row, ic_col) = (ic_sequence(ic_row, ic_col) > dloss) ? dloss : ic_sequence(ic_row, ic_col);
//             }
//         }

//         if ((abs((invphi2 - invphi) * h[0]) <= s_tol && abs((invphi2 - invphi) * h[1]) < log_lambda_tol) || tt == 50)
//         {
//             double min_loss;
//             double tmp_loss;
//             if (closs < dloss)
//             {
//                 best_arg[0] = c[0];
//                 best_arg[1] = c[1];
//                 min_loss = closs;

//                 beta1 = beta_temp1;
//                 coef01 = coef0_temp1;
//                 ic1 = closs;
//                 train_loss1 = train_loss_temp1;
//             }
//             else
//             {
//                 best_arg[0] = d[0];
//                 best_arg[1] = d[1];
//                 min_loss = dloss;

//                 beta1 = beta_temp2;
//                 coef01 = coef0_temp2;
//                 ic1 = dloss;
//                 train_loss1 = train_loss_temp2;
//             }
//             for (int i = 1; i < abs((invphi2 - invphi) * h[0]); i++)
//             {
//                 algorithm->update_train_mask(full_mask);
//                 algorithm->update_sparsity_level(int(c[0] + sign(h[0]) * i));
//                 algorithm->update_lambda_level(exp(c[1]));
//                 algorithm->update_beta_init(beta_init);
//                 algorithm->update_coef0_init(coef0_init);
//                 algorithm->update_group_XTX(full_group_XTX);
//                 algorithm->fit();
//                 if (algorithm->warm_start)
//                 {
//                     beta_init = algorithm->get_beta();
//                     coef0_init = algorithm->get_coef0();
//                 }
//                 tmp_loss = metric->ic(algorithm, data);

//                 ic_row = int(c[0]);
//                 ic_col = floor((c[1] - log_lambda_min) / d_lambda);
//                 if (std::isnan(ic_sequence(ic_row, ic_col)))
//                 {
//                     ic_sequence(ic_row, ic_col) = tmp_loss;
//                 }
//                 else
//                 {
//                     ic_sequence(ic_row, ic_col) = (ic_sequence(ic_row, ic_col) > tmp_loss) ? tmp_loss : ic_sequence(ic_row, ic_col);
//                 }
//                 if (tmp_loss < min_loss)
//                 {
//                     best_arg[0] = c[0] + sign(h[0]) * i;
//                     best_arg[1] = c[1];
//                     min_loss = tmp_loss;

//                     beta1 = algorithm->get_beta();
//                     coef01 = algorithm->get_coef0();
//                     train_loss1 = metric->train_loss(algorithm, data);
//                     ic1 = min_loss;
//                 }
//             }
//             return;
//         }
//     }
// }

int GDC(int a, int b)
{
  int Max, Min;
  Max = a > b ? a : b;
  if (a == Max)
    Min = b;
  else
    Min = a;
  int z = Min;
  while (Max % Min != 0)
  {
    z = Max % Min;
    Max = Min;
    Min = z;
  }
  return z;
}

// void seq_search(Data &data, Algorithm *algorithm, Metric *metric, double p[], double u[], int s_min, int s_max, double log_lambda_min, double log_lambda_max, double best_arg[],
//                 Eigen::VectorXd &beta1, double &coef01, double &train_loss1, double &ic1, int nlambda, Eigen::MatrixXd &ic_sequence)
// {
//     vector<Eigen::MatrixXd> full_group_XTX = group_XTX(data.x, data.g_index, data.g_size, data.n, data.p, data.g_num, algorithm->model_type);

//     int n = data.get_n();
//     Eigen::VectorXi full_mask(n);
//     for (int i = 0; i < n; i++)
//     {
//         full_mask(i) = int(i);
//     }
//     Eigen::VectorXd beta_init = Eigen::VectorXd::Zero(data.get_p());
//     double coef0_init = 0.0;
//     Eigen::VectorXd beta_warm(data.get_p());
//     double coef0_warm = 0.0;

//     double d_lambda = (log_lambda_max - log_lambda_min) / (nlambda - 1);
//     int k_lambda = abs(round(u[1] / d_lambda));
//     cout << "d_lambda: " << d_lambda << ", k_lambda: " << k_lambda << ", u[1]: " << u[1] << ", u[0]: " << u[0] << endl;
//     if (abs(u[0]) != 1 && k_lambda != 1)
//     {
//         if (k_lambda == 0 && u[0] != 0)
//         {
//             u[0] = u[0] / abs(u[0]);
//         }
//         else if (u[0] == 0 && k_lambda != 0)
//         {
//             u[1] = u[1] / k_lambda;
//         }
//         else
//         {
//             int gdc;
//             gdc = GDC(k_lambda, abs(int(u[0])));
//             if (gdc)
//             {
//                 u[0] = round(u[0] / gdc);
//                 u[1] = u[1] / gdc;
//             }
//         }
//     }
//     cout << "u[0]: " << u[0] << ", u[1]: " << u[1] << endl;

//     int i = 0;
//     int j = 0;
//     int ic_row, ic_col;
//     Eigen::MatrixXd beta_all_1 = Eigen::MatrixXd::Zero(data.get_p(), (s_max - s_min + 1) * nlambda);
//     Eigen::MatrixXd beta_all_2 = Eigen::MatrixXd::Zero(data.get_p(), (s_max - s_min + 1) * nlambda);
//     Eigen::VectorXd coef0_all_1 = Eigen::VectorXd::Zero((s_max - s_min + 1) * nlambda);
//     Eigen::VectorXd coef0_all_2 = Eigen::VectorXd::Zero((s_max - s_min + 1) * nlambda);
//     Eigen::VectorXd train_loss_1 = Eigen::VectorXd::Zero((s_max - s_min + 1) * nlambda);
//     Eigen::VectorXd train_loss_2 = Eigen::VectorXd::Zero((s_max - s_min + 1) * nlambda);
//     Eigen::VectorXd ic_sequence_1 = Eigen::VectorXd::Zero((s_max - s_min + 1) * nlambda);
//     Eigen::VectorXd ic_sequence_2 = Eigen::VectorXd::Zero((s_max - s_min + 1) * nlambda);
//     beta_warm.setZero();

//     algorithm->update_train_mask(full_mask);
//     algorithm->update_sparsity_level(p[0] + i * u[0]);
//     algorithm->update_lambda_level(exp(p[1] + i * u[1]));
//     algorithm->update_beta_init(beta_init);
//     algorithm->update_coef0_init(coef0_init);
//     algorithm->update_group_XTX(full_group_XTX);

//     algorithm->fit();
//     if (algorithm->warm_start)
//     {
//         beta_init = algorithm->get_beta();
//         coef0_init = algorithm->get_coef0();
//     }
//     ic_sequence_1(i) = metric->ic(algorithm, data);
//     beta_all_1.col(i) = algorithm->get_beta();
//     coef0_all_1(i) = algorithm->get_coef0();
//     train_loss_1(i) = metric->train_loss(algorithm, data);

//     ic_sequence_2(j) = ic_sequence_1(i);
//     beta_all_2.col(j) = beta_all_1.col(i);
//     coef0_all_2(j) = coef0_all_1(i);
//     train_loss_2(j) = train_loss_1(i);

//     ic_row = p[0] + i * u[0] - s_min;
//     ic_col = (p[1] + i * u[1] - log_lambda_min) / d_lambda;
//     if (std::isnan(ic_sequence(ic_row, ic_col)))
//     {
//         ic_sequence(ic_row, ic_col) = ic_sequence_1(i);
//     }
//     else
//     {
//         ic_sequence(ic_row, ic_col) = ic_sequence(ic_row, ic_col) > ic_sequence_1(i) ? ic_sequence_1(i) : ic_sequence(ic_row, ic_col);
//     }
//     cout << "i: " << i << ", ic_sequence(" << ic_row << "," << ic_col << "): " << ic_sequence(ic_row, ic_col) << endl;

//     i++;
//     j++;
//     beta_warm = beta_init;
//     coef0_warm = coef0_init;
//     cout << "s_max: " << s_max << ", s_min: " << s_min << ", log_lambda_max + d_lambda * 1e-4:" << log_lambda_max + d_lambda * 1e-4 << ", log_lambda_min - d_lambda * 1e-4: " << log_lambda_min - d_lambda * 1e-4 << endl;
//     while ((p[0] + i * u[0] <= s_max) && (p[1] + i * u[1] <= log_lambda_max + d_lambda * 1e-4) && (p[0] + i * u[0] >= s_min) && (p[1] + i * u[1] >= log_lambda_min - d_lambda * 1e-4))
//     {
//         cout << "i:" << i << ", p[0]: " << p[0] << ", u[0]: " << u[0] << ", p[0] + i * u[0]: " << p[0] + i * u[0] << ", p[1]" << p[1] << ", u[1]: " << u[1] << ", p[1] + i * u[1] : " << p[1] + i * u[1] << endl;
//         algorithm->update_train_mask(full_mask);
//         algorithm->update_sparsity_level(p[0] + i * u[0]);
//         algorithm->update_lambda_level(exp(p[1] + i * u[1]));
//         algorithm->update_beta_init(beta_init);
//         algorithm->update_coef0_init(coef0_init);
//         algorithm->update_group_XTX(full_group_XTX);

//         algorithm->fit();
//         if (algorithm->warm_start)
//         {
//             beta_init = algorithm->get_beta();
//             coef0_init = algorithm->get_coef0();
//         }
//         ic_sequence_1(i) = metric->ic(algorithm, data);
//         beta_all_1.col(i) = algorithm->get_beta();
//         coef0_all_1(i) = algorithm->get_coef0();
//         train_loss_1(i) = metric->train_loss(algorithm, data);
//         ic_row = p[0] + i * u[0] - s_min;
//         ic_col = round((p[1] + i * u[1] - log_lambda_min) / d_lambda);
//         if (std::isnan(ic_sequence(ic_row, ic_col)))
//         {
//             ic_sequence(ic_row, ic_col) = ic_sequence_1(i);
//         }
//         else
//         {
//             ic_sequence(ic_row, ic_col) = ic_sequence(ic_row, ic_col) > ic_sequence_1(i) ? ic_sequence_1(i) : ic_sequence(ic_row, ic_col);
//         }
//         cout << "i: " << i << ", ic_sequence(" << ic_row << "," << ic_col << "): " << ic_sequence(ic_row, ic_col) << ", ic_sequence_1(i): " << ic_sequence_1(i) << endl;
//         i++;
//     }
//     cout << "out: i:" << i << ", p[0] - i * u[0]: " << p[0] - i * u[0] << ", p[1] - i * u[1] : " << p[1] - i * u[1] << endl;

//     beta_init = beta_warm;
//     coef0_init = coef0_warm;

//     while ((p[0] - j * u[0] <= s_max) && (p[1] - j * u[1] <= log_lambda_max + d_lambda * 1e-4) && (p[0] - j * u[0] >= s_min) && (p[1] - j * u[1] >= log_lambda_min - d_lambda * 1e-4))
//     {
//         cout << "j:" << j << ", p[0]: " << p[0] << ", u[0]: " << u[0] << ", p[0] - j * u[0]: " << p[0] - j * u[0] << ", p[1]" << p[1] << ", u[1]: " << u[1] << ", p[1] - j * u[1]" << p[1] - j * u[1] << endl;
//         algorithm->update_train_mask(full_mask);
//         algorithm->update_sparsity_level(p[0] - j * u[0]);
//         algorithm->update_lambda_level(exp(p[1] - j * u[1]));
//         algorithm->update_beta_init(beta_init);
//         algorithm->update_coef0_init(coef0_init);
//         algorithm->update_group_XTX(full_group_XTX);

//         algorithm->fit();
//         if (algorithm->warm_start)
//         {
//             beta_init = algorithm->get_beta();
//             coef0_init = algorithm->get_coef0();
//         }
//         ic_sequence_2(j) = metric->ic(algorithm, data);
//         beta_all_2.col(j) = algorithm->get_beta();
//         coef0_all_2(j) = algorithm->get_coef0();
//         train_loss_2(j) = metric->train_loss(algorithm, data);
//         ic_row = p[0] - j * u[0] - s_min;
//         ic_col = round((p[1] - j * u[1] - log_lambda_min) / d_lambda);
//         if (std::isnan(ic_sequence(ic_row, ic_col)))
//         {
//             ic_sequence(ic_row, ic_col) = ic_sequence_2(j);
//         }
//         else
//         {
//             ic_sequence(ic_row, ic_col) = ic_sequence(ic_row, ic_col) > ic_sequence_2(j) ? ic_sequence_2(j) : ic_sequence(ic_row, ic_col);
//         }
//         cout << "j: " << j << ", ic_sequence(" << ic_row << "," << ic_col << "): " << ic_sequence(ic_row, ic_col) << "ic_sequence_2(j): " << ic_sequence_2(j) << endl;

//         j++;
//     }
//     cout << "out: j:" << j << ", p[0] - j * u[0]: " << p[0] - j * u[0] << ", p[1] - j * u[1] : " << p[1] - j * u[1] << endl;
//     int minPosition_1, minPosition_2;
//     ic_sequence_1 = ic_sequence_1.head(i).eval();
//     ic_sequence_2 = ic_sequence_2.head(j).eval();
//     ic_sequence_1.minCoeff(&minPosition_1);
//     ic_sequence_2.minCoeff(&minPosition_2);
//     cout << "minPosition_1: " << minPosition_1 << ", minPosition_2: " << minPosition_2 << ",  ic_sequence_1.minCoeff(&minPosition_1): " << ic_sequence_1(minPosition_1) << ", ic_sequence_2.minCoeff(&minPosition_2): " << ic_sequence_2(minPosition_2) << endl;
//     cout << "ic_sequence_2(minPosition_2): " << ic_sequence_2(minPosition_2) << ", ic_sequence_2(minPosition_2+1): " << ic_sequence_2(minPosition_2 + 1) << ", ic_sequence_2(minPosition_2+1)< ic_sequence_2(minPosition_2): " << (ic_sequence_2(minPosition_2 + 1) < ic_sequence_2(minPosition_2)) << endl;
//     int minPosition;
//     if (ic_sequence_1(minPosition_1) < ic_sequence_2(minPosition_2))
//     {
//         minPosition = minPosition_1;
//         ic1 = ic_sequence_1(minPosition);
//         train_loss1 = train_loss_1(minPosition);
//         beta1 = beta_all_1.col(minPosition);
//         coef01 = coef0_all_1(minPosition);
//     }
//     else
//     {
//         minPosition = -minPosition_2;
//         ic1 = ic_sequence_2(minPosition_2);
//         train_loss1 = train_loss_2(minPosition_2);
//         beta1 = beta_all_2.col(minPosition_2);
//         coef01 = coef0_all_2(minPosition_2);
//     }
//     best_arg[0] = p[0] + (minPosition)*u[0];
//     best_arg[1] = p[1] + (minPosition)*u[1];
//     cout << "minPosition :" << minPosition << ",  best_arg[0]:" << best_arg[0] << " best_arg[1]: " << best_arg[1] << ", p[1] + (minPosition)*u[1]= " << p[1] << "+" << minPosition << "* " << u[1] << "= " << p[1] + (minPosition)*u[1] << ", (minPosition)*u[1]: " << (minPosition)*u[1] << endl;
//     return;
// }

// List pgs_path(Data &data, Algorithm *algorithm, Metric *metric, int s_min, int s_max, double log_lambda_min, double log_lambda_max, int powell_path, int nlambda)
// {
//     int n = data.get_n();
//     Eigen::VectorXi full_mask(n);
//     vector<Eigen::MatrixXd> full_group_XTX = group_XTX(data.x, data.g_index, data.g_size, data.n, data.p, data.g_num, algorithm->model_type);
//     for (int i = 0; i < n; i++)
//     {
//         full_mask(i) = i;
//     }
//     Eigen::MatrixXd beta_all = Eigen::MatrixXd::Zero(data.get_p(), 100);
//     Eigen::VectorXd coef0_all = Eigen::VectorXd::Zero(100);
//     Eigen::VectorXd train_loss_all = Eigen::VectorXd::Zero(100);
//     Eigen::VectorXd ic_all = Eigen::VectorXd::Zero(100);
//     Eigen::VectorXd lambda_chosen = Eigen::VectorXd::Zero(100);

//     Eigen::VectorXd beta_temp = Eigen::VectorXd::Zero(data.get_p());
//     double coef0_temp;
//     double train_loss_temp;
//     double ic_temp;
//     //double lambda_temp;

//     if (powell_path == 1)
//         nlambda = 100;
//     Eigen::MatrixXd ic_sequence = NAN * Eigen::MatrixXd::Ones(s_max - s_min + 1, nlambda);

//     double P[3][3], U[2][2];
//     int i;

//     P[0][0] = double(s_min);
//     P[0][1] = log_lambda_min;

//     U[1][0] = 1.; //search directions
//     U[1][1] = 0.;
//     U[0][0] = 0.;
//     U[0][1] = (log_lambda_max - log_lambda_min) / (nlambda - 1);

//     int ttt = 0;
//     if (powell_path == 1)
//         golden_section_search(data, algorithm, metric, P[0], U[1], s_min, s_max, log_lambda_min, log_lambda_max, P[0], beta_temp, coef0_temp, train_loss_temp, ic_temp, ic_sequence);
//     else
//         seq_search(data, algorithm, metric, P[0], U[1], s_min, s_max, log_lambda_min, log_lambda_max, P[0], beta_temp, coef0_temp, train_loss_temp, ic_temp, nlambda, ic_sequence);
//     beta_all.col(ttt) = beta_temp;
//     coef0_all(ttt) = coef0_temp;
//     train_loss_all(ttt) = train_loss_temp;
//     ic_all(ttt) = ic_temp;
//     lambda_chosen(ttt) = exp(P[0][1]);

//     while (ttt < 11)
//     {
//         ttt++;
//         for (i = 0; i < 2; i++)
//         {
//             if (powell_path == 1)
//                 golden_section_search(data, algorithm, metric, P[i], U[i], s_min, s_max, log_lambda_min, log_lambda_max, P[i + 1], beta_temp, coef0_temp, train_loss_temp, ic_temp, ic_sequence);
//             else
//                 seq_search(data, algorithm, metric, P[i], U[i], s_min, s_max, log_lambda_min, log_lambda_max, P[i + 1], beta_temp, coef0_temp, train_loss_temp, ic_temp, nlambda, ic_sequence);
//             beta_all.col(ttt) = beta_temp;
//             coef0_all(ttt) = coef0_temp;
//             train_loss_all(ttt) = train_loss_temp;
//             ic_all(ttt) = ic_temp;
//             lambda_chosen(ttt) = exp(P[i + 1][1]);
//             ttt++;
//         }
//         U[0][0] = U[1][0];
//         U[0][1] = U[1][1];
//         U[1][0] = P[2][0] - P[0][0];
//         U[1][1] = P[2][1] - P[0][1];
//         if ((!(abs(U[1][0]) <= 0.0001 && abs(U[1][1]) <= 0.0001)) && ttt < 11)
//         {
//             if (powell_path == 1)
//                 golden_section_search(data, algorithm, metric, P[0], U[1], s_min, s_max, log_lambda_min, log_lambda_max, P[0], beta_temp, coef0_temp, train_loss_temp, ic_temp, ic_sequence);
//             else
//                 seq_search(data, algorithm, metric, P[0], U[1], s_min, s_max, log_lambda_min, log_lambda_max, P[0], beta_temp, coef0_temp, train_loss_temp, ic_temp, nlambda, ic_sequence);
//             beta_all.col(ttt) = beta_temp;
//             coef0_all(ttt) = coef0_temp;
//             train_loss_all(ttt) = train_loss_temp;
//             ic_all(ttt) = ic_temp;
//             lambda_chosen(ttt) = exp(P[0][1]);
//         }
//         else
//         {
//             // P[0] is the best parameter.
//             algorithm->update_train_mask(full_mask);
//             algorithm->update_sparsity_level(int(P[0][0]));
//             algorithm->update_lambda_level(exp(P[0][1]));
//             algorithm->update_group_XTX(full_group_XTX);
//             algorithm->fit();

//             Eigen::VectorXd best_beta = algorithm->get_beta();
//             double best_coef0 = algorithm->get_coef0();
//             double best_train_loss = metric->train_loss(algorithm, data);
//             double best_ic = metric->ic(algorithm, data);

//             beta_all.col(ttt) = best_beta;
//             coef0_all(ttt) = best_coef0;
//             train_loss_all(ttt) = best_train_loss;
//             ic_all(ttt) = best_ic;
//             lambda_chosen(ttt) = exp(P[0][1]);

//             ttt++;
//             beta_all = beta_all.leftCols(ttt).eval();
//             coef0_all = coef0_all.head(ttt).eval();
//             train_loss_all = train_loss_all.head(ttt).eval();
//             ic_all = ic_all.head(ttt).eval();
//             lambda_chosen = lambda_chosen.head(ttt).eval();

//             if (data.is_normal)
//             {
//                 if (algorithm->model_type == 1)
//                 {
//                     for (int k = 0; k < ttt; k++)
//                     {
//                         beta_all.col(k) = sqrt(double(n)) * beta_all.col(k).cwiseQuotient(data.x_norm);
//                         coef0_all(k) = data.y_mean - beta_all.col(k).dot(data.x_mean);
//                     }
//                 }

//                 else if (data.data_type == 2)
//                 {
//                     for (int k = 0; k < ttt; k++)
//                     {
//                         beta_all.col(k) = sqrt(double(n)) * beta_all.col(k).cwiseQuotient(data.x_norm);
//                         coef0_all(k) = coef0_all(k) - beta_all.col(k).dot(data.x_mean);
//                     }
//                 }
//                 else
//                 {
//                     for (int k = 0; k < ttt; k++)
//                     {
//                         beta_all.col(k) = sqrt(double(n)) * beta_all.col(k).cwiseQuotient(data.x_norm);
//                     }
//                 }
//             }
//             int min_ic_index = 0;
//             double min_ic = ic_all.minCoeff(&min_ic_index);
//             if (min_ic == ic_all(ttt - 1))
//             {
//                 min_ic_index = ttt - 1;
//             }
// #ifdef R_BUILD
//             return List::create(Named("beta") = beta_all.col(min_ic_index).eval(), Named("coef0") = coef0_all(min_ic_index),
//                                 Named("train_loss") = train_loss_all(min_ic_index), Named("ic") = ic_all(min_ic_index),
//                                 Named("lambda") = lambda_chosen(min_ic_index),
//                                 Named("beta_all") = beta_all, Named("coef0_all") = coef0_all,
//                                 Named("train_loss_all") = train_loss_all,
//                                 Named("ic_all") = ic_all,
//                                 Named("lambda_all") = lambda_chosen,
//                                 Named("ic_mat") = ic_sequence);
// #else
//             List mylist;
//             mylist.add("beta", beta_all.col(min_ic_index).eval());
//             mylist.add("coef0", coef0_all(min_ic_index));
//             mylist.add("train_loss", train_loss_all(min_ic_index));
//             mylist.add("ic", ic_all(min_ic_index));
//             mylist.add("lambda", lambda_chosen(min_ic_index));
//             return mylist;
// #endif
//         }
//     }
// #ifdef R_BUILD
//     Rcpp::Rcout << "powell end wrong" << endl;
// #else
//     cout << "powell end wrong" << endl;
// #endif
// #ifdef R_BUILD
//     return List::create(Named("beta") = 0);
// #else
//     List mylist;
//     return mylist;
// #endif
// }