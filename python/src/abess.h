//
// Created by jtwok on 2020/3/8.
//

#ifndef BESS_BESS_H
#define BESS_BESS_H

#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
#else
#include <Eigen/Eigen>
#include "List.h"
#include "Algorithm.h"
#endif

#include <iostream>

template <class T2, class T3>
struct Result
{
    Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic> beta_matrix;
    Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic> coef0_matrix;
    Eigen::MatrixXd ic_matrix;
    Eigen::MatrixXd test_loss_matrix;
    Eigen::MatrixXd train_loss_matrix;
    // Eigen::Matrix<Eigen::VectorXi, Eigen::Dynamic, Eigen::Dynamic> A_matrix;
    Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic> bd_matrix;
};

List abessCpp2(Eigen::MatrixXd x, Eigen::MatrixXd y, int n, int p,
               int data_type, Eigen::VectorXd weight,
               bool is_normal,
               int algorithm_type, int model_type, int max_iter, int exchange_num,
               int path_type, bool is_warm_start,
               int ic_type, double ic_coef, bool is_cv, int Kfold,
               Eigen::VectorXi status,
               Eigen::VectorXi sequence,
               Eigen::VectorXd lambda_seq,
               int s_min, int s_max, int K_max, double epsilon,
               double lambda_min, double lambda_max, int nlambda,
               bool is_screening, int screening_size, int powell_path,
               Eigen::VectorXi g_index,
               Eigen::VectorXi always_select,
               double tau,
               int primary_model_fit_max_iter, double primary_model_fit_epsilon,
               bool early_stop, bool approximate_Newton,
               int thread,
               bool covariance_update,
               bool sparse_matrix,
               int splicing_type);

template <class T1, class T2, class T3, class T4>
List abessCpp(T4 &x, T1 &y, int n, int p,
              int data_type, Eigen::VectorXd weight,
              bool is_normal,
              int algorithm_type, int model_type, int max_iter, int exchange_num,
              int path_type, bool is_warm_start,
              int ic_type, double ic_coef, bool is_cv, int Kfold,
              Eigen::VectorXi status,
              Eigen::VectorXi sequence,
              Eigen::VectorXd lambda_seq,
              int s_min, int s_max, int K_max, double epsilon,
              double lambda_min, double lambda_max, int nlambda,
              bool is_screening, int screening_size, int powell_path,
              Eigen::VectorXi g_index,
              Eigen::VectorXi always_select,
              double tau,
              int primary_model_fit_max_iter, double primary_model_fit_epsilon,
              bool early_stop, bool approximate_Newton,
              int thread,
              bool covariance_update,
              bool sparse_matrix,
              Algorithm<T1, T2, T3, T4> *algorithm, vector<Algorithm<T1, T2, T3, T4> *> algorithm_list);

#ifndef R_BUILD
void pywrap_abess(double *x, int x_row, int x_col, double *y, int y_row, int n, int p, int y_col, int data_type, double *weight, int weight_len,
                  bool is_normal,
                  int algorithm_type, int model_type, int max_iter, int exchange_num,
                  int path_type, bool is_warm_start,
                  int ic_type, double ic_coef, bool is_cv, int K,
                  int *gindex, int gindex_len,
                  int *status, int status_len,
                  int *sequence, int sequence_len,
                  double *lambda_sequence, int lambda_sequence_len,
                  int s_min, int s_max, int K_max, double epsilon,
                  double lambda_min, double lambda_max, int n_lambda,
                  bool is_screening, int screening_size, int powell_path,
                  int *always_select, int always_select_len, double tau,
                  int primary_model_fit_max_iter, double primary_model_fit_epsilon,
                  bool early_stop, bool approximate_Newton,
                  int thread,
                  bool covariance_update,
                  bool sparse_matrix,
                  int splicing_type,
                  double *beta_out, int beta_out_len, double *coef0_out, int coef0_out_len, double *train_loss_out,
                  int train_loss_out_len, double *ic_out, int ic_out_len, double *nullloss_out, double *aic_out,
                  int aic_out_len, double *bic_out, int bic_out_len, double *gic_out, int gic_out_len, int *A_out,
                  int A_out_len, int *l_out);
#endif

#endif //BESS_BESS_H