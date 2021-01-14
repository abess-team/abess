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
#endif

#include <iostream>

List bessCpp(Eigen::MatrixXd x, Eigen::VectorXd y, int data_type, Eigen::VectorXd weight,
             bool is_normal,
             int algorithm_type, int model_type, int max_iter, int exchange_num,
             int path_type, bool is_warm_start,
             int ic_type, bool is_cv, int K,
             Eigen::VectorXd state,
             Eigen::VectorXi sequence,
             Eigen::VectorXd lambda_seq,
             int s_min, int s_max, int K_max, double epsilon,
             double lambda_min, double lambda_max, int nlambda,
             bool is_screening, int screening_size, int powell_path,
             Eigen::VectorXi g_index,
             Eigen::VectorXi always_select,
             double tao);

void pywrap_bess(double *x, int x_row, int x_col, double *y, int y_len, int data_type, double *weight, int weight_len,
                 bool is_normal,
                 int algorithm_type, int model_type, int max_iter, int exchange_num,
                 int path_type, bool is_warm_start,
                 int ic_type, bool is_cv, int K,
                 int *gindex, int gindex_len,
                 double *state, int state_len,
                 int *sequence, int sequence_len,
                 double *lambda_sequence, int lambda_sequence_len,
                 int s_min, int s_max, int K_max, double epsilon,
                 double lambda_min, double lambda_max, int n_lambda,
                 bool is_screening, int screening_size, int powell_path,
                 int *always_select, int always_select_len, double tao,
                 double *beta_out, int beta_out_len, double *coef0_out, int coef0_out_len, double *train_loss_out,
                 int train_loss_out_len, double *ic_out, int ic_out_len, double *nullloss_out, double *aic_out,
                 int aic_out_len, double *bic_out, int bic_out_len, double *gic_out, int gic_out_len, int *A_out,
                 int A_out_len, int *l_out);

#endif //BESS_BESS_H
