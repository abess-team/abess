//
// Created by jiangkangkang on 2020/3/9.
//

#ifndef BESS_UTILITIES_H
#define BESS_UTILITIES_H

#ifndef R_BUILD

#include <Eigen/Eigen>

#else

#include <RcppEigen.h>

#endif

#include <iostream>
#include <vector>
using namespace std;

Eigen::MatrixXd Pointer2MatrixXd(double *x, int x_row, int x_col);
Eigen::MatrixXi Pointer2MatrixXi(int *x, int x_row, int x_col);
Eigen::VectorXd Pointer2VectorXd(double *x, int x_len);
Eigen::VectorXi Pointer2VectorXi(int *x, int x_len);
void MatrixXd2Pointer(Eigen::MatrixXd x_matrix, double *x);
void MatrixXi2Pointer(Eigen::MatrixXi x_matrix, int *x);
void VectorXd2Pointer(Eigen::VectorXd x_vector, double *x);
void VectorXi2Pointer(Eigen::VectorXi x_vector, int *x);

Eigen::VectorXi find_ind(Eigen::VectorXi &L, Eigen::VectorXi &index, Eigen::VectorXi &gsize, int p, int N);
Eigen::MatrixXd X_seg(Eigen::MatrixXd &X, int n, Eigen::VectorXi &ind);
Eigen::Matrix<Eigen::MatrixXd, -1, -1> Phi(Eigen::MatrixXd &X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, double lambda, Eigen::Matrix<Eigen::MatrixXd, -1, -1> group_XTX);
Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhi(Eigen::Matrix<Eigen::MatrixXd, -1, -1> &Phi, int N);
Eigen::Matrix<Eigen::MatrixXd, -1, -1> group_XTX(Eigen::MatrixXd &X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, int model_type);

// void max_k(Eigen::VectorXd &vec, int k, Eigen::VectorXi &result);
void slice_assignment(Eigen::VectorXd &nums, Eigen::VectorXi &ind, double value);

Eigen::VectorXi get_value_index(Eigen::VectorXd &nums, double value);
Eigen::VectorXd vector_slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind);
Eigen::VectorXi vector_slice(Eigen::VectorXi &nums, Eigen::VectorXi &ind);
Eigen::MatrixXd row_slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind);
Eigen::MatrixXd matrix_slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind, int axis);

Eigen::MatrixXd X_seg(Eigen::MatrixXd &X, int n, Eigen::VectorXi &ind);
Eigen::VectorXi Ac(Eigen::VectorXi &A, int N);
Eigen::VectorXi Ac(Eigen::VectorXi &A, Eigen::VectorXi &U);
Eigen::VectorXi diff_union(Eigen::VectorXi A, Eigen::VectorXi &B, Eigen::VectorXi &C);
Eigen::VectorXi min_k(Eigen::VectorXd &nums, int k, bool sort_by_value = false);
Eigen::VectorXi max_k(Eigen::VectorXd &nums, int k, bool sort_by_value = false);

Eigen::VectorXi max_k_2(Eigen::VectorXd &vec, int k);

// to do
void slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind, Eigen::VectorXd &A, int axis = 0);
void slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind, Eigen::MatrixXd &A, int axis = 0);

void slice_restore(Eigen::VectorXd &A, Eigen::VectorXi &ind, Eigen::VectorXd &nums, int axis = 0);
void slice_restore(Eigen::MatrixXd &A, Eigen::VectorXi &ind, Eigen::MatrixXd &nums, int axis = 0);

void coef_set_zero(int p, int M, Eigen::VectorXd &beta, double &coef0);
void coef_set_zero(int p, int M, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0);

Eigen::VectorXd array_product(Eigen::VectorXd &A, Eigen::VectorXd &B, int axis = 0);
Eigen::MatrixXd array_product(Eigen::MatrixXd &A, Eigen::VectorXd &B, int axis = 0);

void array_quotient(Eigen::VectorXd &A, Eigen::VectorXd &B, int axis = 0);
void array_quotient(Eigen::MatrixXd &A, Eigen::VectorXd &B, int axis = 0);

double matrix_dot(Eigen::VectorXd &A, Eigen::VectorXd &B);
Eigen::VectorXd matrix_dot(Eigen::MatrixXd &A, Eigen::VectorXd &B);

#endif //BESS_UTILITIES_H
