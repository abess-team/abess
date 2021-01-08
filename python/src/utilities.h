//
// Created by jiangkangkang on 2020/3/9.
//

#ifndef BESS_UTILITIES_H
#define BESS_UTILITIES_H

#include <iostream>
#include <Eigen/Eigen>
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
std::vector<Eigen::MatrixXd> Phi(Eigen::MatrixXd &X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, double lambda, std::vector<Eigen::MatrixXd> group_XTX);
std::vector<Eigen::MatrixXd> invPhi(std::vector<Eigen::MatrixXd> &Phi, int N);
std::vector<Eigen::MatrixXd> group_XTX(Eigen::MatrixXd &X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, int model_type);

void max_k(Eigen::VectorXd &vec, int k, Eigen::VectorXi &result);
void slice_assignment(Eigen::VectorXd &nums, Eigen::VectorXi &ind, double value);

Eigen::VectorXi get_value_index(Eigen::VectorXd &nums, double value);
Eigen::VectorXd vector_slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind);
Eigen::VectorXi vector_slice(Eigen::VectorXi &nums, Eigen::VectorXi ind);
Eigen::MatrixXd row_slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind);
Eigen::MatrixXd matrix_slice(Eigen::MatrixXd &nums, Eigen::VectorXi ind, int axis);

//Splicing
std::vector<int> diff_union(std::vector<int> A, std::vector<int> B, std::vector<int> C);
std::vector<int> vec_seg(std::vector<int> ind, std::vector<int> L);
std::vector<int> Ac(std::vector<int> A, int N);
// std::vector<int> max_k(Eigen::VectorXd L, int k);
// std::vector<int> min_k(Eigen::VectorXd L, int k);
// Eigen::VectorXi find_ind(std::vector<int> L, Eigen::VectorXi &index, Eigen::VectorXi &gsize, int N, int p);

Eigen::MatrixXd X_seg(Eigen::MatrixXd &X, int n, Eigen::VectorXi &ind);
Eigen::VectorXi Ac(Eigen::VectorXi A, int N);
Eigen::VectorXi diff_union(Eigen::VectorXi A, Eigen::VectorXi B, Eigen::VectorXi C);
Eigen::VectorXi min_k(Eigen::VectorXd L, int k);
Eigen::VectorXi max_k(Eigen::VectorXd L, int k);

#endif //BESS_UTILITIES_H
