//
// Created by jiangkangkang on 2020/3/9.
//

#ifndef BESS_UTILITIES_H
#define BESS_UTILITIES_H

#ifndef R_BUILD

#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>

#else
#include <RcppEigen.h>
#endif

#include <iostream>
#include <vector>
using namespace std;
using namespace Eigen;

template <class T2, class T3, class T5>
struct FIT_ARG
{
    int support_size;
    double lambda;
    T2 beta_init;
    T3 coef0_init;
    T5 bd_init;
    Eigen::VectorXi A_init;

    FIT_ARG(int _support_size, double _lambda, T2 _beta_init, T3 _coef0_init, T5 _bd_init, VectorXi _A_init)
    {
        support_size = _support_size;
        lambda = _lambda;
        beta_init = _beta_init;
        coef0_init = _coef0_init;
        bd_init = _bd_init;
        A_init = _A_init;
    };

    FIT_ARG(){};
};

#ifndef R_BUILD
Eigen::MatrixXd Pointer2MatrixXd(double *x, int x_row, int x_col);
// Eigen::MatrixXi Pointer2MatrixXi(int *x, int x_row, int x_col);
Eigen::VectorXd Pointer2VectorXd(double *x, int x_len);
Eigen::VectorXi Pointer2VectorXi(int *x, int x_len);
void MatrixXd2Pointer(Eigen::MatrixXd x_matrix, double *x);
// void MatrixXi2Pointer(Eigen::MatrixXi x_matrix, int *x);
void VectorXd2Pointer(Eigen::VectorXd x_vector, double *x);
// void VectorXi2Pointer(Eigen::VectorXi x_vector, int *x);
void VectorXd2Pointer(Eigen::Matrix<long double, Eigen::Dynamic, 1> x_vector, long double *x);
#endif

Eigen::VectorXi find_ind(Eigen::VectorXi &L, Eigen::VectorXi &index, Eigen::VectorXi &gsize, int p, int N, int model_type = 0);

Eigen::VectorXi find_ind_graph(Eigen::VectorXi &ind, Eigen::MatrixXi &map, int p);

template <class T4>
T4 X_seg(T4 &X, int n, Eigen::VectorXi &ind)
{
    if (ind.size() == X.cols())
    {
        return X;
    }
    else
    {
        T4 X_new(n, ind.size());
        for (int k = 0; k < ind.size(); k++)
        {
            X_new.col(k) = X.col(ind(k));
        }
        return X_new;
    }
};

template <class T4>
void X_seg(T4 &X, int n, Eigen::VectorXi &ind, T4 &X_seg)
{
    if (ind.size() == X.cols())
    {
        X_seg = X;
    }
    else
    {
        X_seg.resize(n, ind.size());
        for (int k = 0; k < ind.size(); k++)
        {
            X_seg.col(k) = X.col(ind(k));
        }
    }
};

template <class T4>
Eigen::Matrix<T4, -1, -1> group_XTX(T4 &X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, int model_type)
{
    Eigen::Matrix<T4, -1, -1> XTX(N, 1);
    if (model_type == 1 || model_type == 5)
    {
        for (int i = 0; i < N; i++)
        {
            T4 X_ind = X.block(0, index(i), n, gsize(i));
            XTX(i, 0) = X_ind.transpose() * X_ind;
        }
    }
    return XTX;
}

template <class T4>
Eigen::Matrix<Eigen::MatrixXd, -1, -1> Phi(T4 &X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, double lambda, Eigen::Matrix<T4, -1, -1> group_XTX)
{
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> phi(N, 1);
    for (int i = 0; i < N; i++)
    {
        Eigen::MatrixXd lambda_XtX = 2 * lambda * Eigen::MatrixXd::Identity(gsize(i), gsize(i)) + group_XTX(i, 0) / double(n);
        lambda_XtX.sqrt().evalTo(phi(i, 0));
    }
    return phi;
}

Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhi(Eigen::Matrix<Eigen::MatrixXd, -1, -1> &Phi, int N);
// void max_k(Eigen::VectorXd &vec, int k, Eigen::VectorXi &result);
void slice_assignment(Eigen::VectorXd &nums, Eigen::VectorXi &ind, double value);
// Eigen::VectorXi get_value_index(Eigen::VectorXd &nums, double value);
// Eigen::VectorXd vector_slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind);
Eigen::VectorXi vector_slice(Eigen::VectorXi &nums, Eigen::VectorXi &ind);
// Eigen::MatrixXd row_slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind);
// Eigen::MatrixXd matrix_slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind, int axis);

// Eigen::MatrixXd X_seg(Eigen::MatrixXd &X, int n, Eigen::VectorXi &ind);
Eigen::VectorXi Ac(Eigen::VectorXi &A, int N);
// Eigen::VectorXi Ac(Eigen::VectorXi &A, Eigen::VectorXi &U);
Eigen::VectorXi diff_union(Eigen::VectorXi A, Eigen::VectorXi &B, Eigen::VectorXi &C);
Eigen::VectorXi min_k(Eigen::VectorXd &nums, int k, bool sort_by_value = false);
Eigen::VectorXi max_k(Eigen::VectorXd &nums, int k, bool sort_by_value = false);
Eigen::VectorXi min_k(Eigen::Matrix<long double, Eigen::Dynamic, 1> &nums, int k, bool sort_by_value = false);
Eigen::VectorXi max_k(Eigen::Matrix<long double, Eigen::Dynamic, 1> &nums, int k, bool sort_by_value = false);
// Eigen::VectorXi max_k_2(Eigen::VectorXd &vec, int k);

// to do
void slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind, Eigen::VectorXd &A, int axis = 0);
void slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind, Eigen::MatrixXd &A, int axis = 0);
void slice(Eigen::SparseMatrix<double> &nums, Eigen::VectorXi &ind, Eigen::SparseMatrix<double> &A, int axis = 0);
void slice(Eigen::Matrix<long double, Eigen::Dynamic, 1> &nums, Eigen::VectorXi &ind, Eigen::Matrix<long double, Eigen::Dynamic, 1> &A, int axis = 0);

void slice_restore(Eigen::VectorXd &A, Eigen::VectorXi &ind, Eigen::VectorXd &nums, int axis = 0);
void slice_restore(Eigen::MatrixXd &A, Eigen::VectorXi &ind, Eigen::MatrixXd &nums, int axis = 0);
void slice_restore(Eigen::Matrix<long double, Eigen::Dynamic, 1> &A, Eigen::VectorXi &ind, Eigen::Matrix<long double, Eigen::Dynamic, 1> &nums, int axis = 0);

void coef_set_zero(int p, int M, Eigen::VectorXd &beta, double &coef0);
void coef_set_zero(int p, int M, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0);
void coef_set_zero(int p, int M, Eigen::Matrix<long double, Eigen::Dynamic, 1> &beta, double &coef0);

// Eigen::VectorXd array_product(Eigen::VectorXd &A, Eigen::VectorXd &B, int axis = 0);
Eigen::MatrixXd array_product(Eigen::MatrixXd &A, Eigen::VectorXd &B, int axis = 0);

void array_quotient(Eigen::VectorXd &A, Eigen::VectorXd &B, int axis = 0);
void array_quotient(Eigen::MatrixXd &A, Eigen::VectorXd &B, int axis = 0);
void array_quotient(Eigen::Matrix<long double, Eigen::Dynamic, 1> &A, Eigen::VectorXd &B, int axis = 0);

double matrix_dot(Eigen::VectorXd &A, Eigen::VectorXd &B);
double matrix_dot(Eigen::Matrix<long double, Eigen::Dynamic, 1> &A, Eigen::VectorXd &B);
Eigen::VectorXd matrix_dot(Eigen::MatrixXd &A, Eigen::VectorXd &B);

void matrix_sqrt(Eigen::MatrixXd &A, Eigen::MatrixXd &B);
void matrix_sqrt(Eigen::SparseMatrix<double> &A, Eigen::MatrixXd &B);

void add_constant_column(Eigen::MatrixXd &X);
void add_constant_column(Eigen::SparseMatrix<double> &X);

void set_nonzeros(Eigen::MatrixXd &X, Eigen::MatrixXd &x);
void set_nonzeros(Eigen::SparseMatrix<double> &X, Eigen::SparseMatrix<double> &x);

// void overload_ldlt(Eigen::SparseMatrix<double> &X_new, Eigen::SparseMatrix<double> &X, Eigen::VectorXd &Z, Eigen::VectorXd &beta);
// void overload_ldlt(Eigen::MatrixXd &X_new, Eigen::MatrixXd &X, Eigen::VectorXd &Z, Eigen::VectorXd &beta);

// void overload_ldlt(Eigen::SparseMatrix<double> &X_new, Eigen::SparseMatrix<double> &X, Eigen::MatrixXd &Z, Eigen::MatrixXd &beta);
// void overload_ldlt(Eigen::MatrixXd &X_new, Eigen::MatrixXd &X, Eigen::MatrixXd &Z, Eigen::MatrixXd &beta);

// bool check_ill_condition(Eigen::MatrixXd &M);

double matrix_relative_difference(const Eigen::MatrixXd&, const Eigen::MatrixXd&);

#endif //BESS_UTILITIES_H
