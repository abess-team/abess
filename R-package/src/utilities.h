//
// Created by jiangkangkang on 2020/3/9.
//

#ifndef SRC_UTILITIES_H
#define SRC_UTILITIES_H

#ifndef R_BUILD

#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>

#else
#include <RcppEigen.h>
#endif

#include <cfloat>
#include <iostream>
using namespace std;
using namespace Eigen;

/** Result struct
 * @brief Save the sequential fitting result along the parameter searching.
 */
template <class T2, class T3>
struct Result {
    Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic> beta_matrix;            /*!<  */
    Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic> coef0_matrix;           /*!<  */
    Eigen::MatrixXd ic_matrix;                                                /*!<  */
    Eigen::MatrixXd test_loss_matrix;                                         /*!<  */
    Eigen::MatrixXd train_loss_matrix;                                        /*!<  */
    Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic> bd_matrix; /*!<  */
    Eigen::MatrixXd effective_number_matrix;                                  /*!<  */
};

template <class T2, class T3>
struct FIT_ARG {
    int support_size;
    double lambda;
    T2 beta_init;
    T3 coef0_init;
    Eigen::VectorXd bd_init;
    Eigen::VectorXi A_init;

    FIT_ARG(int _support_size, double _lambda, T2 _beta_init, T3 _coef0_init, VectorXd _bd_init, VectorXi _A_init) {
        support_size = _support_size;
        lambda = _lambda;
        beta_init = _beta_init;
        coef0_init = _coef0_init;
        bd_init = _bd_init;
        A_init = _A_init;
    };

    FIT_ARG(){};
};

struct single_parameter {
    int support_size;
    double lambda;

    single_parameter(){};
    single_parameter(int support_size, double lambda) {
        this->support_size = support_size;
        this->lambda = lambda;
    };
};

class Parameters {
   public:
    Eigen::VectorXi support_size_list;
    Eigen::VectorXd lambda_list;
    int s_min = 0;
    int s_max = 0;
    Eigen::Matrix<single_parameter, -1, 1> sequence;

    Parameters() {}
    Parameters(Eigen::VectorXi &support_size_list, Eigen::VectorXd &lambda_list, int s_min, int s_max) {
        this->support_size_list = support_size_list;
        this->lambda_list = lambda_list;
        this->s_min = s_min;
        this->s_max = s_max;
        if (support_size_list.size() > 0) {
            // path = "seq"
            this->build_sequence();
        }
    }

    void build_sequence() {
        // suppose each input vector has size >= 1
        int ind = 0;
        int size1 = (this->support_size_list).size();
        int size2 = (this->lambda_list).size();
        (this->sequence).resize(size1 * size2, 1);

        for (int i1 = 0; i1 < size1; i1++) {  // other order?
            for (int i2 = (1 - pow(-1, i1)) * (size2 - 1) / 2; i2 < size2 && i2 >= 0; i2 = i2 + pow(-1, i1)) {
                int support_size = this->support_size_list(i1);
                double lambda = this->lambda_list(i2);
                single_parameter temp(support_size, lambda);
                this->sequence(ind++) = temp;
            }
        }
    }

    // void print_sequence() {
    //     // for debug
    //     std::cout << "==> Parameter List:" << endl;
    //     for (int i = 0; i < (this->sequence).size(); i++) {
    //         int support_size = (this->sequence(i)).support_size;
    //         double lambda = (this->sequence(i)).lambda;
    //         std::cout << "  support_size = " << support_size << ", lambda = " << lambda << endl;
    //     }
    // }
};

#ifndef R_BUILD
Eigen::MatrixXd Pointer2MatrixXd(double *x, int x_row, int x_col);
Eigen::MatrixXi Pointer2MatrixXi(int *x, int x_row, int x_col);
Eigen::VectorXd Pointer2VectorXd(double *x, int x_len);
Eigen::VectorXi Pointer2VectorXi(int *x, int x_len);
void MatrixXd2Pointer(Eigen::MatrixXd x_matrix, double *x);
// void MatrixXi2Pointer(Eigen::MatrixXi x_matrix, int *x);
void VectorXd2Pointer(Eigen::VectorXd x_vector, double *x);
// void VectorXi2Pointer(Eigen::VectorXi x_vector, int *x);
#endif

Eigen::VectorXi find_ind(Eigen::VectorXi &L, Eigen::VectorXi &index, Eigen::VectorXi &gsize, int beta_size, int N);

template <class T4>
T4 X_seg(T4 &X, int n, Eigen::VectorXi &ind, int model_type) {
    if (ind.size() == X.cols() || model_type == 10 || model_type == 7) {
        return X;
    } else {
        T4 X_new(n, ind.size());
        for (int k = 0; k < ind.size(); k++) {
            X_new.col(k) = X.col(ind(k));
        }
        return X_new;
    }
};

// template <class T4>
// void X_seg(T4 &X, int n, Eigen::VectorXi &ind, T4 &X_seg)
// {
//     if (ind.size() == X.cols())
//     {
//         X_seg = X;
//     }
//     else
//     {
//         X_seg.resize(n, ind.size());
//         for (int k = 0; k < ind.size(); k++)
//         {
//             X_seg.col(k) = X.col(ind(k));
//         }
//     }
// };

template <class T4>
Eigen::Matrix<T4, -1, -1> compute_group_XTX(T4 &X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N) {
    Eigen::Matrix<T4, -1, -1> XTX(N, 1);
    for (int i = 0; i < N; i++) {
        T4 X_ind = X.block(0, index(i), n, gsize(i));
        XTX(i, 0) = X_ind.transpose() * X_ind;
    }
    return XTX;
}

template <class T4>
Eigen::Matrix<Eigen::MatrixXd, -1, -1> Phi(T4 &X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N,
                                           double lambda, Eigen::Matrix<T4, -1, -1> group_XTX) {
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> phi(N, 1);
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd lambda_XtX =
            2 * lambda * Eigen::MatrixXd::Identity(gsize(i), gsize(i)) + group_XTX(i, 0) / double(n);
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

// Eigen::VectorXi max_k_2(Eigen::VectorXd &vec, int k);

// to do
void slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind, Eigen::VectorXd &A, int axis = 0);
void slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind, Eigen::MatrixXd &A, int axis = 0);
void slice(Eigen::SparseMatrix<double> &nums, Eigen::VectorXi &ind, Eigen::SparseMatrix<double> &A, int axis = 0);

void slice_restore(Eigen::VectorXd &A, Eigen::VectorXi &ind, Eigen::VectorXd &nums, int axis = 0);
void slice_restore(Eigen::MatrixXd &A, Eigen::VectorXi &ind, Eigen::MatrixXd &nums, int axis = 0);

void coef_set_zero(int p, int M, Eigen::VectorXd &beta, double &coef0);
void coef_set_zero(int p, int M, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0);

Eigen::VectorXd array_product(Eigen::VectorXd &A, Eigen::VectorXd &B, int axis = 0);
Eigen::MatrixXd array_product(Eigen::MatrixXd &A, Eigen::VectorXd &B, int axis = 0);
// Eigen::SparseMatrix<double> array_product(Eigen::SparseMatrix<double> &A, Eigen::VectorXd &B, int axis = 0);

void array_quotient(Eigen::VectorXd &A, Eigen::VectorXd &B, int axis = 0);
void array_quotient(Eigen::MatrixXd &A, Eigen::VectorXd &B, int axis = 0);

double matrix_dot(Eigen::VectorXd &A, Eigen::VectorXd &B);
Eigen::VectorXd matrix_dot(Eigen::MatrixXd &A, Eigen::VectorXd &B);

void matrix_sqrt(Eigen::MatrixXd &A, Eigen::MatrixXd &B);
void matrix_sqrt(Eigen::SparseMatrix<double> &A, Eigen::MatrixXd &B);

void add_constant_column(Eigen::MatrixXd &X);
void add_constant_column(Eigen::SparseMatrix<double> &X);

void set_nonzeros(Eigen::MatrixXd &X, Eigen::MatrixXd &x);
void set_nonzeros(Eigen::SparseMatrix<double> &X, Eigen::SparseMatrix<double> &x);

// void overload_ldlt(Eigen::SparseMatrix<double> &X_new, Eigen::SparseMatrix<double> &X, Eigen::VectorXd &Z,
// Eigen::VectorXd &beta); void overload_ldlt(Eigen::MatrixXd &X_new, Eigen::MatrixXd &X, Eigen::VectorXd &Z,
// Eigen::VectorXd &beta);

// void overload_ldlt(Eigen::SparseMatrix<double> &X_new, Eigen::SparseMatrix<double> &X, Eigen::MatrixXd &Z,
// Eigen::MatrixXd &beta); void overload_ldlt(Eigen::MatrixXd &X_new, Eigen::MatrixXd &X, Eigen::MatrixXd &Z,
// Eigen::MatrixXd &beta);

// bool check_ill_condition(Eigen::MatrixXd &M);

template <class T2, class T3>
void restore_for_normal(T2 &beta, T3 &coef0, Eigen::Matrix<T2, Dynamic, 1> &beta_matrix,
                        Eigen::Matrix<T3, Dynamic, 1> &coef0_matrix, bool sparse_matrix, int normalize_type, int n,
                        Eigen::VectorXd x_mean, T3 y_mean, Eigen::VectorXd x_norm) {
    if (normalize_type == 0 || sparse_matrix) {
        // no need to restore
        return;
    }

    int sequence_size = beta_matrix.rows();
    if (normalize_type == 1) {
        array_quotient(beta, x_norm, 1);
        beta = beta * sqrt(double(n));
        coef0 = y_mean - matrix_dot(beta, x_mean);
        for (int ind = 0; ind < sequence_size; ind++) {
            array_quotient(beta_matrix(ind), x_norm, 1);
            beta_matrix(ind) = beta_matrix(ind) * sqrt(double(n));
            coef0_matrix(ind) = y_mean - matrix_dot(beta_matrix(ind), x_mean);
        }
    } else if (normalize_type == 2) {
        array_quotient(beta, x_norm, 1);
        beta = beta * sqrt(double(n));
        coef0 = coef0 - matrix_dot(beta, x_mean);
        for (int ind = 0; ind < sequence_size; ind++) {
            array_quotient(beta_matrix(ind), x_norm, 1);
            beta_matrix(ind) = beta_matrix(ind) * sqrt(double(n));
            coef0_matrix(ind) = coef0_matrix(ind) - matrix_dot(beta_matrix(ind), x_mean);
        }
    } else {
        array_quotient(beta, x_norm, 1);
        beta = beta * sqrt(double(n));
        for (int ind = 0; ind < sequence_size; ind++) {
            array_quotient(beta_matrix(ind), x_norm, 1);
            beta_matrix(ind) = beta_matrix(ind) * sqrt(double(n));
        }
    }
    return;
}

template <class T4>
Eigen::VectorXd pi(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &coef) {
    int p = coef.size();
    int n = X.rows();
    Eigen::VectorXd Pi = Eigen::VectorXd::Zero(n);
    if (X.cols() == p - 1) {
        Eigen::VectorXd intercept = Eigen::VectorXd::Ones(n) * coef(0);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
        Eigen::VectorXd eta = X * (coef.tail(p - 1).eval()) + intercept;
        for (int i = 0; i < n; i++) {
            if (eta(i) > 30) {
                eta(i) = 30;
            } else if (eta(i) < -30) {
                eta(i) = -30;
            }
        }
        Eigen::VectorXd expeta = eta.array().exp();
        Pi = expeta.array() / (one + expeta).array();
        return Pi;
    } else {
        Eigen::VectorXd eta = X * coef;
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

        for (int i = 0; i < n; i++) {
            if (eta(i) > 30) {
                eta(i) = 30;
            } else if (eta(i) < -30) {
                eta(i) = -30;
            }
        }
        Eigen::VectorXd expeta = eta.array().exp();
        Pi = expeta.array() / (one + expeta).array();
        return Pi;
    }
}

template <class T4>
void pi(T4 &X, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, Eigen::MatrixXd &pr) {
    int n = X.rows();
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, 1);
    Eigen::MatrixXd Xbeta = X * beta + one * coef0.transpose();
    pr = Xbeta.array().exp();
    Eigen::VectorXd sumpi = pr.rowwise().sum();
    for (int i = 0; i < n; i++) {
        pr.row(i) = pr.row(i) / sumpi(i);
    }

    // return pi;
};

template <class T4>
void pi(T4 &X, Eigen::MatrixXd &y, Eigen::MatrixXd &coef, Eigen::MatrixXd &pr) {
    int n = X.rows();
    // Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, 1);
    Eigen::MatrixXd Xbeta = X * coef;
    pr = Xbeta.array().exp();
    Eigen::VectorXd sumpi = pr.rowwise().sum();
    for (int i = 0; i < n; i++) {
        pr.row(i) = pr.row(i) / sumpi(i);
    }

    // return pi;
};

void add_weight(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd weights);

void add_weight(Eigen::MatrixXd &x, Eigen::MatrixXd &y, Eigen::VectorXd weights);

void add_weight(Eigen::SparseMatrix<double> &x, Eigen::VectorXd &y, Eigen::VectorXd weights);

void add_weight(Eigen::SparseMatrix<double> &x, Eigen::MatrixXd &y, Eigen::VectorXd weights);

#endif  // SRC_UTILITIES_H
