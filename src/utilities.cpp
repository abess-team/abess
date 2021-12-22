//
// Created by jiangkangkang on 2020/3/9.
//

#ifndef R_BUILD
#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>

#else

#include <RcppEigen.h>

#endif

#include <string.h>

#include <algorithm>
#include <iostream>

using namespace std;
using namespace Eigen;

#ifndef R_BUILD
Eigen::MatrixXd Pointer2MatrixXd(double *x, int x_row, int x_col) {
    Eigen::MatrixXd x_matrix(x_row, x_col);
    int i, j;
    for (i = 0; i < x_row; i++) {
        for (j = 0; j < x_col; j++) {
            x_matrix(i, j) = x[i * x_col + j];
        }
    }
    return x_matrix;
}

Eigen::MatrixXi Pointer2MatrixXi(int *x, int x_row, int x_col) {
    Eigen::MatrixXi x_matrix(x_row, x_col);
    int i, j;
    for (i = 0; i < x_row; i++) {
        for (j = 0; j < x_col; j++) {
            x_matrix(i, j) = x[i * x_col + j];
        }
    }
    return x_matrix;
}

Eigen::VectorXd Pointer2VectorXd(double *x, int x_len) {
    Eigen::VectorXd x_vector(x_len);
    int i;
    for (i = 0; i < x_len; i++) {
        x_vector[i] = x[i];
    }
    return x_vector;
}

Eigen::VectorXi Pointer2VectorXi(int *x, int x_len) {
    Eigen::VectorXi x_vector(x_len);
    int i;
    for (i = 0; i < x_len; i++) {
        x_vector[i] = x[i];
    }
    return x_vector;
}

void MatrixXd2Pointer(Eigen::MatrixXd x_matrix, double *x) {
    int x_matrix_row, x_matrix_col, i, j;
    x_matrix_row = x_matrix.rows();
    x_matrix_col = x_matrix.cols();
    for (i = 0; i < x_matrix_row; i++) {
        for (j = 0; j < x_matrix_col; j++) {
            x[i * x_matrix_col + j] = x_matrix(i, j);
        }
    }
    return;
}

// void MatrixXi2Pointer(Eigen::MatrixXi x_matrix, int *x)
// {
//     int x_matrix_row, x_matrix_col, i, j;
//     x_matrix_row = x_matrix.rows();
//     x_matrix_col = x_matrix.cols();
//     for (i = 0; i < x_matrix_row; i++)
//     {
//         for (j = 0; j < x_matrix_col; j++)
//         {
//             x[i * x_matrix_col + j] = x_matrix(i, j);
//         }
//     }
// }

void VectorXd2Pointer(Eigen::VectorXd x_vector, double *x) {
    int x_matrix_len, i;
    x_matrix_len = x_vector.size();

    for (i = 0; i < x_matrix_len; i++) {
        x[i] = x_vector[i];
    }
    return;
}

// void VectorXi2Pointer(Eigen::VectorXi x_vector, int *x)
// {
//     int x_matrix_len, i;
//     x_matrix_len = x_vector.size();

//     for (i = 0; i < x_matrix_len; i++)
//     {
//         x[i] = x_vector[i];
//     }
// }
#endif

Eigen::VectorXi find_ind(Eigen::VectorXi &L, Eigen::VectorXi &index, Eigen::VectorXi &gsize, int beta_size, int N) {
    if (L.size() == N) {
        return Eigen::VectorXi::LinSpaced(beta_size, 0, beta_size - 1);
    } else {
        int mark = 0;
        Eigen::VectorXi ind = Eigen::VectorXi::Zero(beta_size);

        for (int i = 0; i < L.size(); i++) {
            ind.segment(mark, gsize(L(i))) =
                Eigen::VectorXi::LinSpaced(gsize(L(i)), index(L(i)), index(L(i)) + gsize(L(i)) - 1);
            mark = mark + gsize(L(i));
        }
        return ind.head(mark).eval();
    }
}

Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhi(Eigen::Matrix<Eigen::MatrixXd, -1, -1> &Phi, int N) {
    Eigen::Matrix<Eigen::MatrixXd, -1, -1> invPhi(N, 1);
    int row;
    for (int i = 0; i < N; i++) {
        row = (Phi(i, 0)).rows();
        invPhi(i, 0) = (Phi(i, 0)).ldlt().solve(Eigen::MatrixXd::Identity(row, row));
    }
    return invPhi;
}

void slice_assignment(Eigen::VectorXd &nums, Eigen::VectorXi &ind, double value) {
    if (ind.size() != 0) {
        for (int i = 0; i < ind.size(); i++) {
            nums(ind(i)) = value;
        }
    }
    return;
}

// Eigen::VectorXd vector_slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind)
// {
//     Eigen::VectorXd sub_nums(ind.size());
//     if (ind.size() != 0)
//     {
//         for (int i = 0; i < ind.size(); i++)
//         {
//             sub_nums(i) = nums(ind(i));
//         }
//     }
//     return sub_nums;
// }

Eigen::VectorXi vector_slice(Eigen::VectorXi &nums, Eigen::VectorXi &ind) {
    Eigen::VectorXi sub_nums(ind.size());
    if (ind.size() != 0) {
        for (int i = 0; i < ind.size(); i++) {
            sub_nums(i) = nums(ind(i));
        }
    }
    return sub_nums;
}

// Eigen::MatrixXd matrix_slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind, int axis)
// {
//     if (axis == 0)
//     {
//         Eigen::MatrixXd sub_nums(ind.size(), nums.cols());
//         if (ind.size() != 0)
//         {
//             for (int i = 0; i < ind.size(); i++)
//             {
//                 sub_nums.row(i) = nums.row(ind(i));
//             }
//         }
//         return sub_nums;
//     }
//     else
//     {
//         Eigen::MatrixXd sub_nums(nums.rows(), ind.size());
//         if (ind.size() != 0)
//         {
//             for (int i = 0; i < ind.size(); i++)
//             {
//                 sub_nums.col(i) = nums.col(ind(i));
//             }
//         }
//         return sub_nums;
//     }
// }

// Eigen::MatrixXd row_slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind)
// {
//     Eigen::MatrixXd sub_nums(ind.size(), nums.cols());
//     if (ind.size() != 0)
//     {
//         for (int i = 0; i < ind.size(); i++)
//         {
//             sub_nums.row(i) = nums.row(ind(i));
//         }
//     }
//     return sub_nums;
// }

// Eigen::VectorXi get_value_index(Eigen::VectorXd &nums, double value)
// {
//     Eigen::VectorXi ind(nums.size());
//     int cur_index = 0;
//     for (int i = 0; i < nums.size(); i++)
//     {
//         if (nums(i) == value)
//         {
//             ind(cur_index) = i;
//             cur_index += 1;
//         }
//     }
//     return ind.head(cur_index).eval();
// }

// replace B by C in A
// to do : binary search
Eigen::VectorXi diff_union(Eigen::VectorXi A, Eigen::VectorXi &B, Eigen::VectorXi &C) {
    unsigned int k;
    for (unsigned int i = 0; i < B.size(); i++) {
        for (k = 0; k < A.size(); k++) {
            if (B(i) == A(k)) {
                A(k) = C(i);
                break;
            }
        }
    }
    sort(A.data(), A.data() + A.size());
    return A;
}

Eigen::VectorXi min_k(Eigen::VectorXd &vec, int k, bool sort_by_value) {
    Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);  // [0 1 2 3 ... N-1]
    auto rule = [vec](int i, int j) -> bool { return vec(i) < vec(j); };              // sort rule
    std::nth_element(ind.data(), ind.data() + k, ind.data() + ind.size(), rule);
    if (sort_by_value) {
        std::sort(ind.data(), ind.data() + k, rule);
    } else {
        std::sort(ind.data(), ind.data() + k);
    }

    return ind.head(k).eval();
}

Eigen::VectorXi max_k(Eigen::VectorXd &vec, int k, bool sort_by_value) {
    Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);  // [0 1 2 3 ... N-1]
    auto rule = [vec](int i, int j) -> bool { return vec(i) > vec(j); };              // sort rule
    std::nth_element(ind.data(), ind.data() + k, ind.data() + ind.size(), rule);
    if (sort_by_value) {
        std::sort(ind.data(), ind.data() + k, rule);
    } else {
        std::sort(ind.data(), ind.data() + k);
    }
    return ind.head(k).eval();
}

// Eigen::VectorXi max_k_2(Eigen::VectorXd &vec, int k)
// {
//     Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1); //[0 1 2 3 ... N-1]
//     auto rule = [vec](int i, int j) -> bool
//     {
//         return vec(i) > vec(j);
//     }; // sort rule
//     std::nth_element(ind.data(), ind.data() + k, ind.data() + ind.size(), rule);
//     std::sort(ind.data(), ind.data() + k);
//     return ind.head(k).eval();
// }

// Ac
Eigen::VectorXi Ac(Eigen::VectorXi &A, int N) {
    int A_size = A.size();
    if (A_size == 0) {
        return Eigen::VectorXi::LinSpaced(N, 0, N - 1);
    } else if (A_size == N) {
        Eigen::VectorXi I(0);
        return I;
    } else {
        Eigen::VectorXi I(N - A_size);
        int cur_index = 0;
        int A_index = 0;
        for (int i = 0; i < N; i++) {
            if (A_index >= A_size) {
                I(cur_index) = i;
                cur_index += 1;
                continue;
            }
            if (i != A(A_index)) {
                I(cur_index) = i;
                cur_index += 1;
            } else {
                A_index += 1;
            }
        }
        return I;
    }
}

// // Ac
// Eigen::VectorXi Ac(Eigen::VectorXi &A, Eigen::VectorXi &U)
// {
//     int A_size = A.size();
//     int N = U.size();
//     if (A_size == 0)
//     {
//         return U;
//     }
//     else if (A_size == N)
//     {
//         Eigen::VectorXi I(0);
//         return I;
//     }
//     else
//     {
//         Eigen::VectorXi I(N - A_size);
//         int cur_index = 0;
//         int A_index = 0;
//         for (int i = 0; i < N; i++)
//         {
//             if (A_index < A.size() && U(i) == A(A_index))
//             {
//                 A_index += 1;
//                 continue;
//             }
//             else
//             {
//                 I(cur_index) = U(i);
//                 cur_index += 1;
//             }
//         }
//         return I;
//     }
// }

void slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind, Eigen::VectorXd &A, int axis) {
    if (ind.size() == 0) {
        A = Eigen::VectorXd::Zero(0);
    } else {
        A = Eigen::VectorXd::Zero(ind.size());
        for (int i = 0; i < ind.size(); i++) {
            A(i) = nums(ind(i));
        }
    }
    return;
}

void slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind, Eigen::MatrixXd &A, int axis) {
    if (axis == 0) {
        A = Eigen::MatrixXd::Zero(ind.size(), nums.cols());
        if (ind.size() != 0) {
            for (int i = 0; i < ind.size(); i++) {
                A.row(i) = nums.row(ind(i));
            }
        }
    } else {
        A = Eigen::MatrixXd::Zero(nums.rows(), ind.size());
        if (ind.size() != 0) {
            for (int i = 0; i < ind.size(); i++) {
                A.col(i) = nums.col(ind(i));
            }
        }
    }
    return;
}

void slice(Eigen::SparseMatrix<double> &nums, Eigen::VectorXi &ind, Eigen::SparseMatrix<double> &A, int axis) {
    if (axis == 0) {
        Eigen::SparseMatrix<double, Eigen::RowMajor> nums_row(nums);
        Eigen::SparseMatrix<double, Eigen::RowMajor> A_row(ind.size(), nums.cols());
        A_row.reserve(nums.nonZeros());

        if (ind.size() != 0) {
            for (int i = 0; i < ind.size(); i++) {
                A_row.row(i) = nums_row.row(ind(i));
            }
        }

        A = A_row;
    } else {
        A.resize(nums.rows(), ind.size());
        A.reserve(nums.nonZeros());
        if (ind.size() != 0) {
            for (int i = 0; i < ind.size(); i++) {
                A.col(i) = nums.col(ind(i));
            }
        }
    }
    return;
}

void slice_restore(Eigen::VectorXd &A, Eigen::VectorXi &ind, Eigen::VectorXd &nums, int axis) {
    if (ind.size() == 0) {
        nums = Eigen::VectorXd::Zero(nums.size());
    } else {
        nums = Eigen::VectorXd::Zero(nums.size());
        for (int i = 0; i < ind.size(); i++) {
            nums(ind(i)) = A(i);
        }
    }
    return;
}

void slice_restore(Eigen::MatrixXd &A, Eigen::VectorXi &ind, Eigen::MatrixXd &nums, int axis) {
    if (axis == 0) {
        nums = Eigen::MatrixXd::Zero(nums.rows(), nums.cols());
        if (ind.size() != 0) {
            for (int i = 0; i < ind.size(); i++) {
                nums.row(ind(i)) = A.row(i);
            }
        }
    } else {
        nums = Eigen::MatrixXd::Zero(nums.rows(), nums.cols());
        if (ind.size() != 0) {
            for (int i = 0; i < ind.size(); i++) {
                nums.col(ind(i)) = A.col(i);
            }
        }
    }
    return;
}

void coef_set_zero(int p, int M, Eigen::VectorXd &beta, double &coef0) {
    beta = Eigen::VectorXd::Zero(p);
    coef0 = 0.;
    return;
}

void coef_set_zero(int p, int M, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0) {
    beta = Eigen::MatrixXd::Zero(p, M);
    coef0 = Eigen::VectorXd::Zero(M);
    return;
}

Eigen::VectorXd array_product(Eigen::VectorXd &A, Eigen::VectorXd &B, int axis) {
    A = A.array() * B.array();
    return A;
}

Eigen::MatrixXd array_product(Eigen::MatrixXd &A, Eigen::VectorXd &B, int axis) {
    if (axis == 0) {
        for (int i = 0; i < A.rows(); i++) {
            A.row(i) = A.row(i).array() * B.array();
        }
    } else {
        for (int i = 0; i < A.cols(); i++) {
            A.col(i) = A.col(i).array() * B.array();
        }
    }
    return A;
}

// Eigen::SparseMatrix<double> array_product(Eigen::SparseMatrix<double> &A, Eigen::VectorXd &B, int axis)
// {
//     for (int i = 0; i < A.cols(); i++)
//     {
//         A.col(i) = A.col(i) * B;
//     }
//     return A;
// }

void array_quotient(Eigen::VectorXd &A, Eigen::VectorXd &B, int axis) {
    A = A.array() / B.array();
    return;
}
void array_quotient(Eigen::MatrixXd &A, Eigen::VectorXd &B, int axis) {
    if (axis == 0) {
        for (int i = 0; i < A.rows(); i++) {
            A.row(i) = A.row(i).array() / B.array();
        }
    } else {
        for (int i = 0; i < A.cols(); i++) {
            A.col(i) = A.col(i).array() / B.array();
        }
    }
    return;
}

double matrix_dot(Eigen::VectorXd &A, Eigen::VectorXd &B) { return A.dot(B); }

Eigen::VectorXd matrix_dot(Eigen::MatrixXd &A, Eigen::VectorXd &B) { return A.transpose() * B; }

// void matrix_sqrt(Eigen::MatrixXd &A, Eigen::MatrixXd &B)
// {
//     A.sqrt().evalTo(B);
// }

// void matrix_sqrt(Eigen::SparseMatrix<double> &A, Eigen::MatrixXd &B)
// {
//     if (A.rows() == 1)
//     {
//         B = Eigen::MatrixXd::Ones(1, 1) * A.cwiseSqrt();
//     }
//     else
//     {
//         Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>>
//             adjoint_eigen_solver(A);
//         // const auto &eigenvalues = adjoint_eigen_solver.eigenvalues();
//         // CHECK_GT(eigenvalues.minCoeff(), -1e-5) //R.minCoeff() 意思是 min(R(:))最小值
//         //     << "MatrixSqrt failed with negative eigenvalues: "
//         //     << eigenvalues.transpose();

//         B = adjoint_eigen_solver.eigenvectors() * (adjoint_eigen_solver.eigenvalues().cwiseSqrt().asDiagonal()) *
//         adjoint_eigen_solver.eigenvectors().transpose();
//         //    .cwiseMax(Eigen::Matrix<FloatType, N, 1>::Zero()) //R.cwiseMax(P)
//         //    .cwiseSqrt()  // R.cwiseSqrt()
//         //    .asDiagonal() * // x.asDiagonal()
//         //    adjoint_eigen_solver.eigenvectors().transpose();
//     }
// }

void add_constant_column(Eigen::MatrixXd &X) {
    X.col(0) = Eigen::MatrixXd::Ones(X.rows(), 1);
    return;
}

void add_constant_column(Eigen::SparseMatrix<double> &X) {
    for (int i = 0; i < X.rows(); i++) {
        X.insert(i, 0) = 1.0;
    }
    return;
}

// void set_nonzeros(Eigen::MatrixXd &X, Eigen::MatrixXd &x)
// {
//     return;
// }

// void set_nonzeros(Eigen::SparseMatrix<double> &X, Eigen::SparseMatrix<double> &x)
// {
//     X.reserve(x.nonZeros() + x.rows());
// }

// void overload_ldlt(Eigen::SparseMatrix<double> &X_new, Eigen::SparseMatrix<double> &X, Eigen::VectorXd &Z,
// Eigen::VectorXd &beta)
// {
//     // Eigen::SparseMatrix<double> XTX = X_new.transpose() * X;

//     // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
//     // solver.compute(X_new.transpose() * X);
//     // beta = solver.solve(X_new.transpose() * Z);
//     Eigen::MatrixXd XTX = X_new.transpose() * X;
//     beta = (XTX).ldlt().solve(X_new.transpose() * Z);
//     return;
// }

// void overload_ldlt(Eigen::SparseMatrix<double> &X_new, Eigen::SparseMatrix<double> &X, Eigen::MatrixXd &Z,
// Eigen::MatrixXd &beta)
// {
//     // Eigen::SparseMatrix<double> XTX = X_new.transpose() * X;

//     // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
//     // solver.compute(X_new.transpose() * X);
//     // beta = solver.solve(X_new.transpose() * Z);

//     Eigen::MatrixXd XTX = X_new.transpose() * X;

//     beta = (XTX).ldlt().solve(X_new.transpose() * Z);

//     return;
// }

// void overload_ldlt(Eigen::MatrixXd &X_new, Eigen::MatrixXd &X, Eigen::VectorXd &Z, Eigen::VectorXd &beta)
// {
//     beta = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//     return;
// }

// void overload_ldlt(Eigen::MatrixXd &X_new, Eigen::MatrixXd &X, Eigen::MatrixXd &Z, Eigen::MatrixXd &beta)
// {
//     beta = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
//     return;
// }

// bool check_ill_condition(Eigen::MatrixXd &M){
//     Eigen::JacobiSVD<Eigen::MatrixXd> svd(M);
//     double l1 = svd.singularValues()(0);
//     double l2 = svd.singularValues()(svd.singularValues().size()-1);
//     return ((l2 == 0 || l1 / l2 > 1e+10) ? true : false);
// }

// to do
void add_weight(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd weights) {
    Eigen::VectorXd sqrt_weight = weights.array().sqrt();
    int n = x.rows();
    for (int i = 0; i < n; i++) {
        x.row(i) = x.row(i) * sqrt_weight(i);
    }
    array_product(y, sqrt_weight, 1);
};

void add_weight(Eigen::MatrixXd &x, Eigen::MatrixXd &y, Eigen::VectorXd weights) {
    Eigen::VectorXd sqrt_weight = weights.array().sqrt();
    int n = x.rows();
    for (int i = 0; i < n; i++) {
        x.row(i) = x.row(i) * sqrt_weight(i);
    }
    array_product(y, sqrt_weight, 1);
};

void add_weight(Eigen::SparseMatrix<double> &x, Eigen::VectorXd &y, Eigen::VectorXd weights) {
    for (int k = 0; k < x.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(x, k); it; ++it) {
            x.coeffRef(int(it.row()), int(it.col())) = x.coeffRef(int(it.row()), int(it.col())) * weights(it.row());
        }
    }
    Eigen::VectorXd sqrt_weight = weights.array().sqrt();
    array_product(y, sqrt_weight, 1);
};

void add_weight(Eigen::SparseMatrix<double> &x, Eigen::MatrixXd &y, Eigen::VectorXd weights) {
    for (int k = 0; k < x.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(x, k); it; ++it) {
            x.coeffRef(int(it.row()), int(it.col())) = x.coeffRef(int(it.row()), int(it.col())) * weights(it.row());
        }
    }
    Eigen::VectorXd sqrt_weight = weights.array().sqrt();
    array_product(y, sqrt_weight, 1);
};
