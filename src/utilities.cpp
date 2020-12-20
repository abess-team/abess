//
// Created by jiangkangkang on 2020/3/9.
//

#include "utilities.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <vector>
#include <iostream>
#include <Eigen/Eigen>
using namespace std;

Eigen::MatrixXd Pointer2MatrixXd(double *x, int x_row, int x_col)
{
    Eigen::MatrixXd x_matrix(x_row, x_col);
    int i, j;
    for (i = 0; i < x_row; i++)
    {
        for (j = 0; j < x_col; j++)
        {
            x_matrix(i, j) = x[i * x_col + j];
        }
    }
    return x_matrix;
}

Eigen::MatrixXi Pointer2MatrixXi(int *x, int x_row, int x_col)
{
    Eigen::MatrixXi x_matrix(x_row, x_col);
    int i, j;
    for (i = 0; i < x_row; i++)
    {
        for (j = 0; j < x_col; j++)
        {
            x_matrix(i, j) = x[i * x_col + j];
        }
    }
    return x_matrix;
}

Eigen::VectorXd Pointer2VectorXd(double *x, int x_len)
{
    Eigen::VectorXd x_vector(x_len);
    int i;
    for (i = 0; i < x_len; i++)
    {
        x_vector[i] = x[i];
    }
    return x_vector;
}

Eigen::VectorXi Pointer2VectorXi(int *x, int x_len)
{
    Eigen::VectorXi x_vector(x_len);
    int i;
    for (i = 0; i < x_len; i++)
    {
        x_vector[i] = x[i];
    }
    return x_vector;
}

void MatrixXd2Pointer(Eigen::MatrixXd x_matrix, double *x)
{
    int x_matrix_row, x_matrix_col, i, j;
    x_matrix_row = x_matrix.rows();
    x_matrix_col = x_matrix.cols();
    for (i = 0; i < x_matrix_row; i++)
    {
        for (j = 0; j < x_matrix_col; j++)
        {
            x[i * x_matrix_col + j] = x_matrix(i, j);
        }
    }
}

void MatrixXi2Pointer(Eigen::MatrixXi x_matrix, int *x)
{
    int x_matrix_row, x_matrix_col, i, j;
    x_matrix_row = x_matrix.rows();
    x_matrix_col = x_matrix.cols();
    for (i = 0; i < x_matrix_row; i++)
    {
        for (j = 0; j < x_matrix_col; j++)
        {
            x[i * x_matrix_col + j] = x_matrix(i, j);
        }
    }
}

void VectorXd2Pointer(Eigen::VectorXd x_vector, double *x)
{
    int x_matrix_len, i;
    x_matrix_len = x_vector.size();

    for (i = 0; i < x_matrix_len; i++)
    {
        x[i] = x_vector[i];
    }
}

void VectorXi2Pointer(Eigen::VectorXi x_vector, int *x)
{
    int x_matrix_len, i;
    x_matrix_len = x_vector.size();

    for (i = 0; i < x_matrix_len; i++)
    {
        x[i] = x_vector[i];
    }
}

Eigen::VectorXi find_ind(Eigen::VectorXi &L, Eigen::VectorXi &index, Eigen::VectorXi &gsize, int p, int N)
{
    if (L.size() == N)
    {
        return Eigen::VectorXi::LinSpaced(p, 0, p - 1);
    }
    else
    {
        int mark = 0;
        Eigen::VectorXi ind = Eigen::VectorXi::Zero(p);
        for (int i = 0; i < L.size(); i++)
        {
            ind.segment(mark, gsize(L[i])) = Eigen::VectorXi::LinSpaced(gsize(L[i]), index(L[i]), index(L[i]) + gsize(L[i]) - 1);
            mark = mark + gsize(L[i]);
        }
        return ind.head(mark);
    }
}

Eigen::MatrixXd X_seg(Eigen::MatrixXd &X, int n, Eigen::VectorXi &ind)
{
    Eigen::MatrixXd X_new(n, ind.size());
    for (int k = 0; k < ind.size(); k++)
    {
        X_new.col(k) = X.col(ind[k]);
    }
    return X_new;
}

std::vector<Eigen::MatrixXd> Phi(Eigen::MatrixXd &X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, double lambda, std::vector<Eigen::MatrixXd> group_XTX)
{
    std::vector<Eigen::MatrixXd> Phi(N);
    for (int i = 0; i < N; i++)
    {
        Eigen::MatrixXd lambda_XtX = 2 * lambda * Eigen::MatrixXd::Identity(gsize(i), gsize(i)) + group_XTX[i] / double(n);
        lambda_XtX.sqrt().evalTo(Phi[i]);
    }
    return Phi;
}

std::vector<Eigen::MatrixXd> group_XTX(Eigen::MatrixXd &X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, int model_type)
{
    std::vector<Eigen::MatrixXd> XTX(N);
    if (model_type == 1)
    {
        for (int i = 0; i < N; i++)
        {
            Eigen::MatrixXd X_ind = X.block(0, index(i), n, gsize(i));
            XTX[i] = X_ind.transpose() * X_ind;
        }
    }
    return XTX;
}

std::vector<Eigen::MatrixXd> invPhi(std::vector<Eigen::MatrixXd> &Phi, int N)
{
    std::vector<Eigen::MatrixXd> invPhi(N);
    int row;
    for (int i = 0; i < N; i++)
    {
        row = (Phi[i]).rows();
        invPhi[i] = (Phi[i]).ldlt().solve(Eigen::MatrixXd::Identity(row, row));
    }
    return invPhi;
}

void max_k(Eigen::VectorXd &vec, int k, Eigen::VectorXi &result)
{
    Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1); //[0 1 2 3 ... N-1]
    auto rule = [vec](int i, int j) -> bool {
        return vec(i) > vec(j);
    }; // sort rule
    std::nth_element(ind.data(), ind.data() + k, ind.data() + ind.size(), rule);
    std::sort(ind.data(), ind.data() + k);
    result = ind.head(k).eval();
}

void slice_assignment(Eigen::VectorXd &nums, Eigen::VectorXi &ind, double value)
{
    if (ind.size() != 0)
    {
        for (int i = 0; i < ind.size(); i++)
        {
            nums(ind(i)) = value;
        }
    }
}

Eigen::VectorXd slice(Eigen::VectorXd &nums, Eigen::VectorXi &ind)
{
    Eigen::VectorXd sub_nums(ind.size());
    if (ind.size() != 0)
    {
        for (int i = 0; i < ind.size(); i++)
        {
            sub_nums(i) = nums(ind(i));
        }
    }
    return sub_nums;
}

Eigen::VectorXi slice(Eigen::VectorXi &nums, Eigen::VectorXi ind)
{
    Eigen::VectorXi sub_nums(ind.size());
    if (ind.size() != 0)
    {
        for (int i = 0; i < ind.size(); i++)
        {
            sub_nums(i) = nums(ind(i));
        }
    }
    return sub_nums;
}

Eigen::MatrixXd row_slice(Eigen::MatrixXd &nums, Eigen::VectorXi &ind)
{
    Eigen::MatrixXd sub_nums(ind.size(), nums.cols());
    if (ind.size() != 0)
    {
        for (int i = 0; i < ind.size(); i++)
        {
            sub_nums.row(i) = nums.row(ind(i));
        }
    }
    return sub_nums;
}

Eigen::VectorXi get_value_index(Eigen::VectorXd &nums, double value)
{
    Eigen::VectorXi ind(nums.size());
    int cur_index = 0;
    for (int i = 0; i < nums.size(); i++)
    {
        if (nums(i) == value)
        {
            ind(cur_index) = i;
            cur_index += 1;
        }
    }
    return ind.head(cur_index).eval();
}

std::vector<int> diff_union(std::vector<int> A, std::vector<int> B, std::vector<int> C)
{
    unsigned int k;
    for (unsigned int i = 0; i < A.size(); i++)
    {
        for (k = 0; k < B.size(); k++)
        {
            if (A[i] == B[k])
            {
                A.erase(A.begin() + i);
                i--;
                break;
            }
        }
    }
    for (k = 0; k < C.size(); k++)
    {
        A.push_back(C[k]);
    }
    sort(A.begin(), A.end());
    return A;
}

// replace B by C in A
// to do : binary search
Eigen::VectorXi diff_union(Eigen::VectorXi A, Eigen::VectorXi B, Eigen::VectorXi C)
{
    unsigned int k;
    for (unsigned int i = 0; i < B.size(); i++)
    {
        for (k = 0; k < A.size(); k++)
        {
            if (B(i) == A(k))
            {
                A(k) = C(i);
                break;
            }
        }
    }
    sort(A.data(), A.data() + A.size());
    return A;
}

// std::vector<int> max_k(Eigen::VectorXd L, int k)
// {
//     std::vector<int> vec(k);
//     for (int i = 0; i < k; i++)
//     {
//         L.maxCoeff(&vec[i]);
//         L(vec[i]) = -1;
//     }
//     return vec;
// }

// std::vector<int> min_k(Eigen::VectorXd L, int k)
// {
//     std::vector<int> vec(k);
//     for (int i = 0; i < k; i++)
//     {
//         L.minCoeff(&vec[i]);
//         L(vec[i]) = 1e100;
//     }
//     return vec;
// }

// Ac
std::vector<int> Ac(std::vector<int> A, int N)
{
    int A_size = A.size();
    int temp = 0;
    int j = 0;
    if (A_size != 0)
    {
        bool label;
        std::vector<int> vec;
        for (int i = 0; i < N; i++)
        {
            label = false;
            for (; j < A_size; j++)
            {
                if (i == A[j])
                {
                    label = true;
                    temp++;
                    break;
                }
            }
            j = temp;
            if (label == true)
            {
                continue;
            }
            else
            {
                vec.push_back(i);
            }
        }
        return vec;
    }
    else
    {
        std::vector<int> vec(N);
        for (int i = 0; i < N; i++)
        {
            vec[i] = i;
        }
        return vec;
    }
}

// Ac
Eigen::VectorXi Ac(Eigen::VectorXi A, int N)
{
    int A_size = A.size();
    if (A_size == 0)
    {
        return Eigen::VectorXi::LinSpaced(N, 0, N - 1);
    }
    else
    {
        Eigen::VectorXi I(N - A.size());
        int cur_index = 0;
        int A_index = 0;
        for (int i = 0; i < N; i++)
        {
            if (i != A(A_index))
            {
                I(cur_index) = i;
                cur_index += 1;
            }
            else
            {
                A_index += 1;
            }
        }
        return I;
    }
}

Eigen::VectorXi find_ind(std::vector<int> L, Eigen::VectorXi &index, Eigen::VectorXi &gsize, int p, int N)
{
    if (L.size() == N)
    {
        return Eigen::VectorXi::LinSpaced(p, 0, p - 1);
    }
    else
    {
        int mark = 0;
        Eigen::VectorXi ind = Eigen::VectorXi::Zero(p);
        for (unsigned int i = 0; i < L.size(); i++)
        {
            ind.segment(mark, gsize(L[i])) = Eigen::VectorXi::LinSpaced(gsize(L[i]), index(L[i]), index(L[i]) + gsize(L[i]) - 1);
            mark = mark + gsize(L[i]);
        }
        return ind.head(mark);
    }
}

Eigen::MatrixXd X_seg(Eigen::MatrixXd &X, int n, Eigen::VectorXi &ind)
{
    Eigen::MatrixXd X_new(n, ind.size());
    for (int k = 0; k < ind.size(); k++)
    {
        X_new.col(k) = X.col(ind[k]);
    }
    return X_new;
}

std::vector<int> vec_seg(std::vector<int> L, std::vector<int> ind)
{
    std::vector<int> vec(ind.size());
    for (unsigned int i = 0; i < ind.size(); i++)
    {
        vec[i] = L[ind[i]];
    }
    return vec;
}
