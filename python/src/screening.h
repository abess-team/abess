#ifndef screening_H
#define screening_H

//  #define R_BUILD

#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Eigen>
#include "List.h"
#endif

#include <algorithm>
#include <vector>
#include <cmath>
#include "screening.h"
// #include "model_fit.h"
#include "utilities.h"
#include "Data.h"
#include <iostream>
#include <cfloat>

using namespace std;
using namespace Eigen;

template <class T1, class T2, class T3, class T4>
Eigen::VectorXi screening(Data<T1, T2, T3, T4> &data, Algorithm<T1, T2, T3, T4> *algorithm, int screening_size, int &beta_size, Eigen::VectorXi &always_select, double lambda)
{
    int n = data.n;
    int p = data.p;
    int M = data.M;
    int g_num = data.g_num;
    Eigen::VectorXi g_size = data.g_size;
    Eigen::VectorXi g_index = data.g_index;

    Eigen::VectorXi screening_A(screening_size);
    Eigen::VectorXd coef_norm = Eigen::VectorXd::Zero(g_num);

    T2 beta_init;
    T3 coef0_init;
    Eigen::VectorXi A_init;
    Eigen::VectorXd bd_init;

    for (int i = 0; i < g_num; i++)
    {
        int p_tmp = g_size(i);
        T4 x_tmp = data.x.middleCols(g_index(i), p_tmp);
        Eigen::VectorXi g_index_tmp = Eigen::VectorXi::LinSpaced(p_tmp, 0, p_tmp-1);
        Eigen::VectorXi g_size_tmp = Eigen::VectorXi::Ones(p_tmp);
        coef_set_zero(p_tmp, M, beta_init, coef0_init);

        algorithm->update_sparsity_level(p_tmp);
        algorithm->update_lambda_level(lambda);
        algorithm->update_beta_init(beta_init);
        algorithm->update_bd_init(bd_init);
        algorithm->update_coef0_init(coef0_init);
        algorithm->update_A_init(A_init, p_tmp);
        algorithm->fit(x_tmp, data.y, data.weight, g_index_tmp, g_size_tmp, n, p_tmp, p_tmp);

        T2 beta = algorithm->beta;
        coef_norm(i) = beta.squaredNorm() / p_tmp;
    }

    // keep always_select in active_set
    slice_assignment(coef_norm, always_select, DBL_MAX);
    screening_A = max_k(coef_norm, screening_size);

    // data after screening
    Eigen::VectorXi new_g_index(screening_size);
    Eigen::VectorXi new_g_size(screening_size);

    int new_p = 0;
    for (int i = 0; i < screening_size; i++)
    {
        new_p += g_size(screening_A(i));
        new_g_size(i) = g_size(screening_A(i));
    }

    new_g_index(0) = 0;
    for (int i = 0; i < screening_size - 1; i++)
    {
        new_g_index(i + 1) = new_g_index(i) + g_size(screening_A(i));
    }

    Eigen::VectorXi screening_A_ind = find_ind(screening_A, g_index, g_size, beta_size, g_num);
    T4 x_A;
    slice(data.x, screening_A_ind, x_A, 1);

    Eigen::VectorXd new_x_mean, new_x_norm;
    slice(data.x_mean, screening_A_ind, new_x_mean);
    slice(data.x_norm, screening_A_ind, new_x_norm);

    data.x = x_A;
    data.x_mean = new_x_mean;
    data.x_norm = new_x_norm;
    data.p = new_p;
    data.g_num = screening_size;
    data.g_index = new_g_index;
    data.g_size = new_g_size;
    beta_size = algorithm->get_beta_size(n, new_p);

    if (always_select.size() != 0)
    {
        Eigen::VectorXi new_always_select(always_select.size());
        int j = 0;
        for (int i = 0; i < always_select.size(); i++)
        {
            while (always_select(i) != screening_A(j))
                j++;
            new_always_select(i) = j;
        }
        always_select = new_always_select;
    }

    cout<<screening_A_ind<<endl;///
    return screening_A_ind;
}

#endif
