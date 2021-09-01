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
#include "model_fit.h"
#include "utilities.h"
#include "Data.h"
#include <iostream>
#include <cfloat>

using namespace std;
using namespace Eigen;

template <class T4>
Eigen::VectorXi screening(Data<Eigen::VectorXd, Eigen::VectorXd, double, T4> &data, int model_type, int screening_size, Eigen::VectorXi &always_select, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon)
{
    // int n = data.x.rows();
    int p = data.x.cols();
    int M = data.y.cols();
    Eigen::VectorXi screening_A(screening_size);

    int g_num = data.g_num;
    Eigen::VectorXi g_size = data.g_size;
    Eigen::VectorXi g_index = data.g_index;

    Eigen::VectorXd coef_norm = Eigen::VectorXd::Zero(g_num);

    for (int i = 0; i < g_num; i++)
    {

        T4 x_tmp = data.x.middleCols(g_index(i), g_size(i));
        Eigen::VectorXd beta;
        double coef0;
        coef_set_zero(g_size(i), M, beta, coef0);
        if (model_type == 1)
        {
            lm_fit(x_tmp, data.y, data.weight, beta, coef0, DBL_MAX, approximate_Newton, primary_model_fit_max_iter, primary_model_fit_epsilon, 0., 0.);
        }
        else if (model_type == 2)
        {
            logistic_fit(x_tmp, data.y, data.weight, beta, coef0, DBL_MAX, approximate_Newton, primary_model_fit_max_iter, primary_model_fit_epsilon, 0., 0.);
        }
        else if (model_type == 3)
        {
            poisson_fit(x_tmp, data.y, data.weight, beta, coef0, DBL_MAX, approximate_Newton, primary_model_fit_max_iter, primary_model_fit_epsilon, 0., 0.);
        }
        else if (model_type == 4)
        {
            cox_fit(x_tmp, data.y, data.weight, beta, coef0, DBL_MAX, approximate_Newton, primary_model_fit_max_iter, primary_model_fit_epsilon, 0., 0.);
        }
        coef_norm(i) = beta.squaredNorm() / g_size(i);
    }

    // keep always_select in active_set
    slice_assignment(coef_norm, always_select, DBL_MAX);

    screening_A = max_k(coef_norm, screening_size);

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

    Eigen::VectorXi screening_A_ind = find_ind(screening_A, g_index, g_size, p, g_num);
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

    return screening_A_ind;
}

template <class T4>
Eigen::VectorXi screening(Data<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, T4> &data, int model_type, int screening_size, Eigen::VectorXi &always_select, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon)
{
    // int n = data.x.rows();
    int p = data.x.cols();
    int M = data.y.cols();
    Eigen::VectorXi screening_A(screening_size);

    int g_num = data.g_num;
    Eigen::VectorXi g_size = data.g_size;
    Eigen::VectorXi g_index = data.g_index;

    Eigen::VectorXd coef_norm = Eigen::VectorXd::Zero(g_num);

    for (int i = 0; i < g_num; i++)
    {
        T4 x_tmp = data.x.middleCols(g_index(i), g_size(i));
        Eigen::MatrixXd beta;
        Eigen::VectorXd coef0;
        coef_set_zero(g_size(i), M, beta, coef0);
        if (model_type == 5)
        {
            multigaussian_fit(x_tmp, data.y, data.weight, beta, coef0, DBL_MAX, approximate_Newton, primary_model_fit_max_iter, primary_model_fit_epsilon, 0., 0.);
        }
        else if (model_type == 6)
        {
            multinomial_fit(x_tmp, data.y, data.weight, beta, coef0, DBL_MAX, approximate_Newton, primary_model_fit_max_iter, primary_model_fit_epsilon, 0., 0.);
        }
        coef_norm(i) = beta.squaredNorm() / g_size(i);
    }

    // keep always_select in active_set
    slice_assignment(coef_norm, always_select, DBL_MAX);

    screening_A = max_k(coef_norm, screening_size);

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

    Eigen::VectorXi screening_A_ind = find_ind(screening_A, g_index, g_size, p, g_num);
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

    return screening_A_ind;
}

#endif
