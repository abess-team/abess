//  #define R_BUILD
#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen\Eigen>
#include "List.h"
#endif

#include <algorithm>
#include <vector>
#include <cmath>
#include "screening.h"
#include "logistic.h"
#include "poisson.h"
#include "coxph.h"
#include "utilities.h"
#include <iostream>
#include <cfloat>

using namespace std;
using namespace Eigen;

Eigen::VectorXi screening(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, int algorithm_type, int screening_size, Eigen::VectorXi &g_index, Eigen::VectorXi &always_select)
{
    int n = x.rows();
    int p = x.cols();
    Eigen::VectorXi screening_A(screening_size);

    int g_num = (g_index).size();
    Eigen::VectorXi temp = Eigen::VectorXi::Zero(g_num);
    temp.head(g_num - 1) = g_index.tail(g_num - 1);
    temp(g_num - 1) = p;
    Eigen::VectorXi g_size = temp - g_index;

    Eigen::VectorXd coef_norm = Eigen::VectorXd::Zero(g_num);

    for (int i = 0; i < g_num; i++)
    {
        Eigen::MatrixXd x_tmp = x.middleCols(g_index(i), g_size(i));
        Eigen::VectorXd beta;
        if (algorithm_type == 1)
        {
            beta = x_tmp.colPivHouseholderQr().solve(y);
        }
        else if (algorithm_type == 2)
        {
            beta = logit_fit(x_tmp, y, n, g_size(i), weight);
        }
        else if (algorithm_type == 3)
        {
            beta = poisson_fit(x_tmp, y, n, g_size(i), weight);
        }
        else if (algorithm_type == 4)
        {
            beta = cox_fit(x_tmp, y, n, g_size(i), weight);
        }
        coef_norm(i) = beta.tail(g_size(i)).eval().squaredNorm() / g_size(i);
    }

    // keep always_select in active_set
    slice_assignment(coef_norm, always_select, DBL_MAX);

    max_k(coef_norm, screening_size, screening_A);

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

    Eigen::MatrixXd x_A = Eigen::MatrixXd::Zero(n, new_p);
    for (int i = 0; i < screening_size; i++)
    {
        x_A.middleCols(new_g_index(i), new_g_size(i)) = x.middleCols(g_index(screening_A(i)), g_size(screening_A(i)));
    }
    x = x_A;
    g_index = new_g_index;

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

    return screening_A;
}
