#ifndef screening_H
#define screening_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Eigen>
#endif

#include <vector>

using namespace std;
using namespace Eigen;

Eigen::VectorXi screening(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, int algorithm_type, int screening_size, Eigen::VectorXi &g_index, Eigen::VectorXi &always_select);

#endif
