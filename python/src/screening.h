#ifndef screening_H
#define screening_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Eigen>
#endif

#include <vector>
#include "Data.h"

using namespace std;
using namespace Eigen;

Eigen::VectorXi screening(Data<Eigen::VectorXd, Eigen::VectorXd, double> &data, int model_type, int screening_size, Eigen::VectorXi &always_select, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon);
Eigen::VectorXi screening(Data<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> &data, int model_type, int screening_size, Eigen::VectorXi &always_select, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon);

#endif
