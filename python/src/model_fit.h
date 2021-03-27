#ifndef MODEL_FIT_H
#define MODEL_FIT_H

#ifdef R_BUILD

#include <Rcpp.h>
#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
#else
#include <Eigen/Eigen>
#include "List.h"
#endif

#include "utilities.h"
#include <cfloat>

Eigen::VectorXd pi(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &coef, int n);
Eigen::MatrixXd pi(Eigen::MatrixXd &X, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0);
Eigen::MatrixXd pi(Eigen::MatrixXd &X, Eigen::MatrixXd &y, Eigen::MatrixXd &coef);
void multinomial_fit(Eigen::MatrixXd &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);
void multigaussian_fit(Eigen::MatrixXd &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);
Eigen::VectorXd logit_fit(Eigen::MatrixXd x, Eigen::VectorXd y, int n, int p, Eigen::VectorXd weights);
double loglik_logit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd coef, int n, Eigen::VectorXd weights);
void logistic_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);
void lm_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);
void poisson_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);
double loglik_cox(Eigen::MatrixXd &X, Eigen::VectorXd &status, Eigen::VectorXd &beta, Eigen::VectorXd &weights);
void cox_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);

#endif