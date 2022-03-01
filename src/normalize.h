//
// Created by Jin Zhu on 2020/3/8.
//
// #define R_BUILD
#ifndef SRC_NORMALIZE_H
#define SRC_NORMALIZE_H

#ifdef R_BUILD
#include <RcppEigen.h>
#else
#include <Eigen/Eigen>
#endif

void Normalize(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &meanx, double &meany,
               Eigen::VectorXd &normx);
void Normalize(Eigen::MatrixXd &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &meanx,
               Eigen::VectorXd &meany, Eigen::VectorXd &normx);
void Normalize3(Eigen::MatrixXd &X, Eigen::VectorXd &weights, Eigen::VectorXd &meanx, Eigen::VectorXd &normx);
void Normalize4(Eigen::MatrixXd &X, Eigen::VectorXd &weights, Eigen::VectorXd &normx);

void Normalize(Eigen::SparseMatrix<double> &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &meanx,
               double &meany, Eigen::VectorXd &normx);
void Normalize(Eigen::SparseMatrix<double> &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &meanx,
               Eigen::VectorXd &meany, Eigen::VectorXd &normx);
void Normalize3(Eigen::SparseMatrix<double> &X, Eigen::VectorXd &weights, Eigen::VectorXd &meanx,
                Eigen::VectorXd &normx);
void Normalize4(Eigen::SparseMatrix<double> &X, Eigen::VectorXd &weights, Eigen::VectorXd &normx);

#endif  // SRC_NORMALIZE_H
