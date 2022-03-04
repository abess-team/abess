//
// Created by Kangkang Jiang on 2020/3/8.
//
// #define R_BUILD
#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
#else
#include <Eigen/Eigen>
#endif
#include <iostream>

using namespace std;

void Normalize(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &meanx, double &meany,
               Eigen::VectorXd &normx) {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd tmp(n);
    for (int i = 0; i < p; i++) {
        meanx(i) = weights.dot(X.col(i)) / double(n);
    }
    meany = (y.dot(weights)) / double(n);
    for (int i = 0; i < p; i++) {
        X.col(i) = X.col(i).array() - meanx(i);
    }
    y = y.array() - meany;

    for (int i = 0; i < p; i++) {
        tmp = X.col(i);
        tmp = tmp.array().square();
        normx(i) = sqrt(weights.dot(tmp));
        if (normx(i) == 0) {
#ifdef R_BUILD
            cout << "Warning: the variable " << i + 1 << " is constant. ";
#else
            cout << "Warning: the variable " << i << " is constant. ";
#endif
            cout << "It may cause NAN in the result. Please drop this variable or disable the normalization.\n";
        }
    }
    for (int i = 0; i < p; i++) {
        X.col(i) = sqrt(double(n)) * X.col(i) / normx(i);
    }
}

void Normalize(Eigen::MatrixXd &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &meanx,
               Eigen::VectorXd &meany, Eigen::VectorXd &normx) {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd tmp(n);
    for (int i = 0; i < p; i++) {
        meanx(i) = weights.dot(X.col(i)) / double(n);
    }
    meany = y.transpose() * weights / double(n);
    for (int i = 0; i < p; i++) {
        X.col(i) = X.col(i).array() - meanx(i);
    }

    for (int i = 0; i < n; i++) {
        y.row(i) = y.row(i) - meany;
    }
    // y = y.array() - meany;

    for (int i = 0; i < p; i++) {
        tmp = X.col(i);
        tmp = tmp.array().square();
        normx(i) = sqrt(weights.dot(tmp));
        if (normx(i) == 0) {
#ifdef R_BUILD
            cout << "Warning: the variable " << i + 1 << " is constant. ";
#else
            cout << "Warning: the variable " << i << " is constant. ";
#endif
            cout << "It may cause NAN in the result. Please drop this variable or disable the normalization.\n";
        }
    }
    for (int i = 0; i < p; i++) {
        X.col(i) = sqrt(double(n)) * X.col(i) / normx(i);
    }
}

void Normalize3(Eigen::MatrixXd &X, Eigen::VectorXd &weights, Eigen::VectorXd &meanx, Eigen::VectorXd &normx) {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd tmp(n);
    for (int i = 0; i < p; i++) {
        meanx(i) = weights.dot(X.col(i)) / double(n);
    }
    for (int i = 0; i < p; i++) {
        X.col(i) = X.col(i).array() - meanx(i);
    }
    for (int i = 0; i < p; i++) {
        tmp = X.col(i);
        tmp = tmp.array().square();
        normx(i) = sqrt(weights.dot(tmp));
        if (normx(i) == 0) {
#ifdef R_BUILD
            cout << "Warning: the variable " << i + 1 << " is constant. ";
#else
            cout << "Warning: the variable " << i << " is constant. ";
#endif
            cout << "It may cause NAN in the result. Please drop this variable or disable the normalization.\n";
        }
    }
    for (int i = 0; i < p; i++) {
        X.col(i) = sqrt(double(n)) * X.col(i) / normx(i);
    }
}

void Normalize4(Eigen::MatrixXd &X, Eigen::VectorXd &weights, Eigen::VectorXd &normx) {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd tmp(n);
    for (int i = 0; i < p; i++) {
        tmp = X.col(i);
        tmp = tmp.array().square();
        normx(i) = sqrt(weights.dot(tmp));
        if (normx(i) == 0) {
#ifdef R_BUILD
            cout << "Warning: the variable " << i + 1 << " keeps zero. ";
#else
            cout << "Warning: the variable " << i << " keeps zero. ";
#endif
            cout << "It may cause NAN in the result. Please drop this variable or disable the normalization.\n";
        }
    }
    for (int i = 0; i < p; i++) {
        X.col(i) = sqrt(double(n)) * X.col(i) / normx(i);
    }
}

void Normalize(Eigen::SparseMatrix<double> &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &meanx,
               double &meany, Eigen::VectorXd &normx) {
    return;
}
void Normalize(Eigen::SparseMatrix<double> &X, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &meanx,
               Eigen::VectorXd &meany, Eigen::VectorXd &normx) {
    return;
}
void Normalize3(Eigen::SparseMatrix<double> &X, Eigen::VectorXd &weights, Eigen::VectorXd &meanx,
                Eigen::VectorXd &normx) {
    return;
}
void Normalize4(Eigen::SparseMatrix<double> &X, Eigen::VectorXd &weights, Eigen::VectorXd &normx) { return; }
