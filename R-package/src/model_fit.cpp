//
// Created by jk on 2020/3/18.
//

#ifndef SRC_MODELFIT_H
#define SRC_MODELFIT_H
#endif

// #define TEST

#ifdef R_BUILD

#include <Rcpp.h>
#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;

#else

#include <Eigen/Eigen>
#include "List.h"

#endif

#include <cfloat>
#include "utilities.h"
#include <iostream>
using namespace std;

Eigen::VectorXd pi(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &coef, int n)
{
    int p = coef.size();
    Eigen::VectorXd Pi = Eigen::VectorXd::Zero(n);
    if (X.cols() == p - 1)
    {
        Eigen::VectorXd intercept = Eigen::VectorXd::Ones(n) * coef(0);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
        Eigen::VectorXd eta = X * (coef.tail(p - 1).eval()) + intercept;
        for (int i = 0; i < n; i++)
        {
            if (eta(i) > 30)
            {
                eta(i) = 30;
            }
            else if (eta(i) < -30)
            {
                eta(i) = -30;
            }
        }
        Eigen::VectorXd expeta = eta.array().exp();
        Pi = expeta.array() / (one + expeta).array();
        return Pi;
    }
    else
    {
        Eigen::VectorXd eta = X * coef;
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

        for (int i = 0; i < n; i++)
        {
            if (eta(i) > 30)
            {
                eta(i) = 30;
            }
            else if (eta(i) < -30)
            {
                eta(i) = -30;
            }
        }
        Eigen::VectorXd expeta = eta.array().exp();
        Pi = expeta.array() / (one + expeta).array();
        return Pi;
    }
}

Eigen::MatrixXd pi(Eigen::MatrixXd &X, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0)
{
    int n = X.rows();
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, 1);
    Eigen::MatrixXd Xbeta = X * beta + one * coef0.transpose();
    Eigen::MatrixXd pi = Xbeta.array().exp();
    Eigen::VectorXd sumpi = pi.rowwise().sum();
    for (int i = 0; i < n; i++)
    {
        pi.row(i) = pi.row(i) / sumpi(i);
    }
    return pi;
};

Eigen::MatrixXd pi(Eigen::MatrixXd &X, Eigen::MatrixXd &y, Eigen::MatrixXd &coef)
{
    int n = X.rows();
    // Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, 1);
    Eigen::MatrixXd Xbeta = X * coef;
    Eigen::MatrixXd pi = Xbeta.array().exp();
    Eigen::VectorXd sumpi = pi.rowwise().sum();
    for (int i = 0; i < n; i++)
    {
        pi.row(i) = pi.row(i) / sumpi(i);
    }
    return pi;
};

void multinomial_fit(Eigen::MatrixXd &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{
#ifdef TEST
    clock_t t1 = clock();
#endif
    // cout << "primary_fit-----------" << endl;
    // if (X.cols() == 0)
    // {
    //   coef0 = -log(y.colwise().sum().eval() - 1.0);
    //   return;
    // }

#ifdef TEST
    std::cout << "primary_model_fit 1" << endl;
#endif
    int n = x.rows();
    int p = x.cols();
    int M = y.cols();
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
    X.rightCols(p) = x;
    // Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
    // Eigen::MatrixXd X_new_transpose = Eigen::MatrixXd::Zero(p + 1, n);
    Eigen::MatrixXd beta0 = Eigen::MatrixXd::Zero(p + 1, M);
#ifdef TEST
    std::cout << "primary_model_fit 2" << endl;
#endif
    beta0.row(0) = coef0;
    beta0.block(1, 0, p, M) = beta;
#ifdef TEST
    std::cout << "primary_model_fit 3" << endl;
#endif
    // Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::MatrixXd Pi = pi(X, y, beta0);
    Eigen::MatrixXd log_Pi = Pi.array().log();
    array_product(log_Pi, weights, 1);
    double loglik1 = DBL_MAX, loglik0 = (log_Pi.array() * y.array()).sum();
#ifdef TEST
    std::cout << "primary_model_fit 4" << endl;
#endif

    int j;
    if (approximate_Newton)
    {
        Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, M);
        double t = 2 * (Pi.array() * (one - Pi).array()).maxCoeff();
        Eigen::MatrixXd res = X.transpose() * (y - Pi) / t;
        // ConjugateGradient<MatrixXd, Lower | Upper> cg;
        // cg.compute(X.adjoint() * X);
        Eigen::MatrixXd XTX = X.transpose() * X;
        Eigen::MatrixXd invXTX = XTX.ldlt().solve(Eigen::MatrixXd::Identity(p + 1, p + 1));

        Eigen::MatrixXd beta1;
        for (j = 0; j < primary_model_fit_max_iter; j++)
        {
#ifdef TEST
            std::cout << "primary_model_fit 3: " << j << endl;
#endif

            // beta1 = beta0 + cg.solve(res);
            beta1 = beta0 + invXTX * res;
            // cout << "beta1: " << beta1 << endl;

            // double app_loss0, app_loss1, app_loss2;
            // app_loss0 = ((y - Pi) / t).squaredNorm();
            // app_loss1 = (-X * beta0 - (y - Pi) / t).squaredNorm();
            // app_loss2 = (X * (beta1 - beta0) - (y - Pi) / t).squaredNorm();
            // cout << "app_loss0: " << app_loss0 << endl;
            // cout << "app_loss1: " << app_loss1 << endl;
            // cout << "app_loss2: " << app_loss2 << endl;

            Pi = pi(X, y, beta1);
            log_Pi = Pi.array().log();
            array_product(log_Pi, weights, 1);
            loglik1 = (log_Pi.array() * y.array()).sum();
            // cout << "loglik1: " << loglik1 << endl;
            // cout << "loglik0: " << loglik0 << endl;

            // cout << "j=" << j << " loglik: " << loglik1 << endl;
            // cout << "j=" << j << " loglik diff: " << loglik0 - loglik1 << endl;
            bool condition1 = -(loglik1 + (primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
            // bool condition1 = false;
            bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < primary_model_fit_epsilon;
            bool condition3 = abs(loglik1) < min(1e-3, tau);
            bool condition4 = loglik1 < loglik0;
            // bool condition4 = false;
            if (condition1 || condition2 || condition3 || condition4)
            {
                break;
            }
            loglik0 = loglik1;
            for (int m1 = 0; m1 < M; m1++)
            {
                beta0.col(m1) = beta1.col(m1) - beta1.col(M - 1);
            }
            // beta0 = beta1;
            t = 2 * (Pi.array() * (one - Pi).array()).maxCoeff();
            res = X.transpose() * (y - Pi) / t;
        }
    }
    else
    {
        /* Newton */
        Eigen::MatrixXd W(M * n, M * n);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
        for (int m1 = 0; m1 < M; m1++)
        {
            for (int m2 = m1; m2 < M; m2++)
            {
                if (m1 == m2)
                {
                    W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
#ifdef TEST
                    std::cout << "primary_model_fit 5" << endl;

#endif
                    Eigen::VectorXd PiPj = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array();
                    // cout << "PiPj: " << PiPj << endl;
                    for (int i = 0; i < PiPj.size(); i++)
                    {
                        if (PiPj(i) < 0.001)
                        {
                            PiPj(i) = 0.001;
                        }
                    }
                    W.block(m1 * n, m2 * n, n, n).diagonal() = PiPj;

#ifdef TEST
                    std::cout << "primary_model_fit 6" << endl;
                    cout << "W m1 m2: " << W.block(m1 * n, m2 * n, n, n) << endl;
#endif
                }
                else
                {
                    W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
#ifdef TEST
                    std::cout << "primary_model_fit 5" << endl;

#endif
                    Eigen::VectorXd PiPj = Pi.col(m1).array() * Pi.col(m2).array();
                    // cout << "PiPj: " << PiPj << endl;
                    for (int i = 0; i < PiPj.size(); i++)
                    {
                        if (PiPj(i) < 0.001)
                        {
                            PiPj(i) = 0.001;
                        }
                    }
                    W.block(m1 * n, m2 * n, n, n).diagonal() = -PiPj;
                    W.block(m2 * n, m1 * n, n, n) = W.block(m1 * n, m2 * n, n, n);

                    // cout << "W m1 m2: " << W.block(m1 * n, m2 * n, n, n) << endl;
                }
            }
        }

#ifdef TEST
        std::cout << "primary_model_fit 7" << endl;
#endif
        // cout << "W: " << W << endl;

        Eigen::MatrixXd XTWX(M * (p + 1), M * (p + 1));
        Eigen::MatrixXd XTW(M * (p + 1), M * n);
        for (int m1 = 0; m1 < M; m1++)
        {
            for (int m2 = m1; m2 < M; m2++)
            {
                XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) = X.transpose() * W.block(m1 * n, m2 * n, n, n);
                XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X;
                XTW.block(m2 * (p + 1), m1 * n, (p + 1), n) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n);
                XTWX.block(m2 * (p + 1), m1 * (p + 1), (p + 1), (p + 1)) = XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1));
            }
        }

#ifdef TEST
        std::cout << "primary_model_fit 8" << endl;
#endif

        // Eigen::Matrix<Eigen::MatrixXd, -1, -1> res(M, 1);
        Eigen::VectorXd res(M * n);
        for (int m1 = 0; m1 < M; m1++)
        {
            res.segment(m1 * n, n) = y.col(m1).eval() - Pi.col(m1).eval();
        }

#ifdef TEST
        std::cout << "primary_model_fit 9" << endl;
        cout << "res: " << res << endl;
#endif

        Eigen::VectorXd Xbeta(M * n);
        for (int m1 = 0; m1 < M; m1++)
        {
            Xbeta.segment(m1 * n, n) = X * beta0.col(m1).eval();
        }

#ifdef TEST
        std::cout << "primary_model_fit 10" << endl;
        cout << "Xbeta: " << Xbeta << endl;
#endif

        Eigen::VectorXd Z = Xbeta + W.ldlt().solve(res);
#ifdef TEST
        std::cout << "primary_model_fit 11" << endl;
#endif

#ifdef TEST
        std::cout << "primary_model_fit 2" << endl;
#endif

        Eigen::MatrixXd beta1;
        Eigen::VectorXd beta0_tmp;
        for (j = 0; j < primary_model_fit_max_iter; j++)
        {
#ifdef TEST
            std::cout << "primary_model_fit 3: " << j << endl;
#endif
            beta0_tmp = XTWX.ldlt().solve(XTW * Z);
            for (int m1 = 0; m1 < M; m1++)
            {
                beta0.col(m1) = beta0_tmp.segment(m1 * (p + 1), (p + 1)) - beta0_tmp.segment((M - 1) * (p + 1), (p + 1));
            }
            // cout << "beta0" << beta0 << endl;

            Pi = pi(X, y, beta0);
            log_Pi = Pi.array().log();
            array_product(log_Pi, weights, 1);
            loglik1 = (log_Pi.array() * y.array()).sum();
            // cout << "loss" << loglik1 << endl;
            // cout << "j=" << j << " loglik: " << loglik1 << endl;
            // cout << "j=" << j << " loglik diff: " << loglik0 - loglik1 << endl;
            bool condition1 = -(loglik1 + (primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
            // bool condition1 = false;
            bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < primary_model_fit_epsilon;
            bool condition3 = abs(loglik1) < min(1e-3, tau);
            bool condition4 = loglik1 < loglik0;
            if (condition1 || condition2 || condition3 || condition4)
            {
                // cout << "condition1:" << condition1 << endl;
                // cout << "condition2:" << condition2 << endl;
                // cout << "condition3:" << condition3 << endl;
                break;
            }
            loglik0 = loglik1;

            for (int m1 = 0; m1 < M; m1++)
            {
                for (int m2 = m1; m2 < M; m2++)
                {
                    if (m1 == m2)
                    {
                        // W(m1, m2) = Eigen::MatrixXd::Zero(n, n);
                        // W(m1, m2).diagonal() = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array();

                        W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
                        Eigen::VectorXd PiPj = Pi.col(m1).array() * (one - Pi.col(m1).eval()).array();
                        for (int i = 0; i < PiPj.size(); i++)
                        {
                            if (PiPj(i) < 0.001)
                            {
                                PiPj(i) = 0.001;
                            }
                        }
                        W.block(m1 * n, m2 * n, n, n).diagonal() = PiPj;
                    }
                    else
                    {
                        W.block(m1 * n, m2 * n, n, n) = Eigen::MatrixXd::Zero(n, n);
                        Eigen::VectorXd PiPj = Pi.col(m1).array() * Pi.col(m2).array();
                        for (int i = 0; i < PiPj.size(); i++)
                        {
                            if (PiPj(i) < 0.001)
                            {
                                PiPj(i) = 0.001;
                            }
                        }
                        W.block(m1 * n, m2 * n, n, n).diagonal() = -PiPj;
                        W.block(m2 * n, m1 * n, n, n) = W.block(m1 * n, m2 * n, n, n);
                    }
                }
            }

            for (int m1 = 0; m1 < M; m1++)
            {
                for (int m2 = m1; m2 < M; m2++)
                {
                    XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) = X.transpose() * W.block(m1 * n, m2 * n, n, n);
                    XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1)) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n) * X;
                    XTW.block(m2 * (p + 1), m1 * n, (p + 1), n) = XTW.block(m1 * (p + 1), m2 * n, (p + 1), n);
                    XTWX.block(m2 * (p + 1), m1 * (p + 1), (p + 1), (p + 1)) = XTWX.block(m1 * (p + 1), m2 * (p + 1), (p + 1), (p + 1));
                }
            }

            for (int m1 = 0; m1 < M; m1++)
            {
                res.segment(m1 * n, n) = y.col(m1).eval() - Pi.col(m1).eval();
            }

            for (int m1 = 0; m1 < M; m1++)
            {
                Xbeta.segment(m1 * n, n) = X * beta0.col(m1).eval();
            }

            Z = Xbeta + W.ldlt().solve(res);
        }
    }

#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << j << endl;
#endif

    beta = beta0.block(1, 0, p, M);
    coef0 = beta0.row(0).eval();
};

void multigaussian_fit(Eigen::MatrixXd &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{
    // CG:
    // Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower | Eigen::Upper> cg;
    // cg.compute(x.adjoint() * x + lambda * Eigen::MatrixXd::Identity(x.cols(), x.cols()));
    // beta = cg.solveWithGuess(x.adjoint() * y, beta);

    // colPivHouseholderQr
    if (lambda == 0.0) {
        beta = x.colPivHouseholderQr().solve(y);
    } else {
        beta = (x.adjoint() * x + lambda * Eigen::MatrixXd::Identity(x.cols(), x.cols())).colPivHouseholderQr().solve(x.adjoint() * y);
    }
}

Eigen::VectorXd logit_fit(Eigen::MatrixXd x, Eigen::VectorXd y, int n, int p, Eigen::VectorXd weights)
{
    if (n <= p)
    {
        Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, n);
        Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
        X.rightCols(n - 1) = x.leftCols(n - 1);
        Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd Pi = pi(X, y, beta0, n);
        Eigen::VectorXd log_Pi = Pi.array().log();
        Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
        double loglik0 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
        Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
        Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);
        W = W.cwiseProduct(weights);
        for (int i = 0; i < n; i++)
        {
            X_new.row(i) = X.row(i) * W(i);
        }
        beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);

        double loglik1;

        int j;
        for (j = 0; j < 30; j++)
        {
            Pi = pi(X, y, beta1, n);
            log_Pi = Pi.array().log();
            log_1_Pi = (one - Pi).array().log();
            loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
            if (abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < 1e-6)
            {
                break;
            }
            beta0 = beta1;
            loglik0 = loglik1;
            W = Pi.cwiseProduct(one - Pi);
            Z = X * beta0 + (y - Pi).cwiseQuotient(W);
            W = W.cwiseProduct(weights);
            for (int i = 0; i < n; i++)
            {
                X_new.row(i) = X.row(i) * W(i);
            }
            beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
        }
        return beta0;
    }

    else
    {
        Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
        X.rightCols(p) = x;
        Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
        Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
        Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p + 1);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
        Eigen::VectorXd Pi = pi(X, y, beta0, n);
        Eigen::VectorXd log_Pi = Pi.array().log();
        Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
        double loglik0 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
        Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
        Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);
        W = W.cwiseProduct(weights);
        for (int i = 0; i < n; i++)
        {
            X_new.row(i) = X.row(i) * W(i);
        }
        beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
        double loglik1;

        int j;
        for (j = 0; j < 30; j++)
        {
            Pi = pi(X, y, beta1, n);
            log_Pi = Pi.array().log();
            log_1_Pi = (one - Pi).array().log();
            loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
            if (abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < 1e-6)
            {
                break;
            }
            beta0 = beta1;
            loglik0 = loglik1;
            W = Pi.cwiseProduct(one - Pi);
            Z = X * beta0 + (y - Pi).cwiseQuotient(W);
            W = W.cwiseProduct(weights);
            for (int i = 0; i < n; i++)
            {
                X_new.row(i) = X.row(i) * W(i);
            }
            beta1 = (X_new.transpose() * X).ldlt().solve(X_new.transpose() * Z);
        }
        return beta0;
    }
}

double loglik_logit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd coef, int n, Eigen::VectorXd weights)
{
    Eigen::VectorXd Pi = pi(X, y, coef, n);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd log_Pi = Pi.array().log();
    Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
    return (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
}

void logistic_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{
#ifdef TEST
    clock_t t1 = clock();
#endif
    // cout << "primary_fit-----------" << endl;
    if (x.cols() == 0)
    {
        coef0 = -log(1 / y.mean() - 1);
        return;
    }

    int n = x.rows();
    int p = x.cols();
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
    X.rightCols(p) = x;
    Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p + 1);
    Eigen::MatrixXd X_new_transpose = Eigen::MatrixXd::Zero(p + 1, n);
    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
    beta0(0) = coef0;
    beta0.tail(p) = beta;
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd Pi = pi(X, y, beta0, n);
    Eigen::VectorXd log_Pi = Pi.array().log();
    Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
    double loglik1 = DBL_MAX, loglik0 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
    Eigen::VectorXd W = Pi.cwiseProduct(one - Pi);
    for (int i = 0; i < n; i++)
    {
        if (W(i) < 0.001)
            W(i) = 0.001;
    }
    Eigen::VectorXd Z = X * beta0 + (y - Pi).cwiseQuotient(W);

    // cout << "l0 loglik: " << loglik0 << endl;

    int j;
    double step = 1;
    Eigen::VectorXd g(p + 1);
    Eigen::VectorXd beta1;
    for (j = 0; j < primary_model_fit_max_iter; j++)
    {
        // To do: Approximate Newton method
        if (approximate_Newton)
        {
            Eigen::VectorXd h_diag(p + 1);
            for (int i = 0; i < p + 1; i++)
            {
                h_diag(i) = 1.0 / X.col(i).cwiseProduct(W).cwiseProduct(weights).dot(X.col(i));
            }
            g = X.transpose() * ((y - Pi).cwiseProduct(weights));
            beta1 = beta0 + step * g.cwiseProduct(h_diag);
            Pi = pi(X, y, beta1, n);
            log_Pi = Pi.array().log();
            log_1_Pi = (one - Pi).array().log();
            loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);

            while (loglik1 < loglik0 && step > primary_model_fit_epsilon)
            {
                step = step / 2;
                beta1 = beta0 + step * g.cwiseProduct(h_diag);
                Pi = pi(X, y, beta1, n);
                log_Pi = Pi.array().log();
                log_1_Pi = (one - Pi).array().log();
                loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
            }

            // cout << "j=" << j << " loglik: " << loglik1 << endl;
            // cout << "j=" << j << " loglik diff: " << loglik1 - loglik0 << endl;
            bool condition1 = -(loglik1 + (primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
            // bool condition1 = false;
            if (condition1)
                break;

            if (loglik1 > loglik0)
            {
                beta0 = beta1;
                loglik0 = loglik1;
                W = Pi.cwiseProduct(one - Pi);
                for (int i = 0; i < n; i++)
                {
                    if (W(i) < 0.001)
                        W(i) = 0.001;
                }
            }

            if (step < primary_model_fit_epsilon)
            {
                break;
            }
        }
        else
        {
            for (int i = 0; i < p + 1; i++)
            {
                X_new.col(i) = X.col(i).cwiseProduct(W).cwiseProduct(weights);
            }
            X_new_transpose = X_new.transpose();

            beta0 = (X_new_transpose * X).ldlt().solve(X_new_transpose * Z);

            // CG
            // ConjugateGradient<MatrixXd, Lower | Upper> cg;
            // cg.compute(X_new_transpose * X);
            // beta0 = cg.solveWithGuess(X_new_transpose * Z, beta0);

            Pi = pi(X, y, beta0, n);
            log_Pi = Pi.array().log();
            log_1_Pi = (one - Pi).array().log();
            loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
            // cout << "j=" << j << " loglik: " << loglik1 << endl;
            // cout << "j=" << j << " loglik diff: " << loglik0 - loglik1 << endl;
            bool condition1 = -(loglik1 + (primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
            // bool condition1 = false;
            bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < primary_model_fit_epsilon;
            bool condition3 = abs(loglik1) < min(1e-3, tau);
            if (condition1 || condition2 || condition3)
            {
                // cout << "condition1:" << condition1 << endl;
                // cout << "condition2:" << condition2 << endl;
                // cout << "condition3:" << condition3 << endl;
                break;
            }

            loglik0 = loglik1;
            W = Pi.cwiseProduct(one - Pi);
            for (int i = 0; i < n; i++)
            {
                if (W(i) < 0.001)
                    W(i) = 0.001;
            }
            Z = X * beta0 + (y - Pi).cwiseQuotient(W);
        }
    }
#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << j << endl;
#endif
    beta = beta0.tail(p).eval();
    coef0 = beta0(0);
};

void lm_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{
    // beta = (X.adjoint() * X + lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).colPivHouseholderQr().solve(X.adjoint() * y);

    // beta = (X.adjoint() * X + lambda_level * Eigen::MatrixXd::Identity(X.cols(), X.cols())).ldlt().solve(X.adjoint() * y);

    // if (X.cols() == 0)
    // {
    //   coef0 = y.mean();
    //   return;
    // }

    // CG
    // Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower | Eigen::Upper> cg;
    // cg.compute(x.adjoint() * x + lambda * Eigen::MatrixXd::Identity(x.cols(), x.cols()));
    // beta = cg.solveWithGuess(x.adjoint() * y, beta);

    // colPivHouseholderQr
    if (lambda == 0.0) {
        beta = x.colPivHouseholderQr().solve(y);
    } else {
        beta = (x.adjoint() * x + lambda * Eigen::MatrixXd::Identity(x.cols(), x.cols())).colPivHouseholderQr().solve(x.adjoint() * y);
    }

#ifdef TEST
    cout << "xx: " << x.adjoint() * x << endl;
    cout << "xy: " << x.adjoint() * y << endl;
#endif
};

void poisson_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{
#ifdef TEST
    clock_t t1 = clock();
    std::cout << "primary_fit-----------" << endl;
#endif
    if (x.cols() == 0)
    {
        coef0 = log(y.mean());
        return;
    }

    int n = x.rows();
    int p = x.cols();
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
    X.rightCols(p) = x;
    Eigen::MatrixXd X_trans = X.transpose();
    Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(p + 1, n);
    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
    beta0.tail(p) = beta;
    beta0(0) = coef0;
    Eigen::VectorXd eta = X * beta0;
    Eigen::VectorXd expeta = eta.array().exp();
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
    double loglik0 = (y.cwiseProduct(eta) - expeta).dot(weights);
    double loglik1;

    int j;
    for (j = 0; j < primary_model_fit_max_iter; j++) {
        for (int i = 0; i < n; i++)
        {
            temp.col(i) = X_trans.col(i) * expeta(i) * weights(i);
        }
        z = eta + (y - expeta).cwiseQuotient(expeta);
        beta0 = (temp * X).ldlt().solve(temp * z);
        eta = X * beta0;
        for (int i = 0; i < n; i++)
        {
            if (eta(i) < -30.0)
                eta(i) = -30.0;
            if (eta(i) > 30.0)
                eta(i) = 30.0;
        }
        expeta = eta.array().exp();
        loglik1 = (y.cwiseProduct(eta) - expeta).dot(weights);
        bool condition1 = -(loglik1 + (primary_model_fit_max_iter - j - 1) * (loglik1 - loglik0)) + tau > loss0;
        // bool condition1 = false;
        bool condition2 = abs(loglik0 - loglik1) / (0.1 + abs(loglik1)) < primary_model_fit_epsilon;
        bool condition3 = abs(loglik1) < min(1e-3, tau);
        if (condition1 || condition2 || condition3)
        {
            // cout << "condition1:" << condition1 << endl;
            // cout << "condition2:" << condition2 << endl;
            // cout << "condition3:" << condition3 << endl;
            break;
        }
        loglik0 = loglik1;
    }
#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << j << endl;
#endif
    beta = beta0.tail(p).eval();
    coef0 = beta0(0);
};

double loglik_cox(Eigen::MatrixXd &X, Eigen::VectorXd &status, Eigen::VectorXd &beta, Eigen::VectorXd &weights)
{
    int n = X.rows();
    Eigen::VectorXd eta = X * beta;
    for (int i = 0; i < n; i++)
    {
        if (eta(i) > 30)
        {
            eta(i) = 30;
        }
        else if (eta(i) < -30)
        {
            eta(i) = -30;
        }
    }
    Eigen::VectorXd expeta = eta.array().exp();
    Eigen::VectorXd cum_expeta(n);
    cum_expeta(n - 1) = expeta(n - 1);
    for (int i = n - 2; i >= 0; i--)
    {
        cum_expeta(i) = cum_expeta(i + 1) + expeta(i);
    }
    Eigen::VectorXd ratio = (expeta.cwiseQuotient(cum_expeta)).array().log();
    return (ratio.cwiseProduct(status)).dot(weights);
}

void cox_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{
#ifdef TEST
    clock_t t1 = clock();
    std::cout << "primary_fit-----------" << std::endl;
#endif
    if (x.cols() == 0)
    {
        coef0 = 0.;
        return;
    }

    int n = x.rows();
    int p = x.cols();
    Eigen::VectorXd theta(n);
    Eigen::MatrixXd one = (Eigen::MatrixXd::Ones(n, n)).triangularView<Eigen::Upper>();
    Eigen::MatrixXd x_theta(n, p);
    Eigen::VectorXd xij_theta(n);
    Eigen::VectorXd cum_theta(n);
    Eigen::VectorXd g(p);
    Eigen::VectorXd beta0 = beta, beta1;
    Eigen::VectorXd cum_eta(n);
    Eigen::VectorXd cum_eta2(n);
    Eigen::VectorXd cum_eta3(n);
    Eigen::MatrixXd h(p, p);
    Eigen::VectorXd eta;

    Eigen::VectorXd d(p);
    double loglik1, loglik0 = loglik_cox(x, y, weight, beta0);
    // beta = Eigen::VectorXd::Zero(p);

    double step = 1.0;
    int l;
    for (l = 1; l <= primary_model_fit_max_iter; l++)
    {
        eta = x * beta0;
        for (int i = 0; i <= n - 1; i++)
        {
            if (eta(i) < -30.0)
                eta(i) = -30.0;
            if (eta(i) > 30.0)
                eta(i) = 30.0;
        }
        eta = weight.array() * eta.array().exp();
        cum_eta(n - 1) = eta(n - 1);
        for (int k = n - 2; k >= 0; k--)
        {
            cum_eta(k) = cum_eta(k + 1) + eta(k);
        }
        cum_eta2(0) = (y(0) * weight(0)) / cum_eta(0);
        for (int k = 1; k <= n - 1; k++)
        {
            cum_eta2(k) = (y(k) * weight(k)) / cum_eta(k) + cum_eta2(k - 1);
        }
        cum_eta3(0) = (y(0) * weight(0)) / pow(cum_eta(0), 2);
        for (int k = 1; k <= n - 1; k++)
        {
            cum_eta3(k) = (y(k) * weight(k)) / pow(cum_eta(k), 2) + cum_eta3(k - 1);
        }
        h = -cum_eta3.replicate(1, n);
        h = h.cwiseProduct(eta.replicate(1, n));
        h = h.cwiseProduct(eta.replicate(1, n).transpose());
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                h(j, i) = h(i, j);
            }
        }
        h.diagonal() = cum_eta2.cwiseProduct(eta) + h.diagonal();
        g = weight.cwiseProduct(y) - cum_eta2.cwiseProduct(eta);

        if (approximate_Newton)
        {
            d = (x.transpose() * g).cwiseQuotient((x.transpose() * h * x).diagonal());
        }
        else
        {
            d = (x.transpose() * h * x).ldlt().solve(x.transpose() * g);
        }

        beta1 = beta0 + step * d;

        loglik1 = loglik_cox(x, y, weight, beta1);

        while (loglik1 < loglik0 && step > primary_model_fit_epsilon)
        {
            step = step / 2;
            beta1 = beta0 + step * d;
            loglik1 = loglik_cox(x, y, weight, beta1);
        }

        bool condition1 = -(loglik1 + (primary_model_fit_max_iter - l - 1) * (loglik1 - loglik0)) + tau > loss0;
        if (condition1)
        {
            loss0 = -loglik0;
            // beta = beta0;
            break;
        }

        if (loglik1 > loglik0)
        {
            beta0 = beta1;
            loglik0 = loglik1;
        }

        if (step < primary_model_fit_epsilon)
        {
            loss0 = -loglik0;
            // cout << "condition2" << endl;
            break;
        }
    }
#ifdef TEST
    clock_t t2 = clock();
    std::cout << "primary fit time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "primary fit iter : " << l << endl;
#endif

    beta = beta0;
};

double loglik_poiss(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &coef, int n, Eigen::VectorXd &weights)
{
    int p = x.cols();
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p + 1);
    X.rightCols(p) = x;
    Eigen::VectorXd eta = X * coef;
    for (int i = 0; i <= n - 1; i++)
    {
        if (eta(i) < -30.0)
            eta(i) = -30.0;
        if (eta(i) > 30.0)
            eta(i) = 30.0;
    }
    Eigen::VectorXd expeta = eta.array().exp();
    return (y.cwiseProduct(eta) - expeta).dot(weights);
}
