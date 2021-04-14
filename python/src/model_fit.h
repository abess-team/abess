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

template <class T4>
Eigen::VectorXd pi(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &coef)
{
    int p = coef.size();
    int n = X.rows();
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

template <class T4>
void pi(T4 &X, Eigen::MatrixXd &y, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, Eigen::MatrixXd &pr)
{
    int n = X.rows();
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, 1);
    Eigen::MatrixXd Xbeta = X * beta + one * coef0.transpose();
    pr = Xbeta.array().exp();
    Eigen::VectorXd sumpi = pr.rowwise().sum();
    for (int i = 0; i < n; i++)
    {
        pr.row(i) = pr.row(i) / sumpi(i);
    }
    // cout << "pi: " << pi.block(0, 0, 5, y.cols());
    // return pi;
};

template <class T4>
void pi(T4 &X, Eigen::MatrixXd &y, Eigen::MatrixXd &coef, Eigen::MatrixXd &pr)
{
    int n = X.rows();
    // Eigen::MatrixXd one = Eigen::MatrixXd::Ones(n, 1);
    Eigen::MatrixXd Xbeta = X * coef;
    pr = Xbeta.array().exp();
    Eigen::VectorXd sumpi = pr.rowwise().sum();
    for (int i = 0; i < n; i++)
    {
        pr.row(i) = pr.row(i) / sumpi(i);
    }
    // cout << "pi: " << pi.block(0, 0, 5, y.cols());
    // return pi;
};

void multinomial_fit(Eigen::MatrixXd &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);
void multigaussian_fit(Eigen::MatrixXd &x, Eigen::MatrixXd &y, Eigen::VectorXd &weights, Eigen::MatrixXd &beta, Eigen::VectorXd &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);
Eigen::VectorXd logit_fit(Eigen::MatrixXd x, Eigen::VectorXd y, int n, int p, Eigen::VectorXd weights);

template <class T4>
double loglik_logit(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &coef, int n, Eigen::VectorXd weights)
{
    Eigen::VectorXd Pi = pi(X, y, coef);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd log_Pi = Pi.array().log();
    Eigen::VectorXd log_1_Pi = (one - Pi).array().log();
    return (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);
}

template <class T4>
void logistic_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda)
{
    // cout << "primary_fit-----------" << endl;
    if (x.cols() == 0)
    {
        coef0 = -log(1 / y.mean() - 1);
        return;
    }

    int n = x.rows();
    int p = x.cols();

    // to ensure
    T4 X(n, p + 1);
    X.rightCols(p) = x;

    // to do !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // X.col(0) = Eigen::MatrixXd::Ones(n, 1);

    T4 X_new(n, p + 1);
    T4 X_new_transpose(p + 1, n);

    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p + 1);
    beta0(0) = coef0;
    beta0.tail(p) = beta;
    Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

    Eigen::VectorXd Pi = pi(X, y, beta0);

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
            Pi = pi(X, y, beta1);
            log_Pi = Pi.array().log();
            log_1_Pi = (one - Pi).array().log();
            loglik1 = (y.cwiseProduct(log_Pi) + (one - y).cwiseProduct(log_1_Pi)).dot(weights);

            while (loglik1 < loglik0 && step > primary_model_fit_epsilon)
            {
                step = step / 2;
                beta1 = beta0 + step * g.cwiseProduct(h_diag);
                Pi = pi(X, y, beta1);
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

            // to ensure
            // beta0 = (X_new_transpose * X).llt().solve(X_new_transpose * Z);

            // CG
            ConjugateGradient<T4, Lower | Upper> cg;
            cg.compute(X_new_transpose * X);
            beta0 = cg.solve(X_new_transpose * Z);

            Pi = pi(X, y, beta0);
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
    beta = beta0.tail(p).eval();
    coef0 = beta0(0);
};
void lm_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);
void poisson_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);
double loglik_cox(Eigen::MatrixXd &X, Eigen::VectorXd &status, Eigen::VectorXd &beta, Eigen::VectorXd &weights);
void cox_fit(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, Eigen::VectorXd &beta, double &coef0, double loss0, bool approximate_Newton, int primary_model_fit_max_iter, double primary_model_fit_epsilon, double tau, double lambda);

template <class T4>
double loglik_poiss(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &coef, int n, Eigen::VectorXd &weights)
{
    int p = x.cols();
    T4 X(n, p + 1);
    X.rightCols(p) = x;
    add_constant_column(X);
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

#endif