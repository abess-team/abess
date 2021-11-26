//
// Created by Jin Zhu on 2020/3/8.
//
#ifndef SRC_PATH_H
#define SRC_PATH_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]s]
using namespace Eigen;
#else

#include <Eigen/Eigen>
#include "List.h"

#endif

#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"
#include "abess.h"
#include "utilities.h"

template <class T1, class T2, class T3, class T4>
void sequential_path_cv(Data<T1, T2, T3, T4> &data, Algorithm<T1, T2, T3, T4> *algorithm, Metric<T1, T2, T3, T4> *metric, Eigen::VectorXi &sequence, Eigen::VectorXd &lambda_seq, bool early_stop, int k, Result<T2, T3> &result)
{
    int beta_size = data.beta_size;
    int p = data.p;
    int N = data.g_num;
    int M = data.M;
    Eigen::VectorXi g_index = data.g_index;
    Eigen::VectorXi g_size = data.g_size;
    int sequence_size = sequence.size();
    int lambda_size = lambda_seq.size();
    // int early_stop_s = sequence_size;

    Eigen::VectorXi train_mask, test_mask;
    T1 train_y, test_y;
    Eigen::VectorXd train_weight, test_weight;
    T4 train_x, test_x;
    int train_n = 0, test_n = 0;

    // train & test data
    if (!metric->is_cv)
    {
        train_x = data.x;
        train_y = data.y;
        train_weight = data.weight;
        train_n = data.n;
    }
    else
    {
        train_mask = metric->train_mask_list[k];
        test_mask = metric->test_mask_list[k];
        slice(data.x, train_mask, train_x);
        slice(data.x, test_mask, test_x);
        slice(data.y, train_mask, train_y);
        slice(data.y, test_mask, test_y);
        slice(data.weight, train_mask, train_weight);
        slice(data.weight, test_mask, test_weight);

        train_n = train_mask.size();
        test_n = test_mask.size();
    }

    Eigen::Matrix<T2, Dynamic, Dynamic> beta_matrix(sequence_size, lambda_size);
    Eigen::Matrix<T3, Dynamic, Dynamic> coef0_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd train_loss_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd ic_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd test_loss_matrix(sequence_size, lambda_size);
    Eigen::Matrix<VectorXd, Dynamic, Dynamic> bd_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd effective_number_matrix(sequence_size, lambda_size);

    T2 beta_init;
    T3 coef0_init;
    coef_set_zero(beta_size, M, beta_init, coef0_init);
    Eigen::VectorXi A_init;
    Eigen::VectorXd bd_init;

    for (int i = 0; i < sequence_size; i++)
    {
        for (int j = (1 - pow(-1, i)) * (lambda_size - 1) / 2; j < lambda_size && j >= 0; j = j + pow(-1, i))
        {
            algorithm->update_sparsity_level(sequence(i));
            algorithm->update_lambda_level(lambda_seq(j));
            algorithm->update_beta_init(beta_init);
            algorithm->update_bd_init(bd_init);
            algorithm->update_coef0_init(coef0_init);
            algorithm->update_A_init(A_init, N);

            algorithm->fit(train_x, train_y, train_weight, g_index, g_size, train_n, p, N);

            if (algorithm->warm_start)
            {
                beta_init = algorithm->get_beta();
                coef0_init = algorithm->get_coef0();
                bd_init = algorithm->get_bd();
            }

            // evaluate the beta
            if (metric->is_cv)
            {
                test_loss_matrix(i, j) = metric->neg_loglik_loss(test_x, test_y, test_weight, g_index, g_size, test_n, p, N, algorithm);
            }
            else
            {
                ic_matrix(i, j) = metric->ic(train_n, M, N, algorithm);
            }

            // save for best_model fit
            beta_matrix(i, j) = algorithm->beta;
            coef0_matrix(i, j) = algorithm->coef0;
            train_loss_matrix(i, j) = algorithm->get_train_loss();
            bd_matrix(i, j) = algorithm->bd;
            effective_number_matrix(i, j) = algorithm->get_effective_number();
        }

        // To be ensured
        // if (early_stop && lambda_size <= 1 && i >= 3)
        // {
        //     bool condition1 = ic_sequence(i, 0) > ic_sequence(i - 1, 0);
        //     bool condition2 = ic_sequence(i - 1, 0) > ic_sequence(i - 2, 0);
        //     bool condition3 = ic_sequence(i - 2, 0) > ic_sequence(i - 3, 0);
        //     if (condition1 && condition2 && condition3)
        //     {
        //         early_stop_s = i + 1;
        //         break;
        //     }
        // }
    }

    // if (early_stop)
    // {
    //     ic_sequence = ic_sequence.block(0, 0, early_stop_s, lambda_size).eval();
    // }

    result.beta_matrix = beta_matrix;
    result.coef0_matrix = coef0_matrix;
    result.train_loss_matrix = train_loss_matrix;
    result.bd_matrix = bd_matrix;
    result.ic_matrix = ic_matrix;
    result.test_loss_matrix = test_loss_matrix;
    result.effective_number_matrix = effective_number_matrix;
}

template <class T1, class T2, class T3, class T4>
void gs_path(Data<T1, T2, T3, T4> &data, vector<Algorithm<T1, T2, T3, T4> *> algorithm_list, Metric<T1, T2, T3, T4> *metric, int s_min, int s_max, Eigen::VectorXi &sequence, Eigen::VectorXd &lambda_seq, Result<T2, T3> &result)
{
    int sequence_size = s_max - s_min + 5;
    sequence = Eigen::VectorXi::Zero(sequence_size);
    double lambda = lambda_seq[0];

    // golden search 
    Eigen::Matrix<T2, Dynamic, Dynamic> beta_matrix(sequence_size, 1);
    Eigen::Matrix<T3, Dynamic, Dynamic> coef0_matrix(sequence_size, 1);
    Eigen::MatrixXd train_loss_matrix(sequence_size, 1);
    Eigen::MatrixXd ic_matrix(sequence_size, 1);
    Eigen::MatrixXd test_loss_matrix(sequence_size, 1);
    Eigen::Matrix<VectorXd, Dynamic, Dynamic> bd_matrix(sequence_size, 1);
    Eigen::MatrixXd effective_number_matrix(sequence_size, 1);

    T2 beta_init;
    T3 coef0_init;
    coef_set_zero(data.beta_size, data.M, beta_init, coef0_init);
    Eigen::VectorXi A_init;
    Eigen::VectorXd bd_init;
    FIT_ARG<T2, T3> fit_arg(0, 0, beta_init, coef0_init, bd_init, A_init);

    int ind = -1;
    int left = round(0.618 * s_min + 0.382 * s_max);
    int right = round(0.382 * s_min + 0.618 * s_max);
    bool fit_l = true, fit_r = (left != right);
    double loss_l = 0, loss_r = 0;
    while (true){
        // cout<<" ==> gs: "<<s_min<<" - "<<s_max<<endl;
        if (fit_l)
        {
            fit_l = false;
            fit_arg.support_size = left;
            loss_l = metric->fit_and_evaluate_in_metric(algorithm_list, data, fit_arg);

            // record: left
            sequence(++ind) = left;
            if (metric->is_cv)
            {
                test_loss_matrix(ind, 0) = loss_l;
            }
            else
            {
                beta_matrix(ind, 0) = algorithm_list[0]->beta;
                coef0_matrix(ind, 0) = algorithm_list[0]->coef0;
                train_loss_matrix(ind, 0) = algorithm_list[0]->get_train_loss();
                bd_matrix(ind, 0) = algorithm_list[0]->bd;
                effective_number_matrix(ind, 0) = algorithm_list[0]->get_effective_number();
                ic_matrix(ind, 0) = loss_l;
            }
        }

        if (fit_r)
        {
            fit_r = false;
            fit_arg.support_size = right;
            loss_r = metric->fit_and_evaluate_in_metric(algorithm_list, data, fit_arg);

            // record: pos 2
            sequence(++ind) = right;
            if (metric->is_cv)
            {
                test_loss_matrix(ind, 0) = loss_r;
            }
            else
            {
                beta_matrix(ind, 0) = algorithm_list[0]->beta;
                coef0_matrix(ind, 0) = algorithm_list[0]->coef0;
                train_loss_matrix(ind, 0) = algorithm_list[0]->get_train_loss();
                bd_matrix(ind, 0) = algorithm_list[0]->bd;
                effective_number_matrix(ind, 0) = algorithm_list[0]->get_effective_number();
                ic_matrix(ind, 0) = loss_r;
            }
        }

        // update split point
        if (loss_l < loss_r)
        {
            s_max = right;
            right = left;
            loss_r = loss_l;
            left = round(0.618 * s_min + 0.382 * s_max);
            fit_l = true;
        } 
        else 
        {
            s_min = left;
            left = right;
            loss_l = loss_r;
            right = round(0.382 * s_min + 0.618 * s_max);
            fit_r = true;
        }
        if (left == right) break;
    }
    // cout<<"left==right | s_min = "<<s_min<<" | s_max = "<<s_max<<endl;
    
    T2 best_beta;
    // T3 best_coef0;
    // double best_train_loss = 0;
    double best_loss = DBL_MAX;
    for (int s = s_min; s <= s_max; s++)
    {
        fit_arg.support_size = s;
        fit_arg.beta_init = beta_init;
        fit_arg.coef0_init = coef0_init;
        fit_arg.bd_init = bd_init;
        double loss = metric->fit_and_evaluate_in_metric(algorithm_list, data, fit_arg);

        if (loss < best_loss)
        {
            // record
            sequence(++ind) = s;
            if (metric->is_cv)
            {
                test_loss_matrix(ind, 0) = loss;
                best_loss = loss;
            }
            else
            {
                beta_matrix(ind, 0) = algorithm_list[0]->beta;
                coef0_matrix(ind, 0) = algorithm_list[0]->coef0;
                train_loss_matrix(ind, 0) = algorithm_list[0]->get_train_loss();
                bd_matrix(ind, 0) = algorithm_list[0]->bd;
                effective_number_matrix(ind, 0) = algorithm_list[0]->get_effective_number();
                ic_matrix(ind, 0) = loss;
                best_loss = loss;
            }
        }
    }

    ind++;
    result.beta_matrix = beta_matrix.block(0, 0, ind, 1);
    result.coef0_matrix = coef0_matrix.block(0, 0, ind, 1);
    result.train_loss_matrix = train_loss_matrix.block(0, 0, ind, 1);
    result.bd_matrix = bd_matrix.block(0, 0, ind, 1);
    result.ic_matrix = ic_matrix.block(0, 0, ind, 1);
    result.test_loss_matrix = test_loss_matrix.block(0, 0, ind, 1);
    result.effective_number_matrix = effective_number_matrix.block(0, 0, ind, 1);
    sequence = sequence.head(ind).eval();
    lambda_seq = lambda_seq.head(1).eval();
}

// double det(double a[], double b[]);

// calculate the intersection of two lines
// if parallal, need_flag = false.
// void line_intersection(double line1[2][2], double line2[2][2], double intersection[], bool &need_flag);

// boundary: s=smin, s=max, lambda=lambda_min, lambda_max
// line: crosses p and is parallal to u
// calculate the intersections between boundary and line
// void cal_intersections(double p[], double u[], int s_min, int s_max, double lambda_min, double lambda_max, int a[], int b[]);

// template <class T1, class T2, class T3>
// void golden_section_search(Data<T1, T2, T3> &data, Algorithm<T1, T2, T3> *algorithm, Metric<T1, T2, T3> *metric, double p[], double u[], int s_min, int s_max, double log_lambda_min, double log_lambda_max, double best_arg[],
//                            T2 &beta1, T3 &coef01, double &train_loss1, double &ic1, Eigen::MatrixXd &ic_sequence);

// template <class T1, class T2, class T3>
// void seq_search(Data<T1, T2, T3> &data, Algorithm<T1, T2, T3> *algorithm, Metric<T1, T2, T3> *metric, double p[], double u[], int s_min, int s_max, double log_lambda_min, double log_lambda_max, double best_arg[],
//                 T2 &beta1, T3 &coef01, double &train_loss1, double &ic1, int nlambda, Eigen::MatrixXd &ic_sequence);

// List pgs_path(Data &data, Algorithm *algorithm, Metric *metric, int s_min, int s_max, double log_lambda_min, double log_lambda_max, int powell_path, int nlambda);

#endif //SRC_PATH_H