//
// Created by jtwok on 2020/3/8.
//
#ifndef SRC_PATH_H
#define SRC_PATH_H

#define TEST

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
using namespace Eigen;
#else

#include <Eigen/Eigen>
#include "List.h"

#endif

#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"
#include "abess.h"

template <class T1, class T2, class T3, class T4>
void sequential_path_cv(Data<T1, T2, T3, T4> &data, Algorithm<T1, T2, T3, T4> *algorithm, Metric<T1, T2, T3, T4> *metric, Eigen::VectorXi &sequence, Eigen::VectorXd &lambda_seq, bool early_stop, int k, Result<T2, T3> &result)
{
#ifdef TEST
    clock_t t0, t1, t2;

#endif
    int p = data.get_p();
    int N = data.g_num;
    int M = data.y.cols();
    Eigen::VectorXi g_index = data.g_index;
    Eigen::VectorXi g_size = data.g_size;
    Eigen::VectorXi status = data.status;
    int sequence_size = sequence.size();
    int lambda_size = lambda_seq.size();
    // int early_stop_s = sequence_size;

    Eigen::VectorXi train_mask, test_mask;
    T1 train_y, test_y;
    Eigen::VectorXd train_weight, test_weight;
    T4 train_x, test_x;
    int train_n = 0, test_n = 0;
#ifdef TEST
    t1 = clock();
#endif
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
#ifdef TEST
    t2 = clock();
    std::cout << "train_x time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
    cout << "path 1" << endl;
#endif
    Eigen::Matrix<T4, -1, -1> train_group_XTX = group_XTX<T4>(train_x, g_index, g_size, train_n, p, N, algorithm->model_type);
#ifdef TEST
    cout << "path 1.5" << endl;
#endif
    algorithm->update_group_XTX(train_group_XTX);
    algorithm->PhiG.resize(0, 0);

#ifdef TEST
    cout << "path 2" << endl;
#endif

    if (algorithm->covariance_update)
    {
        algorithm->covariance_update_flag = Eigen::VectorXi::Zero(p);
        algorithm->XTy = train_x.transpose() * train_y;
        algorithm->XTone = train_x.transpose() * Eigen::MatrixXd::Ones(train_n, M);
    }
#ifdef TEST
    cout << "path 3" << endl;
#endif

    Eigen::Matrix<T2, Dynamic, Dynamic> beta_matrix(sequence_size, lambda_size);
    Eigen::Matrix<T3, Dynamic, Dynamic> coef0_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd train_loss_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd ic_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd test_loss_matrix(sequence_size, lambda_size);
    Eigen::Matrix<VectorXd, Dynamic, Dynamic> bd_matrix(sequence_size, lambda_size);

    T2 beta_init;
    T3 coef0_init;
    coef_set_zero(p, M, beta_init, coef0_init);
    Eigen::VectorXi A_init;
    Eigen::VectorXd bd_init;

    for (int i = 0; i < sequence_size; i++)
    {
#ifdef TEST
        std::cout << "\n sequence= " << sequence(i);
#endif
        for (int j = (1 - pow(-1, i)) * (lambda_size - 1) / 2; j < lambda_size && j >= 0; j = j + pow(-1, i))
        {
#ifdef TEST
            std::cout << " =========j: " << j << ", lambda= " << lambda_seq(j) << ", T: " << sequence(i) << endl;
            t0 = clock();
            t1 = clock();
#endif
            algorithm->update_sparsity_level(sequence(i));
            algorithm->update_lambda_level(lambda_seq(j));
            algorithm->update_beta_init(beta_init);
            algorithm->update_bd_init(bd_init);
            algorithm->update_coef0_init(coef0_init);
            algorithm->update_A_init(A_init, N);

            algorithm->fit(train_x, train_y, train_weight, g_index, g_size, train_n, p, N, status);
#ifdef TEST
            t2 = clock();
            std::cout << "fit time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
            t1 = clock();
#endif

            if (algorithm->warm_start)
            {
                beta_init = algorithm->get_beta();
                coef0_init = algorithm->get_coef0();
                bd_init = algorithm->get_bd();
            }
#ifdef TEST
            t2 = clock();
            std::cout << "warm start time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
            t1 = clock();
#endif

            // evaluate the beta
            if (metric->is_cv)
            {
                test_loss_matrix(i, j) = metric->neg_loglik_loss(test_x, test_y, test_weight, g_index, g_size, test_n, p, N, algorithm);
            }
            else
            {
                ic_matrix(i, j) = metric->ic(train_n, M, N, algorithm);
            }
#ifdef TEST
            t2 = clock();
            std::cout << "ic time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
            t1 = clock();
#endif

            // save for best_model fit
            beta_matrix(i, j) = algorithm->beta;
            coef0_matrix(i, j) = algorithm->coef0;
            train_loss_matrix(i, j) = algorithm->get_train_loss();
            bd_matrix(i, j) = algorithm->bd;

#ifdef TEST
            t2 = clock();
            std::cout << "save time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
            std::cout << "path i= " << i << " j= " << j << " time = " << ((double)(t2 - t0) / CLOCKS_PER_SEC) << endl;
#endif
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
}

// List gs_path(Data &data, Algorithm *algorithm, Metric *metric,
//              int s_min, int s_max, int K_max, double epsilon);

double det(double a[], double b[]);

// calculate the intersection of two lines
// if parallal, need_flag = false.
void line_intersection(double line1[2][2], double line2[2][2], double intersection[], bool &need_flag);

// boundary: s=smin, s=max, lambda=lambda_min, lambda_max
// line: crosses p and is parallal to u
// calculate the intersections between boundary and line
void cal_intersections(double p[], double u[], int s_min, int s_max, double lambda_min, double lambda_max, int a[], int b[]);

// template <class T1, class T2, class T3>
// void golden_section_search(Data<T1, T2, T3> &data, Algorithm<T1, T2, T3> *algorithm, Metric<T1, T2, T3> *metric, double p[], double u[], int s_min, int s_max, double log_lambda_min, double log_lambda_max, double best_arg[],
//                            T2 &beta1, T3 &coef01, double &train_loss1, double &ic1, Eigen::MatrixXd &ic_sequence);

// template <class T1, class T2, class T3>
// void seq_search(Data<T1, T2, T3> &data, Algorithm<T1, T2, T3> *algorithm, Metric<T1, T2, T3> *metric, double p[], double u[], int s_min, int s_max, double log_lambda_min, double log_lambda_max, double best_arg[],
//                 T2 &beta1, T3 &coef01, double &train_loss1, double &ic1, int nlambda, Eigen::MatrixXd &ic_sequence);

// List pgs_path(Data &data, Algorithm *algorithm, Metric *metric, int s_min, int s_max, double log_lambda_min, double log_lambda_max, int powell_path, int nlambda);

#endif //SRC_PATH_H