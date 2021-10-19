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

template <class T1, class T2, class T3, class T4, class T5>
void sequential_path_cv(Data<T1, T2, T3, T4, T5> &data, Eigen::MatrixXd sigma, Algorithm<T1, T2, T3, T4, T5> *algorithm, Metric<T1, T2, T3, T4, T5> *metric, Eigen::VectorXi &sequence, Eigen::VectorXd &lambda_seq, bool early_stop, int k, Result<T2, T3, T5> &result)
{

    int p = data.get_p();
    int N = data.g_num;
    if (algorithm->model_type == 8 || algorithm->model_type == 9){
        N = p*(p+1) / 2;
    }
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

    if (algorithm->model_type == 1 || algorithm->model_type == 5){
        Eigen::Matrix<T4, -1, -1> train_group_XTX = group_XTX<T4>(train_x, g_index, g_size, train_n, p, N, algorithm->model_type);
        algorithm->update_group_XTX(train_group_XTX);
        algorithm->PhiG.resize(0, 0);
    }

    if (algorithm->covariance_update)
    {
        algorithm->covariance = new Eigen::VectorXd *[p];
        algorithm->covariance_update_flag = new bool[p];
        for (int i = 0; i < p; i++)
            algorithm->covariance_update_flag[i] = false;
        algorithm->XTy = train_x.transpose() * train_y;
        algorithm->XTone = train_x.transpose() * Eigen::MatrixXd::Ones(train_n, M);
    }

    Eigen::Matrix<T2, Dynamic, Dynamic> beta_matrix(sequence_size, lambda_size);
    Eigen::Matrix<T3, Dynamic, Dynamic> coef0_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd train_loss_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd ic_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd test_loss_matrix(sequence_size, lambda_size);
    Eigen::Matrix<T5, Dynamic, Dynamic> bd_matrix(sequence_size, lambda_size);
    Eigen::MatrixXd effective_number_matrix(sequence_size, lambda_size);

    //init beta & coef0
    T2 beta_init;
    T3 coef0_init;
    if (algorithm->model_type == 8 || algorithm->model_type == 9){
        coef_set_zero(p*(p+1) / 2, M, beta_init, coef0_init);
    }else{
        coef_set_zero(p, M, beta_init, coef0_init);
    }
    
    Eigen::VectorXi A_init;
    T5 bd_init;

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

            algorithm->fit(train_x, train_y, train_weight, g_index, g_size, train_n, p, N, status, sigma);

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

    if (algorithm->covariance_update)
    {
        for (int i = 0; i < p; i++)
            if (algorithm->covariance_update_flag[i])
                delete algorithm->covariance[i];
        delete[] algorithm->covariance;
        delete[] algorithm->covariance_update_flag;
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

template <class T1, class T2, class T3, class T4, class T5>
void gs_path(Data<T1, T2, T3, T4, T5> &data, Algorithm<T1, T2, T3, T4, T5> *algorithm, vector<Algorithm<T1, T2, T3, T4, T5> *> algorithm_list, Metric<T1, T2, T3, T4, T5> *metric, int s_min, int s_max, Eigen::VectorXi &sequence, Eigen::VectorXd &lambda_seq, int K_max, double epsilon, bool is_parallel, Result<T2, T3, T5> &result)
{
    int p = data.get_p();
    // int n = data.get_n();
    // int i;

    int sequence_size = s_max - s_min + 5;
    sequence = Eigen::VectorXi::Zero(sequence_size);
    double lambda = lambda_seq[0];

    if (algorithm->model_type == 1 || algorithm->model_type == 5){
        Eigen::Matrix<T4, -1, -1> train_group_XTX = group_XTX<T4>(data.x, data.g_index, data.g_size, data.n, p, data.g_num, algorithm->model_type);
        algorithm->update_group_XTX(train_group_XTX);
        algorithm->PhiG.resize(0, 0);
    }

    if (algorithm->covariance_update)
    {
        algorithm->covariance = new Eigen::VectorXd *[data.p];
        algorithm->covariance_update_flag = new bool[data.p];
        for (int i = 0; i < data.p; i++)
            algorithm->covariance_update_flag[i] = false;
        algorithm->XTy = data.x.transpose() * data.y;
        algorithm->XTone = data.x.transpose() * Eigen::MatrixXd::Ones(data.n, data.M);
    }

    if (metric->is_cv)
    {
        for (int k = 0; k < metric->Kfold; k++)
        {
            if (algorithm->model_type == 1 || algorithm->model_type == 5){
                Eigen::Matrix<T4, -1, -1> tmp_group_XTX = group_XTX<T4>(metric->train_X_list[k], data.g_index, data.g_size, metric->train_mask_list[k].size(), data.p, data.g_num, algorithm->model_type);
                algorithm_list[k]->update_group_XTX(tmp_group_XTX);
                algorithm_list[k]->PhiG.resize(0, 0);
            }

            if (algorithm_list[k]->covariance_update)
            {
                algorithm_list[k]->covariance = new Eigen::VectorXd *[data.p];
                algorithm_list[k]->covariance_update_flag = new bool[data.p];
                for (int i = 0; i < data.p; i++)
                    algorithm_list[k]->covariance_update_flag[i] = false;
                algorithm_list[k]->XTy = metric->train_X_list[k].transpose() * metric->train_y_list[k];
                algorithm_list[k]->XTone = metric->train_X_list[k].transpose() * Eigen::MatrixXd::Ones(metric->train_mask_list[k].size(), data.M);
            }
        }
    }

    // Eigen::VectorXi full_mask = Eigen::VectorXi::LinSpaced(n, 0, n - 1);

    // Eigen::Matrix<T2, Dynamic, Dynamic> beta_sequence(sequence_size, lambda_size);
    // Eigen::Matrix<T3, Dynamic, Dynamic> coef0_matrix(sequence_size, lambda_size);
    // Eigen::MatrixXd train_loss_matrix(sequence_size, lambda_size);
    // Eigen::MatrixXd ic_matrix(sequence_size, lambda_size);

    // Eigen::MatrixXd beta_sequence = Eigen::MatrixXd::Zero(p, 4);
    // Eigen::VectorXd coef0_sequence = Eigen::VectorXd::Zero(4);
    // Eigen::VectorXd train_loss_sequence = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd ic_sequence = Eigen::VectorXd::Zero(4);

    // Eigen::MatrixXd beta_all = Eigen::MatrixXd::Zero(p, 100);
    // Eigen::VectorXd coef0_all = Eigen::VectorXd::Zero(100);
    // Eigen::VectorXd train_loss_all = Eigen::VectorXd::Zero(100);
    // Eigen::VectorXd ic_all = Eigen::VectorXd::Zero(100);

    // Eigen::VectorXd beta_init = Eigen::VectorXd::Zero(p);
    // double coef0_init = 0.0;
    Eigen::Matrix<T2, Dynamic, Dynamic> beta_matrix(sequence_size, 1);
    Eigen::Matrix<T3, Dynamic, Dynamic> coef0_matrix(sequence_size, 1);
    Eigen::MatrixXd train_loss_matrix(sequence_size, 1);
    Eigen::MatrixXd ic_matrix(sequence_size, 1);
    Eigen::MatrixXd test_loss_matrix(sequence_size, 1);
    Eigen::Matrix<T5, Dynamic, Dynamic> bd_matrix(sequence_size, 1);
    Eigen::MatrixXd effective_number_matrix(sequence_size, 1);

    T2 beta_init;
    T3 coef0_init;
    if (algorithm->model_type == 8 || algorithm->model_type == 9){
        coef_set_zero(data.p * (data.p + 1) / 2, data.M, beta_init, coef0_init);
    }else{
        coef_set_zero(data.p, data.M, beta_init, coef0_init);
    }
    Eigen::VectorXi A_init;
    T5 bd_init;

    int Tmin = s_min;
    int Tmax = s_max;
    int Tl = round(0.618 * Tmin + 0.382 * Tmax);
    int Tr = round(0.382 * Tmin + 0.618 * Tmax);
    // double icTl;
    // double icTr;

    FIT_ARG<T2, T3, T5> fit_arg(Tl, lambda, beta_init, coef0_init, bd_init, A_init);
    // algorithm->update_train_mask(full_mask);
    // algorithm->update_sparsity_level(T1);
    // algorithm->update_beta_init(beta_init);
    // algorithm->update_coef0_init(coef0_init);
    // algorithm->update_group_XTX(full_group_XTX);

    // algorithm->fit();

    ic_sequence(1) = metric->fit_and_evaluate_in_metric(algorithm, data, algorithm_list, fit_arg);
    sequence(0) = Tl;

    // evaluate the beta
    if (metric->is_cv)
    {
        test_loss_matrix(0, 0) = ic_sequence(1);
    }
    else
    {
        ic_matrix(0, 0) = ic_sequence(1);
    }

    if (algorithm->warm_start)
    {
        beta_init = algorithm->get_beta();
        coef0_init = algorithm->get_coef0();
        bd_init = algorithm->get_bd();
    }

    beta_matrix(0, 0) = algorithm->beta;
    coef0_matrix(0, 0) = algorithm->coef0;
    train_loss_matrix(0, 0) = algorithm->get_train_loss();
    bd_matrix(0, 0) = algorithm->bd;
    effective_number_matrix(0, 0) = algorithm->get_effective_number();

    // beta_matrix.col(1) = algorithm->get_beta();
    // coef0_sequence(1) = algorithm->get_coef0();
    // // train_loss_sequence(1) = metric->train_loss(algorithm, data);
    // // ic_sequence(1) = metric->ic(algorithm, data);
    // beta_all.col(0) = beta_matrix.col(1);
    // coef0_all(0) = coef0_sequence(1);
    // train_loss_all(0) = train_loss_sequence(1);
    // ic_all(0) = ic_sequence(1);
    // icT1 = ic_sequence(1);

    // algorithm->update_train_mask(full_mask);
    // algorithm->update_sparsity_level(T2);
    // algorithm->update_beta_init(beta_init);
    // algorithm->update_coef0_init(coef0_init);
    // algorithm->update_group_XTX(full_group_XTX);
    fit_arg.support_size = Tr;
    fit_arg.beta_init = beta_init;
    fit_arg.coef0_init = coef0_init;
    fit_arg.bd_init = bd_init;

    ic_sequence(2) = metric->fit_and_evaluate_in_metric(algorithm, data, algorithm_list, fit_arg);
    sequence(1) = Tr;

    // evaluate the beta
    if (metric->is_cv)
    {
        test_loss_matrix(1, 0) = ic_sequence(2);
    }
    else
    {
        ic_matrix(1, 0) = ic_sequence(2);
    }

    if (algorithm->warm_start)
    {
        beta_init = algorithm->get_beta();
        coef0_init = algorithm->get_coef0();
        bd_init = algorithm->get_bd();
    }

    beta_matrix(1, 0) = algorithm->beta;
    coef0_matrix(1, 0) = algorithm->coef0;
    train_loss_matrix(1, 0) = algorithm->get_train_loss();
    bd_matrix(1, 0) = algorithm->bd;
    effective_number_matrix(1, 0) = algorithm->get_effective_number();

    // algorithm->fit();
    // if (algorithm->warm_start)
    // {
    //     beta_init = algorithm->get_beta();
    //     coef0_init = algorithm->get_coef0();
    // }

    // beta_matrix.col(2) = algorithm->get_beta();
    // coef0_sequence(2) = algorithm->get_coef0();
    // train_loss_sequence(2) = metric->train_loss(algorithm, data);
    // ic_sequence(2) = metric->ic(algorithm, data);
    // beta_all.col(1) = beta_matrix.col(2);
    // coef0_all(1) = coef0_sequence(2);
    // train_loss_all(1) = train_loss_sequence(2);
    // ic_all(1) = ic_sequence(2);

    // icT2 = metric->ic(algorithm, data);

    int iter = 2;
    while (Tl != Tr)
    {
        if (ic_sequence(1) < ic_sequence(2))
        {
            Tmax = Tr;
            // beta_matrix.col(3) = beta_matrix.col(2);
            // coef0_sequence(3) = coef0_sequence(2);
            // train_loss_sequence(3) = train_loss_sequence(2);
            ic_sequence(3) = ic_sequence(2);

            Tr = Tl;
            // beta_matrix.col(2) = beta_matrix.col(1);
            // coef0_sequence(2) = coef0_sequence(1);
            // train_loss_sequence(2) = train_loss_sequence(1);
            ic_sequence(2) = ic_sequence(1);
            // icT2 = ic_sequence(1);

            Tl = round(0.618 * Tmin + 0.382 * Tmax);

            fit_arg.support_size = Tl;
            fit_arg.beta_init = beta_init;
            fit_arg.coef0_init = coef0_init;
            fit_arg.bd_init = bd_init;
            ic_sequence(1) = metric->fit_and_evaluate_in_metric(algorithm, data, algorithm_list, fit_arg);
            sequence(iter) = Tl;

            // evaluate the beta
            if (metric->is_cv)
            {
                test_loss_matrix(iter, 0) = ic_sequence(1);
            }
            else
            {
                ic_matrix(iter, 0) = ic_sequence(1);
            }

            if (algorithm->warm_start)
            {
                beta_init = algorithm->get_beta();
                coef0_init = algorithm->get_coef0();
                bd_init = algorithm->get_bd();
            }

            beta_matrix(iter, 0) = algorithm->beta;
            coef0_matrix(iter, 0) = algorithm->coef0;
            train_loss_matrix(iter, 0) = algorithm->get_train_loss();
            bd_matrix(iter, 0) = algorithm->bd;
            effective_number_matrix(iter, 0) = algorithm->get_effective_number();
            // algorithm->update_train_mask(full_mask);
            // algorithm->update_sparsity_level(T1);
            // algorithm->update_beta_init(beta_init);
            // algorithm->update_coef0_init(coef0_init);
            // algorithm->update_group_XTX(full_group_XTX);
            // algorithm->fit();
            // if (algorithm->warm_start)
            // {
            //     beta_init = algorithm->get_beta();
            //     coef0_init = algorithm->get_coef0();
            // }
            // beta_matrix.col(1) = algorithm->get_beta();
            // coef0_sequence(1) = algorithm->get_coef0();
            // train_loss_sequence(1) = metric->train_loss(algorithm, data);
            // ic_sequence(1) = metric->ic(algorithm, data);

            // beta_all.col(iter) = beta_matrix.col(1);
            // coef0_all(iter) = coef0_sequence(1);
            // train_loss_all(iter) = train_loss_sequence(1);
            // ic_all(iter) = ic_sequence(1);
            iter++;

            // icT1 = metric->ic(algorithm, data);
        }
        else
        {
            Tmin = Tl;
            // beta_matrix.col(0) = beta_matrix.col(1);
            // coef0_sequence(0) = coef0_sequence(1);
            // train_loss_sequence(0) = train_loss_sequence(1);
            ic_sequence(0) = ic_sequence(1);

            Tl = Tr;
            // beta_matrix.col(1) = beta_matrix.col(2);
            // coef0_sequence(1) = coef0_sequence(2);
            // train_loss_sequence(1) = train_loss_sequence(2);
            ic_sequence(1) = ic_sequence(2);
            // icT1 = ic_sequence(2);

            Tr = round(0.382 * Tmin + 0.618 * Tmax);
            fit_arg.support_size = Tr;
            fit_arg.beta_init = beta_init;
            fit_arg.coef0_init = coef0_init;
            fit_arg.bd_init = bd_init;
            ic_sequence(2) = metric->fit_and_evaluate_in_metric(algorithm, data, algorithm_list, fit_arg);
            sequence(iter) = Tr;

            // evaluate the beta
            if (metric->is_cv)
            {
                test_loss_matrix(iter, 0) = ic_sequence(2);
            }
            else
            {
                ic_matrix(iter, 0) = ic_sequence(2);
            }

            if (algorithm->warm_start)
            {
                beta_init = algorithm->get_beta();
                coef0_init = algorithm->get_coef0();
                bd_init = algorithm->get_bd();
            }

            beta_matrix(iter, 0) = algorithm->beta;
            coef0_matrix(iter, 0) = algorithm->coef0;
            train_loss_matrix(iter, 0) = algorithm->get_train_loss();
            bd_matrix(iter, 0) = algorithm->bd;
            effective_number_matrix(iter, 0) = algorithm->get_effective_number();
            // algorithm->update_train_mask(full_mask);
            // algorithm->update_sparsity_level(T2);
            // algorithm->update_beta_init(beta_init);
            // algorithm->update_coef0_init(coef0_init);
            // algorithm->update_group_XTX(full_group_XTX);
            // algorithm->fit();
            // if (algorithm->warm_start)
            // {
            //     beta_init = algorithm->get_beta();
            //     coef0_init = algorithm->get_coef0();
            // }

            // beta_matrix.col(2) = algorithm->get_beta();
            // coef0_sequence(2) = algorithm->get_coef0();
            // train_loss_sequence(2) = metric->train_loss(algorithm, data);
            // ic_sequence(2) = metric->ic(algorithm, data);

            // beta_all.col(iter) = beta_matrix.col(2);
            // coef0_all(iter) = coef0_sequence(2);
            // train_loss_all(iter) = train_loss_sequence(2);
            // ic_all(iter) = ic_sequence(2);
            iter++;

            // icT2 = metric->ic(algorithm, data);
        };
    }

    T2 best_beta;
    // T3 best_coef0;
    // double best_train_loss = 0;
    double best_ic = DBL_MAX;

    for (int T_tmp = Tmin; T_tmp <= Tmax; T_tmp++)
    {
        fit_arg.support_size = T_tmp;
        fit_arg.beta_init = beta_init;
        fit_arg.coef0_init = coef0_init;
        fit_arg.bd_init = bd_init;
        double ic_tmp = metric->fit_and_evaluate_in_metric(algorithm, data, algorithm_list, fit_arg);

        // algorithm->update_train_mask(full_mask);
        // algorithm->update_sparsity_level(T_tmp);
        // algorithm->update_beta_init(beta_init);
        // algorithm->update_coef0_init(coef0_init);
        // algorithm->update_group_XTX(full_group_XTX);
        // algorithm->fit();
        // if (algorithm->warm_start)
        // {
        //     beta_init = algorithm->get_beta();
        //     coef0_init = algorithm->get_coef0();
        // }
        // double ic_tmp = metric->ic(algorithm, data);
        if (ic_tmp < best_ic)
        {
            // evaluate the beta
            if (metric->is_cv)
            {
                test_loss_matrix(iter, 0) = ic_tmp;
            }
            else
            {
                ic_matrix(iter, 0) = ic_tmp;
            }

            if (algorithm->warm_start)
            {
                beta_init = algorithm->get_beta();
                coef0_init = algorithm->get_coef0();
                bd_init = algorithm->get_bd();
            }

            beta_matrix(iter, 0) = algorithm->beta;
            coef0_matrix(iter, 0) = algorithm->coef0;
            train_loss_matrix(iter, 0) = algorithm->get_train_loss();
            bd_matrix(iter, 0) = algorithm->bd;
            effective_number_matrix(iter, 0) = algorithm->get_effective_number();

            sequence(iter) = T_tmp;
            // best_beta = algorithm->get_beta();
            // best_coef0 = algorithm->get_coef0();
            // best_train_loss = metric->train_loss(algorithm, data);
            // best_ic = ic_tmp;

            // beta_all.col(iter) = best_beta;
            // coef0_all(iter) = best_coef0;
            // train_loss_all(iter) = best_train_loss;
            // ic_all(iter) = best_ic;
            iter++;
        }
    }

    if (algorithm->covariance_update)
    {
        for (int i = 0; i < data.p; i++)
            if (algorithm->covariance_update_flag[i])
                delete algorithm->covariance[i];
        delete[] algorithm->covariance;
        delete[] algorithm->covariance_update_flag;
    }

    if (metric->is_cv)
    {
        for (int k = 0; k < metric->Kfold; k++)
        {
            if (algorithm_list[k]->covariance_update)
            {
                for (int i = 0; i < data.p; i++)
                    if (algorithm_list[k]->covariance_update_flag[i])
                        delete algorithm_list[k]->covariance[i];
                delete[] algorithm_list[k]->covariance;
                delete[] algorithm_list[k]->covariance_update_flag;
            }
        }
    }

    result.beta_matrix = beta_matrix.block(0, 0, iter, 1);
    result.coef0_matrix = coef0_matrix.block(0, 0, iter, 1);
    result.train_loss_matrix = train_loss_matrix.block(0, 0, iter, 1);
    result.bd_matrix = bd_matrix.block(0, 0, iter, 1);
    result.ic_matrix = ic_matrix.block(0, 0, iter, 1);
    result.test_loss_matrix = test_loss_matrix.block(0, 0, iter, 1);
    result.effective_number_matrix = effective_number_matrix.block(0, 0, iter, 1);
    sequence = sequence.head(iter).eval();
}

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