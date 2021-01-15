// #define R_BUILD
#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace std;
#else

#include <Eigen/Eigen>
#include "List.h"

#endif

#include <iostream>
#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"
#include "path.h"
#include "utilities.h"
#include "abess.h"
#include "screening.h"
#include <vector>
#include <omp.h>

using namespace Eigen;
using namespace std;

// [[Rcpp::export]]
List abessCpp(Eigen::MatrixXd x, Eigen::VectorXd y, int data_type, Eigen::VectorXd weight,
              bool is_normal,
              int algorithm_type, int model_type, int max_iter, int exchange_num,
              int path_type, bool is_warm_start,
              int ic_type, double ic_coef, bool is_cv, int Kfold,
              Eigen::VectorXd state,
              Eigen::VectorXi sequence,
              Eigen::VectorXd lambda_seq,
              int s_min, int s_max, int K_max, double epsilon,
              double lambda_min, double lambda_max, int nlambda,
              bool is_screening, int screening_size, int powell_path,
              Eigen::VectorXi g_index,
              Eigen::VectorXi always_select,
              double tau,
              int primary_model_fit_max_iter, double primary_model_fit_epsilon,
              bool early_stop, bool approximate_Newton,
              int thread)
{
    // to do: -openmp
    clock_t t1, t2;
    // t1 = clock();
    std::srand(123);

    bool is_parallel = thread != 1;

    // Eigen::initParallel();
    Eigen::setNbThreads(thread);
    cout << Eigen::nbThreads() << " Threads for eigen." << endl;
    cout << omp_get_thread_num() << " Threads for omp." << endl;
    int p = x.cols();
    Eigen::VectorXi screening_A;
    if (is_screening)
    {
        screening_A = screening(x, y, weight, model_type, screening_size, g_index, always_select);
    }
    Data data(x, y, data_type, weight, is_normal, g_index);

    Algorithm *algorithm = nullptr;

    //////////////////// function generate_algorithm_pointer() ////////////////////////////

    if (algorithm_type == 6)
    {
        if (model_type == 1)
        {
            algorithm = new abessLm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon);
        }
        else if (model_type == 2)
        {
            algorithm = new abessLogistic(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon);
        }
    }

    algorithm->set_warm_start(is_warm_start);
    algorithm->always_select = always_select;
    // algorithm->tau = tau;
    algorithm->exchange_num = exchange_num;
    algorithm->approximate_Newton = approximate_Newton;

    Metric *metric = new Metric(ic_type, ic_coef, is_cv, Kfold);

    // For CV:
    // 1:mask
    // 2:warm start save
    // 3:group_XTX
    vector<Algorithm *> algorithm_list(Kfold);
    if (is_cv)
    {
        metric->set_cv_train_test_mask(data.get_n());

        metric->set_cv_initial_model_param(Kfold, data.get_p());
        metric->set_cv_initial_A(Kfold, data.get_p());
        // metric->set_cv_initial_I(Kfold, data.get_p());
        metric->set_cv_initial_coef0(Kfold, data.get_p());

        if (model_type == 1)
            metric->cal_cv_group_XTX(data);

        if (is_parallel)
        {
            for (int i = 0; i < Kfold; i++)
            {
                if (algorithm_type == 6)
                {
                    if (model_type == 1)
                    {
                        algorithm_list[i] = new abessLm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon);
                    }
                    else if (model_type == 2)
                    {
                        algorithm_list[i] = new abessLogistic(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon);
                    }
                }
            }
        }
    }
    // t2 = clock();
    // std::cout << "preprocess time: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;

    // calculate loss for each parameter parameter combination
    t1 = clock();
    cout << "build Result" << endl;
    Result result;
    vector<Result> result_list(Kfold);
    cout << "build Result" << endl;
    if (path_type == 1)
    {
        if (is_cv)
        {
            //////////////////////////////////can parallel///////////////////////////////////
            if (is_parallel)
            {
#pragma omp parallel for
                for (int i = 0; i < Kfold; i++)
                {
                    sequential_path_cv(data, algorithm_list[i], metric, sequence, lambda_seq, early_stop, i, result_list[i]);
                }
            }
            else
            {
                for (int i = 0; i < Kfold; i++)
                {
                    sequential_path_cv(data, algorithm, metric, sequence, lambda_seq, early_stop, i, result_list[i]);
                }
            }
        }
        else
        {
            sequential_path_cv(data, algorithm, metric, sequence, lambda_seq, early_stop, -1, result);
        }
    }
    // else
    // {
    //     if (algorithm_type == 5 || algorithm_type == 3)
    //     {
    //         double log_lambda_min = log(max(lambda_min, 1e-5));
    //         double log_lambda_max = log(max(lambda_max, 1e-5));

    //         result = pgs_path(data, algorithm, metric, s_min, s_max, log_lambda_min, log_lambda_max, powell_path, nlambda);
    //     }
    //     else
    //     {
    //         result = gs_path(data, algorithm, metric, s_min, s_max, K_max, epsilon);
    //     }
    // }
    t2 = clock();
    std::cout << "path time : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;

    // Get bestmodel index && fit bestmodel
    ////////////////////////////put in abess.cpp///////////////////////////////////////
    // get bestmodel index
    int min_loss_index_row = 0, min_loss_index_col = 0;
    Eigen::Matrix<VectorXd, Dynamic, Dynamic> beta_matrix;
    Eigen::MatrixXd coef0_matrix;
    Eigen::Matrix<VectorXi, Dynamic, Dynamic> A_matrix;
    Eigen::Matrix<VectorXd, Dynamic, Dynamic> bd_matrix;
    Eigen::MatrixXd ic_matrix;
    Eigen::MatrixXd test_loss_sum = Eigen::MatrixXd::Zero(sequence.size(), lambda_seq.size());

    if (path_type == 1)
    {
        if (is_cv)
        {
            // cout << "abess 1" << endl;
            Eigen::MatrixXd test_loss_tmp;
            for (int i = 0; i < Kfold; i++)
            {
                // cout << "abess 2" << endl;
                test_loss_tmp = result_list[i].test_loss_matrix;
                // cout << "abess 2.1" << endl;
                test_loss_sum = test_loss_sum + test_loss_tmp / Kfold;
                // cout << "abess 2.2" << endl;
            }
            test_loss_sum.minCoeff(&min_loss_index_row, &min_loss_index_col);
            // cout << "abess 3" << endl;
            beta_matrix = result_list[0].beta_matrix;
            coef0_matrix = result_list[0].coef0_matrix;
            // A_matrix = result_list[0].A_matrix;
            bd_matrix = result_list[0].bd_matrix;
            // cout << "abess 4" << endl;
        }
        else
        {
            beta_matrix = result.beta_matrix;
            coef0_matrix = result.coef0_matrix;
            // A_matrix = result.A_matrix;
            // bd_matrix = result.bd_matrix;
            ic_matrix = result.ic_matrix;

            ic_matrix.minCoeff(&min_loss_index_row, &min_loss_index_col);
        }
    }

    // fit best model
    int best_s = sequence(min_loss_index_row);
    double best_lambda = lambda_seq(min_loss_index_col);

    Eigen::VectorXd best_beta;
    double best_coef0;
    double best_train_loss;
    double best_ic;

    if (is_cv)
    {
        vector<Eigen::MatrixXd> full_group_XTX = group_XTX(data.x, data.g_index, data.g_size, data.n, data.p, data.g_num, algorithm->model_type);

        algorithm->update_sparsity_level(best_s);
        algorithm->update_lambda_level(best_lambda);
        algorithm->update_beta_init(beta_matrix(min_loss_index_row, min_loss_index_col));
        algorithm->update_coef0_init(coef0_matrix(min_loss_index_row, min_loss_index_col));
        // algorithm->update_A_init(A_matrix(min_loss_index_row, min_loss_index_col), data.g_num);
        algorithm->update_bd_init(bd_matrix(min_loss_index_row, min_loss_index_col));
        algorithm->update_group_XTX(full_group_XTX);
        // cout << "abess 5" << endl;

        algorithm->fit(data.x, data.y, data.weight, data.g_index, data.g_size, data.n, data.p, data.g_num);

        best_beta = algorithm->get_beta();
        best_coef0 = algorithm->get_coef0();
        best_train_loss = metric->neg_loglik_loss(data.x, data.y, data.weight, data.g_index, data.g_size, data.n, data.p, data.g_num, algorithm);
        best_ic = test_loss_sum(min_loss_index_row, min_loss_index_col);
        // cout << "abess 6" << endl;
    }
    else
    {
        best_beta = beta_matrix(min_loss_index_row, min_loss_index_col);
        best_coef0 = coef0_matrix(min_loss_index_row, min_loss_index_col);
        best_train_loss = 0; ///////// to do ////////
        best_ic = ic_matrix(min_loss_index_row, min_loss_index_col);
    }

    //////////////Restore best_fit_result for normal//////////////
    if (data.is_normal)
    {
        if (algorithm->model_type == 1)
        {
            best_beta = sqrt(double(data.n)) * best_beta.cwiseQuotient(data.x_norm);
            best_coef0 = data.y_mean - best_beta.dot(data.x_mean);
        }
        else if (data.data_type == 2)
        {
            best_beta = sqrt(double(data.n)) * best_beta.cwiseQuotient(data.x_norm);
            best_coef0 = best_coef0 - best_beta.dot(data.x_mean);
        }
        else
        {
            best_beta = sqrt(double(data.n)) * best_beta.cwiseQuotient(data.x_norm);
        }
    }

    //////////////Restore all_fit_result for normal////////////////////////
    // if (data.is_normal)
    // {
    //     if (algorithm->model_type == 1)
    //     {
    //         for (j = 0; j < lambda_size; j++)
    //         {
    //             for (i = 0; i < early_stop_s; i++)
    //             {
    //                 beta_matrix[j].col(i) = sqrt(double(n)) * beta_matrix[j].col(i).cwiseQuotient(data.x_norm);
    //                 coef0_sequence[j](i) = data.y_mean - beta_matrix[j].col(i).dot(data.x_mean);
    //             }
    //         }
    //     }
    //     else if (data.data_type == 2)
    //     {
    //         for (j = 0; j < lambda_size; j++)
    //         {
    //             for (i = 0; i < early_stop_s; i++)
    //             {
    //                 beta_matrix[j].col(i) = sqrt(double(n)) * beta_matrix[j].col(i).cwiseQuotient(data.x_norm);
    //                 coef0_sequence[j](i) = coef0_sequence[j](i) - beta_matrix[j].col(i).dot(data.x_mean);
    //             }
    //         }
    //     }
    //     else
    //     {
    //         for (j = 0; j < lambda_size; j++)
    //         {
    //             for (i = 0; i < early_stop_s; i++)
    //             {
    //                 beta_matrix[j].col(i) = sqrt(double(n)) * beta_matrix[j].col(i).cwiseQuotient(data.x_norm);
    //             }
    //         }
    //     }
    // }

    //////////////////////////// save result with List//////////////////////////////////////
    List out_result;
#ifdef R_BUILD
    out_result = List::create(Named("beta") = best_beta,
                              Named("coef0") = best_coef0,
                              Named("train_loss") = best_train_loss,
                              Named("ic") = best_ic,
                              Named("lambda") = best_lambda);
    //   Named("beta_all") = beta_matrix,
    //   Named("coef0_all") = coef0_sequence,
    //   Named("train_loss_all") = loss_sequence,
    //   Named("ic_all") = ic_sequence);
#else
    out_result.add("beta", best_beta);
    out_result.add("coef0", best_coef0);
    out_result.add("train_loss", best_train_loss);
    out_result.add("ic", best_ic);
    out_result.add("lambda", best_lambda);
#endif

    // Restore best_fit_result for screening
    if (is_screening)
    {
        Eigen::VectorXd beta_screening_A;
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);

#ifndef R_BUILD
        out_result.get_value_by_name("beta", beta_screening_A);
        for (unsigned int i = 0; i < screening_A.size(); i++)
        {
            beta(screening_A(i)) = beta_screening_A(i);
        }
        out_result.add("beta", beta);
        out_result.add("screening_A", screening_A);
#else
        beta_screening_A = out_result["beta"];
        for (int i = 0; i < screening_A.size(); i++)
        {
            beta(screening_A(i)) = beta_screening_A(i);
        }
        out_result["beta"] = beta;
        out_result.push_back(screening_A, "screening_A");
        cout << "screening AA";
#endif
    }

    delete algorithm;
    delete metric;
    // cout << "abess 8" << endl;
    return out_result;
}

#ifndef R_BUILD

void pywrap_abess(double *x, int x_row, int x_col, double *y, int y_len, int data_type, double *weight, int weight_len,
                  bool is_normal,
                  int algorithm_type, int model_type, int max_iter, int exchange_num,
                  int path_type, bool is_warm_start,
                  int ic_type, double ic_coef, bool is_cv, int Kfold,
                  int *gindex, int gindex_len,
                  double *state, int state_len,
                  int *sequence, int sequence_len,
                  double *lambda_sequence, int lambda_sequence_len,
                  int s_min, int s_max, int K_max, double epsilon,
                  double lambda_min, double lambda_max, int n_lambda,
                  bool is_screening, int screening_size, int powell_path,
                  int *always_select, int always_select_len, double tau,
                  int primary_model_fit_max_iter, double primary_model_fit_epsilon,
                  bool early_stop, bool approximate_Newton,
                  int thread,
                  double *beta_out, int beta_out_len, double *coef0_out, int coef0_out_len, double *train_loss_out,
                  int train_loss_out_len, double *ic_out, int ic_out_len, double *nullloss_out, double *aic_out,
                  int aic_out_len, double *bic_out, int bic_out_len, double *gic_out, int gic_out_len, int *A_out,
                  int A_out_len, int *l_out)
{
    Eigen::MatrixXd x_Mat;
    Eigen::VectorXd y_Vec;
    Eigen::VectorXd weight_Vec;
    Eigen::VectorXi gindex_Vec;
    Eigen::VectorXd state_Vec;
    Eigen::VectorXi sequence_Vec;
    Eigen::VectorXd lambda_sequence_Vec;
    Eigen::VectorXi always_select_Vec;

    clock_t t1, t2;
    // t1 = clock();
    x_Mat = Pointer2MatrixXd(x, x_row, x_col);
    y_Vec = Pointer2VectorXd(y, y_len);
    weight_Vec = Pointer2VectorXd(weight, weight_len);
    state_Vec = Pointer2VectorXd(state, state_len);
    gindex_Vec = Pointer2VectorXi(gindex, gindex_len);
    sequence_Vec = Pointer2VectorXi(sequence, sequence_len);
    lambda_sequence_Vec = Pointer2VectorXd(lambda_sequence, lambda_sequence_len);
    always_select_Vec = Pointer2VectorXi(always_select, always_select_len);
    // t2 = clock();
    // std::cout << "pointer to data: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;

    t1 = clock();
    List mylist = abessCpp(x_Mat, y_Vec, data_type, weight_Vec,
                           is_normal,
                           algorithm_type, model_type, max_iter, exchange_num,
                           path_type, is_warm_start,
                           ic_type, ic_coef, is_cv, Kfold,
                           state_Vec,
                           sequence_Vec,
                           lambda_sequence_Vec,
                           s_min, s_max, K_max, epsilon,
                           lambda_min, lambda_max, n_lambda,
                           is_screening, screening_size, powell_path,
                           gindex_Vec,
                           always_select_Vec, tau,
                           primary_model_fit_max_iter, primary_model_fit_epsilon,
                           early_stop, approximate_Newton,
                           thread);
    t2 = clock();
    std::cout << "get result : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;

    // t1 = clock();
    Eigen::VectorXd beta;
    double coef0 = 0;
    double train_loss = 0;
    double ic = 0;
    mylist.get_value_by_name("beta", beta);
    mylist.get_value_by_name("coef0", coef0);
    mylist.get_value_by_name("train_loss", train_loss);
    mylist.get_value_by_name("ic", ic);

    VectorXd2Pointer(beta, beta_out);
    *coef0_out = coef0;
    *train_loss_out = train_loss;
    *ic_out = ic;
    // t2 = clock();
    // std::cout << "result to pointer: " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;
}
#endif
