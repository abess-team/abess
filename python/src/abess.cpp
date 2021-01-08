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
//#include "bess.h"
#include "screening.h"
#include <vector>
#include <omp.h>

// #ifdef OTHER_ALGORITHM2
// #include "PrincipalBallAlgorithm.h"
// #endif

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
    // clock_t t1, t2;
    // t1 = clock();
    srand(123);

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

    // bool is_parallel = true;

    /// ### keep
    // if (algorithm_type == 1 || algorithm_type == 5) {
    //     if (model_type == 1) {
    //         data.add_weight();
    //         algorithm = new L0L2Lm(data, algorithm_type, max_iter);
    //     } else if (model_type == 2) {
    //         algorithm = new L0L2Logistic(data, algorithm_type, max_iter);
    //     } else if (model_type == 3) {
    //         algorithm = new L0L2Poisson(data, algorithm_type, max_iter);
    //     } else {
    //         algorithm = new L0L2Cox(data,algorithm_type, max_iter);
    //     }
    // }
    //    else if (algorithm_type == 2 || algorithm_type == 3) {
    //     if (model_type == 1) {
    //         data.add_weight();
    //         algorithm = new GroupPdasLm(data,algorithm_type, max_iter);
    //         // algorithm->PhiG = Phi(data.x, g_index, data.get_g_size(), data.get_n(), data.get_p(), data.get_g_num(), 0.);
    //         // algorithm->invPhiG = invPhi(algorithm->PhiG, data.get_g_num());
    //     } else if (model_type == 2) {
    //         algorithm = new GroupPdasLogistic(data, algorithm_type, max_iter);
    //     } else if (model_type == 3) {
    //         algorithm = new GroupPdasPoisson(data, algorithm_type, max_iter);
    //     } else {
    //         algorithm = new GroupPdasCox(data, algorithm_type, max_iter);
    //     }
    // }

    // if (algorithm_type == 1 || algorithm_type == 5 || algorithm_type == 2 || algorithm_type == 3)
    // {
    //     if (model_type == 1)
    //     {
    //         data.add_weight();
    //         algorithm = new GroupPdasLm(data, algorithm_type, max_iter);
    //     }
    //     else if (model_type == 2)
    //     {
    //         algorithm = new GroupPdasLogistic(data, algorithm_type, max_iter);
    //     }
    //     else if (model_type == 3)
    //     {
    //         algorithm = new GroupPdasPoisson(data, algorithm_type, max_iter);
    //     }
    //     else
    //     {
    //         algorithm = new GroupPdasCox(data, algorithm_type, max_iter);
    //     }
    // }

    //////////////////// function generate_algorithm_pointer() ////////////////////////////

    if (algorithm_type == 6)
    {
        if (model_type == 2)
        {
            algorithm = new abessLogistic(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon);
        }
    }

    algorithm->set_warm_start(is_warm_start);
    algorithm->always_select = always_select;
    // algorithm->tau = tau;
    algorithm->exchange_num = exchange_num;
    algorithm->approximate_Newton = approximate_Newton;

    // #ifdef OTHER_ALGORITHM2
    //     if (algorithm_type == 7)
    //     {
    //         if (model_type == 1)
    //         {
    //             data.add_weight();
    //             algorithm = new PrincipalBallLm(data, max_iter);
    //         }
    //     }
    // #endif

    Metric *metric = new Metric(ic_type, ic_coef, is_cv, Kfold);
    // if (model_type == 1)
    // {
    //     metric = new LmMetric(ic_type, ic_coef, is_cv, Kfold);
    // }
    // else if (model_type == 2)
    // {
    //     metric = new LogisticMetric(ic_type, ic_coef, is_cv, Kfold);
    // }
    // else if (model_type == 3)
    // {
    //     metric = new PoissonMetric(ic_type, ic_coef, is_cv, Kfold);
    // }
    // else
    // {
    //     metric = new CoxMetric(ic_type, ic_coef, is_cv, Kfold);
    // }

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
                    if (model_type == 2)
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
    List result;
    vector<List> result_list(Kfold);
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
                    result_list[i] = sequential_path_cv(data, algorithm_list[i], metric, sequence, lambda_seq, early_stop, i);
                }
            }
            else
            {
                for (int i = 0; i < Kfold; i++)
                {
                    result_list[i] = sequential_path_cv(data, algorithm, metric, sequence, lambda_seq, early_stop, i);
                }
            }
        }
        else
        {
            result = sequential_path_cv(data, algorithm, metric, sequence, lambda_seq, early_stop, -1);
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

    // Get bestmodel index && fit bestmodel
    ////////////////////////////put in abess.cpp///////////////////////////////////////
    // get bestmodel index
    int min_loss_index_row = 0, min_loss_index_col = 0;
    Eigen::Matrix<VectorXd, Dynamic, Dynamic> beta_matrix;
    Eigen::MatrixXd coef0_matrix;
    Eigen::Matrix<VectorXi, Dynamic, Dynamic> A_matrix;
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
                result_list[i].get_value_by_name("test_loss_matrix", test_loss_tmp);
                // cout << "abess 2.1" << endl;
                test_loss_sum = test_loss_sum + test_loss_tmp / Kfold;
                // cout << "abess 2.2" << endl;
            }
            test_loss_sum.minCoeff(&min_loss_index_row, &min_loss_index_col);
            // cout << "abess 3" << endl;
            result_list[0].get_value_by_name("beta_matrix", beta_matrix);
            result_list[0].get_value_by_name("coef0_matrix", coef0_matrix);
            result_list[0].get_value_by_name("A_matrix", A_matrix);
            // cout << "abess 4" << endl;
        }
        else
        {
            result.get_value_by_name("beta_matrix", beta_matrix);
            result.get_value_by_name("coef0_matrix", coef0_matrix);
            // result.get_value_by_name("A_matrix", A_matrix);
            result.get_value_by_name("ic_matrix", ic_matrix);

            ic_matrix.minCoeff(&min_loss_index_row, &min_loss_index_col);
        }
    }

    // cout << "abess end" << endl;

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
        algorithm->update_A_init(A_matrix(min_loss_index_row, min_loss_index_col), data.g_num);
        // algorithm->update_I_init(I_init);
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
        best_ic = test_loss_sum(min_loss_index_row, min_loss_index_col);
    }

    // if (!metric->is_cv)
    // {
    //     best_beta = beta_matrix[min_loss_index_col].col(min_loss_index_row).eval();
    //     best_coef0 = coef0_sequence[min_loss_index_col](min_loss_index_row);
    //     // best_train_loss = loss_sequence[min_loss_index_col](min_loss_index_row);
    //     best_ic = ic_sequence(min_loss_index_row, min_loss_index_col);
    // }
    // else
    // {
    //     // algorithm->update_train_mask(full_mask);
    //     algorithm->update_sparsity_level(best_s);
    //     algorithm->update_lambda_level(best_lambda);
    //     algorithm->update_beta_init(beta_matrix[min_loss_index_col].col(min_loss_index_row).eval());
    //     algorithm->update_coef0_init(coef0_sequence[min_loss_index_col](min_loss_index_row));
    //     algorithm->update_A_init(A_sequence[min_loss_index_row][min_loss_index_col], N);
    //     // algorithm->update_I_init(I_init);
    //     algorithm->update_group_XTX(full_group_XTX);

    //     algorithm->fit();

    //     best_beta = algorithm->get_beta();
    //     best_coef0 = algorithm->get_coef0();
    //     // best_train_loss = metric->train_loss(algorithm, data);
    //     best_ic = ic_sequence(min_loss_index_row, min_loss_index_col);
    // }

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
    // cout << "abess 7" << endl;

    // List result;
    result.add("beta", best_beta);
    result.add("coef0", best_coef0);
    result.add("train_loss", best_train_loss);
    result.add("ic", best_ic);
    result.add("lambda", best_lambda);

    // Restore best_fit_result for screening
    if (is_screening)
    {
        Eigen::VectorXd beta_screening_A;
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);

#ifndef R_BUILD
        result.get_value_by_name("beta", beta_screening_A);
        for (unsigned int i = 0; i < screening_A.size(); i++)
        {
            beta(screening_A(i)) = beta_screening_A(i);
        }
        result.add("beta", beta);
        result.add("screening_A", screening_A);
#else
        beta_screening_A = result["beta"];
        for (int i = 0; i < screening_A.size(); i++)
        {
            beta(screening_A(i)) = beta_screening_A(i);
        }
        result["beta"] = beta;
        result.push_back(screening_A, "screening_A");
        cout << "screening AA";
#endif
    }

    delete algorithm;
    delete metric;
    cout << "abess 8" << endl;
    return result;
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

    // clock_t t1, t2;
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

    // t1 = clock();
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
    // t2 = clock();
    // std::cout << "get result : " << ((double)(t2 - t1) / CLOCKS_PER_SEC) << endl;

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
