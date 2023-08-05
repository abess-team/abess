/**
 * @file workflow.h
 * @brief The main workflow for abess.
 * @details It receives all inputs from API, runs the whole abess process
 * and then return the results as a list.
 */

#ifndef SRC_WORKFLOW_H
#define SRC_WORKFLOW_H

// #define R_BUILD
#ifdef R_BUILD

#include <Rcpp.h>
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;

#else
#include <Eigen/Eigen>

#include "List.h"

#endif

#include <iostream>
#include <vector>

#include "Algorithm.h"
#include "Data.h"
#include "Metric.h"
#include "abessOpenMP.h"
#include "path.h"
#include "screening.h"
#include "utilities.h"

typedef Eigen::Triplet<double> triplet;

using namespace Eigen;
using namespace std;

// <Eigen::VectorXd, Eigen::VectorXd, double, Eigen::MatrixXd>                          for Univariate Dense
// <Eigen::VectorXd, Eigen::VectorXd, double, Eigen::SparseMatrix<double> >             for Univariate Sparse
// <Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>                 for Multivariable Dense
// <Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::SparseMatrix<double> >    for Multivariable Sparse

/**
 * @brief The main workflow for abess.
 * @tparam T1 for y, XTy, XTone
 * @tparam T2 for beta
 * @tparam T3 for coef0
 * @tparam T4 for X
 * @param x sample matrix
 * @param y response matrix
 * @param n sample size
 * @param p number of variables
 * @param normalize_type type of normalize
 * @param weight weight of each sample
 * @param algorithm_type type of algorithm
 * @param path_type type of path: 1 for sequencial search and 2 for golden section search
 * @param is_warm_start whether enable warm-start
 * @param eval_type type of information criterion, or test loss in CV
 * @param Kfold number of folds, used for CV
 * @param parameters parameters to be selected, including `support_size`, `lambda`
 * @param screening_size size of screening
 * @param g_index the first position of each group
 * @param early_stop whether enable early-stop
 * @param thread number of threads used for parallel computing
 * @param sparse_matrix whether sample matrix `x` is sparse matrix
 * @param cv_fold_id user-specified cross validation division
 * @param A_init initial active set
 * @param algorithm_list the algorithm pointer
 * @return the result of abess, including the best model parameters
 */
template <class T1, class T2, class T3, class T4>
List abessWorkflow(T4 &x, T1 &y, int n, int p, int normalize_type, Eigen::VectorXd weight, int algorithm_type,
                   int path_type, bool is_warm_start, int eval_type, double ic_coef, int Kfold, Parameters parameters,
                   int screening_size, Eigen::VectorXi g_index, bool early_stop, int thread, bool sparse_matrix,
                   Eigen::VectorXi &cv_fold_id, Eigen::VectorXi &A_init, double beta_low, double beta_high,
                   vector<Algorithm<T1, T2, T3, T4> *> algorithm_list) {
#ifndef R_BUILD
    std::srand(123);
#endif

    int algorithm_list_size = algorithm_list.size();
    for (int i = 0; i < algorithm_list_size; i++) {
        algorithm_list[i]->update_beta_range(beta_low, beta_high);
    }

    // Size of the candidate set:
    //     usually it is equal to `p`, the number of variable,
    //     but it could be different in e.g. RPCA.
    int beta_size = algorithm_list[0]->get_beta_size(n, p);

    // Data packing & normalize:
    //     pack & initial all information of data,
    //     including normalize.
    Data<T1, T2, T3, T4> data(x, y, normalize_type, weight, g_index, sparse_matrix, beta_size);
    if (algorithm_list[0]->model_type == 1 || algorithm_list[0]->model_type == 5) {
        add_weight(data.x, data.y, data.weight);
    }

    // Screening:
    //     if there are too many noise variables,
    //     screening can choose the `screening_size` most important variables
    //     and then focus on them later.
    Eigen::VectorXi screening_A;
    if (screening_size >= 0) {
        screening_A = screening<T1, T2, T3, T4>(data, algorithm_list, screening_size, beta_size,
                                                parameters.lambda_list(0), A_init);
    }

    // Prepare for CV:
    //     if CV is enable,
    //     specify train and test data,
    //     and initialize the fitting argument inside each fold.
    Metric<T1, T2, T3, T4> *metric = new Metric<T1, T2, T3, T4>(eval_type, ic_coef, Kfold);
    if (Kfold > 1) {
        metric->set_cv_train_test_mask(data, data.n, cv_fold_id);
        metric->set_cv_init_fit_arg(beta_size, data.M);
        // metric->set_cv_initial_model_param(Kfold, data.p);
        // metric->set_cv_initial_A(Kfold, data.p);
        // metric->set_cv_initial_coef0(Kfold, data.p);
        // if (model_type == 1)
        //   metric->cal_cv_group_XTX(data);
    }

    // Fitting and loss:
    //     follow the search path,
    //     fit on each parameter combination,
    //     and calculate ic/loss.
    vector<Result<T2, T3>> result_list(Kfold);
    if (path_type == 1) {
        // sequentical search
#pragma omp parallel for
        for (int i = 0; i < Kfold; i++) {
            sequential_path_cv<T1, T2, T3, T4>(data, algorithm_list[i], metric, parameters, early_stop, i, A_init,
                                               result_list[i]);
        }
    } else {
        // if (algorithm_type == 5 || algorithm_type == 3)
        // {
        //     double log_lambda_min = log(max(lambda_min, 1e-5));
        //     double log_lambda_max = log(max(lambda_max, 1e-5));

        //     result = pgs_path(data, algorithm, metric, s_min, s_max, log_lambda_min, log_lambda_max, powell_path,
        //     nlambda);
        // }

        // golden section search
        gs_path<T1, T2, T3, T4>(data, algorithm_list, metric, parameters, A_init, result_list);
    }

    for (int k = 0; k < Kfold; k++) {
        algorithm_list[k]->clear_setting();
    }

    // Get bestmodel && fit bestmodel:
    //     choose the best model with lowest ic/loss
    //     and if CV, refit on full data.
    int min_loss_index = 0;
    int sequence_size = (parameters.sequence).size();
    Eigen::Matrix<T2, Dynamic, 1> beta_matrix(sequence_size, 1);
    Eigen::Matrix<T3, Dynamic, 1> coef0_matrix(sequence_size, 1);
    Eigen::Matrix<VectorXd, Dynamic, 1> bd_matrix(sequence_size, 1);
    Eigen::MatrixXd ic_matrix(sequence_size, 1);
    Eigen::MatrixXd test_loss_sum = Eigen::MatrixXd::Zero(sequence_size, 1);
    Eigen::MatrixXd train_loss_matrix(sequence_size, 1);
    Eigen::MatrixXd effective_number_matrix(sequence_size, 1);

    if (Kfold == 1) {
        // not CV: choose lowest ic
        beta_matrix = result_list[0].beta_matrix;
        coef0_matrix = result_list[0].coef0_matrix;
        ic_matrix = result_list[0].ic_matrix;
        train_loss_matrix = result_list[0].train_loss_matrix;
        effective_number_matrix = result_list[0].effective_number_matrix;
        ic_matrix.col(0).minCoeff(&min_loss_index);
    } else {
        // CV: choose lowest test loss
        for (int i = 0; i < Kfold; i++) {
            test_loss_sum += result_list[i].test_loss_matrix;
        }
        test_loss_sum /= ((double)Kfold);
        test_loss_sum.col(0).minCoeff(&min_loss_index);

        Eigen::VectorXi used_algorithm_index = Eigen::VectorXi::Zero(algorithm_list_size);

        // refit on full data
#pragma omp parallel for
        for (int ind = 0; ind < sequence_size; ind++) {
            int support_size = parameters.sequence(ind).support_size;
            double lambda = parameters.sequence(ind).lambda;

            int algorithm_index = omp_get_thread_num();
            used_algorithm_index(algorithm_index) = 1;

            T2 beta_init;
            T3 coef0_init;
            Eigen::VectorXi A_init;  // start from a clear A_init (not from the given one)
            coef_set_zero(beta_size, data.M, beta_init, coef0_init);
            Eigen::VectorXd bd_init = Eigen::VectorXd::Zero(data.g_num);

            // warmstart from CV's result
            for (int j = 0; j < Kfold; j++) {
                beta_init = beta_init + result_list[j].beta_matrix(ind) / Kfold;
                coef0_init = coef0_init + result_list[j].coef0_matrix(ind) / Kfold;
                bd_init = bd_init + result_list[j].bd_matrix(ind) / Kfold;
            }

            // fitting
            algorithm_list[algorithm_index]->update_sparsity_level(support_size);
            algorithm_list[algorithm_index]->update_lambda_level(lambda);
            algorithm_list[algorithm_index]->update_beta_init(beta_init);
            algorithm_list[algorithm_index]->update_coef0_init(coef0_init);
            algorithm_list[algorithm_index]->update_bd_init(bd_init);
            algorithm_list[algorithm_index]->update_A_init(A_init, data.g_num);
            algorithm_list[algorithm_index]->fit(data.x, data.y, data.weight, data.g_index, data.g_size, data.n, data.p,
                                                 data.g_num);

            // update results
            beta_matrix(ind) = algorithm_list[algorithm_index]->get_beta();
            coef0_matrix(ind) = algorithm_list[algorithm_index]->get_coef0();
            train_loss_matrix(ind) = algorithm_list[algorithm_index]->get_train_loss();
            ic_matrix(ind) = metric->ic(data.n, data.M, data.g_num, algorithm_list[algorithm_index]);
            effective_number_matrix(ind) = algorithm_list[algorithm_index]->get_effective_number();
        }

        for (int i = 0; i < algorithm_list_size; i++) {
            if (used_algorithm_index(i) == 1) {
                algorithm_list[i]->clear_setting();
            }
        }
    }

    // Best result
    double best_support_size = parameters.sequence(min_loss_index).support_size;
    double best_lambda = parameters.sequence(min_loss_index).lambda;

    T2 best_beta;
    T3 best_coef0;
    double best_train_loss, best_ic, best_test_loss;

    best_beta = beta_matrix(min_loss_index);
    best_coef0 = coef0_matrix(min_loss_index);
    best_train_loss = train_loss_matrix(min_loss_index);
    best_ic = ic_matrix(min_loss_index);
    best_test_loss = test_loss_sum(min_loss_index);

    // Restore for normal:
    //    restore the changes if normalization is used.
    restore_for_normal<T2, T3>(best_beta, best_coef0, beta_matrix, coef0_matrix, sparse_matrix, data.normalize_type,
                               data.n, data.x_mean, data.y_mean, data.x_norm);

    // Store in a list for output
    List out_result;
#ifdef R_BUILD
    out_result = List::create(
        Named("beta") = best_beta, Named("coef0") = best_coef0, Named("train_loss") = best_train_loss,
        Named("ic") = best_ic, Named("lambda") = best_lambda, Named("beta_all") = beta_matrix,
        Named("coef0_all") = coef0_matrix, Named("train_loss_all") = train_loss_matrix, Named("ic_all") = ic_matrix,
        Named("effective_number_all") = effective_number_matrix, Named("test_loss_all") = test_loss_sum);
    if (path_type == 2) {
        out_result.push_back(parameters.support_size_list, "sequence");
    }
#else
    out_result.add("beta", best_beta);
    out_result.add("coef0", best_coef0);
    out_result.add("train_loss", best_train_loss);
    out_result.add("test_loss", best_test_loss);
    out_result.add("ic", best_ic);
    out_result.add("lambda", best_lambda);
    // out_result.add("beta_all", beta_matrix);
    // out_result.add("coef0_all", coef0_matrix);
    // out_result.add("train_loss_all", train_loss_matrix);
    // out_result.add("ic_all", ic_matrix);
    // out_result.add("test_loss_all", test_loss_sum);
#endif

    // Restore for screening
    //    restore the changes if screening is used.
    if (screening_size >= 0) {
        T2 beta_screening_A;
        T2 beta;
        T3 coef0;
        beta_size = algorithm_list[0]->get_beta_size(n, p);
        coef_set_zero(beta_size, data.M, beta, coef0);

#ifndef R_BUILD
        out_result.get_value_by_name("beta", beta_screening_A);
        slice_restore(beta_screening_A, screening_A, beta);
        out_result.add("beta", beta);
        out_result.add("screening_A", screening_A);
#else
        beta_screening_A = out_result["beta"];
        slice_restore(beta_screening_A, screening_A, beta);
        out_result["beta"] = beta;
        out_result.push_back(screening_A, "screening_A");
#endif
    }

    delete metric;

    // Return the results
    return out_result;
}

#endif  // SRC_WORKFLOW_H
