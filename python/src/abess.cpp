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

#ifdef OTHER_ALGORITHM1
#include "SpliceAlgorithm.h"
#endif

#ifdef OTHER_ALGORITHM2
#include "PrincipalBallAlgorithm.h"
#endif

using namespace Eigen;
using namespace std;

// [[Rcpp::export]]
List abessCpp(Eigen::MatrixXd x, Eigen::VectorXd y, int data_type, Eigen::VectorXd weight,
              bool is_normal,
              int algorithm_type, int model_type, int max_iter, int exchange_num,
              int path_type, bool is_warm_start,
              int ic_type, bool is_cv, int K,
              Eigen::VectorXd state,
              Eigen::VectorXi sequence,
              Eigen::VectorXd lambda_seq,
              int s_min, int s_max, int K_max, double epsilon,
              double lambda_min, double lambda_max, int nlambda,
              bool is_screening, int screening_size, int powell_path,
              Eigen::VectorXi g_index,
              Eigen::VectorXi always_select,
              double tau,
              int primary_model_fit_max_iter, double primary_model_fit_epsilon)
{
    srand(123);
    int p = x.cols();
    Eigen::VectorXi screening_A;
    if (is_screening)
    {
        screening_A = screening(x, y, weight, model_type, screening_size, g_index, always_select);
    }
    Data data(x, y, data_type, weight, is_normal, g_index);

    Algorithm *algorithm;

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

    if (algorithm_type == 6)
    {
        if (model_type == 2)
        {
            algorithm = new abessLogistic(data, algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon);
        }
    }

    algorithm->set_warm_start(is_warm_start);
    algorithm->always_select = always_select;
    algorithm->tau = tau;
    algorithm->exchange_num = exchange_num;

#ifdef OTHER_ALGORITHM1
    if (algorithm_type == 6)
    {
        if (model_type == 1)
        {
            data.add_weight();
            algorithm = new SpliceLm(data, max_iter);
        }
        algorithm->update_exchange_num(exchange_num);
    }
#endif
#ifdef OTHER_ALGORITHM2
    if (algorithm_type == 7)
    {
        if (model_type == 1)
        {
            data.add_weight();
            algorithm = new PrincipalBallLm(data, max_iter);
        }
    }
#endif

    Metric *metric;
    if (model_type == 1)
    {
        metric = new LmMetric(ic_type, is_cv, K);
    }
    else if (model_type == 2)
    {
        metric = new LogisticMetric(ic_type, is_cv, K);
    }
    else if (model_type == 3)
    {
        metric = new PoissonMetric(ic_type, is_cv, K);
    }
    else
    {
        metric = new CoxMetric(ic_type, is_cv, K);
    }

    // For CV
    if (is_cv)
    {
        metric->set_cv_train_test_mask(data.get_n());
        metric->set_cv_initial_model_param(K, data.get_p());
        if (model_type == 1)
            metric->cal_cv_group_XTX(data);
    }

    List result;
    if (path_type == 1)
    {
        result = sequential_path(data, algorithm, metric, sequence, lambda_seq);
    }
    else
    {
        if (algorithm_type == 5 || algorithm_type == 3)
        {
            double log_lambda_min = log(max(lambda_min, 1e-5));
            double log_lambda_max = log(max(lambda_max, 1e-5));

            result = pgs_path(data, algorithm, metric, s_min, s_max, log_lambda_min, log_lambda_max, powell_path, nlambda);
        }
        else
        {
            result = gs_path(data, algorithm, metric, s_min, s_max, K_max, epsilon);
        }
    }
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
    return result;
}

#ifndef R_BUILD

void pywrap_abess(double *x, int x_row, int x_col, double *y, int y_len, int data_type, double *weight, int weight_len,
                  bool is_normal,
                  int algorithm_type, int model_type, int max_iter, int exchange_num,
                  int path_type, bool is_warm_start,
                  int ic_type, bool is_cv, int K,
                  int *gindex, int gindex_len,
                  double *state, int state_len,
                  int *sequence, int sequence_len,
                  double *lambda_sequence, int lambda_sequence_len,
                  int s_min, int s_max, int K_max, double epsilon,
                  double lambda_min, double lambda_max, int n_lambda,
                  bool is_screening, int screening_size, int powell_path,
                  int *always_select, int always_select_len, double tau,
                  int primary_model_fit_max_iter, double primary_model_fit_epsilon,
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

    x_Mat = Pointer2MatrixXd(x, x_row, x_col);
    y_Vec = Pointer2VectorXd(y, y_len);
    weight_Vec = Pointer2VectorXd(weight, weight_len);
    state_Vec = Pointer2VectorXd(state, state_len);
    gindex_Vec = Pointer2VectorXi(gindex, gindex_len);
    sequence_Vec = Pointer2VectorXi(sequence, sequence_len);
    lambda_sequence_Vec = Pointer2VectorXd(lambda_sequence, lambda_sequence_len);
    always_select_Vec = Pointer2VectorXi(always_select, always_select_len);

    List mylist = abessCpp(x_Mat, y_Vec, data_type, weight_Vec,
                           is_normal,
                           algorithm_type, model_type, max_iter, exchange_num,
                           path_type, is_warm_start,
                           ic_type, is_cv, K,
                           state_Vec,
                           sequence_Vec,
                           lambda_sequence_Vec,
                           s_min, s_max, K_max, epsilon,
                           lambda_min, lambda_max, n_lambda,
                           is_screening, screening_size, powell_path,
                           gindex_Vec,
                           always_select_Vec, tau,
                           primary_model_fit_max_iter, primary_model_fit_epsilon);

    Eigen::VectorXd beta;
    double coef0;
    double train_loss;
    double ic;
    mylist.get_value_by_name("beta", beta);
    mylist.get_value_by_name("coef0", coef0);
    mylist.get_value_by_name("train_loss", train_loss);
    mylist.get_value_by_name("ic", ic);

    VectorXd2Pointer(beta, beta_out);
    *coef0_out = coef0;
    *train_loss_out = train_loss;
    *ic_out = ic;
}
#endif
