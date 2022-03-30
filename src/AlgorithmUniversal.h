#ifndef ALGORITHM_UNIVERSAL_H
#define ALGORITHM_UNIVERSAL_H

#include "Algorithm.h"
#include"UniversalData.h"

class abessUniversal : public Algorithm<int, Eigen::VectorXd, int, UniversalData> {
public:
    abessUniversal(int sample_size, int model_size, int max_iter = 30, int primary_model_fit_max_iter = 10,
        double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5,
        Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0)
        : Algorithm<int, Eigen::VectorXd, int, UniversalData>::Algorithm(
            6, UNIVERSAL_MODEL, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start,
            exchange_num, always_select, splicing_type, sub_search) {};

    double loss_function(UniversalData& X, int& y, Eigen::VectorXd& weights, Eigen::VectorXd& beta, int& coef0, Eigen::VectorXi& A,
        Eigen::VectorXi& g_index, Eigen::VectorXi& g_size, double lambda);

    void sacrifice(UniversalData& X, UniversalData& XA, int& y, Eigen::VectorXd& beta, Eigen::VectorXd& beta_A, int& coef0, Eigen::VectorXi& A,
        Eigen::VectorXi& I, Eigen::VectorXd& weights, Eigen::VectorXi& g_index,
        Eigen::VectorXi& g_size, int N, Eigen::VectorXi& A_ind, Eigen::VectorXd& bd,
        Eigen::VectorXi& U, Eigen::VectorXi& U_ind, int num);

    bool primary_model_fit(UniversalData& X, int& y, Eigen::VectorXd& weights, Eigen::VectorXd& beta, int& coef0, double loss0,
        Eigen::VectorXi& A, Eigen::VectorXi& g_index, Eigen::VectorXi& g_size);

    double effective_number_of_parameter(UniversalData& X, UniversalData& XA, int& y, Eigen::VectorXd& weights, Eigen::VectorXd& beta, Eigen::VectorXd& beta_A,
        int& coef0);
};

#endif