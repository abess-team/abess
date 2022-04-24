#ifndef ALGORITHM_UNIVERSAL_H
#define ALGORITHM_UNIVERSAL_H

#include "Algorithm.h"
#include "UniversalData.h"

class abessUniversal : public Algorithm<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, UniversalData> {
private:
    double enough_small = 1e-9;
public:
    abessUniversal(int max_iter = 30, bool warm_start = true, int exchange_num = 5,
        Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 0, int sub_search = 0)
        : Algorithm<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, UniversalData>::Algorithm(
            6, UNIVERSAL_MODEL, max_iter, 10, 1e-8, warm_start, exchange_num, always_select, splicing_type, sub_search) {};
    ~abessUniversal() {};
    /**
      * Compute the loss of active_data with L2 penalty, where the value of parameter is active_para.
      * Only these three paras will be used.  
      * @param active_data                                          UniversalData which has been limited to active set A
      * @param active_para                                          the value of active parameters adapted to active_data
      * @param intercept                                            the value of intercept
      * @param lambda                                               L2 penalty coefficient
      * 
      * @return a double value indicating the loss
      */
    double loss_function(UniversalData& active_data, Eigen::MatrixXd& y, Eigen::VectorXd& weights, Eigen::VectorXd& active_para, Eigen::VectorXd& intercept, Eigen::VectorXi& A,
        Eigen::VectorXi& g_index, Eigen::VectorXi& g_size, double lambda) override;

    /**
      * optimize the loss of active_data with L2 penalty  
      * Only these two paras will be used.
      * @param active_data                                          UniversalData which will be optimized, it has been limited to active set A
      * @param active_para                                          a column vector of initial values for active parameters
      * @param intercept                                            a column vector of initial values for intercept
      *
      * @return a boolean value indicating successful completion of the optimization algorithm.
      */
    bool primary_model_fit(UniversalData& active_data, Eigen::MatrixXd& y, Eigen::VectorXd& weights, Eigen::VectorXd& active_para, Eigen::VectorXd& intercept, double loss0,
        Eigen::VectorXi& A, Eigen::VectorXi& g_index, Eigen::VectorXi& g_size) override;

    /**
      * compute the sacrifice of data
      * Only these seven paras will be used.
      * @param data                                                 UniversalData which include both active set A and inactive set I
      * @param para                                                 the value of effective parameters adapted to data
      * @param intercept                                            the value of intercept
      * @param A                                                    the index in g_index of group in active set 
      * @param I                                                    the index in g_index of group in inactive set
      * @param g_index                                              the index in para of all groups
      * @param g_size                                               the length of all groups
      * @param sacrifice                                            a column vector which will be replaced by results
      */
    void sacrifice(UniversalData& data, UniversalData& XA, Eigen::MatrixXd& y, Eigen::VectorXd& para, Eigen::VectorXd& beta_A, Eigen::VectorXd& intercept, Eigen::VectorXi& A,
        Eigen::VectorXi& I, Eigen::VectorXd& weights, Eigen::VectorXi& g_index,
        Eigen::VectorXi& g_size, int g_num, Eigen::VectorXi& A_ind, Eigen::VectorXd& sacrifice,
        Eigen::VectorXi& U, Eigen::VectorXi& U_ind, int num) override;
    /**
      * compute the effective number of parameters which will be used to compute information criterion
      * Only these two paras will be used.
      * @param active_data                                          UniversalData which has been limited to active set A
      * @param active_para                                          the value of effective parameters adapted to active_data
      * @param intercept                                            the value of intercept
      *
      * @return a double value indicating the effective number of parameters
      */
    double effective_number_of_parameter(UniversalData& X, UniversalData& active_data, Eigen::MatrixXd& y, Eigen::VectorXd& weights, Eigen::VectorXd& beta, Eigen::VectorXd& active_para,
        Eigen::VectorXd& intercept) override;
};

#endif