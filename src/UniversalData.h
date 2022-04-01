#ifndef SRC_UNIVERSALDATA_H
#define SRC_UNIVERSALDATA_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Eigen>
#endif

#include<functional>

typedef std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, void*)> optim_function;
typedef double (*nlopt_function)(unsigned n, const double* x, double* grad, void* f_data);
// UniversalData includes everything about the statistic model like samples, loss.
// In abess project, UniversalData will be an instantiation of T4 in template class algorithm, other instantiation of T4 often is matrix.
// So in algorithm, UniversalData is just like a matrix, and its instance is often denoted as 'x'.
// In order to work like matrix, UniversalData need the help of utility function like X_seg, slice.
class UniversalData{  
protected:
    int model_size; // length of complete_para
    int sample_size = 1;
    void* function;
    void* data;
    double lambda = 0.;
    Eigen::VectorXi effective_para_index; //  complete_para[effective_para_index[i]] = effective_para[i], ohter location of complete_para is 0
    Eigen::VectorXi* compute_para_index_ptr = NULL; //  when it's NULL, compute_para equals to effective_para
    void get_compute_para(const Eigen::VectorXd& effective_para, Eigen::VectorXd& compute_para) const; // extract compute_para from effective_para
public:
    UniversalData() = default;
    UniversalData(int model_size, int sample_size, void* function, void* data);
    UniversalData(const UniversalData& original, const Eigen::VectorXi& target_para_index); // update effective_para accroding to target_para_index
        optim_function get_optim_function(double lambda); // create a function which can be optimized by OptimLib 
    nlopt_function get_nlopt_function(double lambda); // create a function which can be optimized by nlopt
    double loss(const Eigen::VectorXd& effective_para, double lambda); // compute the loss with effective_para
    void gradient(const Eigen::VectorXd& effective_para, Eigen::VectorXd& gradient, double lambda); // compute the gradient of effective_para
    void hessian(const Eigen::VectorXd& effective_para, Eigen::MatrixXd& hessian, double lambda); // compute the hessian of effective_para
    void hessian(const Eigen::VectorXd& effective_para, Eigen::VectorXd& gradient, Eigen::MatrixXd& hessian, int index, int size, double lambda); // compute the hessian of sequence from index to (index+size-1) in effective_para
    int rows() const; // return sample_size
    int cols() const; // return length of effective_para
};

typedef double(*universal_function)(const Eigen::VectorXd& effective_para, const UniversalData& universal_data, Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian);


#endif
