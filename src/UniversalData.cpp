#include"UniversalData.h"

using namespace Eigen;
using namespace std;

UniversalData::UniversalData(int model_size, int sample_size, function_ptr function) 
    :model_size(model_size), sample_size(sample_size), function(function)
{
    if (this->sample_size < 1) {
        this->sample_size = 1;
    }
    if (this->model_size < 1) {
        this->model_size = 1;
    }
    this->effective_para_index = VectorXi::LinSpaced(model_size, 0, model_size - 1);
    // check data and function
    VectorXd zero = VectorXd::Zero(model_size);
    ((universal_function)this->function)(zero, *this, NULL, NULL);
}

UniversalData::UniversalData(const UniversalData& original, const VectorXi& target_para_index) : UniversalData(original)
{
    // assert(target_para_index.array().maxCoeff() < original.effective_para_index.size());
    this->effective_para_index = VectorXi(target_para_index.size());
    for (int i = 0; i < target_para_index.size(); i++) {
        this->effective_para_index[i] = original.effective_para_index[target_para_index[i]];
    }
}

optim_function UniversalData::get_optim_function(double lambda)
{
    return [this,lambda](const VectorXd& effective_para, VectorXd* gradient, void* data) {
        double value = ((universal_function)this->function)(effective_para, *this, gradient, NULL) + lambda * effective_para.cwiseAbs2().sum();
        if (gradient) {
            *gradient = *gradient + 2 * lambda * effective_para;
        }
        return value;
    };
}

nlopt_function UniversalData::get_nlopt_function(double lambda) {
    if (lambda == 0.) { // optimize for frequent degradation situations
        return [](unsigned n, const double* x, double* grad, void* f_data) {
            UniversalData* data = static_cast<UniversalData*>(f_data);
            Map<VectorXd const> effective_para(x, n);
            VectorXd* gradient = NULL;
            if (grad) { // not use operator new
                Map<VectorXd> grad_tem(grad, n);
                gradient = (VectorXd*)&grad_tem;
            }
            return ((universal_function)data->function)(effective_para, *data, gradient, NULL);
        };
    }
    else {
        this->lambda = lambda;
        return [](unsigned n, const double* x, double* grad, void* f_data) {
            UniversalData* data = static_cast<UniversalData*>(f_data);
            Map<VectorXd const> effective_para(x, n);
            VectorXd* gradient = NULL;
            if (grad) { // not use operator new
                Map<VectorXd> grad_tem(grad, n);
                gradient = (VectorXd*)&grad_tem;
            }
            double value = ((universal_function)data->function)(effective_para, *data, gradient, NULL) + data->lambda * effective_para.cwiseAbs2().sum();
            if (gradient) {
                *gradient = *gradient + 2 * data->lambda * effective_para;
            }
            return value;
        };
    }
   
}

double UniversalData::loss(const VectorXd& effective_para, double lambda)
{
    return ((universal_function)this->function)(effective_para, *this, NULL, NULL) + lambda * effective_para.cwiseAbs2().sum();
}


void UniversalData::gradient(const VectorXd& effective_para, VectorXd& gradient, double lambda)
{
    ((universal_function)this->function)(effective_para, *this, &gradient, NULL);
    gradient = gradient + 2 * lambda * effective_para;
}

int UniversalData::cols() const
{
    return effective_para_index.size();
}

int UniversalData::rows() const
{
    return sample_size;
}

void UniversalData::hessian(const VectorXd& effective_para, MatrixXd& hessian, double lambda)
{
    ((universal_function)this->function)(effective_para, *this, NULL, &hessian);
    hessian = hessian + 2 * lambda * MatrixXd::Identity(hessian.rows(), hessian.cols());
}

void UniversalData::hessian(const VectorXd& effective_para, VectorXd& gradient, MatrixXd& hessian, int index, int size, double lambda)
{
    VectorXi tem = this->effective_para_index.segment(index, size);
    this->compute_para_index_ptr = &tem;
    ((universal_function)this->function)(effective_para, *this, &gradient, &hessian);
    if (size == 1) {
        hessian(0, 0) += 2 * lambda;
    }
    else {
        hessian = hessian + 2 * lambda * MatrixXd::Identity(hessian.rows(), hessian.cols());
    }
    this->compute_para_index_ptr = NULL;

}
