#include"UniversalData.h"

using namespace Eigen;

UniversalData::UniversalData(int dim, void* function, void* data, int sample_size = 1) :dim(dim),function(function),data(data),sample_size(sample_size)
{
    this->effective_para_index = VectorXi::LinSpaced(dim, 0, dim - 1);
}

UniversalData::UniversalData(const UniversalData& original, const VectorXi& target_para_index) : UniversalData(original)
{
    // assert(target_para_index.array().maxCoeff() < original.effective_para_index.size());
    this->effective_para_index = VectorXi(target_para_index.size());
    for (int i = 0; i < target_para_index.size(); i++) {
        this->effective_para_index[i] = original.effective_para_index[target_para_index[i]];
    }
}
void UniversalData::get_compute_para(const VectorXd& effective_para, VectorXd& compute_para) const
{
    if (this->compute_para_index.size() == 0) {
        compute_para = effective_para;
    }
    else {
        // assert(effective_para.size() == this->effective_para_index.size());
        VectorXd complete_para = VectorXd::Zero(this->dim);
        for (int i = 0; i < this->effective_para_index.size(); i++) {
            complete_para[this->effective_para_index[i]] = effective_para[i];
        }
        compute_para = VectorXd(this->compute_para_index.size());
        for (int i = 0; i < compute_para.size(); i++) {
            compute_para[i] = complete_para[this->compute_para_index[i]];
        }
    }
}

optim_function UniversalData::get_optim_function(double lambda) const
{
    auto _func = [this,lambda](const VectorXd& effective_para, VectorXd* gradient, void* data) {
        double value = ((universal_function)this->function)(effective_para, *this, gradient, NULL) + lambda * effective_para.cwiseAbs2().sum();
        *gradient = *gradient + 2 * lambda * effective_para;
        return value;
    };
    return _func;
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
    this->compute_para_index = this->effective_para_index.segment(index, size);
    ((universal_function)this->function)(effective_para, *this, &gradient, &hessian);
    hessian = hessian + 2 * lambda * MatrixXd::Identity(hessian.rows(), hessian.cols());
    VectorXi a;
    this->compute_para_index = a;
}
