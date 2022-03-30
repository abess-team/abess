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

optim_function UniversalData::get_optim_function() const
{
    auto _func = [this](const VectorXd& effective_para, VectorXd* gradient, void* data) {
        return ((universal_function)this->function)(effective_para, *this, gradient, NULL);
    };
    return _func;
}


void UniversalData::gradient(const VectorXd& effective_para, VectorXd& gradient)
{
    ((universal_function)this->function)(effective_para, *this, &gradient, NULL);
}

int UniversalData::cols() const
{
    return effective_para_index.size();
}

int UniversalData::rows() const
{
    return sample_size;
}

void UniversalData::hessian(const VectorXd& effective_para, MatrixXd& hessian)
{
    ((universal_function)this->function)(effective_para, *this, NULL, &hessian);
}

void UniversalData::hessian(const VectorXd& effective_para, VectorXd& gradient, MatrixXd& hessian, int index, int size)
{
    this->compute_para_index = this->effective_para_index.segment(index, size);
    ((universal_function)this->function)(effective_para, *this, &gradient, &hessian);
    VectorXi a;
    this->compute_para_index = a;
}
