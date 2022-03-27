#include"UniversalData.h"

UniversalData::UniversalData(int dim, universal_function function, void* data) :dim(dim),function(function),data(data)
{
    this->effective_para_index = Eigen::VectorXi::LinSpaced(dim, 0, dim - 1);
}

UniversalData::UniversalData(const UniversalData& original, const Eigen::VectorXi& target_para_index) : UniversalData(original)
{
    // assert(target_para_index.array().maxCoeff() < original.effective_para_index.size());
    this->effective_para_index = Eigen::VectorXi(target_para_index.size());
    for (int i = 0; i < target_para_index.size(); i++) {
        this->effective_para_index[i] = original.effective_para_index[target_para_index[i]];
    }
}
void UniversalData::get_compute_para(const Eigen::VectorXd& effective_para, Eigen::VectorXd& compute_para) const
{
    if (this->compute_para_index.size() == 0) {
        compute_para = effective_para;
    }
    else {
        // assert(effective_para.size() == this->effective_para_index.size());
        Eigen::VectorXd complete_para = Eigen::VectorXd::Zero(this->dim);
        for (int i = 0; i < this->effective_para_index.size(); i++) {
            complete_para[this->effective_para_index[i]] = effective_para[i];
        }
        compute_para = Eigen::VectorXd(this->compute_para_index.size());
        for (int i = 0; i < compute_para.size(); i++) {
            compute_para[i] = complete_para[this->compute_para_index[i]];
        }
    }
}

optim_function UniversalData::get_optim_function() const
{
    auto _func = [this](const Eigen::VectorXd& effective_para, Eigen::VectorXd* gradient, void* data) {
        return this->function(effective_para, *this, gradient, NULL);
    };
    return _func;
}


void UniversalData::gradient(const Eigen::VectorXd& effective_para, Eigen::VectorXd& gradient)
{
    this->function(effective_para, *this, &gradient, NULL);
}

void UniversalData::hessian(const Eigen::VectorXd& effective_para, Eigen::MatrixXd& hessian)
{
    this->function(effective_para, *this, NULL, &hessian);
}

void UniversalData::hessian(const Eigen::VectorXd& effective_para, Eigen::VectorXd& gradient, Eigen::MatrixXd& hessian, int index, int size)
{
    this->compute_para_index = this->effective_para_index.segment(index, size);
    this->function(effective_para, *this, &gradient, &hessian);
    Eigen::VectorXi a;
    this->compute_para_index = a;
}
