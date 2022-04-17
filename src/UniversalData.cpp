#include"UniversalData.h"
#include <autodiff/forward/dual/eigen.hpp>

using namespace std;

UniversalData::UniversalData(Index model_size, Index sample_size, ExternData data, UniversalModel* model)
    : data(data), model(model), sample_size(sample_size), model_size(model_size), effective_size(model_size)
{
    if (this->sample_size < 1) {
        this->sample_size = 1;
    }
    if (this->model_size < 1) {
        this->model_size = 1;
        this->effective_size = 1;
    }
    if (!model->slice_by_para) {
        this->effective_para_index = VectorXi::LinSpaced(model_size, 0, model_size - 1);
    }
}

UniversalData UniversalData::slice_by_para(const VectorXi& target_para_index)
{
    UniversalData tem(*this);
    if (model->slice_by_para) {
        tem.data = model->slice_by_para(data, target_para_index);
    }
    else {
        tem.effective_para_index = VectorXi(target_para_index.size());
        for (Index i = 0; i < target_para_index.size(); i++) {
            tem.effective_para_index[i] = this->effective_para_index[target_para_index[i]];
        }
    }
    tem.effective_size = target_para_index.size();
    return tem;
}

UniversalData UniversalData::slice_by_sample(const VectorXi& target_sample_index)
{
    UniversalData tem(*this);
    tem.sample_size = target_sample_index.size();
    tem.data = model->slice_by_sample(data, target_sample_index);
    return tem;
}

optim_function UniversalData::get_optim_function(double lambda)
{
    return [this,lambda](const VectorXd& effective_para, VectorXd* gradient, void* data) {
        if (gradient) {
            Map<VectorXd> grad(gradient->data(), gradient->size());
            return this->gradient(effective_para, grad, lambda);
        }
        else {
            return this->loss(effective_para, lambda);
        }
    };
}

nlopt_function UniversalData::get_nlopt_function(double lambda) 
{
    this->lambda = lambda;
    return [](unsigned n, const double* x, double* grad, void* f_data) {
        UniversalData* data = static_cast<UniversalData*>(f_data);
        Map<VectorXd const> effective_para(x, n);
        if (grad) { // not use operator new
            Map<VectorXd> gradient(grad, n);
            return data->gradient(effective_para, gradient, data->lambda);
        }
        else {
            return data->loss(effective_para, data->lambda);
        }
    };
    
   
}

double UniversalData::loss(const VectorXd& effective_para, double lambda)
{
    if (model->slice_by_para) {
        return model->loss(effective_para,this->data) + lambda * effective_para.squaredNorm();
    }
    else {
        VectorXd complete_para = VectorXd::Zero(this->model_size);
        for (Index i = 0; i < this->effective_size; i++) {
            complete_para[this->effective_para_index[i]] = effective_para[i];
        }
        return model->loss(complete_para, this->data) + lambda * effective_para.squaredNorm();
    }
}


double UniversalData::gradient(const VectorXd& effective_para, Map<VectorXd>& gradient, double lambda)
{
    double value = 0.0;
    gradient.resize(effective_size);

    if (model->slice_by_para) {
        if (model->gradient_user_defined) {
            gradient = model->gradient_user_defined(effective_para, this->data, VectorXi::LinSpaced(effective_size, 0, effective_size - 1));
            value = model->loss(effective_para, this->data);
        }
        else {
            dual v;
            VectorXdual compute_para = effective_para;
            gradient = autodiff::gradient([this](VectorXdual const& compute_para) {
                return this->model->gradient_autodiff(compute_para, this->data);
                }, wrt(compute_para), at(compute_para), v);
            value = val(v);
        }
    }
    else {
        VectorXd complete_para = VectorXd::Zero(this->model_size);
        for (Index i = 0; i < this->effective_size; i++) {
            complete_para[this->effective_para_index[i]] = effective_para[i];
        }
        if (model->gradient_user_defined) {
            gradient = model->gradient_user_defined(complete_para, this->data, this->effective_para_index);
            value = model->loss(complete_para, this->data);
        }
        else {
            dual v;
            VectorXdual compute_para = effective_para;
            gradient = autodiff::gradient([this, &complete_para](VectorXdual const& compute_para) {
                VectorXdual para = complete_para;
                for (Index i = 0; i < compute_para.size(); i++) {
                    para[this->effective_para_index[i]] = compute_para[i];
                }
                return this->model->gradient_autodiff(para, this->data);
                }, wrt(compute_para), at(compute_para), v);
            value = val(v);
        }
    }
   
    gradient += 2 * lambda * effective_para;
    return value + lambda * effective_para.squaredNorm();
}

Index UniversalData::cols() const
{
    return effective_size;
}

Index UniversalData::rows() const
{
    return sample_size;
}

void UniversalData::hessian(const VectorXd& effective_para, VectorXd& gradient, MatrixXd& hessian, Index index, Index size, double lambda) 
{
    gradient.resize(size);
    hessian.resize(size, size);
    VectorXi compute_para_index;
    VectorXd const* para_ptr;
    VectorXd complete_para;

    if (model->slice_by_para) {
        compute_para_index = VectorXi::LinSpaced(size, index, size + index - 1);
        para_ptr = &effective_para;
    }
    else {
        compute_para_index = this->effective_para_index.segment(index, size);
        complete_para = VectorXd::Zero(this->model_size);
        for (Index i = 0; i < this->effective_size; i++) {
            complete_para[this->effective_para_index[i]] = effective_para[i];
        }
        para_ptr = &complete_para;
    }

    if (model->hessian_user_defined) {
        model->hessian_user_defined(*para_ptr, this->data, compute_para_index, gradient, hessian);
    }
    else { // autodiff
        dual2nd v;
        VectorXdual2nd g;
        VectorXdual2nd compute_para = effective_para.segment(index, size);
        hessian = autodiff::hessian([this, para_ptr, &compute_para_index](VectorXdual2nd const& compute_para) {
            VectorXdual2nd para = *para_ptr;
            for (Index i = 0; i < compute_para_index.size(); i++) {
                para[compute_para_index[i]] = compute_para[i];
            }
            return this->model->hessian_autodiff(para, this->data);
            }, wrt(compute_para), at(compute_para), v, g);
        for (Index i = 0; i < size; i++) {
            gradient[i] = val(g[i]);
        }
    }

    if (lambda != 0.0) {
        gradient += 2 * lambda * effective_para.segment(index, size);
        hessian += MatrixXd::Constant(size, size, 2 * lambda);
    }
}

void UniversalModel::set_loss_of_model(function<double(VectorXd const&, ExternData const&)>const& f)
{
    loss = f;
}

void UniversalModel::set_gradient_autodiff(function <dual(VectorXdual const&, ExternData const&)>const& f) {
    gradient_autodiff = f;
    gradient_user_defined = nullptr;
}

void UniversalModel::set_hessian_autodiff(function <dual2nd(VectorXdual2nd const&, ExternData const&)>const& f) {
    hessian_autodiff = f;
    hessian_user_defined = nullptr;
}

void UniversalModel::set_gradient_user_defined(function <VectorXd(VectorXd const&, ExternData const&, VectorXi const&)> const& f)
{
    gradient_user_defined = f;
    gradient_autodiff = nullptr;
}

void UniversalModel::set_hessian_user_defined(function <void(VectorXd const&, ExternData const&, VectorXi const&, VectorXd&, MatrixXd&)> const& f)
{
    hessian_user_defined = f;
    hessian_autodiff = nullptr;
}

void UniversalModel::set_slice_by_sample(function <ExternData(ExternData const&, VectorXi const&)> const& f)
{
    slice_by_sample = f;
}

void UniversalModel::set_slice_by_para(function <ExternData(ExternData const&, VectorXi const&)> const& f)
{
    slice_by_para = f;
}

void UniversalModel::unset_slice_by_sample()
{
    slice_by_sample = nullptr;
}

void UniversalModel::unset_slice_by_para()
{
    slice_by_para = nullptr;
}
