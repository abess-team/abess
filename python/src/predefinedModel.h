#ifndef PREDEFINED_MODEL_H
#define PREDEFINED_MODEL_H
#ifndef R_BUILD

#include <autodiff/forward/dual/eigen.hpp>

#include "UniversalData.h"

struct Data {
    MatrixXd x;
    MatrixXd y;
    Data() {}
    Data(MatrixXd x, MatrixXd y) : x(x), y(y) {}
};

pybind11::object slice_by_para(pybind11::object const& old_data, VectorXi const& target_para_index) {
    Data* new_data = new Data;
    Data* data = old_data.cast<Data*>();
    new_data->x = data->x(Eigen::all, target_para_index);
    new_data->y = data->y;
    return pybind11::cast(new_data);
}

pybind11::object slice_by_sample(pybind11::object const& old_data, VectorXi const& target_sample_index) {
    Data* new_data = new Data;
    Data* data = old_data.cast<Data*>();
    new_data->x = data->x(target_sample_index, Eigen::all);
    new_data->y = data->y(target_sample_index);
    return pybind11::cast(new_data);
}

void deleter(pybind11::object const& data) { delete data.cast<Data*>(); }

template <class T>
T linear_model(Matrix<T, -1, 1> const& para, Matrix<T, -1, 1> const& intercept, pybind11::object const& ex_data) {
    Data* data = ex_data.cast<Data*>();
    Map<Matrix<T, -1, -1, RowMajor> const> Beta(para.data(), data->x.cols(), intercept.size());
    return (VectorXd::Ones(data->x.rows()) * intercept.transpose() + data->x * Beta - data->y).squaredNorm();
}

#endif
#endif
