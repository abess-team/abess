#ifndef PREDEFINED_MODEL_H
#define PREDEFINED_MODEL_H
#ifndef R_BUILD

#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using ExternData = pybind11::object;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::InnerStride;
using autodiff::dual;
using autodiff::VectorXdual;
using autodiff::dual2nd;
using autodiff::VectorXdual2nd;

struct PredefinedData {
    static int data_num;
    MatrixXd x;
    MatrixXd y;
    PredefinedData(MatrixXd x, MatrixXd y) : x(x), y(y) {
        pybind11::print("Constructor", ++data_num);
    }
    ~PredefinedData() {
        pybind11::print("Destructor", --data_num);
    }
};
int PredefinedData::data_num = 0;

ExternData slice_by_para(ExternData const& old_data, VectorXi const& target_para_index) {
    PredefinedData* data = old_data.cast<PredefinedData*>();
    PredefinedData* new_data = new PredefinedData(data->x(Eigen::all, target_para_index),data->y);
    return pybind11::cast(new_data);
}

ExternData slice_by_sample(ExternData const& old_data, VectorXi const& target_sample_index) {  
    PredefinedData* data = old_data.cast<PredefinedData*>();
    PredefinedData* new_data = new PredefinedData(data->x(target_sample_index, Eigen::all), data->y(target_sample_index, Eigen::all));
    return pybind11::cast(new_data);
}

void deleter(ExternData const& data) { delete data.cast<PredefinedData*>(); }

template <class T>
T linear_model(Matrix<T, -1, 1> const& para, Matrix<T, -1, 1> const& intercept, ExternData const& ex_data) {
    PredefinedData* data = ex_data.cast<PredefinedData*>();
    T v = T(0.0);
    Eigen::Index m = intercept.size();
    Eigen::Index p = data->x.cols();
    Eigen::Map<Matrix<T, -1, 1> const, 0, InnerStride<>> beta(NULL, p, InnerStride<>(m));
    for (Eigen::Index i = 0; i < m; i++) {
        new (&beta) Eigen::Map<Matrix<T, -1, 1> const, 0, InnerStride<>>(para.data() + i, p, InnerStride<>(m));
        v += ((data->x * beta - data->y.col(i)).array() + intercept[i]).square().sum();
    }
    return v;
}


#endif
#endif
