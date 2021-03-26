//
// Created by Mamba on 2020/2/18.
//
//  #define R_BUILD
#ifndef SRC_DATA_H
#define SRC_DATA_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Eigen>
#endif
#include <iostream>

#include <vector>
#include "normalize.h"
using namespace std;
using namespace Eigen;

template <class T1, class T2, class T3>
class Data
{
public:
    Eigen::MatrixXd x;
    T1 y;
    Eigen::VectorXd weight;
    Eigen::VectorXd x_mean;
    Eigen::VectorXd x_norm;
    T3 y_mean;
    int n;
    int p;
    int M;
    int data_type;
    bool is_normal;
    int g_num;
    Eigen::VectorXi g_index;
    Eigen::VectorXi g_size;

    Eigen::VectorXi status;

    Data() = default;

    Data(Eigen::MatrixXd &x, T1 &y, int data_type, Eigen::VectorXd &weight, bool is_normal, Eigen::VectorXi &g_index, Eigen::VectorXi &status)
    {
        this->x = x;
        this->y = y;
        this->data_type = data_type;
        this->n = x.rows();
        this->p = x.cols();
        this->M = y.cols();

        this->weight = weight;
        this->is_normal = is_normal;
        this->x_mean = Eigen::VectorXd::Zero(this->p);
        this->x_norm = Eigen::VectorXd::Zero(this->p);

        this->status = status;

        if (is_normal)
        {
            this->normalize();
        }

        this->g_index = g_index;
        this->g_num = (g_index).size();
        if (g_num > 1)
        {
            Eigen::VectorXi temp = Eigen::VectorXi::Zero(g_num);
            for (int i = 0; i < g_num - 1; i++)
                temp(i) = g_index(i + 1);
            temp(g_num - 1) = this->p;
            this->g_size = temp - g_index;
        }
    };

    // to do
    void add_weight()
    {
        for (int i = 0; i < this->n; i++)
        {
            this->x.row(i) = this->x.row(i) * sqrt(this->weight(i));
        }
        array_product(this->y, this->weight, 1);
        // this->y(i) = this->y(i) * sqrt(this->weight(i));
    };

    void normalize()
    {
        if (this->data_type == 1)
        {
            Normalize(this->x, this->y, this->weight, this->x_mean, this->y_mean, this->x_norm);
        }
        else if (this->data_type == 2)
        {
            Normalize3(this->x, this->weight, this->x_mean, this->x_norm);
        }
        else
        {
            Normalize4(this->x, this->weight, this->x_norm);
        }
    };

    Eigen::VectorXi get_g_index()
    {
        return this->g_index;
    };

    int get_g_num()
    {
        return this->g_num;
    };

    Eigen::VectorXi get_g_size()
    {
        return this->g_size;
    };

    int get_n()
    {
        return this->n;
    };

    int get_p()
    {
        return this->p;
    };
};
#endif //SRC_DATA_H
