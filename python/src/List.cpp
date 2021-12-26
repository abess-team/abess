#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
#else

#include <Eigen/Eigen>

#include "List.h"

#endif

#include <iostream>
#include <typeinfo>
#include <vector>

using namespace std;
using namespace Eigen;

// void List::add(string name, int value)
// {
//     //	cout<<"value in int add"<<endl;
//     std::size_t i;
//     for (i = 0; i < vector_int_name.size(); i++)
//     {
//         if (vector_int_name[i] == name)
//         {
//             vector_int[i] = value;
//             return;
//         }
//     }
//     vector_int.push_back(value);
//     vector_int_name.push_back(name);
// }

void List::combine_beta(VectorXd &value) {
    std::size_t i;
    for (i = 0; i < vector_MatrixXd_name.size(); i++) {
        if (vector_MatrixXd_name[i] == "beta") {
            MatrixXd beta_new(vector_MatrixXd[i].rows(), vector_MatrixXd[i].cols() + 1);
            beta_new << vector_MatrixXd[i], value;
            vector_MatrixXd[i] = beta_new;
            return;
        }
    }

    for (i = 0; i < vector_VectorXd_name.size(); i++) {
        if (vector_VectorXd_name[i] == "beta") {
            MatrixXd beta_new(value.size(), 2);
            beta_new << vector_VectorXd[i], value;
            vector_VectorXd_name[i] = "beta0";
            this->add("beta", beta_new);
            return;
        }
    }
}

void List::add(string name, double value) {
    // cout<<"value in double add"<<endl;
    std::size_t i;
    for (i = 0; i < vector_double_name.size(); i++) {
        // cout<<"value in get double"<<endl;
        if (vector_double_name[i] == name) {
            vector_double[i] = value;
            return;
        }
    }
    vector_double.push_back(value);
    vector_double_name.push_back(name);
}

void List::add(string name, MatrixXd &value) {
    std::size_t i;
    for (i = 0; i < vector_MatrixXd_name.size(); i++) {
        if (vector_MatrixXd_name[i] == name) {
            vector_MatrixXd[i] = value;
            return;
        }
    }
    vector_MatrixXd.push_back(value);
    vector_MatrixXd_name.push_back(name);
}

void List::add(string name, VectorXd &value) {
    std::size_t i;
    for (i = 0; i < vector_VectorXd_name.size(); i++) {
        if (vector_VectorXd_name[i] == name) {
            vector_VectorXd[i] = value;
            return;
        }
    }
    vector_VectorXd.push_back(value);
    vector_VectorXd_name.push_back(name);
}

void List::add(string name, VectorXi &value) {
    std::size_t i;
    for (i = 0; i < vector_VectorXi_name.size(); i++) {
        if (vector_VectorXi_name[i] == name) {
            vector_VectorXi[i] = value;
            return;
        }
    }
    vector_VectorXi.push_back(value);
    vector_VectorXi_name.push_back(name);
}

// void List::add(string name, Eigen::Matrix<VectorXi, Dynamic, Dynamic> &value)
// {
//     //	cout<<"value in VectorXi add"<<endl;
//     std::size_t i;
//     for (i = 0; i < vector_Matrix_VectorXi_name.size(); i++)
//     {
//         //		cout<<"value in get VectorXi"<<endl;
//         if (vector_Matrix_VectorXi_name[i] == name)
//         {
//             vector_Matrix_VectorXi[i] = value;
//             return;
//         }
//     }
//     vector_Matrix_VectorXi.push_back(value);
//     vector_Matrix_VectorXi_name.push_back(name);
// }

// void List::get_value_by_name(string name, Eigen::Matrix<VectorXi, Dynamic, Dynamic> &value)
// {
//     std::size_t i;
//     for (i = 0; i < vector_Matrix_VectorXi_name.size(); i++)
//     {
//         //		cout<<"value in get VectorXi"<<endl;
//         if (vector_Matrix_VectorXi_name[i] == name)
//         {
//             value = vector_Matrix_VectorXi[i];
//             break;
//         }
//     }
// }

// void List::add(string name, Eigen::Matrix<VectorXd, Dynamic, Dynamic> &value)
// {
//     //	cout<<"value in VectorXi add"<<endl;
//     std::size_t i;
//     for (i = 0; i < vector_Matrix_VectorXd_name.size(); i++)
//     {
//         //		cout<<"value in get VectorXi"<<endl;
//         if (vector_Matrix_VectorXd_name[i] == name)
//         {
//             vector_Matrix_VectorXd[i] = value;
//             return;
//         }
//     }
//     vector_Matrix_VectorXd.push_back(value);
//     vector_Matrix_VectorXd_name.push_back(name);
// }

// void List::get_value_by_name(string name, Eigen::Matrix<VectorXd, Dynamic, Dynamic> &value)
// {
//     std::size_t i;
//     for (i = 0; i < vector_Matrix_VectorXd_name.size(); i++)
//     {
//         //		cout<<"value in get VectorXi"<<endl;
//         if (vector_Matrix_VectorXd_name[i] == name)
//         {
//             value = vector_Matrix_VectorXd[i];
//             break;
//         }
//     }
// }

// void List::get_value_by_name(string name, int &value)
// {
//     std::size_t i;
//     for (i = 0; i < vector_int_name.size(); i++)
//     {
//         //		cout<<"value in get double"<<endl;
//         if (vector_int_name[i] == name)
//         {
//             value = vector_int[i];
//             break;
//         }
//     }
// }

void List::get_value_by_name(string name, double &value) {
    std::size_t i;
    for (i = 0; i < vector_double_name.size(); i++) {
        if (vector_double_name[i] == name) {
            value = vector_double[i];
            break;
        }
    }
}

void List::get_value_by_name(string name, MatrixXd &value) {
    std::size_t i;
    for (i = 0; i < vector_MatrixXd_name.size(); i++) {
        if (vector_MatrixXd_name[i] == name) {
            value = vector_MatrixXd[i];
            break;
        }
    }
}

void List::get_value_by_name(string name, VectorXd &value) {
    std::size_t i;
    for (i = 0; i < vector_VectorXd_name.size(); i++) {
        if (vector_VectorXd_name[i] == name) {
            value = vector_VectorXd[i];
            break;
        }
    }
}

// void List::get_value_by_name(string name, VectorXi &value)
// {
//     std::size_t i;
//     for (i = 0; i < vector_VectorXi_name.size(); i++)
//     {
//         //		cout<<"value in get VectorXi"<<endl;
//         if (vector_VectorXi_name[i] == name)
//         {
//             value = vector_VectorXi[i];
//             break;
//         }
//     }
// }
