#ifndef SRC_LIST_H
#define SRC_LIST_H

#include <Eigen/Eigen>
#include <iostream>
#include <vector>
using namespace std;
using namespace Eigen;

class List {
   public:
    List(){};
    ~List(){};
    // void add(string name, int value);
    // void get_value_by_name(string name, int &value);
    void add(string name, double value);
    void get_value_by_name(string name, double &value);
    void add(string name, MatrixXd &value);
    void get_value_by_name(string name, MatrixXd &value);
    void add(string name, VectorXd &value);
    void get_value_by_name(string name, VectorXd &value);
    void add(string name, VectorXi &value);
    void combine_beta(VectorXd &value);
    // void get_value_by_name(string name, VectorXi &value);
    // void add(string name, Eigen::Matrix<VectorXd, Dynamic, Dynamic> &value);
    // void get_value_by_name(string name, Eigen::Matrix<VectorXd, Dynamic, Dynamic> &value);
    // void add(string name, Eigen::Matrix<VectorXi, Dynamic, Dynamic> &value);
    // void get_value_by_name(string name, Eigen::Matrix<VectorXi, Dynamic, Dynamic> &value);

   private:
    vector<int> vector_int;
    vector<string> vector_int_name;
    vector<double> vector_double;
    vector<string> vector_double_name;
    vector<Eigen::MatrixXd> vector_MatrixXd;
    vector<string> vector_MatrixXd_name;
    vector<Eigen::VectorXd> vector_VectorXd;
    vector<string> vector_VectorXd_name;
    vector<Eigen::VectorXi> vector_VectorXi;
    vector<string> vector_VectorXi_name;
    vector<Eigen::Matrix<VectorXi, Dynamic, Dynamic>> vector_Matrix_VectorXi;
    vector<string> vector_Matrix_VectorXi_name;
    vector<Eigen::Matrix<VectorXd, Dynamic, Dynamic>> vector_Matrix_VectorXd;
    vector<string> vector_Matrix_VectorXd_name;
};

#endif  // SRC_LIST_H
