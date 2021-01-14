#ifndef LIST_H
#define LIST_H
 
#include <iostream>
#include <Eigen/Eigen>
#include <vector>
using namespace std;
using namespace Eigen;

class List
{
    public:
        List(){};
        ~List(){};
        void add(string name, int value);
        void get_value_by_name(string name, int& value);
		void add(string name, double value);
		void get_value_by_name(string name, double& value);
		void add(string name, MatrixXd value);
		void get_value_by_name(string name, MatrixXd& value);
		void add(string name, VectorXd value);
		void get_value_by_name(string name, VectorXd& value);
		void add(string name, VectorXi value);
		void get_value_by_name(string name, VectorXi& value);
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
};

#endif //LIST_H
