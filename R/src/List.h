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
	// List(const List &mylist)
	// {
	// 	if (mylist.vector_int.size() != 0)
	// 	{
	// 		for (int i = 0; i < mylist.vector_int.size(); i++)
	// 		{
	// 			this->vector_int.push_back(mylist.vector_int[i]);
	// 			this->vector_int_name.push_back(mylist.vector_int_name[i]);
	// 		}
	// 	}
	// 	if (mylist.vector_Matrix_VectorXi.size() != 0)
	// 	{
	// 		for (int i = 0; i < mylist.vector_Matrix_VectorXi.size(); i++)
	// 		{
	// 			this->vector_Matrix_VectorXi.push_back(vector_Matrix_VectorXi[i]);
	// 			this->vector_Matrix_VectorXi_name.push_back(vector_Matrix_VectorXi_name[i]);
	// 		}
	// 	}
	// 	if (mylist.vector_Matrix_VectorXd.size() != 0)
	// 	{
	// 		for (int i = 0; i < mylist.vector_Matrix_VectorXd.size(); i++)
	// 		{
	// 			this->vector_Matrix_VectorXd.push_back(vector_Matrix_VectorXd[i]);
	// 			this->vector_Matrix_VectorXd_name.push_back(vector_Matrix_VectorXd_name[i]);
	// 		}
	// 	}
	// 	if (mylist.vector_MatrixXd.size() != 0)
	// 	{
	// 		for (int i = 0; i < mylist.vector_MatrixXd.size(); i++)
	// 		{
	// 			this->vector_MatrixXd.push_back(vector_MatrixXd[i]);
	// 			this->vector_MatrixXd_name.push_back(vector_Matrix_VectorXd_name[i]);
	// 		}
	// 	}
	// }
	void add(string name, int value);
	void get_value_by_name(string name, int &value);
	void add(string name, double value);
	void get_value_by_name(string name, double &value);
	void add(string name, MatrixXd &value);
	void get_value_by_name(string name, MatrixXd &value);
	void add(string name, VectorXd &value);
	void get_value_by_name(string name, VectorXd &value);
	void add(string name, VectorXi &value);
	void get_value_by_name(string name, VectorXi &value);
	void add(string name, Eigen::Matrix<VectorXd, Dynamic, Dynamic> &value);
	void get_value_by_name(string name, Eigen::Matrix<VectorXd, Dynamic, Dynamic> &value);
	void add(string name, Eigen::Matrix<VectorXi, Dynamic, Dynamic> &value);
	void get_value_by_name(string name, Eigen::Matrix<VectorXi, Dynamic, Dynamic> &value);

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

#endif //LIST_H
