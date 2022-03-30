#include "AlgorithmUniversal.h"

using namespace std;
using namespace Eigen;

double abessUniversal::loss_function(UniversalData& X, int& y, VectorXd& weights, VectorXd& beta, int& coef0, VectorXi& A,
    VectorXi& g_index, VectorXi& g_size, double lambda) {

}

void abessUniversal::sacrifice(UniversalData& X, UniversalData& XA, int& y, VectorXd& beta, VectorXd& beta_A, int& coef0, VectorXi& A,
    VectorXi& I, VectorXd& weights, VectorXi& g_index,
    VectorXi& g_size, int N, VectorXi& A_ind, VectorXd& bd,
    VectorXi& U, VectorXi& U_ind, int num) {

}

double abessUniversal::effective_number_of_parameter(UniversalData& X, UniversalData& XA, int& y, VectorXd& weights, VectorXd& beta, VectorXd& beta_A,
    int& coef0) {

}

bool abessUniversal::primary_model_fit(UniversalData& X, int& y, VectorXd& weights, VectorXd& beta, int& coef0, double loss0,
    VectorXi& A, VectorXi& g_index, VectorXi& g_size) {

}