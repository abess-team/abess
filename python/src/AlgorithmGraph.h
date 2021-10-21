#ifndef SRC_ALGORITHMGRAPH_H
#define SRC_ALGORITHMGRAPH_H

#include "Algorithm.h"
#include<algorithm>
// #include<Eigen/CholmodSupport> // install `suitesparse` first

using namespace Eigen;
using namespace std;

using triplet = Triplet<double>;

template <class T4>
class abessGraph : public Algorithm<VectorXd, VectorXd, double, T4, VectorXd>
{
public:
  abessGraph(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, VectorXi always_select = VectorXi::Zero(0), int splicing_type = 1, int sub_search = 0) : Algorithm<VectorXd, VectorXd, double, T4, VectorXd>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessGraph(){};

  void update_tau(int train_n, int N)
  {
    // cout<<"init tau"<<endl;
    this->tau = 0.01 * (double)this->sparsity_level * log((double)N - (this->x).cols()) * log(log((double)train_n)) / (double)train_n;
  }

  MatrixXd S;

  VectorXi inital_screening(T4 &X, VectorXd &y, VectorXd &beta, double &coef0, VectorXi &A, VectorXi &I, VectorXd &bd, VectorXd &weights,
                            VectorXi &g_index, VectorXi &g_size, int &N)
  {
    // cout<<"==> init\n";///
    if (bd.size() == 0)
    {
      // variable initialization
      // int n = X.rows();
      int p = X.cols();
      // cout<<"p = "<<p<<", betasize = "<<beta.size()<<endl;///
      bd = VectorXd::Zero(p*(p + 1)/2);

      // cov
      MatrixXd X1 = X;
      MatrixXd centered = X1.rowwise() - X1.colwise().mean();
      this->S = (centered.adjoint() * centered) / (X1.rows() - 1);
      for (int i = 0; i < bd.size(); i++){
        int mi = this->map1(i, 0);
        int mj = this->map1(i, 1);
        bd(i) = fabs(this->S(mi, mj) / sqrtf(this->S(mi, mi) * this->S(mj, mj)));
      }

      for (int i = 0; i < this->always_select.size(); i++)
      {
        bd(this->always_select(i)) = DBL_MAX;
      }
    }

    // get Active-set A according to max_k bd
    VectorXi A_new = max_k(bd, this->sparsity_level);
    // cout<<"==> init end\n";///
    return A_new;
  };

  bool primary_model_fit(T4 &x, VectorXd &y, VectorXd &weights, VectorXd &beta, double &coef0, double loss0, VectorXi &A, VectorXi &g_index, VectorXi &g_size)
  {
    int T0 = A.size(), p = (this->S).cols(), T1 = A.size() - this->always_select.size();
    VectorXi dense_number_vec = VectorXi::Zero(p);
    vector<VectorXi> vec_non_zero_index_list(p);
    MatrixXd W = this->S, last_W = this->S;
    MatrixXd A2(2 * T1, 2);
    vector<triplet> trp;

    int ind = 0;
    for (int i = 0; i < T0; i++){
      int mi = this->map1(A(i), 0), mj = this->map1(A(i), 1);
      if (mi == mj) continue; 
      A2(ind, 0) = mi;
      A2(ind, 1) = mj;
      A2(T1 + ind, 0) = mj;
      A2(T1 + ind, 1) = mi;
      dense_number_vec(mi)++;
      dense_number_vec(mj)++;
      ind++;
      // cout<<"("<<mi<<","<<mj<<") ";///
    }
    // cout<<endl;///

    for (int i = 0; i < p; i++) {
      if (dense_number_vec(i) > 0) {
        int dense_number = dense_number_vec(i);
        VectorXi vec_non_zero_index(dense_number);
        int t = 0;
        for (int j = 0; j < 2 * T1; j++) {
          if (A2(j, 1) == i) {
            if (A2(j, 0) == (p - 1)) {
              vec_non_zero_index(t++) = i;// ???
            } else {
              vec_non_zero_index(t++) = A2(j, 0);
            }
          }
        }
        sort(vec_non_zero_index.data(), vec_non_zero_index.data() + vec_non_zero_index.size());
        vec_non_zero_index_list[i] = vec_non_zero_index;
      }
    }

    // cout<<" --> iter\n";///
    int iter = 0;
    while (iter < this->primary_model_fit_max_iter) {
      for (int i = 0; i < p; ++i) {
        // When there is any element in the active set, the update would work
        // else, we should set the corresponding diagonal element in y_inv 1 / covariance(i,i)
        // auto start = high_resolution_clock::now();
        
        if (dense_number_vec(i) > 0) {
          int dense_number = dense_number_vec(i);

          VectorXi vec_non_zero_index = vec_non_zero_index_list[i];

          MatrixXd W11 = MatrixXd::Zero(dense_number, dense_number);
          for (int j = 0; j < dense_number; j++) {
            int k = vec_non_zero_index(j);
            if (k != i) {
              W11(j, j) = W(k, k);
            } else {
              W11(j, j) = W(p - 1, p - 1);
            }
          }
          for (int j = 0; j < dense_number; j++) {
            int index_j = vec_non_zero_index[j];
            for (int k = j + 1; k < dense_number; k++) {
              int index_k = vec_non_zero_index[k];
              if (index_j != i && index_k != i) {
                W11(j, k) = W(index_j, index_k);
                W11(k, j) = W11(j, k);
              } else if (index_j == i && index_k != i) {
                W11(j, k) = W(p - 1, index_k);
                W11(k, j) = W11(j, k);
              } else {
                W11(j, k) = W(index_j, p - 1);
                W11(k, j) = W11(j, k);
              }
            }
          }
          
          VectorXd s12 = VectorXd::Zero(dense_number);
          for (int j = 0; j < dense_number; j++) {
            int k = vec_non_zero_index[j];
            if (k != i) {
              s12(j) = this->S(k, i);
            } else {
              s12(j) = this->S(p - 1, i);
            }
          }
          
          LLT<MatrixXd> lltOfW(W11); // compute the Cholesky decomposition of W11
          VectorXd beta_star;
          if (lltOfW.info() == NumericalIssue)
          {
            // throw 1;
            return false;
          } else {
            // VectorXd beta_star = W11.colPivHouseholderQr().solve(s12);
            beta_star = W11.llt().solve(s12);
          }
          
          VectorXd updated_w12 = VectorXd::Zero(p - 1);
          for (int k = 0; k < dense_number; k++) {
            int index_k = vec_non_zero_index[k];
            for (int j = 0; j < p - 1; j++) {
              if (j != i) {
                if (index_k != i) {
                  updated_w12(j) += beta_star(k) * W(j, index_k);
                } else {
                  updated_w12(j) += beta_star(k) * W(j, p - 1);
                }
              } else {
                if (index_k != i) {
                  updated_w12(i) += beta_star(k) * W(p - 1, index_k);
                } else {
                  updated_w12(i) += beta_star(k) * W(p - 1, p - 1);
                }
              }
            }
          }

          for (int j = 0; j < p - 1; j++) {
            if (j != i) {
              W(i, j) = updated_w12(j);
              W(j, i) = W(i, j);
            } else {
              W(p - 1, j) = updated_w12(j);
              W(j, p - 1) = W(p - 1, j);
            }
          }

          if (iter == (this->primary_model_fit_max_iter - 1)) {
            VectorXd beta_hat = VectorXd::Zero(p - 1);
            for (int j = 0; j < dense_number; j++) {
              beta_hat(vec_non_zero_index[j]) = beta_star(j);
            }
            double temp = updated_w12.dot(beta_hat);

            double temp_diag = 1 / (this->S(i, i) - temp);
            trp.push_back(triplet(i, i, temp_diag));

            for (int j = 0; j < dense_number; j++) {
              int k = vec_non_zero_index[j];
              double temp_entry = - 0.5 * beta_hat(k) * temp_diag;
              if (k != i) {
                trp.push_back(triplet(k, i, temp_entry));
                trp.push_back(triplet(i, k, temp_entry));
              } else {
                trp.push_back(triplet(p - 1, k, temp_entry));
                trp.push_back(triplet(k, p - 1, temp_entry));
              }
            }

            // Here for push_back, it would count twice, so we should divide the corresponding entris by 2,
            // but for y_inv, it would not be the case, the corresponding entries would be updated twice
            // and only the last time would be the result
          }
        } else {
          if (iter == (this->primary_model_fit_max_iter - 1)) {
            trp.push_back(triplet(i, i, 1 / this->S(i, i)));
          }
        }
      }

      // cout<<"\n iter = "<<iter<<" | W: \n";
      // for (int i=0;i<W.rows();i++){
      //   for (int j=0;j<W.cols();j++)
      //     cout<<W(i, j)<<" ";
      //   cout<<endl;
      // }

      if (iter == (this->primary_model_fit_max_iter - 1)){
        break;
      } else if (matrix_relative_difference(W, last_W) < this->primary_model_fit_epsilon) {
        iter = this->primary_model_fit_max_iter - 1;
      } else {
        last_W = W;
        iter++;
      }
    }

    // cout<<" --> iter end | iter = "<<iter<<"\n";///
    for (int i = 0; i < 2 * T1 + p; i++) {
      if (!isfinite(trp[i].value())) 
        // throw 1;
        return false;
    }

    SparseMatrix<double> Omega(p, p);
    Omega.setFromTriplets(trp.begin(), trp.end());

    // cout<<"Omega:\n";///
    // for (int i = 0; i < Omega.rows();i++){///
    //     for (int j = 0; j < Omega.cols();j++)
    //       cout<<Omega.coeff(i, j)<<" ";
    //   cout<<endl;
    // }

    // cout<<" --> restore beta\n";///
    beta = set_beta(Omega, A);
    // cout<<"==> primary fit end\n";///
    return true;
  };

  double neg_loglik_loss(T4 &X, VectorXd &y, VectorXd &weights, VectorXd &beta, double &coef0, VectorXi &A, VectorXi &g_index, VectorXi &g_size)
  {
    // cout<<"==> loss\n";///
    int p = X.cols();
    // cout<<"betasize "<<beta.size()<<" | p "<<p<<endl;///
    SparseMatrix<double> Omega = set_Omega(beta, A, p);

    MatrixXd X1 = X;
    MatrixXd centered = X1.rowwise() - X1.colwise().mean();
    MatrixXd S = (centered.adjoint() * centered) / (X1.rows() - 1);

    SparseLU<SparseMatrix<double>, COLAMDOrdering<int> > SPARSELUSOLVER;
    SPARSELUSOLVER.compute(Omega);
    double log_Omega_det = SPARSELUSOLVER.logAbsDeterminant();
    // cout<<"==> loss end\n";///
    return (S * Omega).trace() - log_Omega_det;
  };

  void sacrifice(T4 &X, T4 &XA, VectorXd &y, VectorXd &beta, VectorXd &beta_A, double &coef0, VectorXi &A, VectorXi &I, VectorXd &weights, VectorXi &g_index, VectorXi &g_size, int N, VectorXi &A_ind, VectorXd &bd, VectorXi &U, VectorXi &U_ind, int num)
  {
    // cout<<"==> sacrifice\n";///
    int p = X.cols();
    SparseMatrix<double> Omega = set_Omega(beta_A, A, p);

    // cout<<"Omega:\n";///
    // for (int i = 0; i < Omega.rows();i++){///
    //     for (int j = 0; j < Omega.cols();j++)
    //       cout<<Omega.coeff(i, j)<<" ";
    //   cout<<endl;
    // }

    // CholmodSimplicialLLT<SparseMatrix<double> > solverD;
    SimplicialLLT<SparseMatrix<double> > solverD;
    SparseMatrix<double> eye(p, p);
    eye.setIdentity();
    Omega.makeCompressed();
    solverD.compute(Omega);
    MatrixXd W = solverD.solve(eye);
    MatrixXd D = W - this->S;

    // cout<<"W:\n";///
    // for (int i = 0; i < W.rows();i++){///
    //     for (int j = 0; j < W.cols();j++)
    //       cout<<W.coeff(i, j)<<" ";
    //   cout<<endl;
    // }

    // cout<<"D:\n";///
    // for (int i = 0; i < D.rows();i++){///
    //     for (int j = 0; j < D.cols();j++)
    //       cout<<D.coeff(i, j)<<" ";
    //   cout<<endl;
    // }

    // backward
    for (int i = 0; i < A.size(); i++){
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);
      bd(A(i)) = fabs(Omega.coeff(mi, mj));
      // cout<<" > backward: ("<<mi<<","<<mj<<") "<<bd(A(i))<<endl;
      // Alter method
      // double delta_ij = W(mi, mi) * W(mj, mj) - W(mi, mj) * W(mi, mj);
      // bd(A(i)) = fabs(-2 * S(mi, mj) * Omega.coeff(mi, mj) - log(1 - 2 * W(mi, mj) * Omega.coeff(mi, mj) - delta_ij * Omega.coeff(mi, mj) * Omega.coeff(mi, mj)));
    }

    // forward
    for (int i = 0; i < I.size(); i++) {
      int mi = this->map1(I(i), 0);
      int mj = this->map1(I(i), 1);
      bd(I(i)) = fabs(D(mi, mj));
      // cout<<" > forward: ("<<mi<<","<<mj<<") "<<bd(I(i))<<endl;
      // Alter method
      // double delta_ij = W(mi, mi) * W(mj, mj) - W(mi, mj) * W(mi, mj);
      // double t_ij = W(mi, mj) / delta_ij;
      // if (this->S(mi, mj) != 0) {
      //   t_ij += (delta_ij - sqrt(delta_ij * delta_ij + 4 * this->S(mi, mj) * this->S(mi, mj) * W(mi, mi) * W(mj, mj))) / (2 * delta_ij * this->S(mi, mj));
      // }
      // bd(I(i)) = fabs(-2 * S(mi, mj) * t_ij + log(1 + 2 * W(mi, mj) * t_ij - delta_ij * t_ij * t_ij));
    }
    // cout<<"==> sacrifice end\n";///
    return;
  };

  VectorXd set_beta(SparseMatrix<double> &Omega, VectorXi &A){
    VectorXd beta = VectorXd::Zero(A.size());
    for (int i = 0; i < A.size(); i++){
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);
      beta(i) = Omega.coeff(mi, mj);
    }
    return beta;
    // for (int i = 0; i < beta.size(); i++) cout<<beta(i)<<" ";cout<<endl;
  }

  SparseMatrix<double> set_Omega(VectorXd &beta, VectorXi &A, int p){
    SparseMatrix<double> Omega(p, p);
    for (int i = 0; i < A.size(); i++){
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);
      Omega.insert(mi, mj) = beta(i);
      if (mi != mj){
        Omega.insert(mj, mi) = beta(i);
      }
    }
    return Omega;
  }

};

#endif // SRC_ALGORITHMGRAPH_H
