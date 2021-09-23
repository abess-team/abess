#ifndef SRC_ALGORITHMISING_H
#define SRC_ALGORITHMISING_H

#include "Algorithm.h"

template <class T4>
class abessIsing : public Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>
{
public:
  abessIsing(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 1, int sub_search = 0) : Algorithm<Eigen::VectorXd, Eigen::VectorXd, double, T4>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessIsing(){};

  double gamma = 0.0;


  MatrixXd SigmaA(Eigen::VectorXi &ind, int p)
  {
    if (ind.size() == p){
      return this->Sigma;
    }

    int len = ind.size();
    MatrixXd SA(len, len);
    
    for (int i = 0; i < len; i++)
      for (int j = 0; j < i + 1; j++)
      {
        SA(i, j) = this->Sigma(ind(i), ind(j));
        SA(j, i) = this->Sigma(ind(j), ind(i));
      }

    return SA;
  }

  Eigen::VectorXi inital_screening(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &bd, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {
    cout<<"==> init"<<endl;///

    if (bd.size() == 0)
    {
      // variable initialization
      int n = X.rows();
      int p = X.cols();
      bd = Eigen::VectorXd::Zero(N);

      // calculate beta & d & h
      Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N, this->model_type);
      Eigen::VectorXi XA_ind = find_ind_graph(A_ind, this->map1, p);
      T4 X_A = X_seg(X, n, XA_ind);
      T2 beta_A;
      slice(beta, A_ind, beta_A);

      this->update_exp_odd();

      Eigen::VectorXi U = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      Eigen::VectorXi U_ind = Eigen::VectorXi::LinSpaced(N, 0, N - 1);
      this->sacrifice(X, X_A, y, beta, beta_A, coef0, A, I, weights, g_index, g_size, N, A_ind, bd, U, U_ind, 0);
      for (int i = 0; i < this->always_select.size(); i++)
      {
        bd(this->always_select(i)) = DBL_MAX;
      }
    }

    // get Active-set A according to max_k bd
    Eigen::VectorXi A_new = max_k(bd, this->sparsity_level);

    return A_new;
  }

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    cout<<" --> primary fit"<<endl;///
    // inner_loop3
    int n = this->x.rows();
    int p = this->x.cols();

    // Eigen::VectorXi SA_ind = find_ind_graph(A, this->map1, p);
    // Eigen::MatrixXd SA = this->SigmaA(SA_ind, p);

    Eigen::MatrixXd prob, first_der, delta_Sigma;
    double loglik = loss0, step = 1.0, alpha = 0.1, scale = 0.5;
    
    // Newton
    int iter = 0;
    while(iter++ <= this->primary_model_fit_max_iter) {
      prob = 1.0 / (1.0 + this->exp_odd.array());
      first_der = first_derivative(weights, prob, A);
      delta_Sigma = compute_delta_Sigma(first_der, weights, prob, A);
      
      this->Sigma += step * delta_Sigma;
      this->update_exp_odd();

      loglik = this->neg_loglik_loss(X, y, weights, beta, coef0, A, g_index, g_size);
      
      // need max iter?
      while(loglik <= loss0 + alpha * step * (first_der.cwiseProduct(delta_Sigma)).sum()) {
        step *= scale; 
        Sigma = last_Sigma + step * delta_Sigma;
        this->update_exp_odd();
        loglik = this->neg_loglik_loss(X, y, weights, beta, coef0, A, g_index, g_size);
      }
      
      // fail to find
      if(!isfinite(loglik - loss0)) {
        return false;
      }
      
      // stop iter
      if (fabs(loglik - loss0) < this->primary_model_fit_epsilon) break;

      loss0 = loglik;
    }
    
    // update beta
    this->update_beta(beta, A_ind);

    return true;
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    cout<<" --> loss"<<endl;///

    double loss;
    int n = X.rows();
    int p = X.cols();

    Eigen::MatrixXd log_par = (Eigen::MatrixXd::Ones(n, p) + this->exp_odd).array().log();
    loss = - ((log_par.rowwise().sum()).dot(weights)) / (double)(this->ising_n) - 
      this->gamma * (this->Sigma.cwiseProduct(this->Sigma).sum());
    
    return loss;
  };

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXd &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, Eigen::VectorXd &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {
    cout<<" --> sacrifice"<<endl;///
    int p = X.cols();
    int n = X.rows();

    // backward
    for (int i = 0; i < A.size(); i++){
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);
      bd(A(i)) = fabs(this->Sigma(mi, mj));
    }

    // forward
    prob = 1.0 / (1.0 + this->exp_odd.array());
    first_der = first_derivative(weights, prob, A);

    for (int i = 0; i < I.size(); i++){
      int mi = this->map1(I(i), 0);
      int mj = this->map1(I(i), 1);
      bd(I(i)) = fabs(first_der(mi, mj));
    }

  };

  void update_exp_odd() {
    int n = this->x.rows();
    int p = this->x.cols();
    
    this->exp_odd = - 2.0 * (this->x * this->Sigma).cwiseProduct(this->x);
    this->exp_odd = this->exp_odd.array().exp();
  }

  void update_beta(Eigen::VectorXd &beta, Eigen::VectorXi &A_ind){
    for (int i = 0; i < A_ind.size(); i++){
      int mi = this->map1(A_ind(i), 0);
      int mj = this->map1(A_ind(i), 1);
      beta(i) = this->Sigma(mi, mj);
    }
  }

  Eigen::MatrixXd compute_delta_Sigma(Eigen::MatrixXd &first_der, 
                                      Eigen::MatrixXd &weights, 
                                      Eigen::MatrixXd &prob, Eigen::MatrixXi &A) {
    // without penalty term
    // gradient of PL, not gradient of loss
    int p = this->x.cols();
    double second_der;
    Eigen::MatrixXd delta_Sigma = Eigen::MatrixXd::Zero(p, p);
    
    for(int i = 0; i < this->sparsity_level; i++) {
      // only for off-diagonal elements
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);

      Eigen::MatrixXd ans = ( - prob.col(mi).array() * (1.0 - prob.col(mi).array())) + 
        ( - prob.col(mj).array() * (1.0 - prob.col(mj).array()));
      
      second_der = (4.0 * ans.dot(weights)) / (double)(this->ising_n);
      delta_Sigma(mi, mj) =  - (first_der(mi, mj) / second_der);
      delta_Sigma(mj, mi) = delta_Sigma(mi, mj);
    }
    return delta_Sigma;
  }

  Eigen::MatrixXd first_derivative(Eigen::MatrixXd &weights, 
                                   Eigen::MatrixXd &prob, Eigen::MatrixXi &A) {
    // gradient of PL, not gradient of loss
    int p = this->x.cols();
    Eigen::MatrixXd first_der = Eigen::MatrixXd::Zero(p, p);
    
    for(int i = 0; i < this->sparsity_level; i++) {
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);

      Eigen::MatrixXd ans = this->x.col(mi).array() * this->x.col(mj).array() * 
        (2.0 - prob.col(mi).array() - prob.col(mj).array());
      
      first_der(mi, mj) = (2.0 * ans.dot(weights)) / (double)(this->ising_n) - 
        4.0 * this->gamma * this->Sigma(mi, mj);
      first_der(mj, mi) = first_der(mi, mj);
    }

    return first_der;
  }
};

#endif // SRC_ALGORITHMISING_H