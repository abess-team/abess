#ifndef SRC_ALGORITHMISING_H
#define SRC_ALGORITHMISING_H

#include "Algorithm.h"

using VL = Eigen::Vector<long double, Eigen::Dynamic>;
using ML = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;

template <class T4>
class abessIsing : public Algorithm<Eigen::VectorXd, VL, double, T4, VL>
{
public:
  abessIsing(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 1, int sub_search = 0) : Algorithm<Eigen::VectorXd, VL, double, T4, VL>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessIsing(){};

  long double gamma = 0.0;

  ML thetaA(Eigen::VectorXi &ind, int p)
  {
    if (ind.size() == p){
      return this->theta;
    }

    int len = ind.size();
    ML tA(len, len);
    
    for (int i = 0; i < len; i++)
      for (int j = 0; j < i + 1; j++)
      {
        tA(i, j) = this->theta(ind(i), ind(j));
        tA(j, i) = this->theta(ind(j), ind(i));
      }

    return tA;
  }

  Eigen::VectorXi inital_screening(T4 &X, Eigen::VectorXd &y, VL &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, VL &bd, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {
    cout<<"==> init | N = "<<N<<" | Asize = "<<A.size()<<endl;

    if (bd.size() == 0)
    {
      // variable initialization
      int p = X.cols();
      bd = VL::Zero(N);

      // calculate beta & d & h
      Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, p, N, this->model_type);
      // Eigen::VectorXi XA_ind = find_ind_graph(A_ind, this->map1, p);
      // T4 X_A = X_seg(X, n, XA_ind);
      T4 X_A = X;
      VL beta_A;
      slice(beta, A_ind, beta_A);

      this->update_exp_odd(X);

      // cout<<" --> exp_odd:\n";
      // for (int i=0;i<this->exp_odd.rows();i++){
      //   for (int j=0;j<this->exp_odd.cols();j++)
      //     cout<<this->exp_odd(i,j)<<" ";
      //   cout<<endl;
      // }

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

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, VL &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    cout<<" --> primary fit"<<endl;///
    // inner_loop3

    ML prob, first_der, delta_theta, last_theta = this->theta;
    long double loglik = loss0, step = 1.0, alpha = 0.1, scale = 0.5;
    
    // Newton
    int iter = 0;
    while(iter++ <= this->primary_model_fit_max_iter) {
      prob = 1.0 / (1.0 + this->exp_odd.array());
      first_der = first_derivative(x, weights, prob, A);
      delta_theta = compute_delta_theta(x, first_der, weights, prob, A);
      
      this->theta += step * delta_theta;
      this->update_exp_odd(x);

      loglik = this->neg_loglik_loss(x, y, weights, beta, coef0, A, g_index, g_size);
      
      // need max iter?
      int t=0;
      while(step > 0 && loglik <= loss0 + alpha * step * (first_der.cwiseProduct(delta_theta)).sum()) {
        cout<<"   ~~> t = "<<t++<<" | loglik = "<<loglik<<" | step = "<<step<<endl;
        step *= scale; 
        this->theta += step * delta_theta;
        this->update_exp_odd(x);
        loglik = this->neg_loglik_loss(x, y, weights, beta, coef0, A, g_index, g_size);
      }
      
      // fail to find
      if(!isfinite(loglik - loss0)) {
        this->theta = last_theta;
        return false;
      }
      
      // stop iter
      if (fabs(loglik - loss0) < this->primary_model_fit_epsilon) break;

      loss0 = loglik;
    }
    
    // update beta
    this->update_beta(beta, A);

    return true;
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, VL &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    cout<<" --> loss"<<endl;///

    int n = X.rows();
    int p = X.cols();

    VL w = weights.cast<long double>();

    ML log_par = (ML::Ones(n, p) + this->exp_odd).array().log();
    double loss = - ((log_par.rowwise().sum()).dot(w)) / this->ising_n - 
      this->gamma * (this->theta.cwiseProduct(this->theta).sum());
    
    return loss;
  };

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, VL &beta, VL &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, VL &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {
    cout<<" --> sacrifice"<<endl;///

    // backward
    for (int i = 0; i < A.size(); i++){
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);
      bd(A(i)) = fabs(this->theta(mi, mj));
    }

    // forward
    ML prob = 1.0 / (1.0 + this->exp_odd.array());
    ML first_der = first_derivative(X, weights, prob, A);

    for (int i = 0; i < I.size(); i++){
      int mi = this->map1(I(i), 0);
      int mj = this->map1(I(i), 1);
      bd(I(i)) = fabs(first_der(mi, mj));
    }

  };

  void update_exp_odd(T4 &x) {
    cout<<" --> update_exp_odd\n";///

    ML x1 = x.template cast<long double>();
    
    this->exp_odd = - 2.0 * (x1 * this->theta).cwiseProduct(x1);
    this->exp_odd = this->exp_odd.array().exp();
  }

  void update_beta(VL &beta, Eigen::VectorXi &A_ind){
    cout<<" --> update_beta\n";///
    for (int i = 0; i < A_ind.size(); i++){
      int mi = this->map1(A_ind(i), 0);
      int mj = this->map1(A_ind(i), 1);
      beta(i) = this->theta(mi, mj);
    }
  }

  ML compute_delta_theta(T4 &x,
                                      ML &first_der, 
                                      Eigen::VectorXd &weights, 
                                      ML &prob, Eigen::VectorXi &A) {
    // without penalty term
    // gradient of PL, not gradient of loss
    cout<<" --> compute_delta_theta\n";///
    int p = x.cols();
    ML delta_theta = ML::Zero(p, p);
    VL w = weights.cast<long double>();
    
    for(int i = 0; i < this->sparsity_level; i++) {
      // only for off-diagonal elements
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);

      VL ans = ( prob.col(mi).cwiseProduct(prob.col(mi)) - prob.col(mi) ) + 
        ( prob.col(mj).cwiseProduct(prob.col(mj)) - prob.col(mj) );
      
      long double second_der = (4.0 * ans.dot(w)) / this->ising_n;
      delta_theta(mi, mj) =  - (first_der(mi, mj) / second_der);
      delta_theta(mj, mi) = delta_theta(mi, mj);
    }
    return delta_theta;
  }

  ML first_derivative(T4 &x,
                                   Eigen::VectorXd &weights, 
                                   ML &prob, Eigen::VectorXi &A) {
    cout<<" --> first_derivative\n";///
    // gradient of PL, not gradient of loss
    int p = x.cols();
    ML first_der = ML::Zero(p, p);
    
    for(int i = 0; i < A.size(); i++) {
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);

      Eigen::VectorXd temp = x.col(mi).cwiseProduct(x.col(mj));
      VL xij = temp.cast<long double>();
      VL ans =  2.0 * xij - xij.cwiseProduct(prob.col(mi) + prob.col(mj));
      VL w = weights.cast<long double>();
      
      first_der(mi, mj) = (2.0 * ans.dot(w)) / this->ising_n - 
        4.0 * this->gamma * this->theta(mi, mj);
      first_der(mj, mi) = first_der(mi, mj);
    }

    return first_der;
  }
};

#endif // SRC_ALGORITHMISING_H