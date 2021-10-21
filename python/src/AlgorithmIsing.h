#ifndef SRC_ALGORITHMISING_H
#define SRC_ALGORITHMISING_H

#include "Algorithm.h"
#include<cmath>

using VL = Eigen::Matrix<long double, Eigen::Dynamic, 1>;
using ML = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;

template <class T4>
class abessIsing : public Algorithm<Eigen::VectorXd, VL, double, T4, VL>
{
public:
  abessIsing(int algorithm_type, int model_type, int max_iter = 30, int primary_model_fit_max_iter = 10, double primary_model_fit_epsilon = 1e-8, bool warm_start = true, int exchange_num = 5, bool approximate_Newton = false, Eigen::VectorXi always_select = Eigen::VectorXi::Zero(0), int splicing_type = 1, int sub_search = 0) : Algorithm<Eigen::VectorXd, VL, double, T4, VL>::Algorithm(algorithm_type, model_type, max_iter, primary_model_fit_max_iter, primary_model_fit_epsilon, warm_start, exchange_num, approximate_Newton, always_select, false, splicing_type, sub_search){};

  ~abessIsing(){};

  // ML Xij;
  // Eigen::VectorXi Xij_flag;

  void update_tau(int train_n, int N)
  {
    // cout<<"init tau"<<endl;
    this->tau = min(1e-5, 0.1 / this->ising_n);
  }

  // ML thetaA(Eigen::VectorXi &ind, int p)
  // {
  //   if (ind.size() == p){
  //     return this->theta;
  //   }
  //   int len = ind.size();
  //   ML tA(len, len);
  //   for (int i = 0; i < len; i++)
  //     for (int j = 0; j < i + 1; j++)
  //     {
  //       tA(i, j) = this->theta(ind(i), ind(j));
  //       tA(j, i) = this->theta(ind(j), ind(i));
  //     }
  //   return tA;
  // }

  Eigen::VectorXi inital_screening(T4 &X, Eigen::VectorXd &y, VL &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, VL &bd, Eigen::VectorXd &weights,
                                   Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int &N)
  {
    // cout<<"==> init | N = "<<N<<" | Asize = "<<A.size()<<endl;

    if (bd.size() == 0)
    {
      // chisq
      int p = X.cols();
      int n = X.rows();
      bd = VL::Zero(N);

      ML chisq_matrix = ML::Zero(p, p);
      VL chisq_res = VL::Zero(N);

      for (int i = 1; i < p; i++){
        for (int j = 0; j < i; j++){
          //chisq test

          ML obs = ML::Zero(2, 2);
          for (int k = 0; k < n; k++){
            double x1 = X.coeff(k, i), x2 = X.coeff(k, j);
            if (x1 == -1){
              if (x2 == -1) obs(0, 0) += (long double) weights(k);
                else obs(0, 1) += (long double) weights(k);
            }else{
              if (x2 == -1) obs(1, 0) += (long double) weights(k);
                else obs(1, 1) += (long double) weights(k);
            }
          }
          
          ML exp = ML::Zero(2, 2);
          exp(0, 0) = (obs(0, 0) + obs(1, 0)) * (obs(0, 0) + obs(0, 1));
          exp(0, 1) = (obs(0, 1) + obs(1, 1)) * (obs(0, 0) + obs(0, 1));
          exp(1, 0) = (obs(0, 0) + obs(1, 0)) * (obs(1, 0) + obs(1, 1));
          exp(1, 1) = (obs(0, 1) + obs(1, 1)) * (obs(1, 0) + obs(1, 1));
          exp = exp / this->ising_n;

          // chisq_res -> bd
          int ind = this->map2(i, j);
          bd(ind) = pow(obs(0, 0) - exp(0, 0), 2) / exp(0, 0);
          bd(ind) += pow(obs(0, 1) - exp(0, 1), 2) / exp(0, 1);
          bd(ind) += pow(obs(1, 0) - exp(1, 0), 2) / exp(1, 0);
          bd(ind) += pow(obs(1, 1) - exp(1, 1), 2) / exp(1, 1);
        }
      }

      for (int i = 0; i < this->always_select.size(); i++)
      {
        bd(this->always_select(i)) = DBL_MAX;
      }

      // this->Xij = ML::Zero(n, N + p);
      // this->Xij_flag = Eigen::VectorXi::Zero(N + p);
    }

    // get Active-set A according to max_k bd
    Eigen::VectorXi A_new = max_k(bd, this->sparsity_level);

    // cout<<" A_init = ";///
    // for (int i=0;i<A_new.size();i++) cout<<this->map1(A_new(i), 0)<<","<<this->map1(A_new(i), 1)<<endl;cout<<endl;

    // cout<<" chisq = ";///
    // for (int i=0;i<bd.size();i++) cout<<bd(i)<<"("<<this->map1(i, 0)<<","<<this->map1(i, 1)<<") "<<endl;cout<<endl;

    return A_new;
  }

  bool primary_model_fit(T4 &x, Eigen::VectorXd &y, Eigen::VectorXd &weights, VL &beta, double &coef0, double loss0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    // cout<<" --> primary fit | beta size = "<<beta.size()<<endl;///
    // inner_loop3

    ML theta = this->set_theta(beta, A, x.cols());
    ML prob, first_der, delta_theta, last_theta, exp_odd;
    long double l0, l1 = loss0 + 1, step = 1.0, alpha = 0.1, scale = 0.5;

    l0 = (long double) this->neg_loglik_loss(x, y, weights, beta, coef0, A, g_index, g_size);

    // Newton
    int iter = 0;
    while (iter++ <= this->primary_model_fit_max_iter) {
      // cout << "   ~~> iter = "<<iter<<" | loss = "<<l0<<endl;///

      last_theta = theta;

      exp_odd = this->compute_exp_odd(x, theta);
      prob = 1.0 / (1.0 + exp_odd.array());
      first_der = compute_first_derivative(x, weights, theta, prob, A);
      delta_theta = compute_delta_theta(x, first_der, weights, prob, A);
      
      // step = 1.0;
      step *= 2;
      theta += step * delta_theta;

      VL beta_temp = this->set_beta(theta, A);
      l1 = (long double) this->neg_loglik_loss(x, y, weights, beta_temp, coef0, A, g_index, g_size);

      int c=0;
      while (step > 0 && l1 >= l0 - alpha * step * (first_der.cwiseProduct(delta_theta)).sum()) {
        c++;
        step *= scale; 
        theta = last_theta + step * delta_theta;
        beta_temp = this->set_beta(theta, A);
        l1 = (long double) this->neg_loglik_loss(x, y, weights, beta_temp, coef0, A, g_index, g_size);
      }
      
      bool condition1 = l1 - (this->primary_model_fit_max_iter - iter - 1) * (l0 - l1) + this->tau > loss0;
      if(condition1 || (!isfinite(l0 - l1))) {
        return false;
      }

      // stop iter
      // this->primary_model_fit_epsilon = this->tau * 0.1;
      if (l0 - l1 < this->tau * 0.1) break;
      
      l0 = l1;

    }
    // cout << " --> primary fit loss = "<<l0<<" | iter = "<<iter<<endl;///

    if (l1 > loss0){
      return false;
    }else{
      // update beta
      beta = this->set_beta(theta, A);
      return true;
    }
    
  };

  double neg_loglik_loss(T4 &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, VL &beta, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size)
  {
    // cout<<" --> loss = ";///

    int n = X.rows();
    int p = X.cols();

    VL w = weights.cast<long double>();

    ML theta = this->set_theta(beta, A, p);
    ML exp_odd = this->compute_exp_odd(X, theta);
    ML log_par = (ML::Ones(n, p) + exp_odd).array().log();
    // cout << "log par:\n";///
    // for (int i=0;i<log_par.rows();i++){
    //   for (int j=0;j<log_par.cols();j++)
    //     cout<<log_par(i,j)<<" ";
    //   cout<<endl;
    // }
    long double loglik = - ((log_par.rowwise().sum()).dot(w)) / this->ising_n - 
      (long double) this->lambda_level * (theta.cwiseProduct(theta).sum());
    
    // cout << loss << endl;
    return - loglik;
  };

  void sacrifice(T4 &X, T4 &XA, Eigen::VectorXd &y, VL &beta, VL &beta_A, double &coef0, Eigen::VectorXi &A, Eigen::VectorXi &I, Eigen::VectorXd &weights, Eigen::VectorXi &g_index, Eigen::VectorXi &g_size, int N, Eigen::VectorXi &A_ind, VL &bd, Eigen::VectorXi &U, Eigen::VectorXi &U_ind, int num)
  {
    // cout<<" --> sacrifice : \n";///

    ML theta = this->set_theta(beta_A, A, X.cols());
    ML exp_odd = this->compute_exp_odd(X, theta);

    // backward
    for (int i = 0; i < A.size(); i++){
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);
      bd(A(i)) = fabs(theta(mi, mj));
    }

    // forward
    ML prob = 1.0 / (1.0 + exp_odd.array());
    ML first_der = compute_first_derivative(X, weights, theta, prob, I);

    for (int i = 0; i < I.size(); i++){
      int mi = this->map1(I(i), 0);
      int mj = this->map1(I(i), 1);
      bd(I(i)) = fabs(first_der(mi, mj));
    }

    // for (int i=0;i<bd.size();i++) cout<<bd(i)<<"("<<this->map1(i, 0)<<","<<this->map1(i, 1)<<") ";cout<<endl;///
  };


  VL set_beta(ML &theta, Eigen::VectorXi &A){

    VL beta = VL::Zero(A.size());
    for (int i = 0; i < A.size(); i++){
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);
      if (mi == mj) continue; // keep diag zero
      beta(i) = theta(mi, mj);
    }
    return beta;
    // for (int i = 0; i < beta.size(); i++) cout<<beta(i)<<" ";cout<<endl;
  }

  ML set_theta(VL &beta, Eigen::VectorXi &A, int p){

    ML theta = ML::Zero(p, p);
    for (int i = 0; i < A.size(); i++){
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);
      if (mi == mj) continue; // keep diag zero
      theta(mi, mj) = beta(i);
      theta(mj, mi) = beta(i);
    }
    return theta;
  }

  ML compute_exp_odd(T4 &x, ML &theta) {
    // cout<<" --> update_exp_odd\n";///

    ML x1 = x.template cast<long double>();

    ML odd = - 2.0 * (x1 * theta).cwiseProduct(x1);
    return odd.array().exp();
  }

  ML compute_delta_theta(T4 &x, ML &first_der, 
                         Eigen::VectorXd &weights, 
                         ML &prob, Eigen::VectorXi &A) {
    // without penalty term
    // gradient of PL, not gradient of loss
    // cout<<" --> compute_delta_theta\n";///
    int p = x.cols();
    ML delta_theta = ML::Zero(p, p);
    VL w = weights.cast<long double>();
    
    for (int i = 0; i < A.size(); i++) {
      // only for off-diagonal elements
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);
      if (mi == mj) continue;

      VL ans = ( prob.col(mi).cwiseProduct(prob.col(mi)) - prob.col(mi) ) + 
        ( prob.col(mj).cwiseProduct(prob.col(mj)) - prob.col(mj) );
      
      long double second_der =  (long double) (ans.dot(w) * 4.0) /  (long double) this->ising_n;
      delta_theta(mi, mj) =  -  (long double) (first_der(mi, mj) / second_der);
      delta_theta(mj, mi) = delta_theta(mi, mj);
    }
    return delta_theta;
  }

  ML compute_first_derivative(T4 &x, Eigen::VectorXd &weights, ML &theta,
                      ML &prob, Eigen::VectorXi &A) {
    // cout<<" --> first_derivative\n";///
    // gradient of PL, not gradient of loss
    int p = x.cols();
    ML first_der = ML::Zero(p, p);
    VL w = weights.cast<long double>();
    ML x1 = x.template cast<long double>();
    
    for (int i = 0; i < A.size(); i++) {
      int mi = this->map1(A(i), 0);
      int mj = this->map1(A(i), 1);
      if (mi == mj) continue;
      
      // VL xij = compute_Xij(x1, mi, mj);
      VL xij = x1.col(mi).cwiseProduct(x1.col(mj));
      VL ans =  xij * 2.0 - xij.cwiseProduct(prob.col(mi) + prob.col(mj));
      
      first_der(mi, mj) =  (long double) (ans.dot(w) * 2.0) /  (long double) this->ising_n - 
        this->lambda_level * 4.0 * theta(mi, mj);
      first_der(mj, mi) = first_der(mi, mj);
    }

    return first_der;
  }

  // VL compute_Xij(ML &x, int i, int j){
  //   int N = (int) x.cols() * (x.cols() - 1) / 2;
  //   int ind = (i != j) ? (this->map2(i, j)) : (N + i);

  //   if (this->Xij_flag(ind) == 0){
  //     this->Xij.col(ind) = x.col(i).cwiseProduct(x.col(j));
  //     this->Xij_flag(ind) = 1;
  //   }

  //   return this->Xij.col(ind);
  // }
};

#endif // SRC_ALGORITHMISING_H