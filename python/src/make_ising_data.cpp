
#ifdef R_BUILD
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::interfaces(r,cpp)]]
#include <Rcpp.h>
// [[Rcpp::plugins(cpp11)]]
#include <RcppEigen.h>
using namespace Rcpp;
#else
#include <Eigen/Eigen>
#endif

#include <iostream>
#include <random>
#include <vector>
#include "utilities.h"

using namespace std;

Eigen::MatrixXd comp_conf(int num_conf, int p){
  Eigen::MatrixXi conf = Eigen::MatrixXi::Zero(num_conf, p);
  Eigen::VectorXi num = Eigen::VectorXi::LinSpaced(num_conf, 0, num_conf - 1);

  for (int i = p - 1; i >= 0; i--){
    conf.col(i) = num - num / 2 * 2;
    num /= 2;
  }
  conf = conf.array() * 2 - 1;
  return conf.cast<double>();
}

// [[Rcpp::export]]
Eigen::MatrixXd sample_by_conf(long long n, Eigen::MatrixXd theta, int seed) {
  int p = theta.rows();
  int num_conf = pow(2, p);
  
  Eigen::MatrixXd table = comp_conf(num_conf, p);
  Eigen::VectorXd weight(num_conf);
  
  Eigen::VectorXd vec_diag = theta.diagonal();
  Eigen::MatrixXd theta_diag = vec_diag.asDiagonal();
  Eigen::MatrixXd theta_off = theta - theta_diag;
  
  for (int num = 0; num < num_conf; num++) {
    Eigen::VectorXd conf = table.row(num);
    weight(num) = 0.5 * (double) (conf.transpose() * theta_off * conf) + (double) (vec_diag.transpose() * conf);
  }
  weight = weight.array().exp();
  
  std::vector<double> w;
  w.resize(weight.size());
  Eigen::VectorXd::Map(&w[0], weight.size()) = weight;
  
  // int sd = (((long long int)time(0)) * 2718) % 314159265;
  // Rcout << "Seed: "<< sd << endl;
  std::default_random_engine generator(seed);
  // std::default_random_engine generator(1);
  std::discrete_distribution<int> distribution(std::begin(w), std::end(w));
  
  Eigen::VectorXd freq = Eigen::VectorXd::Zero(num_conf);
  
  for (long long int i = 0; i < n; i++) {
    int num = distribution(generator);
    freq(num)++;
  }
  
  Eigen::MatrixXd data(num_conf, p + 1);
  data.col(0) = freq;
  data.rightCols(p) = table;
  return data;
}

// The following is for Gibbs sampling
void iteration(Eigen::VectorXd &sample, Eigen::MatrixXd &theta,
               Eigen::VectorXd &value, int set_seed, int iter_time) {
  double foo, prob_v1, v1 = value(0), v2 = value(1);
  int p = sample.size();

  while (iter_time-- > 0){
    static std::default_random_engine generator(set_seed);
    
    for (int i = 0; i < p; i++) {
      sample(i) = v2 - v1;
      foo = sample(i) * (sample.dot(theta.row(i)) - theta(i, i) * sample(i) + theta(i, i));
      prob_v1 = 1 / (1.0 + exp(foo));
      std::bernoulli_distribution distribution(prob_v1);
      sample(i) = (v1 - v2) * (int)(distribution(generator)) + v2;
    }
    // return sample;
  }
}

// [[Rcpp::export]]
Eigen::MatrixXd Ising_Gibbs(Eigen::MatrixXd theta, int n_sample, int burn, int skip,
                            Eigen::VectorXd value, bool using_seed = false, int set_seed = 1) {
  
  if (using_seed == false){
    set_seed = (((long long)time(0)) * 2718) % 314159265;
  } 
  int p = theta.cols();
  Eigen::MatrixXd data = Eigen::MatrixXd::Zero(n_sample, p);
  Eigen::VectorXd sample = Eigen::VectorXd::Zero(p);

  // gendata
  iteration(sample, theta, value, set_seed, burn);
  data.row(0) = sample;

  for (int i = 1; i < n_sample; i++) {
    iteration(sample, theta, value, set_seed, skip);
    data.row(i) = sample;
  }
  return data;
}

#ifndef R_BUILD

void ising_sample_by_conf_wrap(long long n, double *theta, int theta_row, int theta_col, 
                               int seed_train, int seed_valid,
                               double *train_out, int train_out_len,
                               double *valid_out, int valid_out_len){
  int num_conf = pow(2, theta_col);                               
  Eigen::MatrixXd theta_Mat(theta_row, theta_col);
  Eigen::MatrixXd train_out_Mat(num_conf, theta_col + 1);
  Eigen::MatrixXd valid_out_Mat(num_conf, theta_col + 1);

  theta_Mat = Pointer2MatrixXd(theta, theta_row, theta_col);

  // cout<<"cpp theta: "<<endl;///
  // for (int i=0;i<theta_Mat.rows();i++){
  //   for (int j=0;j<theta_Mat.cols();j++) 
  //     cout<<theta_Mat(i,j)<<" ";cout<<endl;
  // }

  train_out_Mat = sample_by_conf(n, theta_Mat, seed_train);
  // cout<<"cpp: "<<endl;///
  // for (int i=0;i<10;i++){
  //   for (int j=0;j<train_out_Mat.cols();j++) 
  //     cout<<train_out_Mat(i,j)<<" ";cout<<endl;
  // }
  valid_out_Mat = sample_by_conf(n, theta_Mat, seed_valid);

  MatrixXd2Pointer(train_out_Mat, train_out);
  MatrixXd2Pointer(valid_out_Mat, valid_out);
}

void ising_gibbs_wrap(double *theta, int theta_row, int theta_col, 
                      int n_sample, int burn, int skip,
                      double *value, int value_len, bool using_seed, int set_seed,
                      double *data_out, int data_out_len){
  Eigen::MatrixXd theta_Mat(theta_row, theta_col);
  Eigen::VectorXd value_Vec(value_len);
  Eigen::MatrixXd data_out_Mat(2 * n_sample, theta_col);

  theta_Mat = Pointer2MatrixXd(theta, theta_row, theta_col);
  value_Vec = Pointer2VectorXd(value, value_len);

  data_out_Mat = Ising_Gibbs(theta_Mat, 2 * n_sample, burn, skip, value_Vec, using_seed, set_seed);

  MatrixXd2Pointer(data_out_Mat, data_out);
}

#endif // R_BUILD