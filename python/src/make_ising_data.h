#ifndef SRC_MAKE_ISING_DATA_H
#define SRC_MAKE_ISING_DATA_H

#include <Eigen/Eigen>

Eigen::MatrixXd comp_conf(int num_conf, int p);
Eigen::MatrixXd sample_by_conf(long long n, Eigen::MatrixXd theta, int seed);
void iteration(Eigen::VectorXd &sample, Eigen::MatrixXd &theta,
               Eigen::VectorXd &value, int set_seed, int iter_time);
Eigen::MatrixXd Ising_Gibbs(Eigen::MatrixXd theta, int n_sample, int burn, int skip,
                            Eigen::VectorXd value, bool using_seed = false, int set_seed = 1);

#ifndef R_BUILD

void ising_sample_by_conf_wrap(long long n, double *theta, int theta_row, int theta_col, 
                               int seed_train, int seed_valid,
                               double *train_out, int train_out_len,
                               double *valid_out, int valid_out_len);

void ising_gibbs_wrap(double *theta, int theta_row, int theta_col, 
                      int n_sample, int burn, int skip,
                      double *value, int value_len, bool using_seed, int set_seed,
                      double *data_out, int data_out_len);

#endif //R_BUILD
#endif //SRC_MAKE_ISING_DATA_H