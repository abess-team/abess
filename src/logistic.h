#ifndef logistic_H
#define logistic_H

Eigen::VectorXd pi(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd coef, int n);

Eigen::VectorXd logit_fit(Eigen::MatrixXd x, Eigen::VectorXd y, int n, int p, Eigen::VectorXd weights);

double loglik_logit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd coef, int n, Eigen::VectorXd weights);

#endif
