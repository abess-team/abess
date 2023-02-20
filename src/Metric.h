//
// Created by Jin Zhu on 2020/2/18.
//
// #define R_BUILD
#ifndef SRC_METRICS_H
#define SRC_METRICS_H

#ifdef R_BUILD
#include <Rcpp.h>
using namespace Rcpp;
#endif

#include <algorithm>
#include <random>
#include <vector>

#include "Algorithm.h"
#include "Data.h"
#include "utilities.h"

template <class T1, class T2, class T3, class T4>
// To do: calculate loss && all to one && lm poisson cox
class Metric {
   public:
    bool is_cv;
    int Kfold;
    int eval_type;
    double ic_coef;

    bool raise_warning = true;

    // Eigen::Matrix<T2, Dynamic, 1> cv_initial_model_param;
    // Eigen::Matrix<T3, Dynamic, 1> cv_initial_coef0;

    // std::vector<Eigen::VectorXi> cv_initial_A;
    // std::vector<Eigen::VectorXi> cv_initial_I;

    std::vector<Eigen::VectorXi> train_mask_list;
    std::vector<Eigen::VectorXi> test_mask_list;

    std::vector<T4> train_X_list;
    std::vector<T4> test_X_list;
    std::vector<T1> train_y_list;
    std::vector<T1> test_y_list;
    std::vector<Eigen::VectorXd> train_weight_list;
    std::vector<Eigen::VectorXd> test_weight_list;

    std::vector<FIT_ARG<T2, T3>> cv_init_fit_arg;

    // std::vector<std::vector<T4>> group_XTX_list;

    Metric() = default;

    Metric(int eval_type, double ic_coef = 1.0, int Kfold = 5) {
        this->is_cv = Kfold > 1;
        this->eval_type = eval_type;
        this->Kfold = Kfold;
        this->ic_coef = ic_coef;
        if (is_cv) {
            cv_init_fit_arg.resize(Kfold);
            train_X_list.resize(Kfold);
            test_X_list.resize(Kfold);
            train_y_list.resize(Kfold);
            test_y_list.resize(Kfold);
            test_weight_list.resize(Kfold);
            train_weight_list.resize(Kfold);
        }
    };

    void set_cv_init_fit_arg(int beta_size, int M) {
        for (int i = 0; i < this->Kfold; i++) {
            T2 beta_init;
            T3 coef0_init;
            coef_set_zero(beta_size, M, beta_init, coef0_init);
            Eigen::VectorXi A_init;
            Eigen::VectorXd bd_init;

            FIT_ARG<T2, T3> fit_arg(0, 0., beta_init, coef0_init, bd_init, A_init);

            cv_init_fit_arg[i] = fit_arg;
        }
    }

    // void set_cv_initial_model_param(int Kfold, int p)
    // {
    //   this->cv_initial_model_param = Eigen::MatrixXd::Zero(p, Kfold);
    // };

    // void set_cv_initial_A(int Kfold, int p)
    // {
    //   vector<Eigen::VectorXi> tmp(Kfold);
    //   this->cv_initial_A = tmp;
    // };

    // void set_cv_initial_coef0(int Kfold, int p)
    // {
    //   vector<double> tmp(Kfold);
    //   for (int i = 0; i < Kfold; i++)
    //     tmp[i] = 0;
    //   this->cv_initial_coef0 = tmp;
    // };

    // void update_cv_initial_model_param(Eigen::VectorXd model_param, int k)
    // {
    //   this->cv_initial_model_param.col(k) = model_param;
    // }

    // void update_cv_initial_A(Eigen::VectorXi A, int k)
    // {
    //   this->cv_initial_A[k] = A;
    // }

    // void update_cv_initial_coef0(double coef0, int k)
    // {
    //   this->cv_initial_coef0[k] = coef0;
    // }

    void set_cv_train_test_mask(Data<T1, T2, T3, T4> &data, int n, Eigen::VectorXi &cv_fold_id) {
        Eigen::VectorXi index_list(n);
        std::vector<int> index_vec((unsigned int)n);
        std::vector<Eigen::VectorXi> group_list((unsigned int)this->Kfold);
        for (int i = 0; i < n; i++) {
            index_vec[i] = i;
        }

        if (cv_fold_id.size() == 0) {
            // std::random_device rd;
            std::mt19937 g(123);
            std::shuffle(index_vec.begin(), index_vec.end(), g);

            for (int i = 0; i < n; i++) {
                index_list(i) = index_vec[i];
            }

            Eigen::VectorXd loss_list(this->Kfold);
            int group_size = int(n / this->Kfold);
            for (int k = 0; k < (this->Kfold - 1); k++) {
                group_list[k] = index_list.segment(int(k * group_size), group_size);
            }
            group_list[this->Kfold - 1] =
                index_list.segment(int((this->Kfold - 1) * group_size), n - int(int(this->Kfold - 1) * group_size));
        } else {
            // given cv_fold_id
            auto rule = [cv_fold_id](int i, int j) -> bool { return cv_fold_id(i) < cv_fold_id(j); };
            std::sort(index_vec.begin(), index_vec.end(), rule);

            for (int i = 0; i < n; i++) {
                index_list(i) = index_vec[i];
            }

            int k = 0, st = 0, ed = 1;
            while (k < this->Kfold && ed < n) {
                int mask = cv_fold_id(index_list(st));
                while (ed < n && mask == cv_fold_id(index_list(ed))) ed++;

                group_list[k] = index_list.segment(st, ed - st);
                st = ed;
                ed++;
                k++;
            }
        }
        for (int k = 0; k < this->Kfold; k++) {
            std::sort(group_list[k].data(), group_list[k].data() + group_list[k].size());
        }

        // cv train-test partition:
        std::vector<Eigen::VectorXi> train_mask_list_tmp((unsigned int)this->Kfold);
        std::vector<Eigen::VectorXi> test_mask_list_tmp((unsigned int)this->Kfold);
        for (int k = 0; k < this->Kfold; k++) {
            int train_x_size = n - group_list[k].size();
            // get train_mask
            Eigen::VectorXi train_mask(train_x_size);
            int i = 0;
            for (int j = 0; j < this->Kfold; j++) {
                if (j != k) {
                    for (int s = 0; s < group_list[j].size(); s++) {
                        train_mask(i) = group_list[j](s);
                        i++;
                    }
                }
            }
            std::sort(train_mask.data(), train_mask.data() + train_mask.size());
            train_mask_list_tmp[k] = train_mask;
            test_mask_list_tmp[k] = group_list[k];

            slice(data.x, train_mask, this->train_X_list[k]);
            slice(data.x, group_list[k], this->test_X_list[k]);
            slice(data.y, train_mask, this->train_y_list[k]);
            slice(data.y, group_list[k], this->test_y_list[k]);
            slice(data.weight, train_mask, this->train_weight_list[k]);
            slice(data.weight, group_list[k], this->test_weight_list[k]);
        }
        this->train_mask_list = train_mask_list_tmp;
        this->test_mask_list = test_mask_list_tmp;
    };

    // void cal_cv_group_XTX(Data<T1, T2, T3> &data)
    // {
    //   int p = data.p;
    //   Eigen::VectorXi index = data.g_index;
    //   Eigen::VectorXi gsize = data.g_size;
    //   int N = data.g_num;

    //   std::vector<std::vector<Eigen::MatrixXd>> group_XTX_list_tmp(this->Kfold);

    //   for (int k = 0; k < this->Kfold; k++)
    //   {
    //     int train_size = this->train_mask_list[k].size();
    //     Eigen::MatrixXd train_x(train_size, p);

    //     for (int i = 0; i < train_size; i++)
    //     {
    //       train_x.row(i) = data.x.row(this->train_mask_list[k](i));
    //     };
    //     group_XTX_list_tmp[k] = group_XTX(train_x, index, gsize, train_size, p, N, 1);
    //   }
    //   this->group_XTX_list = group_XTX_list_tmp;
    // }

    double ic(int train_n, int M, int N, Algorithm<T1, T2, T3, T4> *algorithm) {
        // information criterioin: for non-CV
        double loss;
        if (algorithm->model_type == 1 || algorithm->model_type == 5) {
            loss = train_n *
                   log(algorithm->get_train_loss() - algorithm->lambda_level * algorithm->beta.cwiseAbs2().sum());
        } else {
            loss = 2 * (algorithm->get_train_loss() - algorithm->lambda_level * algorithm->beta.cwiseAbs2().sum());
        }
        // 0. only loss
        if (this->eval_type == 0) {
            return loss;
        }
        // 1. AIC
        if (this->eval_type == 1) {
            return loss + 2.0 * algorithm->get_effective_number();
        }
        // 2. BIC
        if (this->eval_type == 2) {
            return loss + this->ic_coef * log(double(train_n)) * algorithm->get_effective_number();
        }
        // 3. GIC
        if (this->eval_type == 3) {
            return loss +
                   this->ic_coef * log(double(N)) * log(log(double(train_n))) * algorithm->get_effective_number();
        }
        // 4. EBIC
        if (this->eval_type == 4) {
            return loss +
                   this->ic_coef * (log(double(train_n)) + 2 * log(double(N))) * algorithm->get_effective_number();
        }
        // 5. HIC
        if (this->eval_type == 5) {
            return train_n *
                       (algorithm->get_train_loss() - algorithm->lambda_level * algorithm->beta.cwiseAbs2().sum()) +
                   this->ic_coef * log(double(N)) * log(log(double(train_n))) * algorithm->get_effective_number();
        }
        if (this->raise_warning) {
#ifdef R_BUILD
            Rcout << "[warning] No available IC type for training. Use loss instead. "
                  << "(E" << this->eval_type << "M" << algorithm->model_type << ")" << endl;
#else
            cout << "[warning] No available IC type for training. Use loss instead. "
                 << "(E" << this->eval_type << "M" << algorithm->model_type << ")" << endl;
#endif
            this->raise_warning = false;
        }
        // return 0;
        return loss;
    };

    double test_loss(T4 &test_x, T1 &test_y, Eigen::VectorXd &test_weight, Eigen::VectorXi &g_index,
                     Eigen::VectorXi &g_size, int test_n, int p, int N, Algorithm<T1, T2, T3, T4> *algorithm) {
        // test loss: for CV
        Eigen::VectorXi A = algorithm->get_A_out();
        T2 beta = algorithm->get_beta();
        T3 coef0 = algorithm->get_coef0();

        Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, beta.rows(), N);
        T4 test_X_A = X_seg(test_x, test_n, A_ind, algorithm->model_type);

        T2 beta_A;
        slice(beta, A_ind, beta_A);

        // 0. only test loss
        if (this->eval_type == 0) {
            return algorithm->loss_function(test_X_A, test_y, test_weight, beta_A, coef0, A, g_index, g_size,
                                            algorithm->lambda_level);
        }
        // 1. negative AUC (for logistic)
        if (this->eval_type == 1 && algorithm->model_type == 2) {
            // compute probability
            Eigen::VectorXd test_y_temp = test_y;
            Eigen::VectorXd proba = test_X_A * beta_A + coef0 * Eigen::VectorXd::Ones(test_n);
            proba = proba.array().exp();
            proba = proba.cwiseQuotient(Eigen::VectorXd::Ones(test_n) + proba);
            return -this->binary_auc_score(test_y_temp, proba);
        }
        // 2. 3. negative AUC, One vs One/Rest (for multinomial)
        if (algorithm->model_type == 6) {
            int M = test_y.cols();
            // compute probability
            Eigen::MatrixXd proba = test_X_A * beta_A;
            proba = rowwise_add(proba, coef0);
            proba = proba.array().exp();
            Eigen::VectorXd proba_rowsum = proba.rowwise().sum();
            proba = proba.cwiseQuotient(proba_rowsum.replicate(1, p));
            // compute AUC
            if (this->eval_type == 2) {
                // (One vs One) the AUC of all possible pairwise combinations of classes
                double auc = 0;
                for (int i = 0; i < M - 1; i++) {
                    for (int j = i + 1; j < M; j++) {
                        int nij = 0;
                        Eigen::VectorXd test_y_i(test_n), test_y_j(test_n), proba_i(test_n), proba_j(test_n);
                        // extract samples who belongs to class i or j
                        for (int k = 0; k < test_n; k++) {
                            if (test_y(k, i) + test_y(k, j) > 0) {
                                test_y_i(nij) = test_y(k, i);
                                test_y_j(nij) = test_y(k, j);
                                proba_i(nij) = proba(k, i);
                                proba_j(nij) = proba(k, j);
                                nij++;
                            }
                        }
                        test_y_i = test_y_i.head(nij).eval();
                        test_y_j = test_y_j.head(nij).eval();
                        proba_i = proba_i.head(nij).eval();
                        proba_j = proba_j.head(nij).eval();
                        // get AUC
                        auc += this->binary_auc_score(test_y_i, proba_i);
                        auc += this->binary_auc_score(test_y_j, proba_j);
                    }
                }
                return -auc / (p * (p - 1));
            }
            if (this->eval_type == 3) {
                // (One vs Rest) the AUC of each class against the rest
                double auc = 0;
                for (int i = 0; i < M; i++) {
                    Eigen::VectorXd test_y_single = test_y.col(i);
                    Eigen::VectorXd proba_single = proba.col(i);
                    auc += this->binary_auc_score(test_y_single, proba_single);
                }
                return -auc / p;
            }
        }
        if (this->raise_warning) {
#ifdef R_BUILD
            Rcout << "[warning] No available CV score for training. Use test_loss instead. "
                  << "(E" << this->eval_type << "M" << algorithm->model_type << ")" << endl;
#else
            cout << "[warning] No available CV score for training. Use test_loss instead. "
                 << "(E" << this->eval_type << "M" << algorithm->model_type << ")" << endl;
#endif
            this->raise_warning = false;
        }
        // return 0;
        return algorithm->loss_function(test_X_A, test_y, test_weight, beta_A, coef0, A, g_index, g_size,
                                        algorithm->lambda_level);
    };

    double binary_auc_score(Eigen::VectorXd &true_label, Eigen::VectorXd &pred_proba) {
        // sort proba from large to small
        int n = true_label.rows();
        Eigen::VectorXi sort_ind = max_k(pred_proba, n, true);

        // use each value as threshold to get TPR, FPR
        double tp = 0, fp = 0, positive = true_label.sum();
        double last_tpr = 0, last_fpr = 0, auc = 0;

        if (positive == 0 || positive == n) {
#ifdef R_BUILD
            Rcout << "[Warning] There is only one class in the test data, "
                  << "the result may be meaningless. Please use another type of loss, "
                  << "or try to specify cv_fold_id." << endl;
#else
            cout << "[Warning] There is only one class in the test data, "
                 << "the result may be meaningless. Please use another type of loss, "
                 << "or try to specify cv_fold_id." << endl;
#endif
        }

        for (int i = 0; i < n; i++) {
            // current threshold: pred_proba(sort_ind(i))
            int k = sort_ind(i);
            tp += true_label(k);
            fp += 1 - true_label(k);
            // skip same threshold
            if (i < n - 1) {
                int kk = sort_ind(i + 1);
                if (pred_proba(k) == pred_proba(kk)) continue;
            }
            // compute tpr, fpr
            double tpr = tp / positive;
            double fpr = fp / (n - positive);
            if (fpr > last_fpr) {
                auc += (tpr + last_tpr) / 2 * (fpr - last_fpr);
            }
            last_tpr = tpr;
            last_fpr = fpr;
        }
        return auc;
    };

    // to do
    Eigen::VectorXd fit_and_evaluate_in_metric(std::vector<Algorithm<T1, T2, T3, T4> *> algorithm_list,
                                               Data<T1, T2, T3, T4> &data, FIT_ARG<T2, T3> &fit_arg) {
        Eigen::VectorXd loss_list(this->Kfold);

        if (!is_cv) {
            algorithm_list[0]->update_sparsity_level(fit_arg.support_size);
            algorithm_list[0]->update_lambda_level(fit_arg.lambda);
            algorithm_list[0]->update_beta_init(fit_arg.beta_init);
            algorithm_list[0]->update_bd_init(fit_arg.bd_init);
            algorithm_list[0]->update_coef0_init(fit_arg.coef0_init);
            algorithm_list[0]->update_A_init(fit_arg.A_init, data.g_num);

            algorithm_list[0]->fit(data.x, data.y, data.weight, data.g_index, data.g_size, data.n, data.p, data.g_num);

            if (algorithm_list[0]->get_warm_start()) {
                fit_arg.beta_init = algorithm_list[0]->get_beta();
                fit_arg.coef0_init = algorithm_list[0]->get_coef0();
                fit_arg.bd_init = algorithm_list[0]->get_bd();
            }

            loss_list(0) = this->ic(data.n, data.M, data.g_num, algorithm_list[0]);
        } else {
            Eigen::VectorXi g_index = data.g_index;
            Eigen::VectorXi g_size = data.g_size;
            int p = data.p;
            int N = data.g_num;

#pragma omp parallel for
            // parallel
            for (int k = 0; k < this->Kfold; k++) {
                // get test_x, test_y
                int test_n = this->test_mask_list[k].size();
                int train_n = this->train_mask_list[k].size();

                // train & test data
                // Eigen::MatrixXd train_x = matrix_slice(data.x, this->train_mask_list[k], 0);
                // Eigen::MatrixXd test_x = matrix_slice(data.x, this->test_mask_list[k], 0);
                // Eigen::VectorXd train_y = vector_slice(data.y, this->train_mask_list[k]);
                // Eigen::VectorXd test_y = vector_slice(data.y, this->test_mask_list[k]);
                // Eigen::VectorXd train_weight = vector_slice(data.weight, this->train_mask_list[k]);
                // Eigen::VectorXd test_weight = vector_slice(data.weight, this->test_mask_list[k]);

                // Eigen::VectorXd beta_init;
                algorithm_list[k]->update_sparsity_level(fit_arg.support_size);
                algorithm_list[k]->update_lambda_level(fit_arg.lambda);

                algorithm_list[k]->update_beta_init(this->cv_init_fit_arg[k].beta_init);
                algorithm_list[k]->update_bd_init(this->cv_init_fit_arg[k].bd_init);
                algorithm_list[k]->update_coef0_init(this->cv_init_fit_arg[k].coef0_init);
                algorithm_list[k]->update_A_init(this->cv_init_fit_arg[k].A_init, N);
                // beta_init = this->cv_initial_model_param.col(k).eval();
                // algorithm->update_beta_init(beta_init);
                // algorithm->update_coef0_init(this->cv_initial_coef0[k]);
                // algorithm->update_A_init(this->cv_initial_A[k], N);
                // algorithm->update_train_mask(this->train_mask_list[k]);
                // ??????????????????????????????????????????????????????????????
                algorithm_list[k]->fit(this->train_X_list[k], this->train_y_list[k], this->train_weight_list[k],
                                       g_index, g_size, train_n, p, N);

                if (algorithm_list[k]->get_warm_start()) {
                    this->cv_init_fit_arg[k].beta_init = algorithm_list[k]->get_beta();
                    this->cv_init_fit_arg[k].coef0_init = algorithm_list[k]->get_coef0();
                    this->cv_init_fit_arg[k].bd_init = algorithm_list[k]->get_bd();
                    // this->update_cv_initial_model_param(algorithm->get_beta(), k);
                    // this->update_cv_initial_A(algorithm->get_A_out(), k);
                    // this->update_cv_initial_coef0(algorithm->get_coef0(), k);
                }

                loss_list(k) = this->test_loss(this->test_X_list[k], this->test_y_list[k], this->test_weight_list[k],
                                               g_index, g_size, test_n, p, N, algorithm_list[k]);
            }
        }

        return loss_list;
    };
};

#endif  // SRC_METRICS_H
