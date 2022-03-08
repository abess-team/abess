//
// Created by Jin Zhu on 2020/2/18.
//
// #define R_BUILD
#ifndef SRC_METRICS_H
#define SRC_METRICS_H

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
    int ic_type;
    // Eigen::Matrix<T2, Dynamic, 1> cv_initial_model_param;
    // Eigen::Matrix<T3, Dynamic, 1> cv_initial_coef0;

    std::vector<Eigen::VectorXi> cv_initial_A;
    std::vector<Eigen::VectorXi> cv_initial_I;

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

    double ic_coef;

    Metric() = default;

    Metric(int ic_type, double ic_coef = 1.0, int Kfold = 5) {
        this->is_cv = Kfold > 1;
        this->ic_type = ic_type;
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
        double loss;
        if (algorithm->model_type == 1 || algorithm->model_type == 5) {
            loss = train_n *
                   log(algorithm->get_train_loss() - algorithm->lambda_level * algorithm->beta.cwiseAbs2().sum());
        } else {
            loss = 2 * (algorithm->get_train_loss() - algorithm->lambda_level * algorithm->beta.cwiseAbs2().sum());
        }

        if (ic_type == 0) {
            return loss;
        } else if (ic_type == 1) {
            return loss + 2.0 * algorithm->get_effective_number();
        } else if (ic_type == 2) {
            return loss + this->ic_coef * log(double(train_n)) * algorithm->get_effective_number();
        } else if (ic_type == 3) {
            return loss +
                   this->ic_coef * log(double(N)) * log(log(double(train_n))) * algorithm->get_effective_number();
        } else if (ic_type == 4) {
            return loss +
                   this->ic_coef * (log(double(train_n)) + 2 * log(double(N))) * algorithm->get_effective_number();
        } else
            return 0;
    };

    double loss_function(T4 &train_x, T1 &train_y, Eigen::VectorXd &train_weight, Eigen::VectorXi &g_index,
                         Eigen::VectorXi &g_size, int train_n, int p, int N, Algorithm<T1, T2, T3, T4> *algorithm) {
        Eigen::VectorXi A = algorithm->get_A_out();
        T2 beta = algorithm->get_beta();
        T3 coef0 = algorithm->get_coef0();

        Eigen::VectorXi A_ind = find_ind(A, g_index, g_size, beta.rows(), N);
        T4 X_A = X_seg(train_x, train_n, A_ind, algorithm->model_type);

        T2 beta_A;
        slice(beta, A_ind, beta_A);

        // Eigen::VectorXd beta_A(A_ind.size());
        // for (int k = 0; k < A_ind.size(); k++)
        // {
        //   beta_A(k) = beta(A_ind(k));
        // }
        return algorithm->loss_function(X_A, train_y, train_weight, beta_A, coef0, A, g_index, g_size, 0.0);
    }

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

                loss_list(k) =
                    this->loss_function(this->test_X_list[k], this->test_y_list[k], this->test_weight_list[k], g_index,
                                        g_size, test_n, p, N, algorithm_list[k]);
            }
        }

        return loss_list;
    };
};

#endif  // SRC_METRICS_H
