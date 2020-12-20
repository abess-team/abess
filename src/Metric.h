//
// Created by Mamba on 2020/2/18.
//
// #define R_BUILD
#ifndef SRC_METRICS_H
#define SRC_METRICS_H

#include "Data.h"
#include "Algorithm.h"
#include "coxph.h"
#include <vector>
#include <random>
#include <algorithm>
#include "utilities.h"

class Metric
{
public:
    bool is_cv;
    int K;
    int ic_type;
    Eigen::MatrixXd cv_initial_model_param;
    std::vector<Eigen::VectorXi> train_mask_list;
    std::vector<Eigen::VectorXi> test_mask_list;

    std::vector<std::vector<Eigen::MatrixXd>> group_XTX_list;

    Metric() = default;

    Metric(int ic_type, bool is_cv, int K = 0)
    {
        this->is_cv = is_cv;
        this->ic_type = ic_type;
        this->K = K;
    };

    void set_cv_initial_model_param(int K, int p)
    {
        this->cv_initial_model_param = Eigen::MatrixXd::Zero(K, p);
    };

    void update_cv_initial_model_param(Eigen::VectorXd model_param, int k)
    {
        this->cv_initial_model_param.row(k) = model_param;
    }

    void set_cv_train_test_mask(int n)
    {
        Eigen::VectorXi index_list(n);
        std::vector<int> index_vec((unsigned int)n);
        for (int i = 0; i < n; i++)
        {
            index_vec[i] = i;
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(index_vec.begin(), index_vec.end(), g);

        for (int i = 0; i < n; i++)
        {
            index_list(i) = index_vec[i];
        }

        Eigen::VectorXd loss_list(this->K);
        std::vector<Eigen::VectorXi> group_list((unsigned int)this->K);
        int group_size = int(n / this->K);
        for (int k = 0; k < (this->K - 1); k++)
        {
            group_list[k] = index_list.segment(int(k * group_size), group_size);
        }
        group_list[this->K - 1] = index_list.segment(int((this->K - 1) * group_size),
                                                     n - int(int(this->K - 1) * group_size));
        for (int k = 0; k < this->K; k++)
        {
            std::sort(group_list[k].data(), group_list[k].data() + group_list[k].size());
        }

        // cv train-test partition:
        std::vector<Eigen::VectorXi> train_mask_list_tmp((unsigned int)this->K);
        std::vector<Eigen::VectorXi> test_mask_list_tmp((unsigned int)this->K);
        for (int k = 0; k < this->K; k++)
        {
            int train_x_size = n - group_list[k].size();
            // get train_mask
            Eigen::VectorXi train_mask(train_x_size);
            int i = 0;
            for (int j = 0; j < this->K; j++)
            {
                if (j != k)
                {
                    for (int s = 0; s < group_list[j].size(); s++)
                    {
                        train_mask(i) = group_list[j](s);
                        i++;
                    }
                }
            }
            std::sort(train_mask.data(), train_mask.data() + train_mask.size());
            train_mask_list_tmp[k] = train_mask;
            test_mask_list_tmp[k] = group_list[k];
        }
        this->train_mask_list = train_mask_list_tmp;
        this->test_mask_list = test_mask_list_tmp;
    };

    void cal_cv_group_XTX(Data &data)
    {
        int p = data.p;
        Eigen::VectorXi index = data.g_index;
        Eigen::VectorXi gsize = data.g_size;
        int N = data.g_num;

        std::vector<std::vector<Eigen::MatrixXd>> group_XTX_list_tmp(this->K);

        for (int k = 0; k < this->K; k++)
        {
            int train_size = this->train_mask_list[k].size();
            Eigen::MatrixXd train_x(train_size, p);

            for (int i = 0; i < train_size; i++)
            {
                train_x.row(i) = data.x.row(this->train_mask_list[k](i));
            };
            group_XTX_list_tmp[k] = group_XTX(train_x, index, gsize, train_size, p, N, 1);
        }
        this->group_XTX_list = group_XTX_list_tmp;
    }

    virtual double test_loss(Algorithm *algorithm, Data &data) = 0;

    virtual double train_loss(Algorithm *algorithm, Data &data) = 0;

    virtual double ic(Algorithm *algorithm, Data &data) = 0;
};

class LmMetric : public Metric
{
public:
    LmMetric(int ic_type, bool is_cv, int K = 0) : Metric(ic_type, is_cv, K){};

    double train_loss(Algorithm *algorithm, Data &data)
    {
        return (data.y - data.x * algorithm->get_beta()).array().square().sum() / (data.get_n());
    };

    double test_loss(Algorithm *algorithm, Data &data)
    {
        if (!this->is_cv)
        {
            return (data.y - data.x * algorithm->get_beta()).array().square().sum() / (data.get_n());
        }
        else
        {

            int k;
            int p = data.get_p();

            Eigen::VectorXd loss_list(this->K);
            for (k = 0; k < this->K; k++)
            {
                int test_size = this->test_mask_list[k].size();
                Eigen::MatrixXd test_x(test_size, p);
                Eigen::VectorXd test_y(test_size);
                Eigen::VectorXd test_weight(test_size);

                for (int i = 0; i < test_size; i++)
                {
                    test_x.row(i) = data.x.row(this->test_mask_list[k](i));
                    test_y(i) = data.y(this->test_mask_list[k](i));
                    test_weight(i) = data.weight(this->test_mask_list[k](i));
                };

                if (algorithm->get_warm_start())
                {
                    algorithm->update_beta_init(this->cv_initial_model_param.row(k));
                }

                algorithm->update_train_mask(this->train_mask_list[k]);
                algorithm->update_group_XTX(this->group_XTX_list[k]);
                algorithm->fit();
                if (algorithm->get_warm_start())
                {
                    this->update_cv_initial_model_param(algorithm->get_beta(), k);
                }

                loss_list(k) = (test_y - test_x * algorithm->get_beta()).array().square().sum() / double(2 * test_size);
            }

            return loss_list.mean();
        }
    };

    double ic(Algorithm *algorithm, Data &data)
    {
        if (this->is_cv)
        {
            return this->test_loss(algorithm, data);
        }
        else
        {
            if (algorithm->algorithm_type == 1 || algorithm->algorithm_type == 5)
            {
                if (ic_type == 1)
                {
                    return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
                           2.0 * algorithm->get_sparsity_level();
                }
                else if (ic_type == 2)
                {
                    return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
                           log(double(data.get_n())) * algorithm->get_sparsity_level();
                }
                else if (ic_type == 3)
                {
                    return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
                           log(double(data.get_p())) * log(log(double(data.get_n()))) * algorithm->get_sparsity_level();
                }
                else if (ic_type == 4)
                {
                    return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
                           (log(double(data.get_n())) + 2 * log(double(data.get_p()))) * algorithm->get_sparsity_level();
                }
                else
                    return 0;
            }
            else
            {
                if (ic_type == 1)
                {
                    return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
                           2.0 * algorithm->get_group_df();
                }
                else if (ic_type == 2)
                {
                    return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
                           log(double(data.get_n())) * algorithm->get_group_df();
                }
                else if (ic_type == 3)
                {
                    return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
                           log(double(data.get_g_num())) * log(log(double(data.get_n()))) * algorithm->get_group_df();
                }
                else if (ic_type == 4)
                {
                    return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
                           (log(double(data.get_n())) + 2 * log(double(data.get_g_num()))) * algorithm->get_group_df();
                }
                else
                    return 0;
            }
        }
    };
};

class LogisticMetric : public Metric
{
public:
    LogisticMetric(int ic_type, bool is_cv, int K = 0) : Metric(ic_type, is_cv, K){};

    double train_loss(Algorithm *algorithm, Data &data)
    {
        int i;
        int n = data.get_n();
        Eigen::VectorXd coef(n);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

        for (i = 0; i <= n - 1; i++)
        {
            coef(i) = algorithm->get_coef0();
        }
        Eigen::VectorXd xbeta_exp = data.x * algorithm->get_beta() + coef;
        for (int i = 0; i <= n - 1; i++)
        {
            if (xbeta_exp(i) > 30.0)
                xbeta_exp(i) = 30.0;
            if (xbeta_exp(i) < -30.0)
                xbeta_exp(i) = -30.0;
        }
        xbeta_exp = xbeta_exp.array().exp();
        Eigen::VectorXd pr = xbeta_exp.array() / (xbeta_exp + one).array();
        return -2 * (data.weight.array() *
                     ((data.y.array() * pr.array().log()) + (one - data.y).array() * (one - pr).array().log()))
                        .sum();
    }

    double test_loss(Algorithm *algorithm, Data &data)
    {
        if (!is_cv)
        {
            return this->train_loss(algorithm, data);
        }
        else
        {
            int k, i;
            int p = data.get_p();

            Eigen::VectorXd loss_list(this->K);

            for (k = 0; k < this->K; k++)
            {
                //get test_x, test_y
                int test_size = this->test_mask_list[k].size();
                Eigen::MatrixXd test_x(test_size, p);
                Eigen::VectorXd test_y(test_size);
                Eigen::VectorXd test_weight(test_size);

                for (i = 0; i < test_size; i++)
                {
                    test_x.row(i) = data.x.row(this->test_mask_list[k](i));
                    test_y(i) = data.y(this->test_mask_list[k](i));
                    test_weight(i) = data.weight(this->test_mask_list[k](i));
                };

                if (algorithm->get_warm_start())
                {
                    algorithm->update_beta_init(this->cv_initial_model_param.row(k));
                }
                algorithm->update_train_mask(this->train_mask_list[k]);
                algorithm->fit();
                if (algorithm->get_warm_start())
                {
                    this->update_cv_initial_model_param(algorithm->get_beta(), k);
                }

                Eigen::VectorXd coef(test_size);
                Eigen::VectorXd one = Eigen::VectorXd::Ones(test_size);

                for (i = 0; i <= test_size - 1; i++)
                {
                    coef(i) = algorithm->get_coef0();
                }
                Eigen::VectorXd xbeta_exp = test_x * algorithm->get_beta() + coef;
                for (i = 0; i <= test_size - 1; i++)
                {
                    if (xbeta_exp(i) > 25.0)
                        xbeta_exp(i) = 25.0;
                    if (xbeta_exp(i) < -25.0)
                        xbeta_exp(i) = -25.0;
                }
                xbeta_exp = xbeta_exp.array().exp();
                Eigen::VectorXd pr = xbeta_exp.array() / (xbeta_exp + one).array();

                loss_list(k) = -2 * (test_weight.array() * ((test_y.array() * pr.array().log()) +
                                                            (one - test_y).array() * (one - pr).array().log()))
                                        .sum();
            }
            return loss_list.mean();
        }
    }

    double ic(Algorithm *algorithm, Data &data)
    {
        if (this->is_cv)
        {
            return this->test_loss(algorithm, data);
        }
        else
        {
            if (data.get_g_index().size() == data.get_p())
            {
                if (ic_type == 1)
                {
                    return this->train_loss(algorithm, data) +
                           2.0 * algorithm->get_sparsity_level();
                }
                else if (ic_type == 2)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_n())) * algorithm->get_sparsity_level();
                }
                else if (ic_type == 3)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_p())) * log(log(double(data.get_n()))) * algorithm->get_sparsity_level();
                }
                else if (ic_type == 4)
                {
                    return this->train_loss(algorithm, data) +
                           (log(double(data.get_n())) + 2 * log(double(data.get_p()))) * algorithm->get_sparsity_level();
                }
                else
                    return 0;
            }
            else
            {
                if (ic_type == 1)
                {
                    return this->train_loss(algorithm, data) +
                           2.0 * algorithm->get_group_df();
                }
                else if (ic_type == 2)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_n())) * algorithm->get_group_df();
                }
                else if (ic_type == 3)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_g_num())) * log(log(double(data.get_n()))) * algorithm->get_group_df();
                }
                else if (ic_type == 4)
                {
                    return this->train_loss(algorithm, data) +
                           (log(double(data.get_n())) + 2 * log(double(data.get_g_num()))) * algorithm->get_group_df();
                }
                else
                    return 0;
            }
        }
    };
};

class PoissonMetric : public Metric
{
public:
    PoissonMetric(int ic_type, bool is_cv, int K = 0) : Metric(ic_type, is_cv, K){};

    double train_loss(Algorithm *algorithm, Data &data)
    {
        int n = data.get_n();
        int p = data.get_p();
        Eigen::VectorXd coef(p + 1);
        coef(0) = algorithm->get_coef0();
        Eigen::VectorXd beta = algorithm->get_beta();

        for (int i = 0; i < p; i++)
        {
            coef(i + 1) = beta(i);
        }

        return -2 * loglik_poisson(data.x, data.y, coef, n, data.weight);
    }

    double test_loss(Algorithm *algorithm, Data &data)
    {
        if (!is_cv)
        {
            return this->train_loss(algorithm, data);
        }
        else
        {
            int k, i;
            int p = data.get_p();

            Eigen::VectorXd loss_list(this->K);

            for (k = 0; k < this->K; k++)
            {

                int test_size = this->test_mask_list[k].size();
                Eigen::MatrixXd test_x(test_size, p);
                Eigen::VectorXd test_y(test_size);
                Eigen::VectorXd test_weight(test_size);

                for (i = 0; i < test_size; i++)
                {
                    test_x.row(i) = data.x.row(this->test_mask_list[k](i));
                    test_y(i) = data.y(this->test_mask_list[k](i));
                    test_weight(i) = data.weight(this->test_mask_list[k](i));
                };

                if (algorithm->get_warm_start())
                {
                    algorithm->update_beta_init(this->cv_initial_model_param.row(k));
                }
                algorithm->update_train_mask(this->train_mask_list[k]);
                algorithm->fit();
                if (algorithm->get_warm_start())
                {
                    this->update_cv_initial_model_param(algorithm->get_beta(), k);
                }

                Eigen::VectorXd coef(p + 1);
                coef(0) = algorithm->get_coef0();
                Eigen::VectorXd beta = algorithm->get_beta();

                for (int i = 0; i < p; i++)
                {
                    coef(i + 1) = beta(i);
                }
                loss_list(k) = -loglik_poisson(test_x, test_y, coef, test_size, test_weight);
            }

            return loss_list.sum() / loss_list.size();
        }
    }

    double ic(Algorithm *algorithm, Data &data)
    {
        if (this->is_cv)
        {
            return this->test_loss(algorithm, data);
        }
        else
        {
            if (data.get_g_index().size() == data.get_p())
            {
                if (ic_type == 1)
                {
                    return this->train_loss(algorithm, data) +
                           2.0 * algorithm->get_sparsity_level();
                }
                else if (ic_type == 2)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_n())) * algorithm->get_sparsity_level();
                }
                else if (ic_type == 3)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_p())) * log(log(double(data.get_n()))) * algorithm->get_sparsity_level();
                }
                else if (ic_type == 4)
                {
                    return this->train_loss(algorithm, data) +
                           (log(double(data.get_n())) + 2 * log(double(data.get_p()))) * algorithm->get_sparsity_level();
                }
                else
                    return 0;
            }
            else
            {
                if (ic_type == 1)
                {
                    return this->train_loss(algorithm, data) +
                           2.0 * algorithm->get_group_df();
                }
                else if (ic_type == 2)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_n())) * algorithm->get_group_df();
                }
                else if (ic_type == 3)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_g_num())) * log(log(double(data.get_n()))) * algorithm->get_group_df();
                }
                else if (ic_type == 4)
                {
                    return this->train_loss(algorithm, data) +
                           (log(double(data.get_n())) + 2 * log(double(data.get_g_num()))) * algorithm->get_group_df();
                }
                else
                    return 0;
            }
        }
    }
};

class CoxMetric : public Metric
{
public:
    CoxMetric(int ic_type, bool is_cv, int K = 0) : Metric(ic_type, is_cv, K){};

    double train_loss(Algorithm *algorithm, Data &data)
    {
        return -2 * loglik_cox(data.x, data.y, algorithm->get_beta(), data.weight);
    };

    double test_loss(Algorithm *algorithm, Data &data)
    {
        if (!this->is_cv)
        {
            return -2 * loglik_cox(data.x, data.y, algorithm->get_beta(), data.weight);
        }
        else
        {
            int k, i;
            int p = data.get_p();

            Eigen::VectorXd loss_list(this->K);

            for (k = 0; k < this->K; k++)
            {

                int test_size = this->test_mask_list[k].size();

                Eigen::MatrixXd test_x(test_size, p);
                Eigen::VectorXd test_y(test_size);
                Eigen::VectorXd test_weight(test_size);

                for (i = 0; i < test_size; i++)
                {
                    test_x.row(i) = data.x.row(this->test_mask_list[k](i)).eval();
                    test_y(i) = data.y(this->test_mask_list[k](i));
                    test_weight(i) = data.weight(this->test_mask_list[k](i));
                };

                if (algorithm->get_warm_start())
                {
                    algorithm->update_beta_init(this->cv_initial_model_param.row(k).eval());
                }
                algorithm->update_train_mask(this->train_mask_list[k]);
                algorithm->fit();
                if (algorithm->get_warm_start())
                {
                    this->update_cv_initial_model_param(algorithm->get_beta(), k);
                }
                loss_list(k) = -2 * loglik_cox(test_x, test_y, algorithm->get_beta(), test_weight);
            }

            return loss_list.sum() / double(loss_list.size());
        }
    };

    double ic(Algorithm *algorithm, Data &data)
    {
        if (this->is_cv)
        {
            return this->test_loss(algorithm, data);
        }
        else
        {
            if (data.get_g_index().size() == data.get_p())
            {
                if (ic_type == 1)
                {
                    return this->train_loss(algorithm, data) +
                           2.0 * algorithm->get_sparsity_level();
                }
                else if (ic_type == 2)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_n())) * algorithm->get_sparsity_level();
                }
                else if (ic_type == 3)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_p())) * log(log(double(data.get_n()))) * algorithm->get_sparsity_level();
                }
                else if (ic_type == 4)
                {
                    return this->train_loss(algorithm, data) +
                           (log(double(data.get_n())) + 2 * log(double(data.get_p()))) * algorithm->get_sparsity_level();
                }
                else
                    return 0;
            }
            else
            {
                if (ic_type == 1)
                {
                    return this->train_loss(algorithm, data) +
                           2.0 * algorithm->get_group_df();
                }
                else if (ic_type == 2)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_n())) * algorithm->get_group_df();
                }
                else if (ic_type == 3)
                {
                    return this->train_loss(algorithm, data) +
                           log(double(data.get_g_num())) * log(log(double(data.get_n()))) * algorithm->get_group_df();
                }
                else if (ic_type == 4)
                {
                    return this->train_loss(algorithm, data) +
                           (log(double(data.get_n())) + 2 * log(double(data.get_g_num()))) * algorithm->get_group_df();
                }
                else
                    return 0;
            }
        }
    }
};

#endif //SRC_METRICS_H
