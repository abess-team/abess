import numpy as np
from abess.linear import *
# from bess.linear import *

from abess.gen_data import gen_data, gen_data_splicing
# import pandas as pd
# import os
# from time import time
from pytest import approx

from sklearn import linear_model
# sklearn.linear_model.LogisticRegression
# from lifelines import CoxPHFitter


class TestClass:
    def test_gaussian(self):
        n = 1000
        p = 200
        k = 20
        family = "gaussian"
        rho = 0.5
        sigma = 1
        np.random.seed(1)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        sequence = range(0, 50)

        model = abessLm(path_type="seq", sequence=sequence, ic_type='ebic', is_screening=True, screening_size=200,
                        K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                        exchange_num=2, tau=0.1 * np.log(n*p) / n,
                        primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=False)
        model.fit(data.x, data.y)

        nonzero_true = np.nonzero(data.beta)[0]
        nonzero_fit = np.nonzero(model.beta)[0]
        print(nonzero_true)
        print(nonzero_fit)
        new_x = data.x[:, nonzero_fit]
        reg = linear_model.LinearRegression()
        reg.fit(new_x, data.y)
        # assert model.beta[nonzero_fit] == approx(reg.coef_, rel=1e-5, abs=1e-5)
        # assert (nonzero_true == nonzero_fit).all()

    def test_binomial(self):
        n = 200
        p = 100
        k = 20
        family = "binomial"
        rho = 0.5
        sigma = 1
        np.random.seed(3)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        sequence = range(0, int(n/np.log(np.log(n)) / np.log(p)))
        print("logistic abess")

        # model = abessLogistic(path_type="seq", sequence=sequence, ic_type='ebic', is_screening=False, screening_size=30,
        #                       K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
        #                       exchange_num=2, tau=0.1 * np.log(n*p) / n,
        #                       primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        # group = np.linspace(1, p, p)
        # model.fit(data.x, data.y, group=group)

        model2 = abessLogistic(path_type="seq", sequence=sequence, ic_type='ebic', is_screening=True, screening_size=100,
                               K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5,
                               exchange_num=2, tau=0.1 * np.log(n*p) / n,
                               primary_model_fit_max_iter=80, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=1)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.beta)[0]
        nonzero_fit = np.nonzero(model2.beta)[0]
        print(nonzero_true)
        print(nonzero_fit)
        # new_x = data.x[:, nonzero_fit]
        # reg = linear_model.LogisticRegression(penalty="none")
        # reg.fit(new_x, data.y)
        # print(model2.beta[nonzero_fit])
        # print(reg.coef_)
        # assert (nonzero_true == nonzero_fit).all()
        # assert model2.beta[nonzero_fit] == approx(
        #     reg.coef_[0], rel=1e-2, abs=1e-2)

    def test_cox(self):
        n = 200
        p = 100
        k = 10
        family = "cox"
        rho = 0.5
        sigma = 1

        np.random.seed(1)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        sequence = range(1, 21)

        # model = abessCox(path_type="seq", sequence=sequence, ic_type='ebic', is_screening=False, screening_size=30,
        #                  K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
        #                  exchange_num=2, tau=0.1 * np.log(n*p) / n,
        #                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        # start = time()
        # group = np.linspace(1, p, p)
        # model.fit(data.x, data.y, group=group)
        # stop = time()

        model2 = abessCox(path_type="seq", sequence=sequence, ic_type='ebic', is_screening=True, screening_size=100,
                          K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5,
                          exchange_num=2, tau=0.1 * np.log(n*p) / n,
                          primary_model_fit_max_iter=60, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.beta)[0]
        nonzero_fit = np.nonzero(model2.beta)[0]
        print(nonzero_true)
        print(nonzero_fit)

        # new_x = data.x[:, nonzero_fit]

        # survival = pd.DataFrame()
        # for i in range(new_x.shape[1]):
        #     survival["Var" + str(i)] = new_x[:, i]

        # survival["T"] = data.y[:, 0]
        # survival["E"] = data.y[:, 1]

        # cph = CoxPHFitter(penalizer=0, l1_ratio=0)

        # cph.fit(survival, 'T', event_col='E')
        # print(model2.beta[nonzero_fit])
        # print(cph.params_.values)

        # assert model2.beta[nonzero_fit] == approx(
        #     cph.params_.values, rel=2e-1, abs=2e-1)

    def test_poisson(self):
        # to do
        n = 200
        p = 100
        k = 10
        family = "poisson"
        rho = 0.5
        sigma = 1
        np.random.seed(3)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        sequence = range(0, 11)

        # model = abessLogistic(path_type="seq", sequence=sequence, ic_type='ebic', is_screening=False, screening_size=30,
        #                       K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
        #                       exchange_num=2, tau=0.1 * np.log(n*p) / n,
        #                       primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        # group = np.linspace(1, p, p)
        # model.fit(data.x, data.y, group=group)

        model2 = abessPoisson(path_type="seq", sequence=sequence, ic_type='ebic', is_screening=True, screening_size=100,
                              K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5,
                              exchange_num=2, tau=0.1 * np.log(n*p) / n,
                              primary_model_fit_max_iter=80, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=1)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.beta)[0]
        nonzero_fit = np.nonzero(model2.beta)[0]
        print(nonzero_true)
        print(nonzero_fit)
        # new_x = data.x[:, nonzero_fit]
        # reg = linear_model.LogisticRegression(
        #     penalty="none", tol=1e-6, max_iter=200)
        # reg.fit(new_x, data.y)
        # print(model2.beta[nonzero_fit])
        # print(reg.coef_)
        # assert (nonzero_true == nonzero_fit).all()
        # assert model2.beta[nonzero_fit] == approx(
        #     reg.coef_[0], rel=1e-2, abs=1e-2)

    def test_mulgaussian(self):
        n = 1000
        p = 200
        k = 20
        family = "multigaussian"
        rho = 0.5
        M = 3
        np.random.seed(1)
        data = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        sequence = range(0, int(n/np.log(np.log(n)) / np.log(p)))

        model = abessMultigaussian(path_type="seq", sequence=sequence, ic_type='ebic', is_screening=True, screening_size=200,
                                   K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                   exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                   primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=False)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.beta)[0]
        nonzero_fit = np.nonzero(model.beta)[0]
        print(nonzero_true)
        print(nonzero_fit)
        # new_x = data.x[:, nonzero_fit]
        # reg = linear_model.LinearRegression()
        # reg.fit(new_x, data.y)
        # assert model.beta[nonzero_fit] == approx(reg.coef_, rel=1e-5, abs=1e-5)
        # assert (nonzero_true == nonzero_fit).all()

    def test_mulnomial(self):
        n = 200
        p = 100
        k = 20
        family = "multinomial"
        rho = 0.5
        M = 3
        np.random.seed(1)
        data = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        sequence = range(0, 50)

        model = abessMultinomial(path_type="seq", sequence=sequence, ic_type='ebic', is_screening=True, screening_size=100,
                                 K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5,
                                 exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultinomial(path_type="seq", sequence=sequence, ic_type='ebic', is_screening=True, screening_size=100,
                                  K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5,
                                  exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.beta)[0]
        nonzero_fit = np.nonzero(model.beta)[0]
        print(nonzero_true)
        print(nonzero_fit)
        # new_x = data.x[:, nonzero_fit]
        # reg = linear_model.LinearRegression()
        # reg.fit(new_x, data.y)
        # assert model.beta[nonzero_fit] == approx(reg.coef_, rel=1e-5, abs=1e-5)
        # assert (nonzero_true == nonzero_fit).all()
