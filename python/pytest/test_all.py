import numpy as np
from abess.linear import *
from abess.gen_data import gen_data, gen_data_splicing
import pandas as pd
from pytest import approx
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# For Python >=3.6
if sys.version_info[0] == 3 and sys.version_info[1] >= 6:
    from sklearn.linear_model import PoissonRegressor
    from lifelines import CoxPHFitter


class TestClass:
    def test_gaussian(self):
        n = 100
        p = 20
        k = 3
        family = "gaussian"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(2)
        data = gen_data_splicing(family=family, n=n, p=p, k=k, rho=rho, M=M)
        data2 = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data3 = gen_data_splicing(
            family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20

        model = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                        K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                        exchange_num=2, tau=0.1 * np.log(n*p) / n,
                        primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=True)
        model.fit(data.x, data.y)
        model.predict(data.x)

        model2 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=1, covariance_update=True)
        model2.fit(data.x, data.y)

        model3 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=0, covariance_update=False, sparse_matrix=True)
        model3.fit(data.x, data.y)

        model4 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=0, covariance_update=True)
        model4.fit(data.x, data.y)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        new_x = data.x[:, nonzero_fit]
        reg = LinearRegression()
        reg.fit(new_x, data.y.reshape(-1))
        assert model.coef_[nonzero_fit] == approx(
            reg.coef_, rel=1e-5, abs=1e-5)
        assert (nonzero_true == nonzero_fit).all()

    def test_binomial(self):
        n = 100
        p = 20
        k = 3
        family = "binomial"
        rho = 0.5
        sigma = 1
        np.random.seed(1)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)
        print("logistic abess")

        model = abessLogistic(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=30,
                              K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                              exchange_num=2, tau=0.1 * np.log(n*p) / n,
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessLogistic(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                               K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                               exchange_num=2, tau=0.1 * np.log(n*p) / n,
                               primary_model_fit_max_iter=80, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)
        model2.predict(data.x)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model2.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        assert (nonzero_true == nonzero_fit).all()

        if sys.version_info[1] >= 6:
            new_x = data.x[:, nonzero_fit]
            reg = LogisticRegression(penalty="none")
            reg.fit(new_x, data.y)
            print(model2.coef_[nonzero_fit])
            print(reg.coef_)
            assert model2.coef_[nonzero_fit] == approx(
                reg.coef_[0], rel=1e-2, abs=1e-2)

    def test_cox(self):
        n = 100
        p = 20
        k = 3
        family = "cox"
        rho = 0.5
        sigma = 1

        # np.random.seed(3)
        np.random.seed(3)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)

        model = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)
        model.predict(data.x)

        model2 = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                          K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                          exchange_num=2, tau=0.1 * np.log(n*p) / n,
                          primary_model_fit_max_iter=60, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model2.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        assert (nonzero_true == nonzero_fit).all()

        if sys.version_info[1] >= 6:
            new_x = data.x[:, nonzero_fit]
            survival = pd.DataFrame()
            for i in range(new_x.shape[1]):
                survival["Var" + str(i)] = new_x[:, i]
            survival["T"] = data.y[:, 0]
            survival["E"] = data.y[:, 1]
            cph = CoxPHFitter(penalizer=0, l1_ratio=0)
            cph.fit(survival, 'T', event_col='E')
            print(model2.coef_[nonzero_fit])
            print(cph.params_.values)

            assert model2.coef_[nonzero_fit] == approx(
                cph.params_.values, rel=5e-1, abs=5e-1)

    def test_poisson(self):
        # to do
        n = 100
        p = 20
        k = 3
        family = "poisson"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(3)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data2 = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessPoisson(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                             K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                             exchange_num=2, tau=0.1 * np.log(n*p) / n,
                             primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessPoisson(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                              K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                              exchange_num=2, tau=0.1 * np.log(n*p) / n,
                              primary_model_fit_max_iter=80, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)
        model2.predict(data.x)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model2.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        assert (nonzero_true == nonzero_fit).all()

        if sys.version_info[1] >= 6:
            new_x = data.x[:, nonzero_fit]
            reg = PoissonRegressor(
                alpha=0, tol=1e-6, max_iter=200)
            reg.fit(new_x, data.y)
            print(model2.coef_[nonzero_fit])
            print(reg.coef_)
            assert model2.coef_[nonzero_fit] == approx(
                reg.coef_, rel=1e-2, abs=1e-2)

    def test_mulgaussian(self):
        n = 100
        p = 20
        k = 3
        family = "multigaussian"
        rho = 0.5
        M = 3
        np.random.seed(1)
        data = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, int(n/np.log(np.log(n)) / np.log(p)))

        model = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                   K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                   exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                   primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=False)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                    K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                    exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                    primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=True, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        # new_x = data.x[:, nonzero_fit]
        # reg = linear_model.LinearRegression()
        # reg.fit(new_x, data.y)
        # assert model.coef_[nonzero_fit] == approx(reg.coef_, rel=1e-5, abs=1e-5)
        assert (nonzero_true == nonzero_fit).all()

    def test_mulnomial(self):
        n = 100
        p = 20
        k = 3
        family = "multinomial"
        rho = 0.5
        M = 3
        np.random.seed(5)
        data = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                 K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                 exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                  K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5,
                                  exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        model3 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                  K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                  exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model3.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        # new_x = data.x[:, nonzero_fit]
        # reg = linear_model.LinearRegression()
        # reg.fit(new_x, data.y)
        # assert model.coef_[nonzero_fit] == approx(reg.coef_, rel=1e-5, abs=1e-5)
        assert (nonzero_true == nonzero_fit).all()

    def test_gaussian_sklearn(self):
        n = 100
        p = 20
        k = 3
        family = "gaussian"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(2)
        # data = gen_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
        data = gen_data(n, p, family=family, k=k, rho=rho)
        # data3 = gen_data_splicing(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20
        support_size = np.linspace(0, s_max, s_max+1, dtype = "int32")
        alpha = [0., 0.1, 0.2, 0.3, 0.4]

        model = abessLm()
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        gcv = GridSearchCV(
            model,
            param_grid={"support_size": support_size,
                        "alpha": alpha},
            cv=cv,
            n_jobs=5).fit(data.x, data.y)

        assert gcv.best_params_["support_size"] == k
        assert gcv.best_params_["alpha"] == 0.

    
    def test_cox_sklearn(self):
        n = 100
        p = 20
        k = 3
        family = "cox"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(3)
        # data = gen_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
        data = gen_data(n, p, family=family, k=k, rho=rho)
        # data3 = gen_data_splicing(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 10
        support_size = np.linspace(1, s_max, s_max+1, dtype = "int32")
        alpha = [0., 0.1, 0.2, 0.3]

        model = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        gcv = GridSearchCV(
            model,
            param_grid={"support_size": support_size,
                        "alpha": alpha},
            cv=cv,
            n_jobs=1).fit(data.x, data.y)

        assert gcv.best_params_["support_size"] == k
        assert gcv.best_params_["alpha"] == 0.

    def test_PCA(self):
        n = 1000
        p = 20
        s = 10
        group_size = 5
        group_num = 4

        np.random.seed(2)
        x1 = np.random.randn(n, 1)
        x1 /= np.linalg.norm(x1)
        X = x1.dot(np.random.randn(1, p)) + 0.01 * np.random.randn(n, p)
        X = X - X.mean(axis=0)
        g_index = np.arange(group_num)
        g_index = g_index.repeat(group_size)

        # Check1: give X
        model = abessPCA(support_size=range(s, s + 1))
        model.fit(X, is_normal = False)
        coef1 = np.nonzero(model.coef_)[0]

        assert len(coef1) == s

        # Check2: give Sigma
        model.fit(Sigma = X.T.dot(X))
        coef2 = np.nonzero(model.coef_)[0]

        assert len(coef2) == s

        # Check3: group
        model = abessPCA(support_size=range(3, 4))
        model.fit(X, group=g_index, is_normal=False)

        coef3 = np.unique(g_index[np.nonzero(model.coef_)])
        assert (coef3.size == 3)

        # Check4: multi
        model = abessPCA(support_size=range(s, s + 1))
        model.fit(X, is_normal = False, number = 3)
        assert (model.coef_.shape[1] == 3)

        for i in range(3):
            coef4 = np.nonzero(model.coef_[:, i])[0]
            assert (len(coef4) == s)



    def test_gaussian_gs(self):
        n = 100
        p = 20
        k = 3
        family = "gaussian"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(2)
        data = gen_data_splicing(family=family, n=n, p=p, k=k, rho=rho, M=M)
        data2 = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data3 = gen_data_splicing(
            family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20

        model = abessLm(path_type="pgs", support_size=[0], ic_type='ebic', is_screening=True, screening_size=20,
                        K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=s_max, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                        exchange_num=2, tau=0.1 * np.log(n*p) / n,
                        primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=True)
        model.fit(data.x, data.y)
        model.predict(data.x)

        model2 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=1, covariance_update=True)
        model2.fit(data.x, data.y)

        model3 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=0, covariance_update=False, sparse_matrix=True)
        model3.fit(data.x, data.y)

        model4 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=0, covariance_update=True)
        model4.fit(data.x, data.y)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        new_x = data.x[:, nonzero_fit]
        reg = LinearRegression()
        reg.fit(new_x, data.y.reshape(-1))
        assert model.coef_[nonzero_fit] == approx(
            reg.coef_, rel=1e-5, abs=1e-5)
        assert (nonzero_true == nonzero_fit).all()

    def test_binomial_gs(self):
        n = 100
        p = 20
        k = 3
        family = "binomial"
        rho = 0.5
        sigma = 1
        np.random.seed(5)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)
        print("logistic abess")

        model = abessLogistic(path_type="pgs", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=30,
                              K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=20, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                              exchange_num=2, tau=0.1 * np.log(n*p) / n,
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessLogistic(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                               K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                               exchange_num=2, tau=0.1 * np.log(n*p) / n,
                               primary_model_fit_max_iter=80, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)
        model2.predict(data.x)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model2.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        assert (nonzero_true == nonzero_fit).all()

        if sys.version_info[1] >= 6:
            new_x = data.x[:, nonzero_fit]
            reg = LogisticRegression(penalty="none")
            reg.fit(new_x, data.y)
            print(model2.coef_[nonzero_fit])
            print(reg.coef_)
            assert model2.coef_[nonzero_fit] == approx(
                reg.coef_[0], rel=1e-2, abs=1e-2)

    def test_cox_gs(self):
        n = 100
        p = 20
        k = 3
        family = "cox"
        rho = 0.5
        sigma = 1

        # np.random.seed(3)
        np.random.seed(3)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)

        model = abessCox(path_type="pgs", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=20, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)
        model.predict(data.x)

        model2 = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                          K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                          exchange_num=2, tau=0.1 * np.log(n*p) / n,
                          primary_model_fit_max_iter=60, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model2.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        assert (nonzero_true == nonzero_fit).all()

        if sys.version_info[1] >= 6:
            new_x = data.x[:, nonzero_fit]
            survival = pd.DataFrame()
            for i in range(new_x.shape[1]):
                survival["Var" + str(i)] = new_x[:, i]
            survival["T"] = data.y[:, 0]
            survival["E"] = data.y[:, 1]
            cph = CoxPHFitter(penalizer=0, l1_ratio=0)
            cph.fit(survival, 'T', event_col='E')
            print(model2.coef_[nonzero_fit])
            print(cph.params_.values)

            assert model2.coef_[nonzero_fit] == approx(
                cph.params_.values, rel=5e-1, abs=5e-1)

    def test_poisson_gs(self):
        # to do
        n = 100
        p = 20
        k = 3
        family = "poisson"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(3)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data2 = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessPoisson(path_type="pgs", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                             K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                             exchange_num=2, tau=0.1 * np.log(n*p) / n,
                             primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessPoisson(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                              K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                              exchange_num=2, tau=0.1 * np.log(n*p) / n,
                              primary_model_fit_max_iter=80, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)
        model2.predict(data.x)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model2.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        assert (nonzero_true == nonzero_fit).all()

        if sys.version_info[1] >= 6:
            new_x = data.x[:, nonzero_fit]
            reg = PoissonRegressor(
                alpha=0, tol=1e-6, max_iter=200)
            reg.fit(new_x, data.y)
            print(model2.coef_[nonzero_fit])
            print(reg.coef_)
            assert model2.coef_[nonzero_fit] == approx(
                reg.coef_, rel=1e-2, abs=1e-2)

    def test_mulgaussian_gs(self):
        n = 100
        p = 20
        k = 3
        family = "multigaussian"
        rho = 0.5
        M = 3
        np.random.seed(1)
        data = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, int(n/np.log(np.log(n)) / np.log(p)))

        model = abessMultigaussian(path_type="pgs", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                   K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                   exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                   primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=False)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                    K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                    exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                    primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=True, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        # new_x = data.x[:, nonzero_fit]
        # reg = linear_model.LinearRegression()
        # reg.fit(new_x, data.y)
        # assert model.coef_[nonzero_fit] == approx(reg.coef_, rel=1e-5, abs=1e-5)
        assert (nonzero_true == nonzero_fit).all()

    def test_mulnomial_gs(self):
        n = 100
        p = 20
        k = 3
        family = "multinomial"
        rho = 0.5
        M = 3
        np.random.seed(5)
        data = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessMultinomial(path_type="pgs", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                 K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                 exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                  K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5,
                                  exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        model3 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                  K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                  exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model3.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        # new_x = data.x[:, nonzero_fit]
        # reg = linear_model.LinearRegression()
        # reg.fit(new_x, data.y)
        # assert model.coef_[nonzero_fit] == approx(reg.coef_, rel=1e-5, abs=1e-5)
        assert (nonzero_true == nonzero_fit).all()


    def test_gaussian_sparse_matrix(self):
        n = 100
        p = 20
        k = 3
        family = "gaussian"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(2)
        data = gen_data_splicing(family=family, n=n, p=p, k=k, rho=rho, M=M)
        data2 = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data3 = gen_data_splicing(
            family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20

        model = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                        K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                        exchange_num=2, tau=0.1 * np.log(n*p) / n,
                        primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=True)
        model.fit(data.x + 1, data.y + 1)
        model.predict(data.x)

        model2 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                        K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                        exchange_num=2, tau=0.1 * np.log(n*p) / n,
                        primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=True, sparse_matrix=True)
        model2.fit(data.x + 1, data.y +1)

        assert model.coef_ == approx(model2.coef_, rel=1e-5, abs=1e-5)
        assert model.intercept_ == approx(model2.intercept_, rel=1e-5, abs=1e-5)

    def test_binomial_sparse_matrix(self):
        n = 100
        p = 20
        k = 3
        family = "binomial"
        rho = 0.5
        sigma = 1
        np.random.seed(1)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)
        print("logistic abess")

        model = abessLogistic(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=30,
                              K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                              exchange_num=2, tau=0.1 * np.log(n*p) / n,
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessLogistic(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=30,
                              K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                              exchange_num=2, tau=0.1 * np.log(n*p) / n,
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=False, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)
        model2.predict(data.x)

        assert model.coef_ == approx(model2.coef_, rel=1e-5, abs=1e-5)
        assert model.intercept_ == approx(model2.intercept_, rel=1e-5, abs=1e-5)

    def test_cox_sparse_matrix(self):
        n = 100
        p = 20
        k = 3
        family = "cox"
        rho = 0.5
        sigma = 1

        # np.random.seed(3)
        np.random.seed(3)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)

        model = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x + 1, data.y, group=group)
        model.predict(data.x)

        model2 = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                         K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                         exchange_num=2, tau=0.1 * np.log(n*p) / n,
                         primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x + 1, data.y, group=group)

        assert model.coef_ == approx(model2.coef_, rel=1e-5, abs=1e-5)
        assert model.intercept_ == approx(model2.intercept_, rel=1e-5, abs=1e-5)

    def test_poisson_sparse_matrix(self):
        # to do
        n = 100
        p = 20
        k = 3
        family = "poisson"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(1)
        data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data2 = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessPoisson(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                             K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                             exchange_num=2, tau=0.1 * np.log(n*p) / n,
                             primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x + 1, data.y, group=group)

        model2 = abessPoisson(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                             K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                             exchange_num=2, tau=0.1 * np.log(n*p) / n,
                             primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x + 1, data.y, group=group)
        model2.predict(data.x)

        assert model.coef_ == approx(model2.coef_, rel=1e-1, abs=1e-1)
        assert model.intercept_ == approx(model2.intercept_, rel=1e-1, abs=1e-1)

    def test_mulgaussian_sparse_matrix(self):
        n = 100
        p = 20
        k = 3
        family = "multigaussian"
        rho = 0.5
        M = 3
        np.random.seed(4)
        data = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, int(n/np.log(np.log(n)) / np.log(p)))

        model = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                   K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                   exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                   primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=True)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                   K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                   exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                   primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, covariance_update=True, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        assert model.coef_[np.nonzero(model.coef_)[0]] == approx(model2.coef_[np.nonzero(model2.coef_)[0]], rel=1e-1, abs=1e-1)
        assert model.intercept_ == approx(model2.intercept_, rel=1e-1, abs=1e-1)

    def test_mulnomial_sparse_matrix(self):
        n = 100
        p = 20
        k = 3
        family = "multinomial"
        rho = 0.5
        M = 3
        np.random.seed(5)
        data = gen_data_splicing(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                 K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                 exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                 K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                                 exchange_num=2, tau=0.1 * np.log(n*p) / n,
                                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        assert model.coef_[np.nonzero(model.coef_)[0]] == approx(model2.coef_[np.nonzero(model2.coef_)[0]], rel=1e-1, abs=1e-1)
        assert model.intercept_ == approx(model2.intercept_, rel=1e-1, abs=1e-1)
