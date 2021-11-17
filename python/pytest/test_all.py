import numpy as np
from abess.linear import *
from abess.pca import *
from abess.datasets import make_glm_data, make_multivariate_glm_data
import pandas as pd
from pytest import approx
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from scipy.sparse import coo_matrix

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
        data = make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M)
        data2 = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data3 = make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20

        model = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                        s_min=1, s_max=p, cv=1,
                        exchange_num=2, 
                        ic_coef=1., thread=5, covariance_update=True)
        model.fit(data.x, data.y)
        model.predict(data.x)

        model2 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         s_min=1, s_max=p, cv=5,
                         exchange_num=2, 
                         ic_coef=1., thread=1, covariance_update=True, always_select=[0])
        model2.fit(data.x, data.y)

        model3 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         s_min=1, s_max=p, cv=5,
                         exchange_num=2, 
                         ic_coef=1., thread=5, covariance_update=False, sparse_matrix=True)
        model3.fit(data.x, data.y)

        model4 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20, alpha=[0.001],
                         s_min=1, s_max=p, cv=1,
                         exchange_num=2, 
                         ic_coef=1., thread=5, covariance_update=True, splicing_type=1)
        model4.fit(data.x, data.y)

        model5 = abessLm(support_size=range(s_max), important_search=0)
        model5.fit(data.x, data.y)

        model6 = abessLm(support_size=range(2, s_max), important_search=5, always_select=[0, 1], covariance_update=True)
        model6.fit(data.x, data.y)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model5.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        new_x = data.x[:, nonzero_fit]
        reg = LinearRegression()
        reg.fit(new_x, data.y.reshape(-1))
        assert model5.coef_[nonzero_fit] == approx(
            reg.coef_, rel=1e-5, abs=1e-5)
        assert (nonzero_true == nonzero_fit).all()
        assert (model6.coef_[0] != 0)

    def test_binomial(self):
        n = 100
        p = 20
        k = 3
        family = "binomial"
        rho = 0.5
        sigma = 1
        np.random.seed(5)
        data = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)
        print("logistic abess")

        model = abessLogistic(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=30,
                              s_min=1, s_max=p, cv=5,
                              exchange_num=2, 
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessLogistic(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                               s_min=1, s_max=p, cv=5,
                               exchange_num=2, 
                               primary_model_fit_max_iter=80, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)
        model2.predict(data.x)

        model3 = abessLogistic(path_type="seq", support_size=support_size, ic_type='aic', is_screening=False, screening_size=30,  alpha=[0.001],
                              s_min=1, s_max=p, cv=1,
                              exchange_num=2, 
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model3.fit(data.x, data.y, group=group)

        model4 = abessLogistic(path_type="seq", support_size=support_size, ic_type='aic', is_screening=True, screening_size=20,  alpha=[0.001],
                              s_min=1, s_max=p, cv=1,
                              exchange_num=2, 
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model4.fit(data.x, data.y, group=group)

        model.predict_proba(data.x)

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
        np.random.seed(2)
        data = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)

        model = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=20, alpha=[0.001],
                         s_min=1, s_max=p, cv=5,
                         exchange_num=2, 
                         primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)
        model.predict(data.x)

        model = abessCox(path_type="seq", support_size=support_size, ic_type='bic', is_screening=True, screening_size=20, alpha=[0.001],
                         s_min=1, s_max=p, cv=1,
                         exchange_num=2, 
                         primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)
        model.predict(data.x)

        model2 = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                          s_min=1, s_max=p, cv=5,
                          exchange_num=2, 
                          primary_model_fit_max_iter=60, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        model3 = abessCox(support_size=support_size, important_search=10)
        model3.fit(data.x, data.y, group=group)

        model4 = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                          s_min=1, s_max=p, cv=5,
                          exchange_num=2, primary_model_fit_epsilon=1,  ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model4.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model3.coef_)[0]
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
        np.random.seed(9)
        data = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        # data2 = make_multivariate_glm_data(
        #     family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessPoisson(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20, alpha=[0.001],
                             s_min=1, s_max=p, cv=5,
                             exchange_num=2, 
                             primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model1 = abessPoisson(path_type="seq", support_size=support_size, ic_type='gic', is_screening=True, screening_size=20, alpha=[0.001],
                              s_min=1, s_max=p, cv=1,
                              exchange_num=2, 
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model1.fit(data.x, data.y, group=group)

        model2 = abessPoisson(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                              s_min=1, s_max=p, cv=5,
                              exchange_num=2, 
                              primary_model_fit_max_iter=80, primary_model_fit_epsilon=1e-6, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)
        model2.predict(data.x)

        model3 = abessPoisson(support_size=support_size, important_search=10)
        model3.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model3.coef_)[0]
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
        data = make_multivariate_glm_data(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, int(n/np.log(np.log(n)) / np.log(p)))

        model = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                   s_min=1, s_max=p, cv=5,
                                   exchange_num=2, 
                                   ic_coef=1., thread=5, covariance_update=False)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)
        model.predict(data.x)

        model2 = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20, alpha=[0.001],
                                    s_min=1, s_max=p, cv=5,
                                    exchange_num=2, 
                                    ic_coef=1., thread=5, covariance_update=True, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        model3 = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20, alpha=[0.001],
                                    s_min=1, s_max=p, cv=1,
                                    exchange_num=2, 
                                    ic_coef=1., thread=5, covariance_update=True)
        group = np.linspace(1, p, p)
        model3.fit(data.x, data.y, group=group)

        model4 = abessMultigaussian(support_size=support_size, important_search=5, covariance_update=True)
        group = np.linspace(1, p, p)
        model4.fit(data.x, data.y, group=group)

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
        support_size = range(0, 20)

        np.random.seed(5)
        data = make_multivariate_glm_data(
            family=family, n=n, p=p,  k=k, rho=rho, M=M + 1)
        model = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                 s_min=1, s_max=p, cv=5,
                                 exchange_num=2, 
                                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)
        model.predict(data.x)

        model = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=20,
                                 s_min=1, s_max=p, cv=5,
                                 exchange_num=2, 
                                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        model.fit(data.x, data.y, group=group)
        model.predict(data.x)

        np.random.seed(5)
        data = make_multivariate_glm_data(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)

        model = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                 s_min=1, s_max=p, cv=5,
                                 exchange_num=2, 
                                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)
        model.predict(data.x)

        model2 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20, alpha=[0.001],
                                  s_min=1, s_max=p, cv=1,
                                  exchange_num=2, 
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        model3 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                  s_min=1, s_max=p, cv=5,
                                  exchange_num=2, 
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model3.fit(data.x, data.y, group=group)

        model4 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20, alpha=[0.001],
                                  s_min=1, s_max=p, cv=1,
                                  exchange_num=2, 
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model4.fit(data.x, data.y, group=group)

        model5 = abessMultinomial(support_size=support_size, important_search=10)
        model5.fit(data.x, data.y, group=group)

        nonzero_true = np.unique(np.nonzero(data.coef_)[0])
        nonzero_fit = np.unique(np.nonzero(model5.coef_)[0])
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
        np.random.seed(7)
        # data = make_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
        data = make_glm_data(n, p, family=family, k=k, rho=rho)
        # data3 = make_multivariate_glm_data(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20
        support_size = np.linspace(0, s_max, s_max+1, dtype="int32")
        alpha = [0., 0.1, 0.2, 0.3, 0.4]

        model = abessLm()
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        gcv = GridSearchCV(
            model,
            param_grid={"support_size": support_size,
                        "important_search": [10],
                        "alpha": alpha},
            cv=cv,
            n_jobs=5).fit(data.x, data.y)

        assert gcv.best_params_["support_size"] == k
        assert gcv.best_params_["alpha"] == 0.

    def test_binomial_sklearn(self):
        n = 100
        p = 20
        k = 3
        family = "binomial"
        rho = 0.5
        sigma = 1
        np.random.seed(3)
        data = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        # data3 = make_multivariate_glm_data(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20
        support_size = np.linspace(0, s_max, s_max+1, dtype="int32")
        alpha = [0., 0.1, 0.2, 0.3, 0.4]

        model = abessLogistic()
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        gcv = GridSearchCV(
            model,
            param_grid={"support_size": support_size,
                        "important_search": [10],
                        "alpha": alpha},
            cv=cv,
            n_jobs=5).fit(data.x, data.y)

        assert gcv.best_params_["support_size"] == k
        assert gcv.best_params_["alpha"] == 0.

    def test_poisson_sklearn(self):
        n = 100
        p = 20
        k = 3
        family = "poisson"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(3)
        # data = make_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
        data = make_glm_data(n, p, family=family, k=k, rho=rho)
        # data3 = make_multivariate_glm_data(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20
        support_size = np.linspace(0, s_max, s_max+1, dtype="int32")
        alpha = [0., 0.1, 0.2, 0.3, 0.4]

        model = abessPoisson()
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        gcv = GridSearchCV(
            model,
            param_grid={"support_size": support_size,
                        "important_search": [10],
                        "alpha": alpha},
            cv=cv,
            n_jobs=1).fit(data.x, data.y)

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
        # data = make_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
        data = make_glm_data(n, p, family=family, k=k, rho=rho)
        # data3 = make_multivariate_glm_data(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 10
        support_size = np.linspace(1, s_max, s_max+1, dtype="int32")
        alpha = [0., 0.1, 0.2, 0.3]

        model = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=20,
                         s_min=1, s_max=p, cv=5,
                         exchange_num=2, 
                         primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        gcv = GridSearchCV(
            model,
            param_grid={"support_size": support_size,
                        "important_search": [10],
                        "alpha": alpha},
            cv=cv,
            n_jobs=1).fit(data.x, data.y)

        assert gcv.best_params_["support_size"] == k
        assert gcv.best_params_["alpha"] == 0.

    # def test_multigaussian_sklearn(self):
    #     n = 100
    #     p = 20
    #     k = 3
    #     family = "multigaussian"
    #     rho = 0.5
    #     sigma = 1
    #     M = 1
    #     np.random.seed(2)
    #     # data = make_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
    #     data = make_multivariate_glm_data(
    #         family=family, n=n, p=p,  k=k, rho=rho, M=M)
    #     # data3 = make_multivariate_glm_data(
    #     #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
    #     s_max = 20
    #     support_size = np.linspace(1, s_max, s_max+1)
    #     alpha = [0., 0.1, 0.2, 0.3, 0.4]

    #     model = abessMultigaussian()
    #     cv = KFold(n_splits=5, shuffle=True, random_state=0)
    #     gcv = GridSearchCV(
    #         model,
    #         param_grid={"support_size": support_size,
    #                     "alpha": alpha},
    #         cv=cv,
    #         n_jobs=1).fit(data.x, data.y)

    #     assert gcv.best_params_["support_size"] == k
    #     assert gcv.best_params_["alpha"] == 0.

    # def test_multinomial_sklearn(self):
    #     n = 100
    #     p = 20
    #     k = 3
    #     family = "multinomial"
    #     rho = 0.5
    #     sigma = 1
    #     M = 1
    #     np.random.seed(2)
    #     # data = make_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
    #     data = make_multivariate_glm_data(
    #         family=family, n=n, p=p,  k=k, rho=rho, M=M)
    #     # data3 = make_multivariate_glm_data(
    #     #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
    #     s_max = 20
    #     support_size = np.linspace(0, s_max, s_max+1, dtype = "int32")
    #     alpha = [0., 0.1, 0.2, 0.3, 0.4]

    #     model = abessMultinomial()
    #     cv = KFold(n_splits=5, shuffle=True, random_state=0)
    #     gcv = GridSearchCV(
    #         model,
    #         param_grid={"support_size": support_size,
    #                     "alpha": alpha},
    #         cv=cv,
    #         n_jobs=1).fit(data.x, data.y)

    #     assert gcv.best_params_["support_size"] == k
    #     assert gcv.best_params_["alpha"] == 0.

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
        model.fit(X, is_normal=False)
        coef1 = np.nonzero(model.coef_)[0]

        assert len(coef1) == s

        model = abessPCA(support_size=s)    # give integer
        model.fit(X, is_normal=False)
        coef1 = np.nonzero(model.coef_)[0]

        assert len(coef1) == s

        # Check2: give Sigma
        model.fit(Sigma=X.T.dot(X), n = 10)
        coef2 = np.nonzero(model.coef_)[0]

        assert len(coef2) == s

        # Check3: group
        model = abessPCA(support_size=range(3, 4))
        model.fit(X, group=g_index, is_normal=False)

        coef3 = np.unique(g_index[np.nonzero(model.coef_)[0]])
        assert (coef3.size == 3)

        # Check4: multi
        model = abessPCA(support_size=[s,s,s])
        model.fit(X, is_normal=False, number=3)
        assert (model.coef_.shape[1] == 3)

        for i in range(3):
            coef4 = np.nonzero(model.coef_[:, i])[0]
            assert (len(coef4) == s)

        model.ratio(X)

        # Check5: sparse
        model = abessPCA(support_size=[s], sparse_matrix=True)
        model.fit(X, is_normal=False)
        coef5 = np.nonzero(model.coef_)[0]
        assert (coef5 == coef1).all()

        temp = coo_matrix(([1, 2, 3], ([0, 1, 2], [0, 1, 2])))
        model = abessPCA(sparse_matrix=True)
        model.fit(temp)

        # Check6: ratio & transform
        model = abessPCA(sparse_matrix=False)
        model.fit(X, is_normal=False)
        model.ratio(X)
        model.transform(X)
        model.ratio(np.ones((1, p)))

        # Check7: ic
        for ic in ['aic', 'bic', 'ebic', 'gic']:
            model = abessPCA(ic_type=ic)
            model.fit(X)

        # Check8: error arg
        try:
            model = abessPCA()
            model.fit()
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessPCA(ic_type='other')
            model.fit(X)
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessPCA()
            model.fit(X, group=[[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessPCA()
            model.fit(X, group=[1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessPCA(support_size=[p+1])
            model.fit(X)
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessPCA(exchange_num=-1)
            model.fit()
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessPCA(thread=-1)
            model.fit(X)
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessPCA(splicing_type=2)
            model.fit(X)
        except ValueError as e:
            print(e)
        else:
            assert False

    def test_gaussian_gs(self):
        n = 100
        p = 20
        k = 3
        family = "gaussian"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(2)
        data = make_multivariate_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
        data2 = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data3 = make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20

        model = abessLm(path_type="pgs", support_size=[0], ic_type='ebic', is_screening=True, screening_size=20,
                        s_min=1, s_max=s_max, cv=5,
                        exchange_num=2, 
                        ic_coef=1., thread=5, covariance_update=True)
        model.fit(data.x, data.y)
        model.predict(data.x)

        model2 = abessLm(path_type="pgs", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         s_min=1, s_max=p, cv=5,
                         exchange_num=2, 
                         ic_coef=1., thread=1, covariance_update=True)
        model2.fit(data.x, data.y)

        model3 = abessLm(path_type="pgs", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         s_min=1, s_max=p, cv=5,
                         exchange_num=2, 
                         ic_coef=1., thread=0, covariance_update=False, sparse_matrix=True)
        model3.fit(data.x, data.y)

        model4 = abessLm(path_type="pgs", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         s_min=1, s_max=p, cv=1,
                         exchange_num=2, 
                         ic_coef=1., thread=0, covariance_update=True)
        model4.fit(data.x, data.y)


    def test_binomial_gs(self):
        n = 100
        p = 20
        k = 3
        family = "binomial"
        rho = 0.5
        sigma = 1
        np.random.seed(5)
        data = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)
        print("logistic abess")

        model = abessLogistic(path_type="pgs", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=30,
                              s_min=1, s_max=20, cv=5,
                              exchange_num=2, 
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessLogistic(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                               s_min=1, s_max=p, cv=5,
                               exchange_num=2, 
                               primary_model_fit_max_iter=80, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5, sparse_matrix=True)
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
        np.random.seed(4)
        data = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)

        model = abessCox(path_type="pgs", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                         s_min=1, s_max=20, cv=5,
                         exchange_num=2, 
                         primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)
        model.predict(data.x)

        model2 = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                          s_min=1, s_max=p, cv=5,
                          exchange_num=2, 
                          primary_model_fit_max_iter=60, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        model3 = abessCox(path_type="pgs", s_min=1, s_max=20, important_search=10)
        model3.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model3.coef_)[0]
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
        np.random.seed(0)
        data = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data2 = make_multivariate_glm_data(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessPoisson(path_type="pgs", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                             s_min=1, s_max=p, cv=5,
                             exchange_num=2, 
                             primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessPoisson(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                              s_min=1, s_max=p, cv=5,
                              exchange_num=2, 
                              primary_model_fit_max_iter=80, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)
        model2.predict(data.x)

        model3 = abessPoisson(path_type="pgs", s_min=1, s_max=20, important_search=10)
        model3.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model3.coef_)[0]
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
        data = make_multivariate_glm_data(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, int(n/np.log(np.log(n)) / np.log(p)))

        model = abessMultigaussian(path_type="pgs", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                   s_min=1, s_max=p, cv=5,
                                   exchange_num=2, 
                                   ic_coef=1., thread=5, covariance_update=False)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                    s_min=1, s_max=p, cv=5,
                                    exchange_num=2, 
                                    ic_coef=1., thread=5, covariance_update=True, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        model3 = abessMultigaussian(path_type="pgs", s_min=1, s_max=20, important_search=10)
        model3.fit(data.x, data.y, group=group)

        nonzero_true = np.nonzero(data.coef_)[0]
        nonzero_fit = np.nonzero(model3.coef_)[0]
        print(nonzero_true)
        print(nonzero_fit)
        # new_x = data.x[:, nonzero_fit]
        # reg = linear_model.LinearRegression()
        # reg.fit(new_x, data.y)
        # assert model.coef_[nonzero_fit] == approx(reg.coef_, rel=1e-5, abs=1e-5)
        assert (nonzero_true == nonzero_fit).all()

    def test_mulnomial_gs(self):#to do
        n = 100
        p = 20
        k = 3
        family = "multinomial"
        rho = 0.5
        M = 3
        np.random.seed(5)
        data = make_multivariate_glm_data(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessMultinomial(path_type="pgs", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                 s_min=1, s_max=p, cv=5,
                                 exchange_num=2, 
                                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                  s_min=1, s_max=p, cv=1,
                                  exchange_num=2, 
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        model3 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                  s_min=1, s_max=p, cv=5,
                                  exchange_num=2, 
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model3.fit(data.x, data.y, group=group)

        model4 = abessMultinomial(path_type="pgs", s_min=1, s_max=p, ic_type='gic', important_search=10)
        group = np.linspace(1, p, p)
        model4.fit(data.x, data.y, group=group)

        model5 = abessMultinomial(support_size=support_size, important_search=10)
        model5.fit(data.x, data.y, group=group)

        nonzero_true = np.unique(np.nonzero(data.coef_)[0])
        nonzero_fit = np.unique(np.nonzero(model5.coef_)[0])
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
        data = make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M)
        data2 = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data3 = make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20

        model = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                        s_min=1, s_max=p, cv=5,
                        exchange_num=2, 
                        ic_coef=1., thread=5, covariance_update=True)
        model.fit(data.x + 1, data.y + 1)
        model.predict(data.x)

        model2 = abessLm(path_type="seq", support_size=range(0, s_max), ic_type='ebic', is_screening=True, screening_size=20,
                         s_min=1, s_max=p, cv=5,
                         exchange_num=2, 
                         ic_coef=1., thread=5, covariance_update=True, sparse_matrix=True)
        model2.fit(data.x + 1, data.y + 1)

        assert model.coef_ == approx(model2.coef_, rel=1e-5, abs=1e-5)
        assert model.intercept_ == approx(
            model2.intercept_, rel=1e-5, abs=1e-5)

    def test_binomial_sparse_matrix(self):
        n = 100
        p = 20
        k = 3
        family = "binomial"
        rho = 0.5
        sigma = 1
        np.random.seed(1)
        data = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)
        print("logistic abess")

        model = abessLogistic(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=30,
                              s_min=1, s_max=p, cv=5,
                              exchange_num=2, 
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessLogistic(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=False, screening_size=30,
                               s_min=1, s_max=p, cv=5,
                               exchange_num=2, 
                               primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6,  ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)
        model2.predict(data.x)

        assert model.coef_ == approx(model2.coef_, rel=1e-5, abs=1e-5)
        assert model.intercept_ == approx(
            model2.intercept_, rel=1e-5, abs=1e-5)

    def test_cox_sparse_matrix(self):
        n = 100
        p = 20
        k = 3
        family = "cox"
        rho = 0.5
        sigma = 1

        # np.random.seed(3)
        np.random.seed(3)
        data = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        support_size = range(0, 20)

        model = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                         s_min=1, s_max=p, cv=5,
                         exchange_num=2, 
                         primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x + 1, data.y, group=group)
        model.predict(data.x)

        model2 = abessCox(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                          s_min=1, s_max=p, cv=5,
                          exchange_num=2, 
                          primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x + 1, data.y, group=group)

        assert model.coef_ == approx(model2.coef_, rel=1e-5, abs=1e-5)
        assert model.intercept_ == approx(
            model2.intercept_, rel=1e-5, abs=1e-5)

    def test_poisson_sparse_matrix(self):
        # to do
        n = 100
        p = 20
        k = 3
        family = "poisson"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(0)
        data = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data2 = make_multivariate_glm_data(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessPoisson(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                             s_min=1, s_max=p, cv=5,
                             exchange_num=2, 
                             primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x + 1, data.y, group=group)

        model2 = abessPoisson(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                              s_min=1, s_max=p, cv=5,
                              exchange_num=2, 
                              primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x + 1, data.y, group=group)
        model2.predict(data.x)

        model3 = abessPoisson(support_size=support_size, important_search=10, sparse_matrix=True)
        model3.fit(data.x + 1, data.y, group=group)

        assert model.coef_ == approx(model3.coef_, rel=1e-1, abs=1e-1)
        assert model.intercept_ == approx(
            model3.intercept_, rel=1e-1, abs=1e-1)

    def test_mulgaussian_sparse_matrix(self):
        n = 100
        p = 20
        k = 3
        family = "multigaussian"
        rho = 0.5
        M = 3
        np.random.seed(4)
        data = make_multivariate_glm_data(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, int(n/np.log(np.log(n)) / np.log(p)))

        model = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                   s_min=1, s_max=p, cv=5,
                                   exchange_num=2, 
                                   ic_coef=1., thread=5, covariance_update=True)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultigaussian(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                    s_min=1, s_max=p, cv=5,
                                    exchange_num=2, 
                                    ic_coef=1., thread=5, covariance_update=True, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        assert model.coef_[np.nonzero(model.coef_)[0]] == approx(
            model2.coef_[np.nonzero(model2.coef_)[0]], rel=1e-1, abs=1e-1)
        assert model.intercept_ == approx(
            model2.intercept_, rel=1e-1, abs=1e-1)

    def test_mulnomial_sparse_matrix(self):
        n = 100
        p = 20
        k = 3
        family = "multinomial"
        rho = 0.5
        M = 3
        np.random.seed(5)
        data = make_multivariate_glm_data(
            family=family, n=n, p=p,  k=k, rho=rho, M=M)
        support_size = range(0, 20)

        model = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                 s_min=1, s_max=p, cv=5,
                                 exchange_num=2, 
                                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5)
        group = np.linspace(1, p, p)
        model.fit(data.x, data.y, group=group)

        model2 = abessMultinomial(path_type="seq", support_size=support_size, ic_type='ebic', is_screening=True, screening_size=20,
                                  s_min=1, s_max=p, cv=5,
                                  exchange_num=2, 
                                  primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6, approximate_Newton=True, ic_coef=1., thread=5, sparse_matrix=True)
        group = np.linspace(1, p, p)
        model2.fit(data.x, data.y, group=group)

        assert model.coef_[np.nonzero(model.coef_)[0]] == approx(
            model2.coef_[np.nonzero(model2.coef_)[0]], rel=1e-1, abs=1e-1)
        assert model.intercept_ == approx(
            model2.intercept_, rel=1e-1, abs=1e-1)

    def test_wrong_arg(self):
        n = 100
        p = 20
        k = 3
        family = "gaussian"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(2)
        model5 = abessLm(path_type="other", support_size=range(0, 5), ic_type='dic', is_screening=True, screening_size=20,
                         s_min=1, s_max=p, cv=5,
                         exchange_num=2, 
                         ic_coef=1., thread=5, covariance_update=True)

    def test_gendata(self):
        n = 100
        p = 20
        k = 10
        families = ['gaussian', 'binomial', 'poisson', 'cox']
        for f in families:
            data = make_glm_data(n=n, p=p, family=f, k=k)
            assert data.x.shape[0] == n and data.x.shape[1] == p and data.y.shape[0] == n

            data2 = make_glm_data(n=n, p=p, family=f, k=k, coef_=data.coef_)
            assert (data.coef_ == data2.coef_).all()

        # no-censoring Cox
        data = make_glm_data(n=n, p=p, k=k, family='cox', censoring=False)

        # snr gaussian
        data = make_glm_data(n=n, p=p, k=k, family='gaussian', snr=0.05)

        # multi-response
        data = make_multivariate_glm_data(n=n, p=p, k=k, M=2)
        assert data.x.shape[0] == n and data.x.shape[1] == p and data.y.shape[0] == n and data.y.shape[1] == 2

        data2 = make_multivariate_glm_data(
            n=n, p=p, k=k, M=2, coef_=data.coef_)
        assert (data.coef_ == data2.coef_).all()

        # error input
        try:
            make_glm_data(n=n, p=p, k=k, family='other')
        except ValueError:
            assert True
        else:
            assert False

        try:
            make_multivariate_glm_data(n=n, p=p, k=k, family='other')
        except ValueError:
            assert True
        else:
            assert False

    def test_check(self):

        # X and y
        model = abessLm()
        try:
            model.fit([['c', 1, 1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1, 1, 1]], [1, 2])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([1], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(X=[[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(y=[1])
        except ValueError as e:
            print(e)
        else:
            assert False

        x = coo_matrix(([1, 2, 3], ([0, 1, 2], [0, 1, 2])))
        y = [1, 2, 3]
        model.fit(x, y)

        # Sigma
        model = abessPCA()
        try:
            model.fit(Sigma=[['c']])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(Sigma=[1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(Sigma=[[np.nan]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(Sigma=[[1, 0], [1, 0]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(Sigma=[[-1, 0], [0, -1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        # group & weight
        model = abessLm()
        try:
            model.fit([[1]], [1], weight=['c'])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], [1], weight=[1, 2])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], [1], weight=[[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], [1], group=[1, 2])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], [1], group=[[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        # others
        try:
            model = abessLm(path_type='other')
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(path_type='seq')
            model.fit([[1]], [1])
            model.fit([[1], [2]], [1, 2])
            model.fit([[1, 1], [2, 2]], [1, 2])

            model = abessLm(path_type='seq', support_size=[3])
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(path_type='pgs')
            model.fit([[1]], [1])

            model = abessLm(path_type='pgs', s_min=1, s_max=0)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(ic_type='other')
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(exchange_num=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(is_screening=True)
            model = abessLm(is_screening=True, screening_size=3)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(is_screening=True, support_size=[
                            2], screening_size=1)
            model.fit([[1, 2, 3]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLogistic(primary_model_fit_max_iter=0.5)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLogistic(primary_model_fit_epsilon=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLogistic(primary_model_fit_epsilon=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(thread=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(splicing_type=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessPCA()
            model.fit([[1]], [1], number=-1)
        except ValueError as e:
            print(e)
        else:
            assert False

        # full
        model = abessLm(support_size=[1])
        model.fit([[1]], [1])

    def test_score(self):
        model = abessLm()
        model.fit([[1]], [1])
        model.score([[1]], [1])

        model = abessMultigaussian()
        model.fit([[1]], [[1, 1]])
        model.score([[1]], [[1, 1]])

        model = abessLogistic()
        model.fit([[1]], [1])
        model.score([[1]], [1])

        model = abessMultinomial()
        model.fit([[1]], [[1, 1]])
        model.score([[1]], [[1, 1]])

        try:
            model = abessLm()
            model.fit([[1]], [1])
            model.score([1, 2], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm()
            model.fit([[1]], [1])
            model.score([[1, 2]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessCox()
            model.fit([[1]], [[1, 1]])
            model.score([[1]], [[1, 1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessCox()
            model.fit([[1]], [[1, 1]])
            model.score([[1], [2]], [[1, 0], [2, 0]])
        except ValueError as e:
            print(e)
        else:
            assert False
        
        try:
            model = abessLm(cv = 2)
            model.fit([[1], [2]], [1, 2], cv_fold_id = [1, 1])
        except ValueError as e:
            print(e)
        else:
            assert False


    def test_other(self):
        n = 100
        p = 20
        k = 3
        family = "gaussian"
        rho = 0.5
        sigma = 1
        M = 1
        np.random.seed(2)
        data = make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M)
        data2 = make_glm_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
        data3 = make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20

        model = abessLm(path_type="seq", support_size=range(0, 10), ic_type='aic', is_screening=False, screening_size=20,
                        s_min=1, s_max=p, cv=1,
                        exchange_num=2,  is_warm_start=False,
                        ic_coef=1., thread=5, covariance_update=False)
        model.fit(data.x, data.y)

        model = abessLm(path_type="pgs", support_size=range(0, 10), ic_type='aic', is_screening=False, screening_size=20,
                        s_min=1, s_max=p, cv=1,
                        exchange_num=2,  is_warm_start=False,
                        ic_coef=1., thread=5, covariance_update=False)
        model.fit(data.x, data.y)

        model = abessLm(support_size=range(0, 10),  cv=2)
        cv_fold_id = [1 for i in range(50)] + [2 for i in range(n - 50)]
        model.fit(data.x, data.y, cv_fold_id = cv_fold_id)