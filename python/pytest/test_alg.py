import sys
import warnings
import abess
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    PoissonRegressor)
from sklearn.utils.estimator_checks import check_estimator
from lifelines import CoxPHFitter
from utilities import (
    assert_nan,
    assert_value,
    assert_fit,
    save_data,
    load_data)

warnings.filterwarnings("ignore")


class TestAlgorithm:
    """
    Test for each algorithm.
    """

    @staticmethod
    def test_gaussian():
        np.random.seed(2)
        n = 100
        p = 20
        k = 3
        family = "gaussian"
        rho = 0.1

        data = abess.make_glm_data(family=family, n=n, p=p, k=k, rho=rho)

        def assert_reg(coef):
            if (sys.version_info[0] < 3 or sys.version_info[1] < 6):
                return
            nonzero = np.nonzero(coef)[0]
            new_x = data.x[:, nonzero]
            reg = LinearRegression()
            reg.fit(new_x, data.y.reshape(-1))
            assert_value(coef[nonzero], reg.coef_)

        # null
        check_estimator(abess.LinearRegression())
        model1 = abess.LinearRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)
        assert_reg(model1.coef_)

        # predict
        y = model1.predict(data.x)
        assert_nan(y)

        # score
        score = model1.score(data.x, data.y)
        assert not np.isnan(score)

        # covariance update
        model2 = abess.LinearRegression(covariance_update=True)
        model2.fit(data.x, data.y)
        assert_value(model1.coef_, model2.coef_)

        model3 = abess.LinearRegression(
            covariance_update=True,
            important_search=10,
            screening_size=20,
            cv=5)
        model3.fit(data.x, data.y)
        assert_fit(model3.coef_, data.coef_)

        model4 = abess.LinearRegression(
            covariance_update=True, path_type='gs', cv=5)
        model4.fit(data.x, data.y)
        assert_fit(model4.coef_, data.coef_)

    @staticmethod
    def test_binomial():
        np.random.seed(2)
        n = 300
        p = 20
        k = 3
        family = "binomial"
        rho = 0.5
        sigma = 1

        data = abess.make_glm_data(
            family=family,
            n=n,
            p=p,
            k=k,
            rho=rho,
            sigma=sigma)

        def assert_reg(coef):
            if sys.version_info[0] + 0.1 * sys.version_info[1] < 3.6:
                return
            nonzero = np.nonzero(coef)[0]
            new_x = data.x[:, nonzero]
            reg = LogisticRegression(penalty="none")
            reg.fit(new_x, data.y)
            assert_value(coef[nonzero], reg.coef_)

        # null
        # check_estimator(abess.LogisticRegression())
        model1 = abess.LogisticRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)
        assert_reg(model1.coef_)

        # predict
        prob = model1.predict_proba(data.x)
        assert_nan(prob)
        y = model1.predict(data.x)
        assert_nan(y)

        # score
        score = model1.score(data.x, data.y)
        assert not np.isnan(score)

        # approximate Newton
        model2 = abess.LogisticRegression(approximate_Newton=True)
        model2.fit(data.x, data.y)
        assert_fit(model1.coef_, model2.coef_)

    @staticmethod
    def test_cox():
        np.random.seed(2)
        n = 200
        p = 20
        k = 3
        family = "cox"
        rho = 0.5
        sigma = 1

        data = abess.make_glm_data(
            n, p, family=family, k=k, rho=rho, sigma=sigma)

        def assert_reg(coef):
            if sys.version_info[0] + 0.1 * sys.version_info[1] < 3.6:
                return
            nonzero = np.nonzero(coef)[0]
            new_x = data.x[:, nonzero]
            survival = pd.DataFrame()
            for i in range(new_x.shape[1]):
                survival["Var" + str(i)] = new_x[:, i]
            survival["T"] = data.y[:, 0]
            survival["E"] = data.y[:, 1]
            cph = CoxPHFitter(penalizer=0, l1_ratio=0)
            cph.fit(survival, 'T', event_col='E')
            assert_value(coef[nonzero], cph.params_.values, rel=5e-1, abs=5e-1)

        # null
        # check_estimator(abess.CoxPHSurvivalAnalysis())
        model1 = abess.CoxPHSurvivalAnalysis()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)
        assert_reg(model1.coef_)

        # predict
        y = model1.predict(data.x)
        assert_nan(y)

        # score
        score = model1.score(data.x, data.y)
        assert not np.isnan(score)

        # approximate Newton
        model2 = abess.CoxPHSurvivalAnalysis(approximate_Newton=True)
        model2.fit(data.x, data.y)
        # assert_fit(model1.coef_, model2.coef_)    # TODO
        assert_reg(model2.coef_)

    @staticmethod
    def test_poisson():
        np.random.seed(9)
        n = 100
        p = 20
        k = 3
        family = "poisson"
        rho = 0.5
        sigma = 1
        data = abess.make_glm_data(
            n, p, family=family, k=k, rho=rho, sigma=sigma)

        def assert_reg(coef):
            if sys.version_info[0] + 0.1 * sys.version_info[1] < 3.6:
                return
            nonzero = np.nonzero(coef)[0]
            new_x = data.x[:, nonzero]
            reg = PoissonRegressor(
                alpha=0, tol=1e-6, max_iter=200)
            reg.fit(new_x, data.y)
            assert_value(coef[nonzero], reg.coef_)

        # null
        # check_estimator(abess.PoissonRegression())
        model1 = abess.PoissonRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)
        assert_reg(model1.coef_)

        # predict
        y = model1.predict(data.x)
        assert_nan(y)

        # score
        score = model1.score(data.x, data.y)
        assert not np.isnan(score)

    @staticmethod
    def test_multigaussian():
        np.random.seed(1)
        n = 100
        p = 20
        k = 3
        family = "multigaussian"
        rho = 0.5
        M = 3
        data = abess.make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M)

        # save_data(data, "multigaussian_seed1_rho0.5")
        data = load_data("multigaussian_seed1_rho0.5")

        # null
        # check_estimator(abess.MultiTaskRegression())
        model1 = abess.MultiTaskRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)

        # predict
        y = model1.predict(data.x)
        assert_nan(y)

        # score
        score = model1.score(data.x, data.y)
        assert not np.isnan(score)

        # covariance update
        model2 = abess.MultiTaskRegression(covariance_update=True)
        model2.fit(data.x, data.y)
        assert_value(model1.coef_, model2.coef_)

        model3 = abess.MultiTaskRegression(
            covariance_update=True,
            important_search=10,
            screening_size=20,
            cv=5)
        model3.fit(data.x, data.y)
        assert_fit(model3.coef_, data.coef_)

        model4 = abess.MultiTaskRegression(
            covariance_update=True, path_type='gs', cv=5)
        model4.fit(data.x, data.y)
        assert_fit(model4.coef_, data.coef_)

    @staticmethod
    def test_multinomial():
        np.random.seed(5)
        n = 100
        p = 20
        k = 3
        family = "multinomial"
        rho = 0.5
        M = 3

        data = abess.make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M)

        # save_data(data, 'multinomial_seed5_rho0.5')
        data = load_data('multinomial_seed5_rho0.5')

        # null
        # check_estimator(abess.MultinomialRegression())
        model1 = abess.MultinomialRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)

        # predict
        y = model1.predict(data.x)
        assert_nan(y)

        # score
        score = model1.score(data.x, data.y)
        assert not np.isnan(score)

        # # approximate Newton
        # model2 = abess.MultinomialRegression(approximate_Newton=True)
        # model2.fit(data.x, data.y)
        # assert_fit(model1.coef_, model2.coef_)

        # categorical y
        cate_y = np.repeat(np.arange(n / 10), 10)
        model1.fit(data.x, cate_y)
        score = model1.score(data.x, cate_y)
        assert not np.isnan(score)

    @staticmethod
    def test_PCA():
        np.random.seed(1)
        n = 1000
        p = 20
        s = 10
        group_size = 5
        group_num = 4
        support_size = np.zeros((p, 1))
        support_size[s - 1, 0] = 1

        x1 = np.random.randn(n, 1)
        x1 /= np.linalg.norm(x1)
        X = x1.dot(np.random.randn(1, p)) + 0.01 * np.random.randn(n, p)
        X = X - X.mean(axis=0)
        g_index = np.arange(group_num)
        g_index = g_index.repeat(group_size)

        # save_data(X, 'PCA_seed1')
        X = load_data('PCA_seed1')

        # null
        model1 = abess.SparsePCA(support_size=support_size)
        model1.fit(X)
        assert np.count_nonzero(model1.coef_) == s

        # ratio & transform
        model1.ratio(X)
        model1.transform(X)
        model1.fit_transform(X)

        # sparse
        model2 = abess.SparsePCA(support_size=s, sparse_matrix=True)
        model2.fit(coo_matrix(X))
        print("coef1: ", np.unique(np.nonzero(model1.coef_)[0]))
        print("coef2: ", np.unique(np.nonzero(model2.coef_)[0]))
        assert_value(model1.coef_, model2.coef_)

        model2 = abess.SparsePCA(support_size=s, sparse_matrix=True)
        model2.fit(X)
        assert_value(model1.coef_, model2.coef_)

        # sigma input
        model3 = abess.SparsePCA(support_size=support_size)
        model3.fit(Sigma=X.T.dot(X))
        model3.fit(Sigma=np.cov(X.T), n=n)
        assert_fit(model1.coef_, model3.coef_)

        # KPCA
        support_size_m = np.hstack((support_size, support_size, support_size))
        model4 = abess.SparsePCA(support_size=support_size_m)
        model4.fit(X, number=3)
        assert model4.coef_.shape[1] == 3

        for i in range(3):
            coef = np.nonzero(model4.coef_[:, i])[0]
            assert len(coef) == s

        model4.ratio(X)

        # group
        support_size_g = np.zeros((4, 1))
        support_size_g[1, 0] = 1
        group = np.repeat([0, 1, 2, 3], [5, 5, 5, 5])
        model5 = abess.SparsePCA(support_size=support_size_g)
        model5.fit(X, group=group)
        coef = g_index[np.nonzero(model5.coef_)[0]]

        assert len(coef) == 10
        assert len(np.unique(coef)) == 2

        # screening
        model6 = abess.SparsePCA(support_size=support_size, screening_size=20)
        model6.fit(X)
        assert_nan(model6.coef_)

        # ic
        for ic in ['aic', 'bic', 'ebic', 'gic']:
            model4 = abess.SparsePCA(support_size=support_size, ic_type=ic)
            model4.fit(X, is_normal=False)

    @staticmethod
    def test_gamma():
        np.random.seed(1)
        data = abess.make_glm_data(n=100, p=10, k=3, family="gamma")

        # null
        # check_estimator(abess.GammaRegression())
        model1 = abess.GammaRegression()
        model1.fit(data.x, data.y)
        assert_nan(model1.coef_)

        # predict
        model1.predict(data.x)

        # score
        score = model1.score(data.x, data.y)
        score = model1.score(data.x, data.y, np.ones(data.x.shape[0]))
        assert not np.isnan(score)

    @staticmethod
    def test_RPCA():
        np.random.seed(2)
        n = 100
        p = 20
        s = 30
        r = 5

        L = np.random.rand(n, r) @ np.random.rand(r, p)
        nonzero = np.random.choice(n * p, s, replace=False)
        S = np.zeros(n * p)
        S[nonzero] = np.random.rand(s) * 10
        S = S.reshape(p, n).T
        X = L + S

        # null
        model1 = abess.RobustPCA(support_size=s)
        model1.fit(X, r=r)
        # assert_fit(model1.coef_, S)

        # sparse
        model2 = abess.RobustPCA(support_size=s)
        model2.fit(coo_matrix(X), r=r)
        assert_value(model1.coef_, model2.coef_)

        model2 = abess.RobustPCA(support_size=s, sparse_matrix=True)
        model2.fit(X, r=r)
        assert_value(model1.coef_, model2.coef_)

        # group
        group = np.arange(n * p)
        model3 = abess.RobustPCA(support_size=s)
        model3.fit(X, r=r, group=group)

        # ic
        for ic in ['aic', 'bic', 'ebic', 'gic']:
            model4 = abess.RobustPCA(support_size=s, ic_type=ic)
            model4.fit(X, r=r)

    @staticmethod
    def test_ordinal():
        np.random.seed(0)
        data = abess.make_glm_data(n=100, p=20, k=5, family="ordinal")

        # null
        model1 = abess.OrdinalRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)

        pred = model1.predict(data.x)
        print((pred != data.y).sum())
        # assert (pred == data.y)

    @staticmethod
    def test_gaussian_sklearn():
        np.random.seed(7)
        n = 100
        p = 20
        k = 3
        family = "gaussian"
        rho = 0.5
        s_max = 20

        data = abess.make_glm_data(n, p, family=family, k=k, rho=rho)

        support_size = np.linspace(0, s_max, s_max + 1, dtype="int32")
        alpha = [0., 0.1, 0.2, 0.3, 0.4]

        try:
            model = abess.LinearRegression()
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
        except BaseException:
            assert False

    @staticmethod
    def test_binomial_sklearn():
        n = 100
        p = 20
        k = 3
        family = "binomial"
        rho = 0.5
        sigma = 1
        np.random.seed(3)
        data = abess.make_glm_data(
            n, p, family=family, k=k, rho=rho, sigma=sigma)
        # data3 = abess.make_multivariate_glm_data(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20
        support_size = np.linspace(0, s_max, s_max + 1, dtype="int32")
        alpha = [0., 0.1, 0.2, 0.3, 0.4]

        model = abess.LogisticRegression()
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

    @staticmethod
    def test_poisson_sklearn():
        n = 100
        p = 20
        k = 3
        family = "poisson"
        rho = 0.5
        # sigma = 1
        # M = 1
        np.random.seed(3)
        data = abess.make_glm_data(n, p, family=family, k=k, rho=rho)
        # data3 = abess.make_multivariate_glm_data(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 20
        support_size = np.linspace(0, s_max, s_max + 1, dtype="int32")
        alpha = [0., 0.1, 0.2, 0.3, 0.4]

        model = abess.PoissonRegression()
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

    @staticmethod
    def test_cox_sklearn():
        n = 100
        p = 20
        k = 3
        family = "cox"
        rho = 0.5
        # sigma = 1
        # M = 1
        np.random.seed(3)
        data = abess.make_glm_data(n, p, family=family, k=k, rho=rho)
        # data3 = abess.make_multivariate_glm_data(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
        s_max = 10
        support_size = np.linspace(1, s_max, s_max + 1, dtype="int32")
        alpha = [0., 0.1, 0.2, 0.3]

        model = abess.CoxPHSurvivalAnalysis(
            path_type="seq", support_size=support_size,
            ic_type='ebic', screening_size=20,
            s_min=1, s_max=p, cv=5,
            exchange_num=2,
            primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-6,
            approximate_Newton=True, ic_coef=1., thread=5)
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

    # @staticmethod
    # def test_multigaussian_sklearn():
    #     n = 100
    #     p = 20
    #     k = 3
    #     family = "multigaussian"
    #     rho = 0.5
    #     sigma = 1
    #     M = 1
    #     np.random.seed(2)
    #     data = abess.make_multivariate_glm_data(
    #         family=family, n=n, p=p,  k=k, rho=rho, M=M)
    #     # data3 = abess.make_multivariate_glm_data(
    #     #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
    #     s_max = 20
    #     support_size = np.linspace(1, s_max, s_max+1)
    #     alpha = [0., 0.1, 0.2, 0.3, 0.4]

    #     model = abess.MultiTaskRegression()
    #     cv = KFold(n_splits=5, shuffle=True, random_state=0)
    #     gcv = GridSearchCV(
    #         model,
    #         param_grid={"support_size": support_size,
    #                     "alpha": alpha},
    #         cv=cv,
    #         n_jobs=1).fit(data.x, data.y)

    #     assert gcv.best_params_["support_size"] == k
    #     assert gcv.best_params_["alpha"] == 0.

    # @staticmethod
    # def test_multinomial_sklearn():
    #     n = 100
    #     p = 20
    #     k = 3
    #     family = "multinomial"
    #     rho = 0.5
    #     sigma = 1
    #     M = 1
    #     np.random.seed(2)
    #     data = abess.make_multivariate_glm_data(
    #         family=family, n=n, p=p,  k=k, rho=rho, M=M)
    #     # data3 = abess.make_multivariate_glm_data(
    #     #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)
    #     s_max = 20
    #     support_size = np.linspace(0, s_max, s_max+1, dtype = "int32")
    #     alpha = [0., 0.1, 0.2, 0.3, 0.4]

    #     model = abess.MultinomialRegression()
    #     cv = KFold(n_splits=5, shuffle=True, random_state=0)
    #     gcv = GridSearchCV(
    #         model,
    #         param_grid={"support_size": support_size,
    #                     "alpha": alpha},
    #         cv=cv,
    #         n_jobs=1).fit(data.x, data.y)

    #     assert gcv.best_params_["support_size"] == k
    #     assert gcv.best_params_["alpha"] == 0.
