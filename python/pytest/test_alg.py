import sys
import abess
import pytest
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    PoissonRegressor)
from sklearn.utils.estimator_checks import check_estimator

try:
    import pandas as pd
    from lifelines import CoxPHFitter
    miss_dep = False
except ImportError:
    miss_dep = True

from utilities import (  # noqa: F401
    assert_nan,
    assert_value,
    assert_fit,
    save_data,
    load_data)


@pytest.mark.filterwarnings("ignore")
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
        test_data = abess.make_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, coef_=data.coef_)

        # save_data(data, 'gaussian')
        # save_data(test_data, 'gaussian_test')
        data = load_data("gaussian")
        test_data = load_data("gaussian_test")

        def assert_reg(coef, fit_intercept=True, rel=0.01, abs=0.01):
            if (sys.version_info[0] < 3 or sys.version_info[1] < 6):
                return
            nonzero = np.nonzero(coef)[0]
            new_x = data.x[:, nonzero]
            reg = LinearRegression(fit_intercept=fit_intercept)
            reg.fit(new_x, data.y.reshape(-1))
            assert_value(coef[nonzero], reg.coef_, rel, abs)

        # null
        check_estimator(abess.LinearRegression())
        model1 = abess.LinearRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)
        assert_reg(model1.coef_)

        model0 = abess.LinearRegression(fit_intercept=False)
        model0.fit(data.x, data.y)
        assert model0.intercept_ == 0
        assert_fit(model0.coef_, data.coef_)
        assert_reg(model0.coef_, fit_intercept=False)

        # predict
        y = model1.predict(test_data.x)
        assert_nan(y)

        # score
        score = model1.score(test_data.x, test_data.y)
        sample_weight = np.random.rand(n)
        score = model1.score(test_data.x, test_data.y,
                             sample_weight=sample_weight)
        assert score > 0.5

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
        cv_fold_id = np.repeat(np.linspace(1, 5, 5), int(n / 5))
        model4.fit(data.x, data.y, cv_fold_id=cv_fold_id)
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
        test_data = abess.make_glm_data(
            family=family,
            n=n,
            p=p,
            k=k,
            rho=rho,
            sigma=sigma,
            coef_=data.coef_)

        # save_data(data, 'binomial')
        # save_data(test_data, 'binomial_test')
        data = load_data("binomial")
        test_data = load_data("binomial_test")

        def assert_reg(coef, fit_intercept=True, rel=0.01, abs=0.01):
            if sys.version_info[0] + 0.1 * sys.version_info[1] < 3.6:
                return
            nonzero = np.nonzero(coef)[0]
            new_x = data.x[:, nonzero]
            reg = LogisticRegression(
                penalty="none", fit_intercept=fit_intercept)
            reg.fit(new_x, data.y)
            assert_value(coef[nonzero], reg.coef_, rel, abs)

        # null
        check_estimator(abess.LogisticRegression())
        model1 = abess.LogisticRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)
        assert_reg(model1.coef_)

        model0 = abess.LogisticRegression(fit_intercept=False)
        model0.fit(data.x, data.y)
        assert model0.intercept_ == 0
        assert_fit(model0.coef_, data.coef_)
        assert_reg(model0.coef_, fit_intercept=False)

        # predict
        prob = model1.predict_proba(test_data.x)
        assert_nan(prob)
        y = model1.predict(test_data.x)
        assert_nan(y)

        # score
        score = model1.score(test_data.x, test_data.y)
        sample_weight = np.random.rand(n)
        score = model1.score(test_data.x, test_data.y,
                             sample_weight=sample_weight)
        assert score > 0.5

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

        # save_data(data, 'cox')
        data = load_data("cox")

        def assert_reg(coef):
            if miss_dep:
                pytest.skip(
                    "Skip because modules 'pandas' or 'lifelines'"
                    " have not been installed.")

            if sys.version_info[0] + 0.1 * sys.version_info[1] < 3.6:
                pytest.skip("Skip because requiring python3.6 or higher.")

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
        check_estimator(abess.CoxPHSurvivalAnalysis())
        model1 = abess.CoxPHSurvivalAnalysis()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)
        assert_reg(model1.coef_)

        # predict
        y = model1.predict(data.x)
        assert_nan(y)

        # score
        score = model1.score(data.x, data.y)
        sample_weight = np.random.rand(n)
        score = model1.score(data.x, data.y,
                             sample_weight=sample_weight)
        assert not np.isnan(score)

        # approximate Newton
        model2 = abess.CoxPHSurvivalAnalysis(approximate_Newton=True)
        model2.fit(data.x, data.y)
        # assert_fit(model1.coef_, model2.coef_)    # TODO
        assert_reg(model2.coef_)

        # survival function
        surv = model1.predict_survival_function(data.x)
        time_points = np.quantile(data.y[:, 0], np.linspace(0, 0.6, 100))
        surv[0](time_points)

    @staticmethod
    def test_poisson():
        np.random.seed(2)
        n = 100
        p = 20
        k = 3
        family = "poisson"
        rho = 0.5
        sigma = 1
        data = abess.make_glm_data(
            n, p, family=family, k=k, rho=rho, sigma=sigma)
        test_data = abess.make_glm_data(
            n, p, family=family, k=k, rho=rho, sigma=sigma, coef_=data.coef_)

        # save_data(data, 'poisson')
        # save_data(test_data, 'poisson_test')
        data = load_data("poisson")
        test_data = load_data("poisson_test")

        def assert_reg(coef, fit_intercept=True, rel=0.1, abs=0.1):
            if sys.version_info[0] + 0.1 * sys.version_info[1] < 3.6:
                return
            nonzero = np.nonzero(coef)[0]
            new_x = data.x[:, nonzero]
            reg = PoissonRegressor(
                fit_intercept=fit_intercept,
                alpha=0, tol=1e-6, max_iter=200)
            reg.fit(new_x, data.y)
            assert_value(coef[nonzero], reg.coef_, rel, abs)

        # null
        check_estimator(abess.PoissonRegression())
        model1 = abess.PoissonRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)
        assert_reg(model1.coef_)

        model0 = abess.PoissonRegression(fit_intercept=False)
        model0.fit(data.x, data.y)
        assert model0.intercept_ == 0
        assert_fit(model0.coef_, data.coef_)
        assert_reg(model0.coef_, fit_intercept=False)

        # predict
        y = model1.predict(test_data.x)
        assert_nan(y)

        # score
        score = model1.score(test_data.x, test_data.y)
        sample_weight = np.random.rand(n)
        score = model1.score(test_data.x, test_data.y,
                             sample_weight=sample_weight)
        assert score > 0.5

        # approximate Newton
        model2 = abess.PoissonRegression(approximate_Newton=True)
        model2.fit(data.x, data.y)
        assert_fit(model1.coef_, model2.coef_)
        assert_reg(model2.coef_)

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
        test_data = abess.make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M, coef_=data.coef_)

        # save_data(data, "multigaussian")
        # save_data(test_data, "multigaussian_test")
        data = load_data("multigaussian")
        test_data = load_data("multigaussian_test")

        # null
        check_estimator(abess.MultiTaskRegression())
        model1 = abess.MultiTaskRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)

        model0 = abess.MultiTaskRegression(fit_intercept=False)
        model0.fit(data.x, data.y)
        assert np.count_nonzero(model0.intercept_) == 0
        assert_fit(model0.coef_, data.coef_)

        # predict
        y = model1.predict(test_data.x)
        assert_nan(y)

        # score
        score = model1.score(test_data.x, test_data.y)
        sample_weight = np.random.rand(n)
        score = model1.score(test_data.x, test_data.y,
                             sample_weight=sample_weight)
        assert score > 0.5

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
        cv_fold_id = np.repeat(np.linspace(1, 5, 5), int(n / 5))
        model4.fit(data.x, data.y, cv_fold_id=cv_fold_id)
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
        test_data = abess.make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M, coef_=data.coef_)

        # save_data(data, 'multinomial')
        # save_data(test_data, 'multinomial_test')
        data = load_data('multinomial')
        test_data = load_data('multinomial_test')

        # null
        check_estimator(abess.MultinomialRegression())
        model1 = abess.MultinomialRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)

        model0 = abess.MultinomialRegression(fit_intercept=False)
        model0.fit(data.x, data.y)
        assert np.count_nonzero(model0.intercept_) == 0
        assert_fit(model0.coef_, data.coef_)

        # predict
        y = model1.predict(test_data.x)
        assert_nan(y)

        # score
        score = model1.score(test_data.x, test_data.y)
        sample_weight = np.random.rand(n)
        score = model1.score(test_data.x, test_data.y,
                             sample_weight=sample_weight)
        assert score > 0.5

        # # approximate Newton
        # model2 = abess.MultinomialRegression(approximate_Newton=True)
        # model2.fit(data.x, data.y)
        # assert_fit(model1.coef_, model2.coef_)

        # categorical y
        cate_y = np.repeat(np.arange(n / 10), 10)
        model1.fit(data.x, cate_y)
        score = model1.score(data.x, cate_y,
                             sample_weight=sample_weight)
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

        # save_data(X, 'PCA')
        X = load_data('PCA')

        # null
        check_estimator(abess.SparsePCA())
        model1 = abess.SparsePCA(support_size=support_size)
        model1.fit(X)
        assert np.count_nonzero(model1.coef_) == s

        # ratio & transform
        model1.ratio(X)
        model1.transform(X)
        model1.fit_transform(X)

        # sparse
        model2 = abess.SparsePCA(support_size=s)
        model2.fit(coo_matrix(X), sparse_matrix=True)
        print("coef1: ", np.unique(np.nonzero(model1.coef_)[0]))
        print("coef2: ", np.unique(np.nonzero(model2.coef_)[0]))
        assert_value(model1.coef_, model2.coef_)

        model2 = abess.SparsePCA(support_size=s)
        model2.fit(X, sparse_matrix=True)
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
        model5 = abess.SparsePCA(support_size=support_size_g, group=group)
        model5.fit(X)
        coef = g_index[np.nonzero(model5.coef_)[0]]

        assert len(coef) == 10
        assert len(np.unique(coef)) == 2

        # screening
        model6 = abess.SparsePCA(support_size=support_size, screening_size=20)
        model6.fit(X)
        assert_nan(model6.coef_)

        # ic
        for ic in ['loss', 'aic', 'bic', 'ebic', 'gic', 'hic']:
            model = abess.SparsePCA(support_size=support_size, ic_type=ic)
            model.fit(X)

        # A_init
        model = abess.SparsePCA(support_size=support_size, A_init=[0, 1, 2])
        model.fit(X)

    @staticmethod
    def test_gamma():
        np.random.seed(0)
        n = 10000
        p = 20
        k = 3
        data = abess.make_glm_data(n=n, p=p, k=k, family="gamma")

        # save_data(data, 'gamma')
        data = load_data('gamma')

        # null
        check_estimator(abess.GammaRegression())
        model1 = abess.GammaRegression(support_size=k)
        model1.fit(data.x, data.y)
        assert_nan(model1.coef_)
        assert_fit(data.coef_, model1.coef_)
        assert_value(data.coef_, model1.coef_, 1., 1.)

        model0 = abess.GammaRegression(support_size=k, fit_intercept=False)
        model0.fit(data.x, data.y)
        assert model0.intercept_ == 0
        assert_nan(model0.coef_)

        # predict
        model1.predict(data.x)

        # score
        score = model1.score(data.x, data.y)
        sample_weight = np.random.rand(n)
        score = model1.score(data.x, data.y,
                             sample_weight=sample_weight)
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

        # save_data(X, 'RPCA')
        X = load_data('RPCA')

        # null
        check_estimator(abess.RobustPCA())
        model1 = abess.RobustPCA(support_size=s)
        model1.fit(X)
        model1.fit(X, r=r)
        # assert_fit(model1.coef_, S)

        # sparse
        model2 = abess.RobustPCA(support_size=s)
        model2.fit(coo_matrix(X), r=r)
        assert_value(model1.coef_, model2.coef_)

        model2 = abess.RobustPCA(support_size=s)
        model2.fit(X, r=r, sparse_matrix=True)
        assert_value(model1.coef_, model2.coef_)

        # # group
        # group = np.arange(n * p)
        # model3 = abess.RobustPCA(support_size=s, group=group)
        # model3.fit(X, r=r)

        # ic
        for ic in ['aic', 'bic', 'ebic', 'gic', 'hic']:
            model4 = abess.RobustPCA(support_size=s, ic_type=ic)
            model4.fit(X, r=r)

        # always select
        model5 = abess.RobustPCA(support_size=s, always_select=[1])
        model5.fit(X, r=r)

    @staticmethod
    def test_ordinal():
        np.random.seed(2)
        data = abess.make_glm_data(n=100, p=20, k=5, family="ordinal")

        # save_data(data, 'ordinal')
        data = load_data('ordinal')

        # null
        check_estimator(abess.OrdinalRegression())
        model1 = abess.OrdinalRegression()
        model1.fit(data.x, data.y)
        assert_fit(model1.coef_, data.coef_)

        # score
        sample_weight = np.random.rand(100)
        score_ordinal = model1.score(data.x, data.y,
                                     sample_weight=sample_weight)
        score_ordinal = model1.score(data.x, data.y)
        y_random = data.y.copy()
        np.random.shuffle(y_random)
        score_random = ndcg_score(data.y.reshape(
            (1, -1)), y_random.reshape((1, -1)))
        assert score_ordinal > score_random

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

        # save_data(data, 'gaussian_sklearn')
        data = load_data('gaussian_sklearn')

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
        n = 500
        p = 20
        k = 3
        family = "binomial"
        rho = 0.5
        sigma = 1
        np.random.seed(2)
        data = abess.make_glm_data(
            n, p, family=family, k=k, rho=rho, sigma=sigma)
        # data3 = abess.make_multivariate_glm_data(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)

        # save_data(data, "binomial_sklearn")
        data = load_data("binomial_sklearn")

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

        # save_data(data, "poisson_sklearn")
        data = load_data("poisson_sklearn")

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
        # assert gcv.best_params_["alpha"] == 0.

    @ staticmethod
    def test_cox_sklearn():
        n = 100
        p = 20
        k = 3
        family = "cox"
        rho = 0.5
        # sigma = 1
        # M = 1
        np.random.seed(1)
        data = abess.make_glm_data(n, p, family=family, k=k, rho=rho)
        # data3 = abess.make_multivariate_glm_data(
        #     family=family, n=n, p=p, k=k, rho=rho, M=M, sparse_ratio=0.1)

        # save_data(data, "cox_sklearn")
        data = load_data("cox_sklearn")

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
