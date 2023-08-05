from time import time
import pytest
import numpy as np
from utilities import (assert_nan, assert_value, assert_fit)
from scipy.sparse import coo_matrix
import abess


@pytest.mark.filterwarnings("ignore")
class TestWorkflow:
    """
    Test for abess workflow in cpp. (Take `LinearRegression` as an example.)
    """

    @staticmethod
    def test_sparse():
        np.random.seed(0)
        n = 100
        p = 20
        k = 5

        # glm
        data = abess.make_glm_data(n=n, p=p, k=k, family='gaussian')

        model1 = abess.LinearRegression()
        model1.fit(data.x, data.y)
        model2 = abess.LinearRegression()
        model2.fit(coo_matrix(data.x), data.y, sparse_matrix=True)
        assert_value(model1.coef_, model2.coef_)
        assert_value(model1.intercept_, model2.intercept_)

        model3 = abess.LinearRegression()
        model3.fit(data.x, data.y, sparse_matrix=True)
        assert_value(model1.coef_, model3.coef_)
        assert_value(model1.intercept_, model2.intercept_)

        # pca
        data_pca = np.random.randn(n, p)

        model1 = abess.SparsePCA()
        model1.fit(data_pca)
        model2 = abess.SparsePCA()
        model2.fit(coo_matrix(data_pca), sparse_matrix=True)
        assert_value(model1.coef_, model2.coef_)

        model3 = abess.SparsePCA()
        model3.fit(data_pca, sparse_matrix=True)
        assert_value(model1.coef_, model3.coef_)

    @staticmethod
    def test_path():
        np.random.seed(0)
        n = 100
        p = 20
        k = 5
        s_min = 0
        s_max = 10
        data = abess.make_glm_data(n=n, p=p, k=k, family='gaussian')

        # null
        model1 = abess.LinearRegression(
            path_type='seq', support_size=range(s_max))
        model1.fit(data.x, data.y)
        model2 = abess.LinearRegression(
            path_type='gs', s_min=s_min, s_max=s_max)
        model2.fit(data.x, data.y)
        assert_fit(model1.coef_, model2.coef_)

        # cv
        t1 = time()
        model1 = abess.LinearRegression(
            path_type='seq', support_size=range(s_max), cv=5)
        model1.fit(data.x, data.y)
        model2 = abess.LinearRegression(
            path_type='gs', s_min=s_min, s_max=s_max, cv=5)
        model2.fit(data.x, data.y)
        t1 = time() - t1
        assert_fit(model1.coef_, model2.coef_)

        # thread
        t2 = time()
        model1 = abess.LinearRegression(
            path_type='seq',
            support_size=range(s_max),
            cv=5,
            thread=0)
        model1.fit(data.x, data.y)
        model2 = abess.LinearRegression(
            path_type='gs',
            s_min=s_min,
            s_max=s_max,
            cv=5,
            thread=0)
        model2.fit(data.x, data.y)
        t2 = time() - t2
        assert_fit(model1.coef_, model2.coef_)
        # assert t2 < t1

        # warm_start
        model1 = abess.LinearRegression(
            path_type='seq',
            support_size=range(s_max),
            is_warm_start=False)
        model1.fit(data.x, data.y)
        model2 = abess.LinearRegression(
            path_type='gs',
            s_min=s_min,
            s_max=s_max,
            is_warm_start=False)
        model2.fit(data.x, data.y)
        assert_value(model1.coef_, model2.coef_, 0, 0)

        model1 = abess.LinearRegression(
            path_type='seq',
            support_size=range(s_max),
            is_warm_start=False,
            cv=5)
        model1.fit(data.x, data.y)
        model2 = abess.LinearRegression(
            path_type='gs',
            s_min=s_min,
            s_max=s_max,
            is_warm_start=False,
            cv=5)
        model2.fit(data.x, data.y)
        assert_value(model1.coef_, model2.coef_, 0, 0)

    @staticmethod
    def test_normalize():
        np.random.seed(0)
        n = 100
        p = 20
        k = 5

        # glm
        data = abess.make_glm_data(n=n, p=p, k=k, family='gaussian')

        model1 = abess.LinearRegression()
        model2 = abess.LinearRegression()
        model1.fit(data.x, data.y)
        model2.fit(data.x, data.y, is_normal=True)
        assert_value(model1.coef_, model2.coef_)
        assert_value(model1.intercept_, model2.intercept_)

    @staticmethod
    def test_clipping():
        np.random.seed(0)
        n = 200
        p = 20
        k = 5

        # range: (0, 10)
        coef1 = np.zeros(p)
        coef1[np.random.choice(p, k, replace=False)
              ] = np.random.uniform(0, 10, size=k)
        data1 = abess.make_glm_data(
            n=n, p=p, k=k, family='gaussian', coef_=coef1)

        model1 = abess.LinearRegression()
        model1.fit(data1.x, data1.y, beta_low=0, beta_high=10)
        assert_value(model1.coef_, data1.coef_, 0.1, 0.1)

        # range: (-100, -90)
        coef2 = np.zeros(p)
        coef2[np.random.choice(p, k, replace=False)
              ] = np.random.uniform(-100, -90, size=k)
        data2 = abess.make_glm_data(
            n=n, p=p, k=k, family='gaussian', coef_=coef2)

        model2 = abess.LinearRegression()
        model2.fit(data2.x, data2.y, beta_low=-100, beta_high=-90)
        assert_value(model2.coef_, data2.coef_, 0.1, 0.1)

        # one-side
        model1.fit(data1.x, data1.y, beta_low=0)
        assert_value(model1.coef_, data1.coef_, 0.1, 0.1)
        model1.fit(data1.x, data1.y, beta_high=10)
        assert_value(model1.coef_, data1.coef_, 0.1, 0.1)

        # force range
        coef3 = np.zeros(p)
        coef3[np.random.choice(p, k, replace=False)
              ] = np.random.choice([-11, 11], size=k)
        data3 = abess.make_glm_data(
            n=n, p=p, k=k, family='gaussian', coef_=coef3)

        model3 = abess.LinearRegression()
        model3.fit(data3.x, data3.y, beta_low=-10, beta_high=10, is_normal=False)
        assert_fit(model3.coef_, data3.coef_)
        assert (model3.coef_ >= -10).all()
        assert (model3.coef_ <= 10).all()
        assert (model3.coef_[data3.coef_ == -11] == -10).all()
        assert (model3.coef_[data3.coef_ == 11] == 10).all()

    @staticmethod
    def test_possible_input():
        np.random.seed(2)
        n = 100
        p = 20
        k = 5
        M = 3
        # s_min = 0
        s_max = 10
        screen = 15
        imp = 5
        data = abess.make_glm_data(n=n, p=p, k=k, family='gaussian')
        data2 = abess.make_glm_data(n=n, p=p, k=k, family='binomial')
        data3 = abess.make_multivariate_glm_data(
            n=n, p=p, k=k, M=M, family='multinomial')

        # alpha
        model = abess.LinearRegression(alpha=[0.1, 0.2, 0.3])
        model.fit(data.x, data.y)
        assert_nan(model.coef_)

        # screening
        model = abess.LinearRegression(
            support_size=range(s_max),
            screening_size=screen)
        model.fit(data.x, data.y)
        assert_nan(model.coef_)

        model = abess.LinearRegression(
            support_size=range(s_max),
            screening_size=0)
        model.fit(data.x, data.y)
        assert_nan(model.coef_)

        # important search
        model = abess.LinearRegression(
            support_size=range(s_max),
            important_search=imp)
        model.fit(data.x, data.y)
        assert_nan(model.coef_)

        # splicing_type
        model1 = abess.LinearRegression(splicing_type=0)
        model1.fit(data.x, data.y)
        model2 = abess.LinearRegression(splicing_type=1)
        model2.fit(data.x, data.y)
        assert_fit(model1.coef_, model2.coef_)

        # always_select
        model = abess.LinearRegression(always_select=[0, 1, 2, 3])
        model.fit(data.x, data.y)
        assert np.prod(model.coef_[0:4]) != 0

        # group
        group = np.repeat([1, 2, 3, 4], [5, 5, 5, 5])
        model = abess.LinearRegression(support_size=2, group=group)
        model.fit(data.x, data.y)
        nonzero = np.nonzero(model.coef_)[0]
        assert len(nonzero) == 2 * 5
        assert len(set(group[nonzero])) == 2

        # ic
        for ic in ['loss', 'aic', 'bic', 'ebic', 'gic', 'hic']:
            model = abess.LinearRegression(ic_type=ic)
            model.fit(data.x, data.y)
        for cv_score in ['test_loss', 'roc_auc']:
            model = abess.LogisticRegression(cv_score=cv_score, cv=5)
            model.fit(data2.x, data2.y)
        for cv_score in ['test_loss', 'roc_auc_ovo', 'roc_auc_ovr']:
            model = abess.MultinomialRegression(cv_score=cv_score, cv=5)
            model.fit(data3.x, data3.y)

        # A_init
        model = abess.LinearRegression(A_init=[0, 1, 2])
        model.fit(data.x, data.y)
