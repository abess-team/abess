import warnings
from time import time
import numpy as np
from utilities import (assert_nan, assert_value, assert_fit)
from scipy.sparse import coo_matrix
import abess

warnings.filterwarnings("ignore")


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

        model1 = abess.LinearRegression(sparse_matrix=False)
        model1.fit(data.x, data.y)
        model2 = abess.LinearRegression(sparse_matrix=True)
        model2.fit(coo_matrix(data.x), data.y)
        assert_value(model1.coef_, model2.coef_)
        assert_value(model1.intercept_, model2.intercept_)

        model3 = abess.LinearRegression(sparse_matrix=True)
        model3.fit(data.x, data.y)
        assert_value(model1.coef_, model3.coef_)
        assert_value(model1.intercept_, model2.intercept_)

        # pca
        data_pca = np.random.randn(n, p)

        model1 = abess.SparsePCA(sparse_matrix=False)
        model1.fit(data_pca)
        model2 = abess.SparsePCA(sparse_matrix=True)
        model2.fit(coo_matrix(data_pca))
        assert_value(model1.coef_, model2.coef_)

        model3 = abess.SparsePCA(sparse_matrix=True)
        model3.fit(data_pca)
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
    def test_possible_input():
        np.random.seed(2)
        n = 100
        p = 20
        k = 5
        # s_min = 0
        s_max = 10
        screen = 15
        imp = 5
        data = abess.make_glm_data(n=n, p=p, k=k, family='gaussian')

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
        model = abess.LinearRegression(support_size=2)
        model.fit(data.x, data.y, group=group)
        nonzero = np.nonzero(model.coef_)[0]
        assert len(nonzero) == 2 * 5
        assert len(set(group[nonzero])) == 2

        # ic
        for ic in ['aic', 'bic', 'ebic', 'gic']:
            model = abess.LinearRegression(ic_type=ic)
            model.fit(data.x, data.y)

        # A_init
        model = abess.LinearRegression()
        model.fit(data.x, data.y, A_init=[0, 1, 2])
