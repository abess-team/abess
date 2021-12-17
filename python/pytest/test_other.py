from abess import *
from utilities import *
import numpy as np


class TestOther:
    """
    Test for other modules in abess package.
    Include: `abess.datasets`
    """

    @staticmethod
    def test_datasets():
        np.random.seed(0)
        n = 100
        p = 20
        k = 5
        M = 3
        rho = 0.
        sigma = 1.

        for family in ['gaussian', 'binomial', 'poisson']:
            data1 = make_glm_data(
                n=n,
                p=p,
                k=k,
                family=family,
                rho=rho,
                sigma=sigma,
                snr=0)
            data1 = make_glm_data(
                n=n, p=p, k=k, family=family, rho=rho, sigma=sigma)
            assert_shape(data1.x, data1.y, n, p, 1)
            data2 = make_glm_data(
                n=n,
                p=p,
                k=k,
                family=family,
                rho=rho,
                sigma=sigma,
                coef_=data1.coef_)
            assert (data1.coef_ == data2.coef_).all()

        for family in ['cox']:
            data1 = make_glm_data(
                n=n,
                p=p,
                k=k,
                family=family,
                rho=rho,
                sigma=sigma,
                censoring=False)
            data1 = make_glm_data(
                n=n, p=p, k=k, family=family, rho=rho, sigma=sigma)
            assert_shape(data1.x, data1.y, n, p, 2)
            data2 = make_glm_data(
                n=n,
                p=p,
                k=k,
                family=family,
                rho=rho,
                sigma=sigma,
                coef_=data1.coef_)
            assert (data1.coef_ == data2.coef_).all()

        for family in ['multigaussian', 'multinomial']:
            data1 = make_multivariate_glm_data(
                n=n, p=p, k=k, family=family, rho=rho, M=M, sparse_ratio=0.1)
            data1 = make_multivariate_glm_data(
                n=n, p=p, k=k, family=family, rho=rho, M=M)
            assert_shape(data1.x, data1.y, n, p, M)
            data2 = make_multivariate_glm_data(
                n=n, p=p, k=k, family=family, rho=rho, M=M, coef_=data1.coef_)
            assert (data1.coef_ == data2.coef_).all()

        data1 = make_multivariate_glm_data(
            n=n, p=p, k=k, family='poisson', rho=rho, M=M)
        assert_shape(data1.x, data1.y, n, p, 1)

        # error input
        try:
            make_glm_data(n=n, p=p, k=k, family='other')
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            make_multivariate_glm_data(n=n, p=p, k=k, family='other')
        except ValueError as e:
            print(e)
        else:
            assert False
