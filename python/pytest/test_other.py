from abess import *
from utilities import *
import numpy as np

class TestOther:
    """
    Test for other modules in abess package.
    Include: `abess.datasets` 
    """
    def test_datasets(self):
        np.random.seed(0)
        n = 100
        p = 20
        k = 5
        M = 3
        rho = 0.
        sigma = 1.
        SNR = 1
        
        for family in ['gaussian', 'binomial', 'poisson']:
            data1 = make_glm_data(n=n, p=p, k=k, family=family, rho=rho, sigma=sigma)
            assert_shape(data1.x, data1.y, n, p, 1)
            data2 = make_glm_data(n=n, p=p, k=k, family=family, rho=rho, sigma=sigma, coef_ = data1.coef_)
            assert (data1.coef_ == data2.coef_).all()

        for family in ['cox']:
            data1 = make_glm_data(n=n, p=p, k=k, family=family, rho=rho, sigma=sigma)
            assert_shape(data1.x, data1.y, n, p, 2)
            data2 = make_glm_data(n=n, p=p, k=k, family=family, rho=rho, sigma=sigma, coef_ = data1.coef_)
            assert (data1.coef_ == data2.coef_).all()
        
        for family in ['multigaussian', 'multinomial']:
            data1 = make_multivariate_glm_data(n=n, p=p, k=k, family=family, rho=rho, SNR=SNR, M=M)
            assert_shape(data1.x, data1.y, n, p, 1)
            data2 = make_multivariate_glm_data(n=n, p=p, k=k, family=family, rho=rho, SNR=SNR, M=M, coef_ = data1.coef_)
            assert (data1.coef_ == data2.coef_).all()

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
