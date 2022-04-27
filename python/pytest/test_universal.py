from python.abess.datasets import sample
from .universal import ConvexSparseSolver
from .datasets import make_multivariate_glm_data
import .pybind_cabess
from abess import MultiTaskRegression
import pytest
import numpy as np
from utilities import  assert_fit

class TestUniversalModel:
    """
    Test for ConvexSparseSolver
    """
    @staticmethod
    def test_linear_model():
        np.random.seed(1)
        n = 100
        p = 20
        k = 3
        family = "multigaussian"
        rho = 0.5
        M = 3
        data = make_multivariate_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
        group = [i for i in range(p) for j in range(M)]
        model1 = ConvexSparseSolver(model_size=p*M, sample_size=n, intercept_size=M, group=group)
        model1.set_loss(pybind_cabess.loss_linear)
        model1.set_gradient_autodiff(pybind_cabess.gradient_linear)
        model1.set_hessian_autodiff(pybind_cabess.hessian_linear)

        model1.fit(pybind_cabess.Data(data.x, data.y))

        #model2 = MultiTaskRegression()
        #model2.fit(data.x, data.y)
        #assert_fit(model1.coef_, model2.coef_)