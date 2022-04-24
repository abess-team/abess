from python.abess.datasets import sample
from .universal import ConvexSparseSolver
from .datasets import make_multivariate_glm_data
import .pybind_cabess
from abess import MultiTaskRegression
import pytest
import numpy as np


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
        data = make_multivariate_glm_data(
            family=family, n=n, p=p, k=k, rho=rho, M=M)

        model1 = ConvexSparseSolver(model_size=p, sample_size=n, intercept_size=M, support_size=k)
        model1.set_loss(pybind_cabess.linear_model_loss)
        model1.set_grandient_autodiff(pybind_cabess.linear_model_gradient)
        model1.set_hessian_autodiff(pybind_cabess.linear_model_hessian)

        model1.fit(pybind_cabess.Data(data.x, data.y))

        model2 = MultiTaskRegression(support_size=k)
        model2.fit(data.x, data.y)
