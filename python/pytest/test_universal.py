from .universal import ConvexSparseSolver
from .datasets import make_multivariate_glm_data
import .pybind_cabess
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
        data = make_multivariate_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)

        ConvexSparseSolver(model_size = p, )
