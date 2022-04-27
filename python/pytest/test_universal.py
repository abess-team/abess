from python.abess.datasets import sample
from python.pytest.utilities import assert_value
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
        model = ConvexSparseSolver(model_size=p*M, sample_size=n, intercept_size=M, group=group)
        model.set_data(pybind_cabess.Data(data.x, data.y))
        model.set_loss(pybind_cabess.loss_linear)
        model.set_gradient_autodiff(pybind_cabess.gradient_linear)
        model.set_hessian_autodiff(pybind_cabess.hessian_linear)

        model.fit()
        coef = model.coef_
        assert_fit(coef, data.coef_)

        model.thread = 0
        model.fit()
        assert_value(model.coef_, coef)

        model.path_type = "gs"
        model.fit()
        assert_value(model.coef_, coef)

        model.cv = 5
        model.set_slice_by_sample(pybind_cabess.slice_by_sample)
        model.set_deleter(pybind_cabess.deleter)
        model.fit()
        assert_value(model.coef_, coef)

        model.path_type = "seq"
        model.set_slice_by_para(pybind_cabess.slice_by_para)
        model.fit()
        assert_value(model.coef_, coef)                

        