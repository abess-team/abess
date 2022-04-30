from abess.universal import ConvexSparseSolver
from abess.datasets import make_multivariate_glm_data
from abess import pybind_cabess
import pytest
import numpy as np
from utilities import assert_fit, assert_value
import jax.numpy as jnp

class TestUniversalModel:
    """
    Test for ConvexSparseSolver
    """

    @staticmethod
    def test_linear_model_autodiff():
        np.random.seed(1)
        n = 100
        p = 20
        k = 3
        family = "multigaussian"
        rho = 0.5
        M = 3
        data = make_multivariate_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
        group = [i for i in range(p) for j in range(M)]
        model = ConvexSparseSolver(
            model_size=p * M, sample_size=n, intercept_size=M, group=group
        )
        model.set_data(pybind_cabess.Data(data.x, data.y))
        model.set_model_autodiff(
            pybind_cabess.loss_linear,
            pybind_cabess.gradient_linear,
            pybind_cabess.hessian_linear,
        )

        model.fit()
        coef = model.coef_
        assert_fit(coef, [c for v in data.coef_ for c in v])

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

    @staticmethod
    def test_linear_model_jax():
        np.random.seed(1)
        n = 100
        p = 20
        k = 3
        family = "multigaussian"
        rho = 0.5
        M = 1
        data = make_multivariate_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
        group = [i for i in range(p) for j in range(M)]
        model = ConvexSparseSolver(
            model_size=p * M, sample_size=n, intercept_size=M, group=group
        )
        
        model.set_data((jnp.array(data.x), jnp.array(data.y)))
        model.set_model_jax(lambda para, intercept, data:jnp.sum(jnp.square(data[1]-data[0]@para-intercept)))

        model.fit()
        coef = model.coef_
        assert_fit(coef, [c for v in data.coef_ for c in v])

        model.thread = 0
        model.fit()
        assert_value(model.coef_, coef)

        model.path_type = "gs"
        model.fit()
        assert_value(model.coef_, coef)
