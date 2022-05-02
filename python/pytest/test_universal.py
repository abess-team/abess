from abess.universal import ConvexSparseSolver
from abess.datasets import make_multivariate_glm_data
from abess import pybind_cabess
import pytest
import numpy as np
from utilities import assert_fit, assert_value
import jax.numpy as jnp
from jax import jit


class TestUniversalModel:
    """
    Test for ConvexSparseSolver
    """

    @staticmethod
    def test_linear_model_autodiff():
        np.random.seed(3)
        n = 30
        p = 5
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
        model.set_slice_by_para(pybind_cabess.slice_by_para)
        model.set_deleter(pybind_cabess.deleter)
        model.path_type = "gs"
        model.fit()
        assert_value(model.coef_, coef)

        model.cv = 5
        model.set_slice_by_sample(pybind_cabess.slice_by_sample)
        model.fit()
        assert_value(model.coef_, coef)

    @staticmethod
    def test_linear_model_jax():
        np.random.seed(1)
        n = 30
        p = 5
        k = 3
        family = "multigaussian"
        rho = 0.5
        M = 3
        data = make_multivariate_glm_data(family=family, n=n, p=p, k=k, rho=rho, M=M)
        group = [i for i in range(p) for j in range(M)]

        model = ConvexSparseSolver(
            model_size=p * M, sample_size=n, intercept_size=M, group=group
        )

        @jit
        def f(para, intercept, data):
            m = jnp.size(intercept)
            p = data[0].shape[1]
            return jnp.sum(
                jnp.square(data[1] - data[0] @ para.reshape(p, m) - intercept)
            )

        model.set_data((jnp.array(data.x), jnp.array(data.y)))
        model.set_model_jax(f)

        model.fit()
        coef = model.coef_
        assert_fit(coef, [c for v in data.coef_ for c in v])

        model.thread = 0
        model.path_type = "gs"
        model.fit()
        assert_value(model.coef_, coef)

        model.cv = 3
        model.set_slice_by_sample(lambda data, ind: (data[0][ind, :], data[1][ind, :]))
        model.fit()
        assert_value(model.coef_, coef)
