import numpy as np
from pytest import approx


def assert_nan(coef):
    assert not np.isnan(np.sum(coef))


def assert_value(coef1, coef2, rel=1e-2, abs=1e-2):
    coef1 = coef1.reshape(-1)
    coef2 = coef2.reshape(-1)
    assert coef1.shape == coef2.shape
    assert coef1 == approx(coef2, rel=rel, abs=abs)


def assert_fit(coef1, coef2):
    assert_nan(coef1)
    assert_nan(coef2)
    pos1 = np.unique(np.nonzero(coef1)[0])
    pos2 = np.unique(np.nonzero(coef2)[0])
    assert pos1.shape == pos2.shape
    assert (pos1 == pos2).all()
    # assert_value(coef1[pos1], coef2[pos2])


def assert_shape(x, y, n, p, M):
    assert x.shape == (n, p)
    assert y.shape[0] == n
    if M > 1:
        assert y.shape[1] == M
