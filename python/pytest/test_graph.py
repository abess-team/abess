import numpy as np
import pytest

from abess.graph import IsingModel, make_ising_data


def test_make_ising_data_basic():
    result = make_ising_data(n=100, p=16, type=3, seed=42, graph_seed=42,
                             beta=0.5, method="gibbs")
    assert set(result.keys()) == {"data", "weight", "theta"}
    data = result["data"]
    assert data.shape == (100, 16)
    assert set(np.unique(data)).issubset({-1.0, 1.0})
    assert result["weight"].shape == (100,)
    assert result["theta"].shape == (16, 16)
    # theta should be symmetric
    np.testing.assert_allclose(result["theta"], result["theta"].T)


def test_make_ising_data_theta_structure():
    # type=3 is 4-nn cyclic on 4x4 grid; each node has 4 neighbors
    result = make_ising_data(n=10, p=16, type=3, graph_seed=0, beta=0.5, method="gibbs")
    theta = result["theta"]
    for i in range(16):
        n_neighbors = np.sum(theta[i, :] != 0)
        assert n_neighbors == 4, f"Node {i} has {n_neighbors} neighbors, expected 4"


def test_ising_model_fit():
    result = make_ising_data(n=500, p=16, type=3, seed=1, graph_seed=1,
                             beta=0.5, method="gibbs")
    data = result["data"]
    weight = result["weight"]
    true_theta = result["theta"]

    model = IsingModel(max_support_size=4, tune_type="gic")
    model.fit(data, sample_weight=weight)

    assert hasattr(model, "theta_")
    assert model.theta_.shape == (16, 16)
    # theta_ should be symmetric
    np.testing.assert_allclose(model.theta_, model.theta_.T, atol=1e-10)
    assert model.n_features_in_ == 16

    # Check graph structure recovery (at least 50% true positive rate)
    true_edges = (true_theta != 0)
    est_edges = (model.theta_ != 0)
    tp = np.sum(true_edges & est_edges)
    total_true = np.sum(true_edges)
    assert tp > 0.5 * total_true, (
        f"True positive rate {tp}/{total_true} is below 50%"
    )


def test_ising_model_score():
    result = make_ising_data(n=300, p=9, type=3, seed=7, graph_seed=7,
                             beta=0.5, method="gibbs")
    data = result["data"]

    model = IsingModel(max_support_size=3, tune_type="gic")
    model.fit(data)

    s = model.score(data)
    assert isinstance(s, float)
    assert s < 0  # pseudo-log-likelihood is negative


def test_ising_model_sklearn_api():
    result = make_ising_data(n=200, p=9, type=3, seed=3, graph_seed=3,
                             beta=0.5, method="gibbs")
    data = result["data"]

    model = IsingModel(max_support_size=3, tune_type="gic")
    assert hasattr(model, "get_params")
    params = model.get_params()
    assert "max_support_size" in params
    assert "tune_type" in params

    model.fit(data)
    assert hasattr(model, "n_features_in_")
    assert hasattr(model, "theta_")

    # set_params
    model.set_params(graph_threshold=0.1)
    assert model.graph_threshold == 0.1


def test_ising_model_no_max_support_size():
    # When max_support_size is None, should default to min(p-2, 100)
    result = make_ising_data(n=100, p=9, type=3, seed=5, graph_seed=5,
                             beta=0.5, method="gibbs")
    data = result["data"]
    model = IsingModel()
    model.fit(data)
    assert model.theta_.shape == (9, 9)
