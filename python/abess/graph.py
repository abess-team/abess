import numpy as np
import numbers
import warnings

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from scipy.sparse import coo_matrix

from .bess_base import bess_base
from .linear import LogisticRegression


# ---------------------------------------------------------------------------
# Helpers for make_ising_data
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gen_4nn_cyclic(p, degree, beta, alpha, graph_type):
    """
    4-nearest-neighbor cyclic grid topology.
    p must be a perfect square.  Returns p×p theta matrix.
    graph_type 3 = ferromagnetic (+beta), 4 = spin-glass (±beta).
    """
    sq = int(round(p ** 0.5))
    if sq * sq != p:
        raise ValueError("p must be a perfect square for graph type 3/4.")
    theta = np.zeros((p, p))
    for node in range(p):
        row, col = divmod(node, sq)
        neighbors = [
            row * sq + (col + 1) % sq,           # right
            row * sq + (col - 1) % sq,           # left
            ((row + 1) % sq) * sq + col,         # down
            ((row - 1) % sq) * sq + col,         # up
        ]
        for nb in neighbors:
            if graph_type == 4:
                # spin-glass: alternate sign based on (row+col) parity
                sign = 1 if (row + col) % 2 == 0 else -1
            else:
                sign = 1
            if theta[node, nb] == 0:
                theta[node, nb] = sign * beta
                theta[nb, node] = sign * beta
    return theta


def _gen_ld_adjmat(p, degree, alpha):
    """Local-degree adjacency matrix (type 6)."""
    rng = np.random.default_rng(None)
    theta = np.zeros((p, p))
    for i in range(p):
        neighbors = list(range(max(0, i - degree), i)) + \
                    list(range(i + 1, min(p, i + degree + 1)))
        for nb in neighbors:
            if theta[i, nb] == 0:
                val = alpha * (rng.random() - 0.5) * 2  # uniform in [-alpha, alpha]
                theta[i, nb] = val
                theta[nb, i] = val
    return theta


def _gen_rr_adjmat(p, degree, beta, alpha, graph_type):
    """Random k-regular graph (types 1, 2, 7) using networkx."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for graph types 1, 2, 7. "
            "Install it with: pip install networkx"
        )
    G = nx.random_regular_graph(degree, p)
    theta = np.zeros((p, p))
    for u, v in G.edges():
        if graph_type == 2:
            # spin-glass
            sign = 1 if np.random.rand() > 0.5 else -1
            val = sign * beta
        elif graph_type == 7:
            # random weight in [-alpha, alpha]
            val = alpha * (np.random.rand() - 0.5) * 2
        else:
            # type 1: ferromagnetic
            val = beta
        theta[u, v] = val
        theta[v, u] = val
    return theta


def _sim_theta(p, graph_type, graph_seed, beta, degree, alpha):
    """Dispatch to the correct topology builder."""
    if graph_seed is not None:
        np.random.seed(graph_seed)
    if graph_type in (3, 4):
        return _gen_4nn_cyclic(p, degree, beta, alpha, graph_type)
    elif graph_type == 6:
        return _gen_ld_adjmat(p, degree, alpha)
    elif graph_type in (1, 2, 7):
        return _gen_rr_adjmat(p, degree, beta, alpha, graph_type)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")


def _gibbs_sample(theta, n_sample, burn=100000, skip=50, seed=1):
    """
    Pure-numpy Gibbs sampler for Ising model.
    theta: p×p interaction matrix (including diagonal for external field).
    Returns: (n_sample × p) array of {-1, 1} samples.
    """
    rng = np.random.default_rng(seed)
    p = theta.shape[0]
    x = rng.choice([-1, 1], size=p).astype(float)

    # burn-in
    for _ in range(burn):
        j = rng.integers(p)
        log_odd = 2.0 * (theta[j, :] @ x - theta[j, j] * x[j])
        prob = _sigmoid(log_odd)
        x[j] = 1.0 if rng.random() < prob else -1.0

    samples = np.empty((n_sample, p), dtype=float)
    for i in range(n_sample):
        for _ in range(skip):
            j = rng.integers(p)
            log_odd = 2.0 * (theta[j, :] @ x - theta[j, j] * x[j])
            prob = _sigmoid(log_odd)
            x[j] = 1.0 if rng.random() < prob else -1.0
        samples[i] = x.copy()

    return samples


def _exact_sample(n, theta, seed):
    """
    Exact sampler: enumerate all 2^p configurations.
    Only feasible for p <= 20.
    Returns (unique_data, frequencies) where frequencies sum to n.
    """
    p = theta.shape[0]
    if p > 20:
        warnings.warn(
            f"p={p} > 20: exact sampling may be very slow. "
            "Consider using method='gibbs' instead."
        )
    rng = np.random.default_rng(seed)
    configs = np.array(
        [[(1 if (k >> j) & 1 else -1) for j in range(p)] for k in range(2 ** p)],
        dtype=float
    )  # shape (2^p, p)
    # theta_off = theta with diagonal zeroed
    theta_off = theta.copy()
    np.fill_diagonal(theta_off, 0.0)
    # log-weight: 0.5 * c @ theta_off @ c
    log_w = 0.5 * np.array([c @ theta_off @ c for c in configs])
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    indices = rng.choice(len(configs), size=n, p=w, replace=True)
    data = configs[indices]
    unique_data, counts = np.unique(data, axis=0, return_counts=True)
    return data, np.ones(n)


def make_ising_data(n, p, type=3, seed=None, graph_seed=None, theta=None,
                    beta=0.7, degree=3, alpha=0.4, method="gibbs"):
    """
    Generate data from an Ising model.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    p : int
        Number of nodes (variables).
    type : int, optional
        Graph topology type:
        1 = random regular (ferro), 2 = random regular (spin-glass),
        3 = 4-nearest-neighbor cyclic (ferro, requires p perfect square),
        4 = 4-nearest-neighbor cyclic (spin-glass),
        6 = local-degree, 7 = random regular (random weight).
        Default: 3.
    seed : int or None, optional
        Random seed for sampling. Default: None.
    graph_seed : int or None, optional
        Random seed for graph generation. Default: None.
    theta : ndarray of shape (p, p), optional
        Custom interaction matrix. If provided, ignores type/beta/degree/alpha.
    beta : float, optional
        Coupling strength. Default: 0.7.
    degree : int, optional
        Node degree for graph construction. Default: 3.
    alpha : float, optional
        Weight scaling for random-weight topologies. Default: 0.4.
    method : str, optional
        Sampling method: "gibbs" (default) or "freq" (exact enumeration,
        feasible only for small p).

    Returns
    -------
    dict with keys:
        "data" : ndarray of shape (n, p), values in {-1, 1}
        "weight" : ndarray of shape (n,), sample weights (all 1 for gibbs)
        "theta" : ndarray of shape (p, p), the interaction matrix
    """
    if theta is None:
        theta = _sim_theta(p, type, graph_seed, beta, degree, alpha)

    if method == "auto":
        method = "freq" if p <= 16 else "gibbs"

    if method == "freq":
        data, weight = _exact_sample(n, theta, seed)
    else:
        data = _gibbs_sample(theta, n, seed=seed if seed is not None else 1)
        weight = np.ones(n)

    return {"data": data, "weight": weight, "theta": theta}


# ---------------------------------------------------------------------------
# IsingModel: SLIDE algorithm (node-wise logistic regression)
# ---------------------------------------------------------------------------

class IsingModel(BaseEstimator):
    """
    Sparse Ising model estimation via SLIDE (Sparse pairwise Logistic regression
    for Ising model via best-subset sElection).

    Fits one LogisticRegression per node (node-wise logistic regression),
    then symmetrizes the resulting coefficient matrix.

    Parameters
    ----------
    max_support_size : int, array-like of shape (p,), or None
        Maximum support size for each node's logistic regression.
        If None, defaults to min(p-2, 100) for each node.
    tune_type : str, optional
        Model selection criterion: "gic", "aic", "bic", "ebic", or "cv".
        Default: "gic".
    ic_coef : float, optional
        Coefficient for information criterion. Default: 1.0.
    cv : int, optional
        Number of cross-validation folds (used when tune_type="cv"). Default: 5.
    graph_threshold : float, optional
        Entries with |theta| <= graph_threshold are zeroed out. Default: 0.0.
    thread : int, optional
        Number of threads for LogisticRegression. Default: 1.
    primary_model_fit_max_iter : int, optional
        Max iterations for inner logistic solver. Default: 100.
    primary_model_fit_epsilon : float, optional
        Convergence tolerance for inner logistic solver. Default: 1e-6.
    """

    def __init__(self, max_support_size=None, tune_type="gic", ic_coef=1.0,
                 cv=5, graph_threshold=0.0, thread=1,
                 primary_model_fit_max_iter=100,
                 primary_model_fit_epsilon=1e-6):
        self.max_support_size = max_support_size
        self.tune_type = tune_type
        self.ic_coef = ic_coef
        self.cv = cv
        self.graph_threshold = graph_threshold
        self.thread = thread
        self.primary_model_fit_max_iter = primary_model_fit_max_iter
        self.primary_model_fit_epsilon = primary_model_fit_epsilon

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit the sparse Ising model via SLIDE.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Binary data with values in {-1, 1}.
        y : ignored
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self
        """
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        n, p = X.shape
        self.n_features_in_ = p

        # Resolve per-node max support size
        if self.max_support_size is None:
            ms_arr = np.full(p, min(p - 2, 100), dtype=int)
        elif isinstance(self.max_support_size, (numbers.Integral, numbers.Real)):
            ms_arr = np.full(p, int(self.max_support_size), dtype=int)
        else:
            ms_arr = np.array(self.max_support_size, dtype=int)
            if ms_arr.shape != (p,):
                raise ValueError(
                    "max_support_size array must have length p."
                )

        # Build tune kwargs
        if self.tune_type == "cv":
            tune_kwargs = dict(cv=self.cv, ic_type="ebic")
        else:
            tune_kwargs = dict(cv=1, ic_type=self.tune_type)

        theta = np.zeros((p, p))

        for j in range(p):
            ms = int(ms_arr[j])
            X_minus = np.delete(X, j, axis=1)   # n × (p-1)
            y_node = X[:, j]                     # {-1, 1}

            model = LogisticRegression(
                support_size=list(range(0, ms + 1)),
                ic_coef=self.ic_coef,
                fit_intercept=False,
                thread=self.thread,
                primary_model_fit_max_iter=self.primary_model_fit_max_iter,
                primary_model_fit_epsilon=self.primary_model_fit_epsilon,
                **tune_kwargs
            )
            # override normalize_type to 0 (no normalization) as in R's SLIDE
            model.normalize_type = 0

            model.fit(X_minus, y_node, sample_weight=sample_weight)
            coef = model.coef_.flatten()  # length p-1

            # Insert 0 at position j
            theta[j, :j] = coef[:j]
            theta[j, j + 1:] = coef[j:]

        # Symmetrize and scale
        theta = (theta + theta.T) / 2.0
        theta /= 2.0

        # Threshold
        theta[np.abs(theta) <= self.graph_threshold] = 0.0

        self.theta_ = theta
        return self

    def score(self, X, y=None, sample_weight=None):
        """
        Compute the pseudo-log-likelihood of X under the fitted Ising model.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
        y : ignored
        sample_weight : ignored

        Returns
        -------
        float
            Mean pseudo-log-likelihood (higher is better).
        """
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        n, p = X.shape
        theta = self.theta_

        pll = 0.0
        for j in range(p):
            # theta[j, -j] @ x_i,-j  for each i
            mask = np.ones(p, dtype=bool)
            mask[j] = False
            eta = X[:, mask] @ theta[j, mask]  # shape (n,)
            log_prob = np.log(_sigmoid(2.0 * X[:, j] * eta))
            pll += log_prob.sum()

        return float(pll / n)


# ---------------------------------------------------------------------------
# abessGraph (kept for backward compatibility)
# ---------------------------------------------------------------------------

def fix_docs(cls):
    # inherit the document from base class
    index = cls.__doc__.find("Examples\n    --------\n")
    if (index != -1):
        cls.__doc__ = cls.__doc__[:index] + \
            cls.__bases__[0].__doc__ + cls.__doc__[index:]
    return cls


@fix_docs
class abessGraph(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for gaussian graph model.

    Parameters
    ----------
    splicing_type: {0, 1}, optional
        The type of splicing in `fit()` (in Algorithm.h).
        "0" for decreasing by half, "1" for decresing by one.
        Default: splicing_type = 0.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.graph import abessGraph
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> model = abessGraph(support_size = 10)
    >>>
    >>> ### X known
    >>>
    >>> model.fit(X)
    >>> print(model.coef_)
    """

    def __init__(self, max_iter=200, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, s_min=None, s_max=None,
                 ic_type="aic", ic_coef=1.0, primary_model_fit_max_iter=20, primary_model_fit_epsilon=1e-3,
                 always_select=[], alpha=None, cv=1,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=1
                 ):
        super(abessGraph, self).__init__(
            algorithm_type="abess", model_type="Graph", data_type=1, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, primary_model_fit_max_iter=primary_model_fit_max_iter, primary_model_fit_epsilon=primary_model_fit_epsilon,
            always_select=always_select, alpha=alpha, cv=cv,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )

    def fit(self, X=None, cv_fold_id=None):
        """
        The fit function is used to transfer the information of data and return the fit result.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Training data.
        cv_fold_id: array_like of shape (n_samples,) , optional
            An array indicates different folds in CV. Samples in the same fold should be given the same number.
            Default: cv_fold_id=None
        """
        try:
            from abess.pybind_cabess import pywrap_GLM as pywrap_abess
        except ImportError:
            raise ImportError(
                "abessGraph requires the compiled C++ extension (pybind_cabess). "
                "Please build the package with: pip install -e python/"
            )

        # Input check
        if isinstance(X, (list, np.ndarray, np.matrix, coo_matrix)):
            if isinstance(X, coo_matrix):
                self.sparse_matrix = True
            X = check_array(X, accept_sparse=True)

            n = X.shape[0]
            p = X.shape[1]
            y = np.zeros((n, 1))
            self.n_features_in_ = p

        else:
            raise ValueError("X should be given in Ising model.")

        # Algorithm_type
        if self.algorithm_type == "abess":
            algorithm_type_int = 6
        else:
            raise ValueError("algorithm_type should not be " +
                             str(self.algorithm_type))

        model_type_int = 9

        # Path_type: seq, pgs
        if self.path_type == "seq":
            path_type_int = 1
        elif self.path_type == "pgs":
            path_type_int = 2
        else:
            raise ValueError("path_type should be \'seq\' or \'pgs\'")

        # Ic_type
        if self.ic_type == "aic":
            ic_type_int = 1
        elif self.ic_type == "bic":
            ic_type_int = 2
        elif self.ic_type == "gic":
            ic_type_int = 3
        elif self.ic_type == "ebic":
            ic_type_int = 4
        else:
            raise ValueError(
                "ic_type should be \"aic\", \"bic\", \"ebic\" or \"gic\"")

        # cv
        if (not isinstance(self.cv, int) or self.cv <= 0):
            raise ValueError("cv should be an positive integer.")
        elif (self.cv > 1):
            self.is_cv = True

        # cv_fold_id
        if cv_fold_id is None:
            cv_fold_id = np.array([], dtype="int32")
        else:
            cv_fold_id = np.array(cv_fold_id, dtype="int32")
            if cv_fold_id.ndim > 1:
                raise ValueError("group should be an 1D array of integers.")
            elif cv_fold_id.size != n:
                raise ValueError(
                    "The length of group should be equal to X.shape[0].")
            elif len(set(cv_fold_id)) != self.cv:
                raise ValueError(
                    "The number of different masks should be equal to `cv`.")

        # path parameter
        if self.support_size is None:
            support_sizes = list(range(0, int(p * (p - 1) / 2)))
        else:
            if isinstance(self.support_size, (numbers.Real, numbers.Integral)):
                support_sizes = np.empty(1, dtype=int)
                support_sizes[0] = self.support_size
            elif (np.any(np.array(self.support_size) > p * (p - 1) / 2) or
                    np.any(np.array(self.support_size) < 0)):
                raise ValueError(
                    "All support_size should be between 0 and X.shape[1]")
            else:
                support_sizes = self.support_size
        support_sizes = np.array(support_sizes).astype('int32')

        # alpha
        if self.alpha is None:
            alphas = [0]
        else:
            if isinstance(self.alpha, (numbers.Real, numbers.Integral)):
                alphas = np.empty(1, dtype=float)
                alphas[0] = self.alpha
            else:
                alphas = self.alpha

        # Exchange_num
        if (not isinstance(self.exchange_num, int) or self.exchange_num <= 0):
            raise ValueError("exchange_num should be an positive integer.")

        # Thread
        if (not isinstance(self.thread, int) or self.thread < 0):
            raise ValueError(
                "thread should be positive number or 0 (maximum supported by your device).")

        # Splicing type
        if (self.splicing_type != 0 and self.splicing_type != 1):
            raise ValueError("splicing type should be 0 or 1.")

        # Important_search
        if (not isinstance(self.important_search, int) or self.important_search < 0):
            raise ValueError(
                "important_search should be a non-negative number.")

        # Sparse X
        if self.sparse_matrix:
            if type(X) != type(coo_matrix((1, 1))):
                nonzero = 0
                tmp = np.zeros([X.shape[0] * X.shape[1], 3])
                for j in range(X.shape[1]):
                    for i in range(X.shape[0]):
                        if X[i, j] != 0.:
                            tmp[nonzero, :] = np.array([X[i, j], i, j])
                            nonzero += 1
                X = tmp[:nonzero, :]
            else:
                tmp = np.zeros([len(X.data), 3])
                tmp[:, 1] = X.row
                tmp[:, 2] = X.col
                tmp[:, 0] = X.data
                ind = np.lexsort((tmp[:, 2], tmp[:, 1]))
                X = tmp[ind, :]

        # always select (diag)
        temp = -1
        for i in range(p):
            temp += (i + 1)
            if temp not in self.always_select:
                self.always_select += [temp]
                support_sizes += 1

        # unused
        new_s_min = 0
        new_s_max = 0
        new_K_max = 0
        new_lambda_min = 0
        new_lambda_max = 0
        new_screening_size = -1
        g_index = list(range(p))
        state = [0]
        Sigma = np.array([[-1.0]])
        is_normal = False
        weight = np.ones(n)

        result = pywrap_abess(X, y, n, p, self.data_type, weight, Sigma,
                              is_normal,
                              algorithm_type_int, model_type_int, self.max_iter, self.exchange_num,
                              path_type_int, self.is_warm_start,
                              ic_type_int, self.ic_coef, self.is_cv, self.cv,
                              g_index,
                              state,
                              support_sizes,
                              alphas,
                              cv_fold_id,
                              new_s_min, new_s_max, new_K_max, self.epsilon,
                              new_lambda_min, new_lambda_max, self.n_lambda,
                              self.is_screening, new_screening_size, self.powell_path,
                              self.always_select, self.tau,
                              self.primary_model_fit_max_iter, self.primary_model_fit_epsilon,
                              self.early_stop, self.approximate_Newton,
                              self.thread,
                              self.covariance_update,
                              self.sparse_matrix,
                              self.splicing_type,
                              self.important_search,
                              int(p * (p + 1) / 2),
                              1 * 1, 1, 1, 1, 1, 1, int(p * (p + 1) / 2)
                              )

        self.coef_ = result[0].reshape(int(p * (p + 1) / 2), 1)
        self.theta_ = np.zeros((p, p))

        i = 0
        j = 0
        for k in range(0, int(p * (p + 1) / 2)):
            self.theta_[i, j] = self.coef_[k, 0]
            self.theta_[j, i] = self.coef_[k, 0]
            i += 1
            if (i > j):
                i = 0
                j += 1

        return self
