import numbers
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils.validation import check_array
from pybind_cabess import pywrap_PCA, pywrap_RPCA
from .bess_base import bess_base
from .utilities import new_data_check


def fix_docs(cls):
    # This function is to inherit the docstring from base class
    # and avoid unnecessary duplications on description.
    index = cls.__doc__.find("Examples\n    --------\n")
    if index != -1:
        cls.__doc__ = cls.__doc__[:index] + \
            cls.__bases__[0].__doc__ + cls.__doc__[index:]
    return cls


@fix_docs
class SparsePCA(bess_base):
    r"""
    Adaptive Best-Subset Selection(ABESS) algorithm for
    principal component analysis.

    Parameters
    ----------
    splicing_type: {0, 1}, optional, default=1
        The type of splicing.
        "0" for decreasing by half, "1" for decresing by one.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.decomposition import SparsePCA
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> model = SparsePCA(support_size = 10)
    >>>
    >>> ### X known
    >>> X = np.random.randn(100, 50)
    >>> model.fit(X)
    SparsePCA(always_select=[], support_size=10)
    >>> # print(model.coef_)
    >>> print(model.coef_[1:6,])
    [[6.36598737e-314]
     [1.06099790e-313]
     [1.48539705e-313]
     [1.90979621e-313]
     [2.33419537e-313]]
    >>>
    >>> ### X unknown, but Sigma known
    >>> model.fit(Sigma = np.cov(X.T))
    SparsePCA(always_select=[], support_size=10)
    >>> # print(model.coef_)
    >>> print(model.coef_[1:6,])
    [[6.36598737e-314]
     [1.06099790e-313]
     [1.48539705e-313]
     [1.90979621e-313]
     [2.33419537e-313]]
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq",
                 is_warm_start=True, support_size=None,
                 s_min=None, s_max=None,
                 ic_type="loss", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=1
                 ):
        super().__init__(
            algorithm_type="abess", model_type="PCA", normalize_type=1,
            path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size,
            s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv,
            screening_size=screening_size,
            always_select=always_select,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )

    def transform(self, X):
        r"""
        For PCA model, apply dimensionality reduction
        to given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, p_features)
            Sample matrix to be transformed.

        """
        X = new_data_check(self, X)

        return X.dot(self.coef_)

    def ratio(self, X):
        r"""
        Give new data, and it returns the explained ratio.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample matrix.
        """
        X = new_data_check(self, X)
        s = np.cov(X.T)
        # if len(self.coef_.shape) == 1:
        #     explain = self.coef_.T.dot(s).dot(self.coef_)
        # else:
        explain = np.sum(np.diag(self.coef_.T.dot(s).dot(self.coef_)))
        # if isinstance(s, (int, float)):
        #     full = s
        # else:
        full = np.sum(np.diag(s))
        return explain / full

    def fit(self, X=None, is_normal=False,
            group=None, Sigma=None, number=1, n=None, A_init=None):
        r"""
        The fit function is used to transfer the information of data and
        return the fit result.

        Parameters
        ----------
        X : array-like, shape(n_samples, p_features)
            Training data.
        is_normal : bool, optional, default=False
            whether normalize the variables array before fitting the algorithm.
        weight : array-like, shape(n_samples,), optional, default=np.ones(n)
            Individual weights for each sample. Only used for is_weight=True.
        group : int, optional, default=np.ones(p)
            The group index for each variable.
        Sigma : array-like, shape(p_features, p_features), optional
            default=np.cov(X.T).
            Sample covariance matrix.
            For PCA, it can be given as input, instead of X.
            But if X is given, Sigma will be set to np.cov(X.T).
        number : int, optional, default=1
            Indicates the number of PCs returned.
        n : int, optional, default=X.shape[0] or 1
            Sample size.

            - if X is given, it would be X.shape[0] by default;
            - if X is not given (Sigma is given), it would be 1 by default.
        """

        # Input check
        if isinstance(X, (list, np.ndarray, np.matrix, coo_matrix)):
            if isinstance(X, coo_matrix):
                self.sparse_matrix = True
            X = check_array(X, accept_sparse=True)

            n = X.shape[0]
            p = X.shape[1]
            X = X - X.mean(axis=0)
            Sigma = np.cov(X.T)
            self.n_features_in_ = p

        elif isinstance(Sigma, (list, np.ndarray, np.matrix)):
            if self.cv > 1:
                raise ValueError("X should be given to use CV.")

            Sigma = check_array(Sigma)

            if (Sigma.shape[0] != Sigma.shape[1] or np.any(Sigma.T != Sigma)):
                raise ValueError("Sigma should be symmetrical matrix.")
            if np.any(np.linalg.eigvals(Sigma) < 0):
                raise ValueError("Sigma should be semi-positive definite.")

            if n is None:
                n = 1
            p = Sigma.shape[0]
            X = np.zeros((1, p))
            self.n_features_in_ = p
            is_normal = False
        else:
            raise ValueError("X or Sigma should be given in PCA.")

        # # Algorithm_type
        # if self.algorithm_type == "abess":
        #     algorithm_type_int = 6
        # else:
        #     raise ValueError("algorithm_type should not be " +
        #                      str(self.algorithm_type))

        # for PCA,
        # model_type_int = 7
        path_type_int = 1

        # Ic_type
        if self.ic_type == "loss":
            ic_type_int = 0
        elif self.ic_type == "aic":
            ic_type_int = 1
        elif self.ic_type == "bic":
            ic_type_int = 2
        elif self.ic_type == "gic":
            ic_type_int = 3
        elif self.ic_type == "ebic":
            ic_type_int = 4
        else:
            raise ValueError(
                "ic_type should be \"loss\", \"aic\", \"bic\","
                " \"ebic\" or \"gic\"")

        # cv
        if (not isinstance(self.cv, int) or self.cv <= 0):
            raise ValueError("cv should be an positive integer.")
        if self.cv > n:
            raise ValueError("cv should be smaller than n.")

        # Group
        if group is None:
            g_index = list(range(p))
        else:
            group = np.array(group)
            if group.ndim > 1:
                raise ValueError("group should be an 1D array of integers.")
            if group.size != p:
                raise ValueError(
                    "The length of group should be equal to X.shape[1].")
            g_index = []
            group.sort()
            group_set = list(set(group))
            j = 0
            for i in group_set:
                while group[j] != i:
                    j += 1
                g_index.append(j)

        # path parameter (note that: path_type_int = 1)
        if self.support_size is None:
            support_sizes = np.ones(((int(p / 3) + 1), number))
        else:
            if isinstance(self.support_size, (numbers.Real, numbers.Integral)):
                support_sizes = np.zeros((self.support_size, 1))
                support_sizes[self.support_size - 1, 0] = 1
            elif (len(self.support_size.shape) != 2 or
                    self.support_size.shape[1] != number):
                raise ValueError(
                    "`support_size` should be 2-dimension and its number of"
                    " columns should be equal to `number`")
            elif self.support_size.shape[0] > p:
                raise ValueError(
                    "`support_size` should not larger than p")
            else:
                support_sizes = self.support_size
        support_sizes = np.array(support_sizes).astype('int32')

        # screening
        if self.screening_size != -1:
            if self.screening_size == 0:
                self.screening_size = min(
                    p, int(n / (np.log(np.log(n)) * np.log(p))))
            elif self.screening_size > p:
                raise ValueError(
                    "screening size should be smaller than X.shape[1].")
            elif self.screening_size < np.nonzero(support_sizes)[0].max() + 1:
                raise ValueError(
                    "screening size should be more than max(support_size).")

        # unused
        new_s_min = 0
        new_s_max = 0
        cv_fold_id = np.array([], dtype="int32")

        # Exchange_num
        if (not isinstance(self.exchange_num, int) or self.exchange_num <= 0):
            raise ValueError("exchange_num should be an positive integer.")

        # Thread
        if (not isinstance(self.thread, int) or self.thread < 0):
            raise ValueError(
                "thread should be positive number or 0"
                " (maximum supported by your device).")

        # Splicing type
        if self.splicing_type not in (0, 1):
            raise ValueError("splicing type should be 0 or 1.")

        # number
        if (not isinstance(number, int) or number <= 0 or number > p):
            raise ValueError(
                "number should be an positive integer and"
                " not bigger than X.shape[1].")

        # # Important_search
        # if (not isinstance(self.important_search, int)
        #         or self.important_search < 0):
        #     raise ValueError(
        #         "important_search should be a non-negative number.")

        # A_init
        if A_init is None:
            A_init = np.array([], dtype="int32")
        else:
            A_init = np.array(A_init, dtype="int32")
            if A_init.ndim > 1:
                raise ValueError(
                    "The initial active set should be an 1D array of"
                    " integers.")
            if (A_init.min() < 0 or A_init.max() > p):
                raise ValueError(
                    "A_init contains wrong index.")

        # Sparse X
        if self.sparse_matrix:
            if not isinstance(X, type(coo_matrix((1, 1)))):
                # print("sparse matrix 1")
                nonzero = 0
                tmp = np.zeros([X.shape[0] * X.shape[1], 3])
                for j in range(X.shape[1]):
                    for i in range(X.shape[0]):
                        if X[i, j] != 0.:
                            tmp[nonzero, :] = np.array([X[i, j], i, j])
                            nonzero += 1
                X = tmp[:nonzero, :]
            else:
                # print("sparse matrix 2")
                tmp = np.zeros([len(X.data), 3])
                tmp[:, 1] = X.row
                tmp[:, 2] = X.col
                tmp[:, 0] = X.data

                ind = np.lexsort((tmp[:, 2], tmp[:, 1]))
                X = tmp[ind, :]

        # normalize
        normalize = 0
        if is_normal:
            normalize = self.normalize_type

        # always_select
        if self.always_select is None:
            always_select_list = np.zeros(0, dtype="int32")
        else:
            always_select_list = np.array(self.always_select, dtype="int32")

        # unused
        early_stop = False

        # wrap with cpp
        weight = np.ones(n)
        result = pywrap_PCA(
            X, weight,
            n, p, normalize, Sigma,
            self.max_iter, self.exchange_num,
            path_type_int, self.is_warm_start,
            ic_type_int, self.ic_coef, self.cv,
            g_index,
            support_sizes,
            cv_fold_id,
            new_s_min, new_s_max,
            self.screening_size,
            always_select_list,
            early_stop,
            self.thread,
            self.sparse_matrix,
            self.splicing_type,
            self.important_search,
            number,
            A_init
        )

        self.coef_ = result[0]
        return self

    def fit_transform(self, X=None, is_normal=False,
                      group=None, Sigma=None, number=1, n=None, A_init=None):
        self.fit(X, is_normal, group, Sigma, number, n, A_init)
        return X.dot(self.coef_)


@fix_docs
class RobustPCA(bess_base):
    r"""
    Adaptive Best-Subset Selection(ABESS) algorithm for
    robust principal component analysis.

    Parameters
    ----------
    splicing_type: {0, 1}, optional, default=1
        The type of splicing.
        "0" for decreasing by half, "1" for decresing by one.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.decomposition import RobustPCA
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> model = RobustPCA(support_size = 10)
    >>>
    >>> ### X known
    >>> X = np.random.randn(100, 50)
    >>> model.fit(X, r = 10)
    RobustPCA(always_select=[], support_size=10)
    >>> print(model.coef_)
    [[0.         0.         0.         ... 0.         3.71203604 0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     ...
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]]

    """

    def __init__(self, max_iter=20, exchange_num=5, is_warm_start=True,
                 support_size=None,
                 ic_type="gic", ic_coef=1.0,
                 always_select=None,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=1
                 ):
        super().__init__(
            algorithm_type="abess", model_type="RPCA", normalize_type=1,
            path_type="seq", max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size,
            s_min=None, s_max=None, cv=1,
            ic_type=ic_type, ic_coef=ic_coef,
            always_select=always_select,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )

    def fit(self, X, r, group=None, A_init=None):
        r"""
        The fit function is used to transfer the information of
        data and return the fit result.

        Parameters
        ----------
        X : array-like, shape(n_samples, p_features)
            Training data.
        r : int
            Rank of the (recovered) information matrix L.
            It should be smaller than rank of X
            (at least smaller than X.shape[1]).
        group : int, optional, default=np.ones(p)
            The group index for each variable.
        """

        # Input check
        if isinstance(X, (list, np.ndarray, np.matrix, coo_matrix)):
            if isinstance(X, coo_matrix):
                self.sparse_matrix = True
            X = check_array(X, accept_sparse=True)

            n = X.shape[0]
            p = X.shape[1]
            self.n_features_in_ = p
        else:
            raise ValueError("X should be an array.")

        # # Algorithm_type
        # if self.algorithm_type == "abess":
        #     algorithm_type_int = 6
        # else:
        #     raise ValueError("algorithm_type should not be " +
        #                      str(self.algorithm_type))

        # for RPCA,
        # model_type_int = 10
        path_type_int = 1

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

        # Group
        if group is None:
            g_index = list(range(n * p))
        else:
            group = np.array(group)
            if group.ndim > 1:
                raise ValueError("group should be an 1D array of integers.")
            if group.size != n * p:
                raise ValueError(
                    "The length of group should be equal to"
                    " (X.shape[0] * X.shape[1]).")
            g_index = []
            group.sort()
            group_set = list(set(group))
            j = 0
            for i in group_set:
                while group[j] != i:
                    j += 1
                g_index.append(j)

        # path parameter (note that: path_type_int = 1)
        if self.support_size is None:
            support_sizes = list(range(0, n * p))
        else:
            if isinstance(self.support_size, (numbers.Real, numbers.Integral)):
                support_sizes = np.empty(1, dtype=int)
                support_sizes[0] = self.support_size
            elif (np.any(np.array(self.support_size) > n * p) or
                    np.any(np.array(self.support_size) < 0)):
                raise ValueError(
                    "All support_size should be between 0 and X.shape[1]")
            else:
                support_sizes = self.support_size
        support_sizes = np.array(support_sizes).astype('int32')

        # alphas
        if r == int(r):
            alphas = np.array([r], dtype=float)
        else:
            raise ValueError("r should be integer")

        # unused
        new_s_min = 0
        new_s_max = 0
        new_lambda_min = 0
        new_lambda_max = 0

        # Exchange_num
        if (not isinstance(self.exchange_num, int) or self.exchange_num <= 0):
            raise ValueError("exchange_num should be an positive integer.")

        # Thread
        if (not isinstance(self.thread, int) or self.thread < 0):
            raise ValueError(
                "thread should be positive number or 0"
                " (maximum supported by your device).")

        # Splicing type
        if self.splicing_type not in (0, 1):
            raise ValueError("splicing type should be 0 or 1.")

        # # Important_search
        # if (not isinstance(self.important_search, int)
        #         or self.important_search < 0):
        #     raise ValueError(
        #         "important_search should be a non-negative number.")

        # A_init
        if A_init is None:
            A_init = np.array([], dtype="int32")
        else:
            A_init = np.array(A_init, dtype="int32")
            if A_init.ndim > 1:
                raise ValueError(
                    "The initial active set should be an 1D array of"
                    " integers.")
            if (A_init.min() < 0 or A_init.max() >= n * p):
                raise ValueError(
                    "A_init contains wrong index.")

        # Sparse X
        if self.sparse_matrix:
            if not isinstance(X, type(coo_matrix((1, 1)))):
                # print("sparse matrix 1")
                nonzero = 0
                tmp = np.zeros([X.shape[0] * X.shape[1], 3])
                for j in range(X.shape[1]):
                    for i in range(X.shape[0]):
                        if X[i, j] != 0.:
                            tmp[nonzero, :] = np.array([X[i, j], i, j])
                            nonzero += 1
                X = tmp[:nonzero, :]
            else:
                # print("sparse matrix 2")
                tmp = np.zeros([len(X.data), 3])
                tmp[:, 1] = X.row
                tmp[:, 2] = X.col
                tmp[:, 0] = X.data

                ind = np.lexsort((tmp[:, 2], tmp[:, 1]))
                X = tmp[ind, :]

        # normalize
        normalize = 0

        # always_select
        if self.always_select is None:
            always_select_list = np.zeros(0, dtype="int32")
        else:
            always_select_list = np.array(self.always_select, dtype="int32")

        # unused
        n_lambda = 100
        early_stop = False

        # wrap with cpp
        result = pywrap_RPCA(
            X, n, p, normalize,
            self.max_iter, self.exchange_num,
            path_type_int, self.is_warm_start,
            ic_type_int, self.ic_coef,
            g_index,
            support_sizes,
            alphas,
            new_s_min, new_s_max,
            new_lambda_min, new_lambda_max, n_lambda,
            self.screening_size,
            always_select_list,
            self.primary_model_fit_max_iter, self.primary_model_fit_epsilon,
            early_stop,
            self.thread,
            self.sparse_matrix,
            self.splicing_type,
            self.important_search,
            A_init
        )

        self.coef_ = result[0].reshape(p, n).T
        self.train_loss_ = result[2]
        return self
