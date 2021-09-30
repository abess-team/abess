from .bess_base import bess_base

from sklearn.utils.validation import check_array
from abess.cabess import pywrap_abess
from scipy.sparse import coo_matrix
import numpy as np
import types
import numbers


def fix_docs(cls):
    # inherit the document from base class
    index = cls.__doc__.find("Examples\n    --------\n")
    if (index != -1):
        cls.__doc__ = cls.__doc__[:index] + \
            cls.__bases__[0].__doc__ + cls.__doc__[index:]

    # for name, func in vars(cls).items():
    #     if isinstance(func, types.FunctionType):
    #         # print(str(func) +  'needs doc')
    #         for parent in cls.__bases__:
    #             parfunc = getattr(parent, name, None)
    #             if parfunc and getattr(parfunc, '__doc__', None):
    #                 func.__doc__ = parfunc.__doc__ + func.__doc__
    return cls


@fix_docs
class abessPCA(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for principal component analysis.

    Parameters
    ----------
    splicing_type: {0, 1}, optional
        The type of splicing in `fit()` (in Algorithm.h). 
        "0" for decreasing by half, "1" for decresing by one.
        Default: splicing_type = 1.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.pca import abessPCA
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> model = abessPCA(support_size = 10)
    >>>
    >>> ### X known
    >>> X = np.random.randn(100, 50)
    >>> model.fit(X)
    >>> print(model.coef_)
    >>>
    >>> ### X unknown, but Sigma known
    >>> model.fit(Sigma = np.cov(X.T))
    >>> print(model.coef_)
    """

    def __init__(self, max_iter=20, exchange_num=5, is_warm_start=True, support_size=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, 
                 always_select=[], 
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=1
                 ):
        super(abessPCA, self).__init__(
            algorithm_type="abess", model_type="PCA", data_type=1, path_type="seq", max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, s_min=s_min, s_max=s_max, 
            ic_type=ic_type, ic_coef=ic_coef, 
            always_select=always_select, 
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )

    def transform(self, X):
        """
        For PCA model, apply dimensionality reduction 
        to given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Test data.

        """
        X = self.new_data_check(X)

        return X.dot(self.coef_)

    def ratio(self, X):
        """
        Give new data, and it returns the explained ratio.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        """
        X = self.new_data_check(X)
        s = np.cov(X.T)
        if len(self.coef_.shape) == 1:
            explain = self.coef_.T.dot(s).dot(self.coef_)
        else:
            explain = np.sum(np.diag(self.coef_.T.dot(s).dot(self.coef_)))
        if isinstance(s, (int, float)):
            full = s
        else:
            full = np.sum(np.diag(s))
        return explain / full

    def fit(self, X=None, is_normal=True, group=None, Sigma=None, number=1, n=None):
        """
        The fit function is used to transfer the information of data and return the fit result.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Training data
        is_normal : bool, optional
            whether normalize the variables array before fitting the algorithm.
            Default: is_normal=True.
        weight : array-like of shape (n_samples,)
            Individual weights for each sample. Only used for is_weight=True.
            Default is 1 for each observation.
        group : int, optional
            The group index for each variable.
            Default: group = \code{numpy.ones(p)}.
        Sigma : array-like of shape (n_features, n_features), optional
            Sample covariance matrix.
            For PCA, it can be given as input, instead of X. But if X is given, Sigma will be set to \code{np.cov(X.T)}.
            Default: Sigma = \code{np.cov(X.T)}.
        number : int, optional 
            Indicates the number of PCs returned. 
            Default: 1
        n : int, optional
            Sample size. If X is given, it would be X.shape[0]; if Sigma is given, it would be 1 by default.
            Default: X.shape[0] or 1.
        """

        # Input check
        if isinstance(X, (list, np.ndarray, np.matrix, coo_matrix)):
            if isinstance(X, coo_matrix):
                self.sparse_matrix = True
            X = check_array(X, accept_sparse=True)

            n = X.shape[0]
            p = X.shape[1]
            M = 1
            y = np.zeros((n, 1))
            X = X - X.mean(axis=0)
            Sigma = np.cov(X.T)
            self.n_features_in_ = p

        elif isinstance(Sigma, (list, np.ndarray, np.matrix)):
            Sigma = check_array(Sigma)

            if (Sigma.shape[0] != Sigma.shape[1] or np.any(Sigma.T != Sigma)):
                raise ValueError("Sigma should be symmetrical matrix.")
            elif np.any(np.linalg.eigvals(Sigma) < 0):
                raise ValueError("Sigma should be semi-positive definite.")

            if (n is None): 
                n = 1
            p = Sigma.shape[0]
            X = np.zeros((1, p))
            y = np.zeros((1, 1))
            M = 1
            self.n_features_in_ = p
            is_normal = False
        else:
            raise ValueError("X or Sigma should be given in PCA.")

        # Algorithm_type
        if self.algorithm_type == "abess":
            algorithm_type_int = 6
        else:
            raise ValueError("algorithm_type should not be " +
                             str(self.algorithm_type))

        # for PCA,
        #   model_type_int = 7,
        #   path_type_int = 1 (seq)
        model_type_int = 7
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

        # cv
        if (not isinstance(self.cv, int) or self.cv <= 0):
            raise ValueError("cv should be an positive integer.")
        elif (self.cv > 1):
            self.is_cv = True
        
        # cv_mask
        cv_mask = np.array(self.cv_mask)
        if len(cv_mask) > 0:
            if (cv_mask.dtype != "int"):
                raise ValueError("cv_mask should be integer.")
            elif cv_mask.ndim > 1:
                raise ValueError("cv_mask should be a 1-D array.")
            elif cv_mask.size != n:
                raise ValueError("X.shape[0] should be equal to cv_mask.size")
        cv_mask = np.array(cv_mask, dtype = "int32")

        # Group
        if group is None:
            g_index = list(range(p))
        else:
            group = np.array(group)
            if group.ndim > 1:
                raise ValueError("group should be an 1D array of integers.")
            elif group.size != p:
                raise ValueError(
                    "The length of group should be equal to X.shape[1].")
            g_index = []
            group.sort()
            group_set = list(set(group))
            j = 0
            for i in group_set:
                while(group[j] != i):
                    j += 1
                g_index.append(j)

        # path parameter (note that: path_type_int = 1)
        if self.support_size is None:
            support_sizes = list(range(0, p))
        else:
            if isinstance(self.support_size, (numbers.Real, numbers.Integral)):
                support_sizes = np.empty(1, dtype=int)
                support_sizes[0] = self.support_size
            elif (np.any(np.array(self.support_size) > p) or
                    np.any(np.array(self.support_size) < 0)):
                raise ValueError(
                    "All support_size should be between 0 and X.shape[1]")
            else:
                support_sizes = self.support_size
        support_sizes = np.array(support_sizes).astype('int32')

        # unused
        new_s_min = 0
        new_s_max = 0
        new_K_max = 0
        new_lambda_min = 0
        new_lambda_max = 0
        alphas = [0]
        new_screening_size = -1

        # Exchange_num
        if (not isinstance(self.exchange_num, int) or self.exchange_num <= 0):
            raise ValueError("exchange_num should be an positive integer.")

        # # Is_screening
        # if self.is_screening:
        #     new_screening_size = p \
        #         if self.screening_size is None else self.screening_size

        #     if self.screening_size > p:
        #         raise ValueError(
        #             "screening size should be smaller than X.shape[1].")
        #     elif self.screening_size < max(support_sizes):
        #         raise ValueError(
        #             "screening size should be more than max(support_size).")
        # else:
        #     new_screening_size = -1

        # Thread
        if (not isinstance(self.thread, int) or self.thread < 0):
            raise ValueError(
                "thread should be positive number or 0 (maximum supported by your device).")

        # Splicing type
        if (self.splicing_type != 0 and self.splicing_type != 1):
            raise ValueError("splicing type should be 0 or 1.")

        # number
        if (not isinstance(number, int) or number <= 0 or number > p):
            raise ValueError(
                "number should be an positive integer and not bigger than X.shape[1].")
        
        # Important_search
        if (not isinstance(self.important_search, int) or self.important_search < 0):
            raise ValueError(
                "important_search should be a non-negative number.")

        # Sparse X
        if self.sparse_matrix:
            if type(X) != type(coo_matrix((1, 1))):
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

        # wrap with cpp
        weight = np.ones(n)
        state = [0]
        result = pywrap_abess(X, y, n, p, self.data_type, weight, Sigma,
                              is_normal,
                              algorithm_type_int, model_type_int, self.max_iter, self.exchange_num,
                              path_type_int, self.is_warm_start,
                              ic_type_int, self.ic_coef, self.is_cv, self.cv,
                              g_index,
                              state,
                              support_sizes,
                              alphas,
                              cv_mask,
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
                              p * M,
                              1 * M, 1, 1, 1, 1, 1, p
                              )

        self.coef_ = result[0]

        # for PCA, "number" indicates the number of PCs returned
        if (number > 1):
            v = self.coef_.copy()
            v = v.reshape(len(v), 1)
            v_all = v.copy()
            while (number > 1):
                number = number - 1
                temp = v.dot(v.T).dot(Sigma)
                Sigma = Sigma + temp.dot(v).dot(v.T) - temp - temp.T

                result = pywrap_abess(X, y, n, p, self.data_type, weight, Sigma,
                                      is_normal,
                                      algorithm_type_int, model_type_int, self.max_iter, self.exchange_num,
                                      path_type_int, self.is_warm_start,
                                      ic_type_int, self.ic_coef, self.is_cv, self.cv,
                                      g_index,
                                      state,
                                      support_sizes,
                                      alphas,
                                      cv_mask,
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
                                      p * M,
                                      1 * M, 1, 1, 1, 1, 1, p
                                      )
                v = result[0]
                v = v.reshape(len(v), 1)
                v_all = np.hstack((v_all, v))

            self.coef_ = v_all
            self.intercept_ = None
            self.train_loss_ = None
            self.ic_ = None

        return self

    def fit_transform(self, X=None, is_normal=True, group=None, Sigma=None, number=1):
        self.fit(X, is_normal, group, Sigma, number)
        return X.dot(self.coef_)
