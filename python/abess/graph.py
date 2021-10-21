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
                 always_select=[], alpha = None, cv = 1,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=1
                 ):
        super(abessGraph, self).__init__(
            algorithm_type="abess", model_type="Graph", data_type=1, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, s_min=s_min, s_max=s_max, 
            ic_type=ic_type, ic_coef=ic_coef, primary_model_fit_max_iter=primary_model_fit_max_iter, primary_model_fit_epsilon=primary_model_fit_epsilon,
            always_select=always_select, alpha = alpha, cv = cv,
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

        # Input check
        if isinstance(X, (list, np.ndarray, np.matrix, coo_matrix)):
            if isinstance(X, coo_matrix):
                self.sparse_matrix = True
            X = check_array(X, accept_sparse=True)

            n = X.shape[0]
            p = X.shape[1]
            M = 1
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
            cv_fold_id = np.array([], dtype = "int32")
        else:
            cv_fold_id = np.array(cv_fold_id, dtype = "int32")
            if cv_fold_id.ndim > 1:
                raise ValueError("group should be an 1D array of integers.")
            elif cv_fold_id.size != n:
                raise ValueError(
                    "The length of group should be equal to X.shape[0].")
            elif len(set(cv_fold_id)) != self.cv:
                raise ValueError(
                    "The number of different masks should be equal to `cv`.")

        # Group
        # if group is None:
        #     g_index = list(range(p))
        # else:
        #     group = np.array(group)
        #     if group.ndim > 1:
        #         raise ValueError("group should be an 1D array of integers.")
        #     elif group.size != p:
        #         raise ValueError(
        #             "The length of group should be equal to X.shape[1].")
        #     g_index = []
        #     group.sort()
        #     group_set = list(set(group))
        #     j = 0
        #     for i in group_set:
        #         while(group[j] != i):
        #             j += 1
        #         g_index.append(j)

        # path parameter (note that: path_type_int = 1)
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

        # wrap with cpp
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

@fix_docs
class abessIsing(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for Ising model.

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
    >>> from abess.graph import abessIsing
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> model = abessIsing(support_size = 10)
    >>>
    >>> ### X known
    >>> 
    >>> model.fit(X)
    >>> print(model.coef_)
    """

    def __init__(self, max_iter=200, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, s_min=None, s_max=None,
                 ic_type="aic", ic_coef=1.0, primary_model_fit_max_iter=500, primary_model_fit_epsilon=1e-6,
                 always_select=[], alpha = None, cv = 1,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=1
                 ):
        super(abessIsing, self).__init__(
            algorithm_type="abess", model_type="Ising", data_type=1, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, s_min=s_min, s_max=s_max, 
            ic_type=ic_type, ic_coef=ic_coef, primary_model_fit_max_iter=primary_model_fit_max_iter, primary_model_fit_epsilon=primary_model_fit_epsilon,
            always_select=always_select, alpha = alpha, cv = cv,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )

    def fit(self, X=None, frequence=None, cv_fold_id=None):
        """
        The fit function is used to transfer the information of data and return the fit result.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Training data.
        frequence : array-like of shape (n_samples,)
            The frequence for each sample. 
            Default is 1 for each observation.
        cv_fold_id: array_like of shape (n_samples,) , optional
            An array indicates different folds in CV. Samples in the same fold should be given the same number.
            Default: cv_fold_id=None
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
            self.n_features_in_ = p

        else:
            raise ValueError("X should be given in Ising model.")

        # Algorithm_type
        if self.algorithm_type == "abess":
            algorithm_type_int = 6
        else:
            raise ValueError("algorithm_type should not be " +
                             str(self.algorithm_type))

        # for Ising,
        #   model_type_int = 8,
        #   path_type_int = 1 (seq)
        model_type_int = 8
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

        # cv_fold_id
        if cv_fold_id is None:
            cv_fold_id = np.array([], dtype = "int32")
        else:
            cv_fold_id = np.array(cv_fold_id, dtype = "int32")
            if cv_fold_id.ndim > 1:
                raise ValueError("group should be an 1D array of integers.")
            elif cv_fold_id.size != n:
                raise ValueError(
                    "The length of group should be equal to X.shape[0].")
            elif len(set(cv_fold_id)) != self.cv:
                raise ValueError(
                    "The number of different masks should be equal to `cv`.")

        # frequence:
        if frequence is None:
            frequence = np.ones(n)
        else:
            frequence = np.array(frequence)
            if (frequence.dtype != "int" and frequence.dtype != "float"):
                raise ValueError("frequence should be numeric.")
            elif frequence.ndim > 1:
                raise ValueError("frequence should be a 1-D array.")
            elif frequence.size != n:
                raise ValueError("X.shape[0] should be equal to frequence.size")

        # Group
        # if group is None:
        #     g_index = list(range(p))
        # else:
        #     group = np.array(group)
        #     if group.ndim > 1:
        #         raise ValueError("group should be an 1D array of integers.")
        #     elif group.size != p:
        #         raise ValueError(
        #             "The length of group should be equal to X.shape[1].")
        #     g_index = []
        #     group.sort()
        #     group_set = list(set(group))
        #     j = 0
        #     for i in group_set:
        #         while(group[j] != i):
        #             j += 1
        #         g_index.append(j)

        # path parameter (note that: path_type_int = 1)
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

        # wrap with cpp
        result = pywrap_abess(X, y, n, p, self.data_type, frequence, Sigma,
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

