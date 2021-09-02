import math
import numbers
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator
from abess.cabess import pywrap_abess


class bess_base(BaseEstimator):
    """
    Parameters
    ----------
    max_iter : int, optional
        Maximum number of iterations taken for the splicing algorithm to converge.
        Due to the limitation of loss reduction, the splicing algorithm must be able to converge.
        The number of iterations is only to simplify the implementation.
        Default: max_iter = 20.
    is_warm_start : bool, optional
        When tuning the optimal parameter combination, whether to use the last solution as a warm start to accelerate the iterative convergence of the splicing algorithm.
        Default:is_warm_start = True.
    path_type : {"seq", "pgs"}
        The method to be used to select the optimal support size.
        For path_type = "seq", we solve the best subset selection problem for each size in support_size.
        For path_type = "gs", we solve the best subset selection problem with support size ranged in (s_min, s_max), where the specific support size to be considered is determined by golden section.
    support_size : array_like, optional
        An integer vector representing the alternative support sizes. Only used for path_type = "seq".
        Default is 0:min(n, round(n/(log(log(n))log(p)))).
    s_min : int, optional
        The lower bound of golden-section-search for sparsity searching.
        Default: s_min = 1.
    s_max : int, optional
        The higher bound of golden-section-search for sparsity searching.
        Default: s_max = min(n, round(n/(log(log(n))log(p)))).
    ic_type : {'aic', 'bic', 'gic', 'ebic'}, optional
        The type of criterion for choosing the support size. Available options are "gic", "ebic", "bic", "aic".
        Default: ic_type = 'ebic'.
    cv : int, optional
        The folds number when Use the Cross-validation method. If cv=1, cross-validation would not be used.
        Default: cv = 1.
    thread: int, optional
        Max number of multithreads. If thread = 0, the program will use the maximum number supported by the device.
        Default: thread = 1. 
    is_screening: bool, optional
        Screen the variables first and use the chosen variables in abess process.
        Default: is_screen = False.
    screen_size: int, optional
        This parameter is only useful when is_screen = True. 
        The number of variables remaining after screening. It should be a non-negative number smaller than p.
        Default: screen_size = None.
    always_select: array_like, optional
        An array contains the indexes of variables we want to consider in the model.
        Default: always_select = [].
    primary_model_fit_max_iter: int, optional
        The maximal number of iteration in `primary_model_fit()` (in Algorithm.h). 
        Default: primary_model_fit_max_iter = 10.
    primary_model_fit_epsilon: double, optional
        The epsilon (threshold) of iteration in `primary_model_fit()` (in Algorithm.h). 
        Default: primary_model_fit_max_iter = 1e-08.

    Returns
    ------- 
    coef_: array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the best subset selection problem.
    ic_: double
        The score under chosen information criterion.

    References
    ----------
    - Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang. A polynomial algorithm for best-subset selection problem. Proceedings of the National Academy of Sciences, 117(52):33117-33123, 2020.


    """

    def __init__(self, algorithm_type, model_type, data_type, path_type, max_iter=20, exchange_num=5, is_warm_start=True,
                 support_size=None, alpha=None, s_min=None, s_max=None, 
                 ic_type="ebic", ic_coef=1.0,
                 cv=1, is_screening=False, screening_size=None, 
                 always_select=[], 
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 approximate_Newton=False,
                 thread=1,
                 covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=0,
                 # early_stop=False, tau=0., K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, n_lambda=100, powell_path=1,
                 ):
        self.algorithm_type = algorithm_type
        self.model_type = model_type
        self.data_type = data_type
        self.path_type = path_type
        self.max_iter = max_iter
        self.exchange_num = exchange_num
        self.is_warm_start = is_warm_start
        self.support_size = support_size
        self.alpha = alpha
        self.s_min = s_min
        self.s_max = s_max
        self.K_max = 1
        self.epsilon = 0.0001
        self.lambda_min = None
        self.lambda_max = None
        self.n_lambda = 100
        self.ic_type = ic_type
        self.ic_coef = ic_coef
        self.is_cv = False
        self.cv = cv
        self.is_screening = is_screening
        self.screening_size = screening_size
        self.powell_path = 1
        self.always_select = always_select
        self.tau = 0.
        self.primary_model_fit_max_iter = primary_model_fit_max_iter
        self.primary_model_fit_epsilon = primary_model_fit_epsilon
        self.early_stop = False
        self.approximate_Newton = approximate_Newton
        self.thread = thread
        self.covariance_update = covariance_update
        self.sparse_matrix = sparse_matrix
        self.splicing_type = splicing_type
        self.important_search = important_search

    def new_data_check(self, X, y=None):
        # Check1 : whether fit had been called
        check_is_fitted(self)

        # Check2 : X validation
        X = check_array(X, accept_sparse=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X.shape[1] should be " +
                             str(self.n_features_in_))

        # Check3 : y validation
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=True, multi_output=True, y_numeric=True)
            return X, y

        return X

    def fit(self, X=None, y=None, is_normal=True, weight=None, group=None):
        """
        The fit function is used to transfer the information of data and return the fit result.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Training data
        y :  array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
            For linear regression problem, y should be a n time 1 numpy array with type \code{double}.
            For classification problem, \code{y} should be a $n \time 1$ numpy array with values \code{0} or \code{1}.
            For count data, \code{y} should be a $n \time 1$ numpy array of non-negative integer.
        is_normal : bool, optional
            whether normalize the variables array before fitting the algorithm.
            Default: is_normal=True.
        weight : array-like of shape (n_samples,)
            Individual weights for each sample. Only used for is_weight=True.
            Default: weight = 1 for each observation.
        group : int, optional
            The group index for each variable.
            Default: group = \code{numpy.ones(p)}.
        """

        # print("fit enter.")#///

        # Input check & init:
        if isinstance(X, (list, np.ndarray, np.matrix, coo_matrix)):
            if isinstance(X, coo_matrix):
                self.sparse_matrix = True

            # Check that X and y have correct shape
            X, y = check_X_y(X, y, accept_sparse=True,
                             multi_output=True, y_numeric=True)

            # Sort for Cox
            if self.model_type == "Cox":
                X = X[y[:, 0].argsort()]
                y = y[y[:, 0].argsort()]
                y = y[:, 1].reshape(-1)

            # Init
            n = X.shape[0]
            p = X.shape[1]
            Sigma = np.matrix(-1)
            self.n_features_in_ = p

            if y.ndim == 1:
                M = 1
                y = y.reshape(len(y), 1)
            else:
                M = y.shape[1]
        else:
            raise ValueError("X should be a matrix or sparse matrix.")

        # Algorithm_type: abess
        if self.algorithm_type == "abess":
            algorithm_type_int = 6
        else:
            raise ValueError("algorithm_type should not be " +
                             str(self.algorithm_type))

        # Model_type: lm, logit, poiss, cox, multi-gaussian, multi-nomial
        if self.model_type == "Lm":
            model_type_int = 1
        elif self.model_type == "Logistic":
            model_type_int = 2
        elif self.model_type == "Poisson":
            model_type_int = 3
        elif self.model_type == "Cox":
            model_type_int = 4
        elif self.model_type == "Multigaussian":
            model_type_int = 5
        elif self.model_type == "Multinomial":
            model_type_int = 6
        else:
            raise ValueError("model_type should not be " +
                             str(self.model_type))

        # Path_type: seq, pgs
        if self.path_type == "seq":
            path_type_int = 1
        elif self.path_type == "pgs":
            path_type_int = 2
        else:
            raise ValueError("path_type should be \'seq\' or \'pgs\'")

        # Ic_type: aic, bic, gic, ebic
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

        # Group:
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

        # Weight:
        if weight is None:
            weight = np.ones(n)
        else:
            weight = np.array(weight)
            if (weight.dtype != "int" and weight.dtype != "float"):
                raise ValueError("weight should be numeric.")
            elif weight.ndim > 1:
                raise ValueError("weight should be a 1-D array.")
            elif weight.size != n:
                raise ValueError("X.shape[0] should be equal to weight.size")

        # Path parameters
        if path_type_int == 1:  # seq
            if self.support_size is None:
                if (n == 1 or p == 1):
                    support_sizes = [0, 1]
                else:
                    support_sizes = list(range(0, max(min(p, int(
                        n / (np.log(np.log(n)) * np.log(p)))), 1)))
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

            if self.alpha is None:
                alphas = [0]
            else:
                if isinstance(self.alpha, (numbers.Real, numbers.Integral)):
                    alphas = np.empty(1, dtype=float)
                    alphas[0] = self.alpha
                else:
                    alphas = self.alpha

            # unused
            new_s_min = 0
            new_s_max = 0
            new_K_max = 0
            new_lambda_min = 0
            new_lambda_max = 0

        elif path_type_int == 2:    # pgs
            new_s_min = 0 \
                if self.s_min is None else self.s_min
            new_s_max = min(p, int(n / (np.log(np.log(n)) * np.log(p)))) \
                if self.s_max is None else self.s_max
            new_lambda_min = 0 # \
                # if self.lambda_min is None else self.lambda_min
            new_lambda_max = 0 # \
                # if self.lambda_max is None else self.lambda_max
            new_K_max = int(math.log(p, 2/(math.sqrt(5) - 1))) # \
                # if self.K_max is None else self.K_max

            if (new_s_max < new_s_min):
                raise ValueError("s_max should be larger than s_min")
            # if (new_lambda_max < new_lambda_min):
            #     raise ValueError(
            #         "lambda_max should be larger than lambda_min.")

            # unused
            support_sizes = [0]
            alphas = [0]
        support_sizes = np.array(support_sizes, dtype='int32')

        # Exchange_num
        if (not isinstance(self.exchange_num, int) or self.exchange_num <= 0):
            raise ValueError("exchange_num should be an positive integer.")
        # elif (self.exchange_num > min(support_sizes)):
        #     print("[Warning]  exchange_num may be larger than sparsity, and it would be set up to sparsity.")

        # Is_screening
        if self.is_screening:
            new_screening_size = min(p, int(n / (np.log(np.log(n)) * np.log(p)))) \
                if self.screening_size is None else self.screening_size

            if self.screening_size > p:
                raise ValueError(
                    "screening size should be smaller than X.shape[1].")
            elif self.screening_size < max(support_sizes):
                raise ValueError(
                    "screening size should be more than max(support_size).")
        else:
            new_screening_size = -1

        # Primary fit parameters
        if (not isinstance(self.primary_model_fit_max_iter, int) or self.primary_model_fit_max_iter <= 0):
            raise ValueError(
                "primary_model_fit_max_iter should be an positive integer.")
        if (self.primary_model_fit_epsilon < 0):
            raise ValueError(
                "primary_model_fit_epsilon should be non-negative.")

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

        # wrap with cpp
        # print("wrap enter.")#///
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

        # print("linear fit end")
        # print(len(result))
        # print(result)
        if M != 1:
            self.coef_ = result[0].reshape(p, M)
        else:
            self.coef_ = result[0]
        self.intercept_ = result[1]

        self.train_loss_ = result[2]
        self.ic_ = result[3]

        return self
