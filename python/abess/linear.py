from abess.cabess import pywrap_abess
import numpy as np
import math
import types
from scipy.sparse import coo_matrix
import numbers

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .metrics import concordance_index_censored

# from time import time

# def fix_docs(cls):
#     for name, func in vars(cls).items():
#         if isinstance(func, types.FunctionType) and not func.__doc__:
#             # print(str(func) +  'needs doc')
#             for parent in cls.__bases__:
#                 parfunc = getattr(parent, name, None)
#                 if parfunc and getattr(parfunc, '__doc__', None):
#                     func.__doc__ = parfunc.__doc__
#                     break
#     return cls


def fix_docs(cls):
    # inherit the ducument from base class
    index = cls.__doc__.find("Examples\n    --------\n")
    if(index != -1):
        cls.__doc__ = cls.__doc__[:index] + \
            cls.__bases__[0].__doc__ + cls.__doc__[index:]

    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType):
            # print(str(func) +  'needs doc')
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__ + func.__doc__
    return cls


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
    K_max : int, optional
        The max search time of golden-section-search for sparsity searching.
        Default: K_max = int(log(p, 2/(math.sqrt(5) - 1))).
    epsilon : double, optional
        The stop condition of golden-section-search for sparsity searching.
        Default: epsilon = 0.0001.
    ic_type : {'aic', 'bic', 'gic', 'ebic'}, optional
        The type of criterion for choosing the support size. Available options are "gic", "ebic", "bic", "aic".
        Default: ic_type = 'ebic'.
    is_cv : bool, optional
        Use the Cross-validation method to choose the support size.
        Default: is_cv = False.
    K : int optional
        The folds number when Use the Cross-validation method.
        Default: K = 5.


    Atrributes
    ----------
    beta : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the best subset selection problem.


    References
    ----------
    - Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang. A polynomial algorithm for best-subset selection problem. Proceedings of the National Academy of Sciences, 117(52):33117-33123, 2020.


    """

    def __init__(self, algorithm_type, model_type, data_type, path_type, max_iter=20, exchange_num=5, is_warm_start=True,
                 support_size=None, alpha=None, s_min=None, s_max=None, K_max=1, epsilon=0.0001, lambda_min=0, lambda_max=0, n_lambda=100,
                 ic_type="ebic", ic_coef=1.0,
                 is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
                 always_select=[], tau=0.,
                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-8,
                 early_stop=False, approximate_Newton=False,
                 thread=1,
                 covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0,
                 input_type=0):
        self.algorithm_type = algorithm_type
        self.model_type = model_type
        self.data_type = data_type
        self.path_type = path_type
        # self.algorithm_type_int = None
        # self.model_type_int = None
        # self.path_type_int = None
        self.max_iter = max_iter
        self.exchange_num = exchange_num
        self.is_warm_start = is_warm_start
        self.support_size = support_size
        self.alpha = alpha
        self.s_min = s_min
        self.s_max = s_max
        self.K_max = K_max
        self.epsilon = epsilon
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        # to do
        self.n_lambda = n_lambda

        self.ic_type = ic_type
        # self.ic_type_int = None
        self.ic_coef = ic_coef
        self.is_cv = is_cv
        self.K = K
        # self.path_len = None
        # self.p = None
        # self.data_type = None
        self.is_screening = is_screening
        self.screening_size = screening_size
        self.powell_path = powell_path
        self.always_select = always_select
        self.tau = tau
        self.primary_model_fit_max_iter = primary_model_fit_max_iter
        self.primary_model_fit_epsilon = primary_model_fit_epsilon
        self.early_stop = early_stop
        self.approximate_Newton = approximate_Newton
        self.thread = thread
        self.covariance_update = covariance_update
        self.sparse_matrix = sparse_matrix
        self.splicing_type = splicing_type
        self.input_type = input_type

    def _arg_check(self):
        """
        Arguments check.

        """
        # print("arg_check")
        # if self.algorithm_type == "Pdas":
        #     self.algorithm_type_int = 1
        # elif self.algorithm_type == "GroupPdas":
        #     self.algorithm_type_int = 2
        # elif self.algorithm_type == "L0L2":
        #     self.algorithm_type_int = 5
        # elif self.algorithm_type == "abess":
        #     self.algorithm_type_int = 6
        # else:
        #     raise ValueError("algorithm_type should not be " +
        #                      str(self.algorithm_type))

        # if self.model_type == "Lm":
        #     self.model_type_int = 1
        # elif self.model_type == "Logistic":
        #     self.model_type_int = 2
        # elif self.model_type == "Poisson":
        #     self.model_type_int = 3
        # elif self.model_type == "Cox":
        #     self.model_type_int = 4
        # elif self.model_type == "Multigaussian":
        #     self.model_type_int = 5
        # elif self.model_type == "Multinomial":
        #     self.model_type_int = 6
        # else:
        #     raise ValueError("model_type should not be " +
        #                      str(self.model_type))

        # if self.path_type == "seq":
        #     # if self.support_size is None:
        #     #     raise ValueError(
        #     #         "When you choose path_type = support_size-search, the parameter \'support_size\' should be given.")
        #     self.path_type_int = 1

        # elif self.path_type == "pgs":
        #     # if self.s_min is None:
        #     #     raise ValueError(
        #     #         " When you choose path_type = golden-section-search, the parameter \'s_min\' should be given.")
        #     #
        #     # if self.s_max is None:
        #     #     raise ValueError(
        #     #         " When you choose path_type = golden-section-search, the parameter \'s_max\' should be given.")
        #     #
        #     # if self.K_max is None:
        #     #     raise ValueError(
        #     #         " When you choose path_type = golden-section-search, the parameter \'K_max\' should be given.")
        #     #
        #     # if self.epsilon is None:
        #     #     raise ValueError(
        #     #         " When you choose path_type = golden-section-search, the parameter \'epsilon\' should be given.")
        #     self.path_type_int = 2
        # else:
        #     raise ValueError("path_type should be \'seq\' or \'pgs\'")

        # if self.ic_type == "aic":
        #     self.ic_type_int = 1
        # elif self.ic_type == "bic":
        #     self.ic_type_int = 2
        # elif self.ic_type == "gic":
        #     self.ic_type_int = 3
        # elif self.ic_type == "ebic":
        #     self.ic_type_int = 4
        # else:
        #     raise ValueError(
        #         "ic_type should be \"aic\", \"bic\", \"ebic\" or \"gic\"")

    def fit(self, X=None, y=None, is_weight=False, is_normal=True, weight=None, state=None, group=None, always_select=None, Sigma=None):
        """
        The fit function is used to transfer the information of data and return the fit result.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y :  array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
            For linear regression problem, y should be a n time 1 numpy array with type \code{double}.
            For classification problem, \code{y} should be a $n \time 1$ numpy array with values \code{0} or \code{1}.
            For count data, \code{y} should be a $n \time 1$ numpy array of non-negative integer.
        is_weight : bool
            whether to weight sample yourself.
            Default: is$\_$weight = False.
        is_normal : bool, optional
            whether normalize the variables array before fitting the algorithm.
            Default: is$\_$normal=True.
        weight : array-like of shape (n_samples,)
            Individual weights for each sample. Only used for is_weight=True.
            Default is 1 for each observation.
        group : int, optional
            The group index for each variable.
            Default: \code{group} = \code{numpy.ones(p)}.
        always_select : array-like
            An integer vector containing the indexes of variables that should always be included in the model.
            Default: None

        """
        # self._arg_check()


        if X is not None:   # input_type=0
            X = np.array(X)
            if (X.dtype != 'int' and X.dtype != 'float'):
                raise ValueError("X should be numeric matrix.")
            elif len(X.shape) != 2:
                raise ValueError("X should be 2-dimension matrix.")

            n = X.shape[0]
            p = X.shape[1]
            if (y is None):
                if (self.model_type == "SPCA"):
                    y = np.zeros(n)
                else:
                    raise ValueError("y should be given in "+str(self.algorithm_type))

            Sigma = np.matrix(-1)

            # Check that X and y have correct shape
            # accept_sparse
            X, y = check_X_y(X, y, ensure_2d=True,
                            accept_sparse=False, multi_output=True, y_numeric=True)

            self.n_features_in_ = X.shape[1]
            self.input_type = 0 
        elif (self.model_type == "SPCA"):   
            if (Sigma is not None):     # input_type=1
                Sigma = np.array(Sigma)
                if (Sigma.dtype != 'int' and Sigma.dtype != 'float'):
                    raise ValueError("Sigma should be numeric matrix.")
                elif (len(Sigma.shape) != 2):
                    raise ValueError("Sigma should be 2-dimension matrix.")
                elif (Sigma.shape[0] != Sigma.shape[1] or np.any(Sigma.T != Sigma)):
                    raise ValueError("Sigma should be symmetrical matrix.")
                elif not np.all(np.linalg.eigvals(Sigma) >= 0):
                    raise ValueError("Sigma should be semi-positive definite.")
                
                n = 1
                p = Sigma.shape[0]
                X = np.zeros((1, p))
                y = np.zeros(1)
                self.n_features_in_ = p
                self.input_type = 1
                is_normal = False # automatically ignore
            else:
                raise ValueError("X or Sigma should be given in SPCA")
        else:
            raise ValueError("X should be given in "+str(self.algorithm_type))
        

        # print("y: ")
        # print(y)

        # print("X: ")
        # print(X)
        # print(X.dtype)

        if self.algorithm_type == "Pdas":
            algorithm_type_int = 1
        elif self.algorithm_type == "GroupPdas":
            algorithm_type_int = 2
        elif self.algorithm_type == "L0L2":
            algorithm_type_int = 5
        elif self.algorithm_type == "abess":
            algorithm_type_int = 6
        else:
            raise ValueError("algorithm_type should not be " +
                             str(self.algorithm_type))

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
        elif self.model_type == "SPCA":
            model_type_int = 7
        else:
            raise ValueError("model_type should not be " +
                             str(self.model_type))

        if self.path_type == "seq":
            # if self.support_size is None:
            #     raise ValueError(
            #         "When you choose path_type = support_size-search, the parameter \'support_size\' should be given.")
            path_type_int = 1

        elif self.path_type == "pgs":
            # if self.s_min is None:
            #     raise ValueError(
            #         " When you choose path_type = golden-section-search, the parameter \'s_min\' should be given.")
            #
            # if self.s_max is None:
            #     raise ValueError(
            #         " When you choose path_type = golden-section-search, the parameter \'s_max\' should be given.")
            #
            # if self.K_max is None:
            #     raise ValueError(
            #         " When you choose path_type = golden-section-search, the parameter \'K_max\' should be given.")
            #
            # if self.epsilon is None:
            #     raise ValueError(
            #         " When you choose path_type = golden-section-search, the parameter \'epsilon\' should be given.")
            path_type_int = 2
        else:
            raise ValueError("path_type should be \'seq\' or \'pgs\'")

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

        if model_type_int == 4:
            X = X[y[:, 0].argsort()]
            y = y[y[:, 0].argsort()]
            y = y[:, 1].reshape(-1)

        if y.ndim == 1:
            M = 1
        else:
            M = y.shape[1]

        # if self.algorithm_type_int == 2:
        if group is None:
            g_index = range(p)

            # raise ValueError(
            #     "When you choose GroupPdas algorithm, the group information should be given")
        elif len(group) != p:
            raise ValueError(
                "The length of group should be equal to the number of variables")
        else:
            g_index = []
            group.sort()
            group_set = list(set(group))
            j = 0
            for i in group_set:
                while(group[j] != i):
                    j += 1
                g_index.append(j)
        # else:
        #     g_index = range(p)

        if is_weight:
            if weight is None:
                raise ValueError(
                    "When you choose is_weight is True, the parameter weight should be given")
            else:
                if n != weight.size:
                    raise ValueError(
                        "X.shape(0) should be equal to weight.size")
        else:
            weight = np.ones(n)

        # To do
        if state is None:
            state = [0]

        # path parameter
        if path_type_int == 1:
            if self.support_size is None:
                if n == 1:
                    support_sizes = [0, 1]
                elif p == 1:
                    support_sizes = [0, 1]
                else:
                    support_sizes = list(range(0, max(min(p, int(
                        n / (np.log(np.log(n)) * np.log(p)))), 1)))
            else:
                if isinstance(self.support_size, (numbers.Real, numbers.Integral)):
                    support_sizes = np.empty(1, dtype=np.int)
                    support_sizes[0] = self.support_size

                else:
                    support_sizes = self.support_size

            if self.alpha is None:
                alphas = [0]
            else:
                if isinstance(self.alpha, (numbers.Real, numbers.Integral)):
                    alphas = np.empty(1, dtype=np.float_)
                    alphas[0] = self.alpha

                else:
                    alphas = self.alpha

            new_s_min = 0
            new_s_max = 0
            new_K_max = 0
            new_lambda_min = 0
            new_lambda_max = 0
            path_len = int(len(support_sizes))
        else:
            support_sizes = [0]
            alphas = [0]
            if self.s_min is None:
                new_s_min = 0
            else:
                new_s_min = self.s_min

            if self.s_max is None:
                new_s_max = min(p, int(n / (np.log(np.log(n)) * np.log(p))))
            else:
                new_s_max = self.s_max

            if self.K_max is None:
                new_K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
            else:
                new_K_max = self.K_max

            if self.lambda_min is None:
                new_lambda_min = 0
            else:
                new_lambda_min = self.lambda_min

            if self.lambda_max is None:
                new_lambda_max = 0
            else:
                new_lambda_max = self.lambda_max

            path_len = new_K_max + 2

        if self.is_screening:
            if self.screening_size:
                if self.screening_size < max(support_sizes):
                    raise ValueError(
                        "screening size should be more than max(support_size).")
                else:
                    new_screening_size = self.screening_size
            else:
                new_screening_size = min(
                    p, int(n / (np.log(np.log(n)) * np.log(p))))
        else:
            new_screening_size = -1

        # print("argument list: ")
        # print("self.data_type: " + str(self.data_type))
        # print("weight: " + str(weight))
        # print("is_normal: " + str(is_normal))
        # print("self.algorithm_type_int: " + str(self.algorithm_type_int))
        # print("self.model_type_int: " + str(self.model_type_int))
        # print("self.max_iter: " + str(self.max_iter))
        # print("self.exchange_num: " + str(self.exchange_num))

        # print("path_type_int: " + str(self.path_type_int))
        # print("self.is_warm_start: " + str(self.is_warm_start))
        # print("self.ic_type_int: " + str(self.ic_type_int))
        # print("self.is_cv: " + str(self.is_cv))
        # print("self.K: " + str(self.K))
        # # print("g_index: " + str(g_index))
        # print("state: " + str(state))
        # print("self.support_size: " + str(self.support_size))
        # print("self.alpha: " + str(self.alpha))

        # print("self.s_min: " + str(self.s_min))
        # print("self.s_max: " + str(self.s_max))
        # print("self.K_max: " + str(self.K_max))
        # print("self.epsilon: " + str(self.epsilon))

        # print("self.lambda_min: " + str(self.lambda_min))
        # print("self.lambda_max: " + str(self.lambda_max))
        # print("self.n_lambda: " + str(self.n_lambda))
        # print("self.is_screening: " + str(self.is_screening))
        # print("self.screening_size: " + str(self.screening_size))
        # print("self.powell_path: " + str(self.powell_path))
        # print("self.tau: " + str(self.tau))

        if y.ndim == 1:
            y = y.reshape(len(y), 1)

        # start = time()
        if self.sparse_matrix:
            # print(type(X))
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

                # print("nonzeros num: " + str(nonzero))
                # coo = coo_matrix(X)
                # X = np.zeros([len(coo.data), 3])
                # print(X[:, 0])
                # print(coo.data)
                # X[:, 0] = coo.data.reshape(-1)
                # X[:, 1] = coo.row.reshape(-1)
                # X[:, 2] = coo.col.reshape(-1)
                # print(X)
            else:
                # print("sparse matrix 2")
                tmp = np.zeros([len(X.data), 3])
                tmp[:, 1] = X.row
                tmp[:, 2] = X.col
                tmp[:, 0] = X.data

                X = tmp
                # print(X)

        # stop = time()
        # print("sparse x time : " + str(stop-start))
        # print("linear.py fit")
        # print(y.shape)

        result = pywrap_abess(X, y, n, p, self.data_type, weight, Sigma,
                              is_normal,
                              algorithm_type_int, model_type_int, self.max_iter, self.exchange_num,
                              path_type_int, self.is_warm_start,
                              ic_type_int, self.ic_coef, self.is_cv, self.K,
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
        # print(self.coef_)
        # print(self.intercept_)
        # print(self.train_loss)
        # print(self.ic)
        # print("linear fit end")
        # self.nullloss_out = result[3]
        # self.aic_sequence = result[4]
        # self.bic_sequence = result[5]
        # self.gic_sequence = result[6]
        # self.A_out = result[7]
        # self.l_out = result[8]

        return self

    def predict(self, X):
        """
        The predict function is used to give prediction for new data.

        We will return the prediction of response variable.
        For linear and poisson regression problem, we return a numpy array of the prediction of the mean.
        For classification problem, we return a \code{dict} of \code{pr} and \code{y}, where \code{pr} is the probability of response variable is 1 and \code{y} is predicted to be 1 if \code{pr} > 0.5 else \code{y} is 0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X.shape[1] should be " + str(self._p))

        if self.model_type == "Lm":
            intercept_ = np.ones(X.shape[0]) * self.intercept_
            return np.dot(X, self.coef_) + intercept_
        elif self.model_type == "Logistic":
            intercept_ = np.ones(X.shape[0]) * self.intercept_
            xbeta = np.dot(X, self.coef_) + intercept_

            y = np.zeros(xbeta.size)
            y[xbeta > 0] = 1

            # xbeta[xbeta > 25] = 25
            # xbeta[xbeta < -25] = -25
            # xbeta_exp = np.exp(xbeta)
            # pr = xbeta_exp / (xbeta_exp + 1)

            # result = dict()
            # result["Y"] = y
            # result["pr"] = pr
            return y
        elif self.model_type == "Poisson":
            intercept_ = np.ones(X.shape[0]) * self.intercept_
            xbeta_exp = np.exp(np.dot(X, self.coef_) + intercept_)
            # result = dict()
            # result["lam"] = xbeta_exp
            return xbeta_exp
        elif self.model_type == "Multigaussian":
            intercept_ = np.ones(X.shape[0]) * self.intercept_
            return np.dot(X, self.coef_) + intercept_
        elif self.model_type == "Multinomial":
            intercept_ = np.ones(X.shape[0]) * self.intercept_
            xbeta = np.dot(X, self.coef_) + intercept_
            return np.argmax(xbeta)

    def score(self, X, y, sample_weight=None):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X.shape[1] should be " + str(self._p))

        if self.model_type == "Lm" or self.model_type == "Multigaussian":
            intercept_ = np.ones(X.shape[0]) * self.intercept_
            y_pre = np.dot(X, self.coef_) + intercept_
            return ((y - y_pre)*(y - y_pre)).sum()

        elif self.model_type == "Logistic":
            intercept_ = np.ones(X.shape[0]) * self.intercept_
            xbeta = np.dot(X, self.coef_) + intercept_
            eta = np.exp(xbeta)
            pr = np.exp(xbeta)
            return (y * np.log(pr) + (np.ones(X.shape[0]) - y) * np.log(np.ones(X.shape[0]) - pr)).sum()

        elif self.model_type == "Multinomial":
            intercept_ = np.ones(X.shape[0]) * self.intercept_
            xbeta = np.dot(X, self.coef_) + intercept_
            eta = np.exp(xbeta)
            for i in range(X.shape[0]):
                pr = eta[i, :] / np.sum(eta[i, :])
            return np.sum(y * np.log(pr))

        elif self.model_type == "Poisson":
            intercept_ = np.ones(X.shape[0]) * self.intercept_
            xbeta_exp = np.exp(np.dot(X, self.coef_) + intercept_)
            result = dict()
            result["lam"] = xbeta_exp
            return result

        elif self.model_type == "Cox":
            risk_score = np.dot(X, self.coef_)
            result = concordance_index_censored(
                np.array(y[:, 1], np.bool_), y[:, 0], risk_score)
            return result[0]


@ fix_docs
class abessLogistic(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for logistic regression.

    Examples
    --------
    >>> ### Sparsity known
    >>> from bess.linear import *
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
    >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
    >>> xbeta = np.matmul(x, beta)
    >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
    >>> y = np.random.binomial(1, p)
    >>> model = GroupPdasLogistic(path_type="seq", support_size=[5])
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    >>> ### Sparsity unknown
    # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
    >>>
    >>> model = GroupPdasLogistic(path_type="seq")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>>
    >>> model = GroupPdasLogistic(path_type="pgs")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, n_lambda=100, ic_type="ebic", ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
                 always_select=[], tau=0.,
                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-8,
                 early_stop=False, approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0
                 ):
        super(abessLogistic, self).__init__(
            algorithm_type="abess", model_type="Logistic", data_type=2, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
            epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=n_lambda, ic_type=ic_type, ic_coef=ic_coef, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
            always_select=always_select, tau=tau,
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
            early_stop=early_stop, approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )


@ fix_docs
class abessLm(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for linear regression.

    Examples
    --------
    >>> ### Sparsity known
    >>> from bess.linear import *
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
    >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
    >>> xbeta = np.matmul(x, beta)
    >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
    >>> y = np.random.binomial(1, p)
    >>> model = GroupPdasLogistic(path_type="seq", support_size=[5])
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    >>> ### Sparsity unknown
    # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
    >>>
    >>> model = GroupPdasLogistic(path_type="seq")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>>
    >>> model = GroupPdasLogistic(path_type="pgs")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, n_lambda=100, ic_type="ebic", ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
                 always_select=[], tau=0.,
                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-8,
                 early_stop=False, approximate_Newton=False,
                 thread=1, covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0
                 ):
        super(abessLm, self).__init__(
            algorithm_type="abess", model_type="Lm", data_type=1, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
            epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=n_lambda, ic_type=ic_type, ic_coef=ic_coef, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
            always_select=always_select, tau=tau,
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
            early_stop=early_stop, approximate_Newton=approximate_Newton,
            thread=thread, covariance_update=covariance_update,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )


@ fix_docs
class abessCox(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for COX proportional hazards model.

    Examples
    --------
    >>> ### Sparsity known
    >>> from bess.linear import *
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
    >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
    >>> xbeta = np.matmul(x, beta)
    >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
    >>> y = np.random.binomial(1, p)
    >>> model = GroupPdasLogistic(path_type="seq", support_size=[5])
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    >>> ### Sparsity unknown
    # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
    >>>
    >>> model = GroupPdasLogistic(path_type="seq")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>>
    >>> model = GroupPdasLogistic(path_type="pgs")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, n_lambda=100, ic_type="ebic", ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
                 always_select=[], tau=0.,
                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-8,
                 early_stop=False, approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0
                 ):
        super(abessCox, self).__init__(
            algorithm_type="abess", model_type="Cox", data_type=3, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
            epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=n_lambda, ic_type=ic_type, ic_coef=ic_coef, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
            always_select=always_select, tau=tau,
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
            early_stop=early_stop, approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )


@ fix_docs
class abessPoisson(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for Poisson regression.


    Examples
    --------
    >>> ### Sparsity known
    >>> from bess.linear import *
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
    >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
    >>> xbeta = np.matmul(x, beta)
    >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
    >>> y = np.random.binomial(1, p)
    >>> model = GroupPdasLogistic(path_type="seq", support_size=[5])
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    >>> ### Sparsity unknown
    # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
    >>>
    >>> model = GroupPdasLogistic(path_type="seq")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>>
    >>> model = GroupPdasLogistic(path_type="pgs")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, n_lambda=100, ic_type="ebic", ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
                 always_select=[], tau=0.,
                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-8,
                 early_stop=False, approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0
                 ):
        super(abessPoisson, self).__init__(
            algorithm_type="abess", model_type="Poisson", data_type=2, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
            epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=n_lambda, ic_type=ic_type, ic_coef=ic_coef, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
            always_select=always_select, tau=tau,
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
            early_stop=early_stop, approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )


@ fix_docs
class abessMultigaussian(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for multitasklearning.

    Examples
    --------
    >>> ### Sparsity known
    >>> from bess.linear import *
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
    >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
    >>> xbeta = np.matmul(x, beta)
    >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
    >>> y = np.random.binomial(1, p)
    >>> model = GroupPdasLogistic(path_type="seq", support_size=[5])
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    >>> ### Sparsity unknown
    # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
    >>>
    >>> model = GroupPdasLogistic(path_type="seq")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>>
    >>> model = GroupPdasLogistic(path_type="pgs")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, n_lambda=100, ic_type="ebic", ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
                 always_select=[], tau=0.,
                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-8,
                 early_stop=False, approximate_Newton=False,
                 thread=1, covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0
                 ):
        super(abessMultigaussian, self).__init__(
            algorithm_type="abess", model_type="Multigaussian", data_type=1, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
            epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=n_lambda, ic_type=ic_type, ic_coef=ic_coef, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
            always_select=always_select, tau=tau,
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
            early_stop=early_stop, approximate_Newton=approximate_Newton,
            thread=thread, covariance_update=covariance_update,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )
        self.data_type = 1


@ fix_docs
class abessMultinomial(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for multiclassification problem.


    Examples
    --------
    >>> ### Sparsity known
    >>> from bess.linear import *
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
    >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
    >>> xbeta = np.matmul(x, beta)
    >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
    >>> y = np.random.binomial(1, p)
    >>> model = GroupPdasLogistic(path_type="seq", support_size=[5])
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    >>> ### Sparsity unknown
    # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
    >>>
    >>> model = GroupPdasLogistic(path_type="seq")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>>
    >>> model = GroupPdasLogistic(path_type="pgs")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, n_lambda=100, ic_type="ebic", ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
                 always_select=[], tau=0.,
                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-8,
                 early_stop=False, approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0
                 ):
        super(abessMultinomial, self).__init__(
            algorithm_type="abess", model_type="Multinomial", data_type=2, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
            epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=n_lambda, ic_type=ic_type, ic_coef=ic_coef, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
            always_select=always_select, tau=tau,
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
            early_stop=early_stop, approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )

        
@fix_docs
class abessPCA(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for COX proportional hazards model.

    Examples
    --------
    >>> ### Sparsity known
    >>> from bess.linear import *
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
    >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
    >>> xbeta = np.matmul(x, beta)
    >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
    >>> y = np.random.binomial(1, p)
    >>> model = GroupPdasLogistic(path_type="seq", support_size=[5])
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    >>> ### Sparsity unknown
    # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
    >>>
    >>> model = GroupPdasLogistic(path_type="seq")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>>
    >>> model = GroupPdasLogistic(path_type="pgs")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, n_lambda=100, ic_type="ebic", ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
                 always_select=[], tau=0.,
                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-8,
                 early_stop=False, approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=1
                 ):
        super(abessPCA, self).__init__(
            algorithm_type="abess", model_type="SPCA", data_type=1, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
            epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=n_lambda, ic_type=ic_type, ic_coef=ic_coef, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
            always_select=always_select, tau=tau,
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
            early_stop=early_stop, approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type
        )
        self.data_type = 1


# @fix_docs
# class PdasLm(bess_base):
#     '''
#     PdasLm

#     The PDAS solution to the best subset selection for linear regression.


#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)   # fix seed to get the same result
#     >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
#     >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
#     >>> noise = np.random.normal(0, 1, 100)
#     >>> y = np.matmul(x, beta) + noise
#     >>> model = PdasLm(path_type="seq", support_size=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasLm(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLm(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     '''

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.):
#         super(PdasLm, self).__init__(
#             algorithm_type="Pdas", model_type="Lm", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.data_type = 1


# @fix_docs
# class PdasLogistic(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
#     >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
#     >>> xbeta = np.matmul(x, beta)
#     >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
#     >>> y = np.random.binomial(1, p)
#     >>> model = PdasLogistic(path_type="seq", support_size=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasLogistic(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLogistic(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(PdasLogistic, self).__init__(
#             algorithm_type="Pdas", model_type="Logistic", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.data_type = 2


# @fix_docs
# class PdasPoisson(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
#     >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
#     >>> lam = np.exp(np.matmul(x, beta))
#     >>> y = np.random.poisson(lam=lam)
#     >>> model = PdasPoisson(path_type="seq", support_size=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasPoisson(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasPoisson(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(PdasPoisson, self).__init__(
#             algorithm_type="Pdas", model_type="Poisson", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau
#         )
#         self.data_type = 2


# @fix_docs
# class PdasCox(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> data = gen_data(100, 200, family="cox", k=5, rho=0, sigma=1, c=10)
#     >>> model = PdasCox(path_type="seq", support_size=[5])
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasCox(path_type="seq")
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasCox(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(PdasCox, self).__init__(
#             algorithm_type="Pdas", model_type="Cox", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.data_type = 3


# @fix_docs
# class L0L2Lm(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)   # fix seed to get the same result
#     >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
#     >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
#     >>> noise = np.random.normal(0, 1, 100)
#     >>> y = np.matmul(x, beta) + noise
#     >>> model = PdasLm(path_type="seq", support_size=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasLm(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLm(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Lm, self).__init__(
#             algorithm_type="L0L2", model_type="Lm", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau
#         )
#         self.data_type = 1


# @fix_docs
# class L0L2Logistic(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)   # fix seed to get the same result
#     >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
#     >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
#     >>> noise = np.random.normal(0, 1, 100)
#     >>> y = np.matmul(x, beta) + noise
#     >>> model = PdasLm(path_type="seq", support_size=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasLm(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLm(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#         """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Logistic, self).__init__(
#             algorithm_type="L0L2", model_type="Logistic", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.data_type = 2


# @fix_docs
# class L0L2Poisson(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
#     >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
#     >>> lam = np.exp(np.matmul(x, beta))
#     >>> y = np.random.poisson(lam=lam)
#     >>> model = PdasPoisson(path_type="seq", support_size=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasPoisson(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasPoisson(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Poisson, self).__init__(
#             algorithm_type="L0L2", model_type="Poisson", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau
#         )
#         self.data_type = 2


# @fix_docs
# class L0L2Cox(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> data = gen_data(100, 200, family="cox", k=5, rho=0, sigma=1, c=10)
#     >>> model = PdasCox(path_type="seq", support_size=[5])
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasCox(path_type="seq")
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasCox(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Cox, self).__init__(
#             algorithm_type="L0L2", model_type="Cox", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.data_type = 3


# @fix_docs
# class GroupPdasLm(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)   # fix seed to get the same result
#     >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
#     >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
#     >>> noise = np.random.normal(0, 1, 100)
#     >>> y = np.matmul(x, beta) + noise
#     >>> model = GroupPdasLm(path_type="seq", support_size=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = GroupPdasLm(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasLm(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#         """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(GroupPdasLm, self).__init__(
#             algorithm_type="GroupPdas", model_type="Lm", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.data_type = 1


# @fix_docs
# class GroupPdasLogistic(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
#     >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
#     >>> xbeta = np.matmul(x, beta)
#     >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
#     >>> y = np.random.binomial(1, p)
#     >>> model = GroupPdasLogistic(path_type="seq", support_size=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = GroupPdasLogistic(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasLogistic(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(GroupPdasLogistic, self).__init__(
#             algorithm_type="GroupPdas", model_type="Logistic", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau
#         )
#         self.data_type = 2


# @fix_docs
# class GroupPdasPoisson(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
#     >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
#     >>> lam = np.exp(np.matmul(x, beta))
#     >>> y = np.random.poisson(lam=lam)
#     >>> model = GroupPdasPoisson(path_type="seq", support_size=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = GroupPdasPoisson(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasPoisson(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(GroupPdasPoisson, self).__init__(
#             algorithm_type="GroupPdas", model_type="Poisson", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.data_type = 2


# @fix_docs
# class GroupPdasCox(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> data = gen_data(100, 200, family="cox", k=5, rho=0, sigma=1, c=10)
#     >>> model = GroupPdasCox(path_type="seq", support_size=[5])
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = GroupPdasCox(path_type="seq")
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasCox(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1
#                  ):
#         super(GroupPdasCox, self).__init__(
#             algorithm_type="GroupPdas", model_type="Cox", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path)
#         self.data_type = 3
