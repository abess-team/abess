import numbers
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator
from pybind_cabess import pywrap_GLM
from sklearn.utils.validation import check_X_y
from .utilities import categorical_to_dummy


class bess_base(BaseEstimator):
    r"""
    Parameters
    ----------
    max_iter : int, optional, default=20
        Maximum number of iterations taken for the
        splicing algorithm to converge.
        Due to the limitation of loss reduction, the splicing
        algorithm must be able to converge.
        The number of iterations is only to simplify the implementation.
    is_warm_start : bool, optional, default=True
        When tuning the optimal parameter combination,
        whether to use the last solution
        as a warm start to accelerate the iterative
        convergence of the splicing algorithm.
    path_type : {"seq", "gs"}, optional, default="seq"
        The method to be used to select the optimal support size.

        - For path_type = "seq", we solve the best subset selection
          problem for each size in support_size.
        - For path_type = "gs", we solve the best subset selection
          problem with support size ranged in (s_min, s_max), where the
          specific support size to be considered is
          determined by golden section.

    support_size : array-like, optional
        default=range(min(n, int(n/(log(log(n))log(p))))).
        An integer vector representing the alternative support sizes.
        Only used when path_type = "seq".
    s_min : int, optional, default=0
        The lower bound of golden-section-search for sparsity searching.
    s_max : int, optional, default=min(n, int(n/(log(log(n))log(p)))).
        The higher bound of golden-section-search for sparsity searching.
    ic_type : {'aic', 'bic', 'gic', 'ebic'}, optional, default='ebic'
        The type of criterion for choosing the support size.
    cv : int, optional, default=1
        The folds number when use the cross-validation method.

        - If cv=1, cross-validation would not be used.
        - If cv>1, support size will be chosen by CV's test loss,
          instead of IC.

    thread : int, optional, default=1
        Max number of multithreads.

        - If thread = 0, the maximum number of threads supported by
          the device will be used.

    screening_size : int, optional, default=-1
        The number of variables remaining after screening.
        It should be a non-negative number smaller than p,
        but larger than any value in support_size.

        - If screening_size=-1, screening will not be used.
        - If screening_size=0, screening_size will be set as
          :math:`\\min(p, int(n / (\\log(\\log(n))\\log(p))))`.

    always_select : array-like, optional, default=[]
        An array contains the indexes of variables
        we want to consider in the model.
    primary_model_fit_max_iter : int, optional, default=10
        The maximal number of iteration for primary_model_fit.
    primary_model_fit_epsilon : float, optional, default=1e-08
        The epsilon (threshold) of iteration for primary_model_fit.

    Attributes
    ----------
    coef_ : array-like, shape(p_features, ) or (p_features, M_responses)
        Estimated coefficients for the best subset selection problem.
    intercept_ : float or array-like, shape(M_responses,)
        The intercept in the model.
    ic_ : float
        If cv=1, it stores the score under chosen information criterion.
    test_loss_ : float
        If cv>1, it stores the test loss under cross-validation.
    train_loss_ : float
        The loss on training data.

    References
    ----------
    - Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang.
      A polynomial algorithm for best-subset selection problem.
      Proceedings of the National Academy of Sciences,
      117(52):33117-33123, 2020.
    """

    # attributes
    coef_ = None
    intercept_ = None
    ic_ = 0
    train_loss_ = 0
    test_loss_ = 0

    def __init__(
        self,
        algorithm_type,
        model_type,
        normalize_type,
        path_type,
        max_iter=20,
        exchange_num=5,
        is_warm_start=True,
        support_size=None,
        alpha=None,
        s_min=None,
        s_max=None,
        ic_type="ebic",
        ic_coef=1.0,
        cv=1,
        screening_size=-1,
        always_select=None,
        primary_model_fit_max_iter=10,
        primary_model_fit_epsilon=1e-8,
        approximate_Newton=False,
        thread=1,
        covariance_update=False,
        sparse_matrix=False,
        splicing_type=0,
        important_search=0,
        # lambda_min=None, lambda_max=None,
        # early_stop=False, n_lambda=100
    ):
        self.algorithm_type = algorithm_type
        self.model_type = model_type
        self.normalize_type = normalize_type
        self.path_type = path_type
        self.max_iter = max_iter
        self.exchange_num = exchange_num
        self.is_warm_start = is_warm_start
        self.support_size = support_size
        self.alpha = alpha
        self.n_features_in_: int
        self.s_min = s_min
        self.s_max = s_max
        # self.lambda_min = None
        # self.lambda_max = None
        # self.n_lambda = 100
        self.ic_type = ic_type
        self.ic_coef = ic_coef
        self.cv = cv
        self.screening_size = screening_size
        self.always_select = always_select
        self.primary_model_fit_max_iter = primary_model_fit_max_iter
        self.primary_model_fit_epsilon = primary_model_fit_epsilon
        # self.early_stop = False
        self.approximate_Newton = approximate_Newton
        self.thread = thread
        self.covariance_update = covariance_update
        self.sparse_matrix = sparse_matrix
        self.splicing_type = splicing_type
        self.important_search = important_search

    def fit(self,
            X=None,
            y=None,
            is_normal=True,
            weight=None,
            group=None,
            cv_fold_id=None,
            A_init=None):
        r"""
        The fit function is used to transfer
        the information of data and return the fit result.

        Parameters
        ----------
        X : array-like of shape(n_samples, p_features)
            Training data matrix. It should be a numpy array.
        y : array-like of shape(n_samples,) or (n_samples, M_responses)
            Training response values. It should be a numpy array.

            - For regression problem, the element of y should be float.
            - For classification problem,
              the element of y should be either 0 or 1.
              In multinomial regression,
              the p features are actually dummy variables.
            - For survival data, y should be a :math:`n \times 2` array,
              where the columns indicates "censoring" and "time",
              respectively.

        is_normal : bool, optional, default=True
            whether normalize the variables array
            before fitting the algorithm.
        weight : array-like, shape (n_samples,), optional, default=np.ones(n)
            Individual weights for each sample. Only used for is_weight=True.
        group : int, optional, default=np.ones(p)
            The group index for each variable.
        cv_fold_id: array-like, shape (n_samples,), optional, default=None
            An array indicates different folds in CV.
            Samples in the same fold should be given the same number.
        """

        # print("fit enter.")#///

        # Input check & init:
        if isinstance(X, (list, np.ndarray, np.matrix, coo_matrix)):
            if isinstance(X, coo_matrix):
                self.sparse_matrix = True

            # Check that X and y have correct shape
            X, y = check_X_y(X,
                             y,
                             accept_sparse=True,
                             multi_output=True,
                             y_numeric=True,
                             dtype='numeric')

            # Sort for Cox
            if self.model_type == "Cox":
                X = X[y[:, 0].argsort()]
                y = y[y[:, 0].argsort()]
                time = y[:, 0].reshape(-1)
                y = y[:, 1].reshape(-1)

            # Dummy y for Multinomial
            if self.model_type in (
                    "Multinomial", "Ordinal") and len(y.shape) == 1:
                y = categorical_to_dummy(y)

            # Init
            n = X.shape[0]
            p = X.shape[1]
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
        elif self.model_type == 'Gamma':
            model_type_int = 8
        elif self.model_type == 'Ordinal':
            model_type_int = 9
        else:
            raise ValueError("model_type should not be " +
                             str(self.model_type))

        # Path_type: seq, gs
        if self.path_type == "seq":
            path_type_int = 1
        elif self.path_type == "gs":
            path_type_int = 2
        else:
            raise ValueError("path_type should be \'seq\' or \'gs\'")

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
        if self.cv > n:
            raise ValueError("cv should be smaller than n.")

        # cv_fold_id
        if cv_fold_id is None:
            cv_fold_id = np.array([], dtype="int32")
        else:
            cv_fold_id = np.array(cv_fold_id, dtype="int32")
            if cv_fold_id.ndim > 1:
                raise ValueError("group should be an 1D array of integers.")
            if cv_fold_id.size != n:
                raise ValueError(
                    "The length of group should be equal to X.shape[0].")
            if len(set(cv_fold_id)) != self.cv:
                raise ValueError(
                    "The number of different masks should be equal to `cv`.")

        # A_init
        if A_init is None:
            A_init = np.array([], dtype="int32")
        else:
            A_init = np.array(A_init, dtype="int32")
            if A_init.ndim > 1:
                raise ValueError("The initial active set should be "
                                 "an 1D array of integers.")
            if (A_init.min() < 0 or A_init.max() >= p):
                raise ValueError("A_init contains wrong index.")

        # Group:
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

        # Weight:
        if weight is None:
            weight = np.ones(n)
        else:
            weight = np.array(weight)
            if weight.dtype not in ("int", "float"):
                raise ValueError("weight should be numeric.")
            if weight.ndim > 1:
                raise ValueError("weight should be a 1-D array.")
            if weight.size != n:
                raise ValueError("X.shape[0] should be equal to weight.size")

        # Path parameters
        if path_type_int == 1:  # seq
            if self.support_size is None:
                if (n == 1 or p == 1):
                    support_sizes = [0, 1]
                else:
                    support_sizes = list(
                        range(
                            0,
                            max(
                                min(p,
                                    int(n / (np.log(np.log(n)) * np.log(p)))),
                                1)))
            else:
                if isinstance(self.support_size,
                              (numbers.Real, numbers.Integral)):
                    support_sizes = np.empty(1, dtype=int)
                    support_sizes[0] = self.support_size
                elif (np.any(np.array(self.support_size) > p)
                      or np.any(np.array(self.support_size) < 0)):
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
            new_lambda_min = 0
            new_lambda_max = 0

        elif path_type_int == 2:  # gs
            new_s_min = 0 \
                if self.s_min is None else self.s_min
            new_s_max = min(p, int(n / (np.log(np.log(n)) * np.log(p)))) \
                if self.s_max is None else self.s_max
            new_lambda_min = 0  # \
            # if self.lambda_min is None else self.lambda_min
            new_lambda_max = 0  # \
            # if self.lambda_max is None else self.lambda_max

            if new_s_max < new_s_min:
                raise ValueError("s_max should be larger than s_min")
            # if new_lambda_max < new_lambda_min:
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
        #     print("[Warning]  exchange_num may be larger than sparsity, "
        #           "and it would be set up to sparsity.")

        # screening
        if self.screening_size != -1:
            if self.screening_size == 0:
                self.screening_size = min(
                    p, int(n / (np.log(np.log(n)) * np.log(p))))
            elif self.screening_size > p:
                raise ValueError(
                    "screening size should be smaller than X.shape[1].")
            elif self.screening_size < max(support_sizes):
                raise ValueError(
                    "screening size should be more than max(support_size).")

        # Primary fit parameters
        if (not isinstance(self.primary_model_fit_max_iter, int)
                or self.primary_model_fit_max_iter <= 0):
            raise ValueError(
                "primary_model_fit_max_iter should be an positive integer.")
        if self.primary_model_fit_epsilon < 0:
            raise ValueError(
                "primary_model_fit_epsilon should be non-negative.")

        # Thread
        if (not isinstance(self.thread, int) or self.thread < 0):
            raise ValueError("thread should be positive number or 0"
                             " (maximum supported by your device).")

        # Splicing type
        if self.splicing_type not in (0, 1):
            raise ValueError("splicing type should be 0 or 1.")

        # Important_search
        if (not isinstance(self.important_search, int)
                or self.important_search < 0):
            raise ValueError(
                "important_search should be a non-negative number.")

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
        n_lambda = 100
        early_stop = False

        # wrap with cpp
        # print("wrap enter.")#///
        result = pywrap_GLM(
            X, y, weight, n, p, normalize, algorithm_type_int, model_type_int,
            self.max_iter, self.exchange_num, path_type_int,
            self.is_warm_start, ic_type_int, self.ic_coef, self.cv, g_index,
            support_sizes, alphas, cv_fold_id, new_s_min, new_s_max,
            new_lambda_min, new_lambda_max, n_lambda, self.screening_size,
            always_select_list, self.primary_model_fit_max_iter,
            self.primary_model_fit_epsilon, early_stop,
            self.approximate_Newton, self.thread, self.covariance_update,
            self.sparse_matrix, self.splicing_type, self.important_search,
            A_init)

        # print("linear fit end")
        # print(len(result))
        # print(result)
        self.coef_ = result[0].squeeze()
        self.intercept_ = result[1].squeeze()
        self.train_loss_ = result[2]
        self.test_loss_ = result[3]
        self.ic_ = result[4]

        if self.model_type == "Cox":
            self._baseline_model.fit(np.dot(X, self.coef_), y, time)
        if self.model_type == "Ordinal":
            self.coef_ = self.coef_[:, 0]

        return self
