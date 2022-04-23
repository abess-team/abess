from modulefinder import Module
from pybind_cabess import pywrap_Universal
from .bess_base import bess_base
import numbers
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from .utilities import categorical_to_dummy

# just for debug
class abessUniversal(bess_base):
    def __init__(self, max_iter=20, exchange_num=5, path_type="seq",
                 is_warm_start=True, support_size=None,
                 alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 thread=1, covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128,
                 # primary_model_fit_max_iter=10,
                 # primary_model_fit_epsilon=1e-8,
                 # approximate_Newton=False
                 ):
        super().__init__(
            algorithm_type="abess", model_type="Lm", normalize_type=1,
            path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size,
            alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv,
            screening_size=screening_size,
            always_select=always_select,
            thread=thread, covariance_update=covariance_update,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )

    def fit(self,
            X=None,
            y=None,
            is_normal=True,
            weight=None,
            group=None,
            cv_fold_id=None,
            A_init=None):
        # Input check & init:
        if isinstance(X, (list, np.ndarray, np.matrix,
                      coo_matrix, csr_matrix)):
            if isinstance(X, (coo_matrix, csr_matrix)):
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
            if (self.model_type in ("Multinomial", "Ordinal")
                    and (len(y.shape) == 1 or y.shape[1] == 1)):
                y = categorical_to_dummy(y.squeeze())

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
            if not isinstance(X, (coo_matrix)):
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
        if n == 1:
            # with only one sample, nothing to be estimated
            result = [np.zeros((p, M)), np.zeros(M), 0, 0, 0]
        else:
            result = pywrap_Universal(
                X, y, weight, n, p, normalize, algorithm_type_int,
                model_type_int,
                self.max_iter, self.exchange_num, path_type_int,
                self.is_warm_start, ic_type_int, self.ic_coef, self.cv,
                g_index,
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
            self.baseline_model.fit(np.dot(X, self.coef_), y, time)
        if self.model_type == "Ordinal" and self.coef_.ndim > 1:
            self.coef_ = self.coef_[:, 0]

        return self

from sklearn.base import BaseEstimator
from pybind_cabess import pywrap_Universal
from pybind_cabess import UniversalModel as Model_cpp
class UniversalModel(BaseEstimator): 
    r"""
    Adaptive Best-Subset Selection(ABESS) algorithm for
    user defined model.
    
    Parameters
    ----------
    model_size : int
        The total number of variables which need be selected, denoted as p.
    intercept_size : int, optional, default=0
        The total number of variables which need not be selected.
        This is for the convenience of some models, like the intercept
        of linear regression.
    sample_size : int, optional, default=1
        sample size, denoted as n.
    max_iter : int, optional, default=20
        Maximum number of iterations taken for the
        splicing algorithm to converge.
        The limitation of loss reduction can guarantee the convergence.
        The number of iterations is only to simplify the implementation.
    max_exchange_num : int, optional, default=2
        Maximum exchange number in splicing.
    splicing_type : {"halve", "taper"}, optional, default="halve"
        The type of reduce the exchange number in each iteration
        from max_exchange_num.
        "halve" for decreasing by half, "taper" for decresing by one.
    path_type : {"seq", "gs"}, optional, default="seq"
        The method to be used to select the optimal support size.
        - For path_type = "seq", we solve the best subset selection
          problem for each size in support_size.
        - For path_type = "gs", we solve the best subset selection
          problem with support size ranged in gs_bound, where the
          specific support size to be considered is
          determined by golden section.
    support_size : array-like, optional
        default=range(min(n, int(n/(log(log(n))log(p))))) or range(p) if n==1
        An integer vector representing the alternative support sizes.
        Used only when path_type = "seq".
    gs_lower_bound : int, optional, default=0
        The lower bound of golden-section-search for sparsity searching.
        Used only when path_type = "gs".
    gs_higher_bound : int, optional, default=min(n, int(n/(log(log(n))log(p)))) or p if n==1
        The higher bound of golden-section-search for sparsity searching.
        Used only when path_type = "gs".
    cv : int, optional, default=1
        The folds number when use the cross-validation method.
        - If cv=1, cross-validation would not be used.
        - If cv>1, support size will be chosen by CV's test loss,
          instead of IC.
    cv_fold_id: array-like, shape (n_samples,), optional, default=None
        An array indicates different folds in CV.
        Samples in the same fold should be given the same number.
        The number of different masks should be equal to `cv`.
        Used only when cv > 1.
    ic_type : {'aic', 'bic', 'gic', 'ebic'}, optional, default='gic'
        The type of information criterion for choosing the support size.
        Used only when cv = 1.
    ic_coef : float, optional, default=1.0
        The coefficient of information criterion.
        Used only when cv = 1.
    regular_coef : array-like, optional, default=[0.0]
        L2 regularization coefficient for computational stability.
    always_select : array-like, optional, default=[]
        An array contains the indexes of variables which must be selected.
    screening_size : int, optional, default=-1
        The number of variables remaining after the screening before variables select.
        It should be a non-negative number smaller than p,
        but larger than any value in support_size.
        - If screening_size=-1, screening will not be used.
        - If screening_size=0, screening_size will be set as
          :math:`\\min(p, int(n / (\\log(\\log(n))\\log(p))))`.
    important_search : int, optional, default=128
        The number of important variables which need be splicing.
        If it's too large, it would greatly increase runtime.
    group : array-like, optional, default=range(p)
        The group index for each variable, and it must be an incremental integer array.
        A total group is seen as a variable.
    init_active_set : array-like, optional, default=[]
        The index of the variable in initial active set.
    is_warm_start : bool, optional, default=True
        When tuning the optimal parameter combination, whether to use the last solution
        as a warm start to accelerate the iterative convergence of the splicing algorithm.
    thread : int, optional, default=1
        Max number of multithreads.
        - If thread = 0, the maximum number of threads supported by
          the device will be used.
    Attributes
    ----------
    coef_ : array-like, shape(p_features, )
        Estimated coefficients for the best subset selection problem.
    intercept_ : array-like, shape(M_responses,)
        The intercept in the model.
    ic_ : float
        If cv=1, it stores the score under chosen information criterion.
    test_loss_ : float
        If cv>1, it stores the test loss under cross-validation.
    train_loss_ : float
        The loss on training data.
    regularization_: float
        The best L2 regularization coefficient. 
    Examples
    --------


    References
    ----------
    - Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang.
      A polynomial algorithm for best-subset selection problem.
      Proceedings of the National Academy of Sciences,
      117(52):33117-33123, 2020.
    """

    # attributes
    model = Model_cpp()
    coef_ = None
    intercept_ = None
    ic_ = 0
    train_loss_ = 0
    test_loss_ = 0
    regularization_ = 0

    def __init__(self, model_size, intercept_size=0, sample_size=1,
                max_iter=20, max_exchange_num=5, splicing_type="halve", 
                path_type="seq", support_size=None, gs_lower_bound=0, gs_higher_bound=None,
                ic_type="gic", ic_coef=1.0, cv=1, cv_fold_id=None, regular_coef=0.0,
                always_select=None, screening_size=-1, important_search=None,
                group=None, init_active_set=[], is_warm_start=True, thread=1):
        self.model_size = model_size
        self.intercept_size = intercept_size
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.max_exchange_num = max_exchange_num
        self.splicing_type = splicing_type
        self.path_type = path_type
        self.support_size = support_size
        self.gs_lower_bound = gs_lower_bound
        self.gs_higher_bound = gs_higher_bound
        self.ic_type = ic_type
        self.ic_coef = ic_coef
        self.cv = cv
        self.cv_fold_id = cv_fold_id
        self.regular_coef = regular_coef
        self.always_select = always_select
        self.screening_size = screening_size
        self.important_search = important_search
        self.group = group
        self.init_active_set = init_active_set
        self.is_warm_start = is_warm_start
        self.thread = thread

    def fit(self, data):
        r"""
        The fit function is used to transfer
        the information of data and return the fit result.

        Parameters
        ----------
        data : user-defined class
            Any class which is match to model which is also user-defined before fit, denoted as ExternData.  
            It cantains all data that model should be known, like samples, responses, weight.
        """
        
        result = pywrap_Universal(data, self.model, self.model_size, self.sample_size, self.intercept_size,
            self.max_iter, self.max_exchange_num, self.path_type, self.is_warm_start, self.ic_type, 
            self.ic_coef, self.cv, self.support_size, self.regular_coef, self.gs_lower_bound, self.gs_higher_bound,
            self.screening_size, self.group, self.always_select, self.thread, self.splicing_type, 
            self.important_search, self.cv_fold_id, self.init_active_set)
        
        self.coef_ = result[0]
        self.intercept_ = result[1].squeeze()
        self.train_loss_ = result[2]
        self.test_loss_ = result[3]
        self.ic_ = result[4]
        self.regularization_ = result[5]
    
    def set_loss(self, func):
        r"""
        Register callback function: loss of model.

        Parameters
        ----------
        func : function {'para': array-like, 'intercept': array-like, 'data': ExternData, 'return': float}
        """
        self.model.set_loss_of_model(func)
    
    def set_gradient_autodiff(self, func):
        r"""
        Register callback function: 

        Parameters
        ----------
        func : function {}
        """
        self.model.set_gradient_autodiff(func)

    def set_hessian_autodiff(self, func):
        r"""
        Register callback function: 

        Parameters
        ----------
        func : function {}
        """
        self.model.set_hessian_autodiff(func)

    def set_gradient(self, func):
        r"""
        Register callback function: 

        Parameters
        ----------
        func : function {}
        """
        self.model.set_gradient_user_defined(func)

    def set_hessian(self, func):
        r"""
        Register callback function: 

        Parameters
        ----------
        func : function {}
        """
        self.model.set_hessian_user_defined(func)

    def set_slice_by_sample(self, func):
        r"""
        Register callback function: 

        Parameters
        ----------
        func : function {}
        """
        self.model.set_slice_by_sample(func)

    def set_slice_by_para(self, func):
        r"""
        Register callback function: 

        Parameters
        ----------
        func : function {}
        """
        self.model.set_slice_by_para(func)

    def set_deleter(self, func):
        r"""
        Register callback function:

        Parameters
        ----------
        func : function {}
        """
        self.model.set_deleter(func)

    def unset_slice_by_sample(self, func):
        r"""
        Register callback function:

        Parameters
        ----------
        func : function {}
        """
        self.model.unset_slice_by_sample(func)

    def unset_slice_by_para(self, func):
        r"""
        Register callback function: 

        Parameters
        ----------
        func : function {}
        """
        self.model.unset_slice_by_para(func)

    def unset_deleter(self, func):
        r"""
        Register callback function: 

        Parameters
        ----------
        func : function {}
        """
        self.model.unset_deleter(func)