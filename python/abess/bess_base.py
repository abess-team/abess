import sys
import numbers
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.exceptions import DataConversionWarning
from .pybind_cabess import pywrap_GLM
from .utilities import categorical_to_dummy


class bess_base(BaseEstimator):
    r"""
    Parameters
    ----------
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
    group : int, optional, default=np.ones(p)
        The group index for each variable.
    alpha : float, optional, default=0
        Constant that multiples the L2 term in loss function, controlling
        regularization strength. It should be non-negative.

        - If alpha = 0, it indicates ordinary least square.

    fit_intercept : bool, optional, default=True
        Whether to consider intercept in the model. We assume that the data
        has been centered if fit_intercept=False.

    ic_type : {'aic', 'bic', 'gic', 'ebic', 'loss'}, optional, default='ebic'
        The type of criterion for choosing the support size if `cv=1`.
    ic_coef : float, optional, default=1.0
        Constant that controls the regularization strength
        on chosen information criterion.
    cv : int, optional, default=1
        The folds number when use the cross-validation method.

        - If cv=1, cross-validation would not be used.
        - If cv>1, support size will be chosen by CV's test loss,
          instead of IC.

    cv_score : {'test_loss', ...}, optional, default='test_loss'
        The score used on test data for CV.

        - All methods support {'test_loss'}.
        - LogisticRegression also supports {'roc_auc'}.
        - MultinomialRegression also supports {'roc_auc_ovo', 'roc_auc_ovr'},
          which indicate "One vs One/Rest" algorithm, respectively.

    thread : int, optional, default=1
        Max number of multithreads.

        - If thread = 0, the maximum number of threads supported by
          the device will be used.

    A_init : array-like, optional, default=None
        Initial active set before the first splicing.
    always_select : array-like, optional, default=None
        An array contains the indexes of variables
        we want to consider in the model. For group selection,
        it should be the indexes of groups (start from 0).

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

    screening_size : int, optional, default=-1
        The number of variables remaining after screening.
        It should be a non-negative number smaller than p,
        but larger than any value in support_size.

        - If screening_size=-1, screening will not be used.
        - If screening_size=0, screening_size will be set as
          :math:`\\min(p, int(n / (\\log(\\log(n))\\log(p))))`.

    primary_model_fit_max_iter : int, optional, default=10
        The maximal number of iteration for primary_model_fit.
    primary_model_fit_epsilon : float, optional, default=1e-08
        The epsilon (threshold) of iteration for primary_model_fit.

    Attributes
    ----------
    coef_ : array-like, shape(p_features, ) or (p_features, M_responses)
        Estimated coefficients for the best subset selection problem.
    intercept_ : float or array-like, shape(M_responses,)
        The intercept in the model when fit_intercept=True.
    train_loss_ : float
        The loss on training data.
    eval_loss_ : float

        - If cv=1, it stores the score under chosen information criterion.
        - If cv>1, it stores the test loss under cross-validation.

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
    train_loss_ = 0
    eval_loss_ = 0

    def __init__(
        self,
        algorithm_type,
        model_type,
        normalize_type,
        path_type="seq",
        support_size=None,
        s_min=None,
        s_max=None,
        group=None,
        alpha=None,
        fit_intercept=True,
        ic_type="ebic",
        ic_coef=1.0,
        cv=1,
        cv_score="test_loss",
        thread=1,
        A_init=None,
        always_select=None,
        max_iter=20,
        exchange_num=5,
        is_warm_start=True,
        splicing_type=0,
        important_search=0,
        screening_size=-1,
        primary_model_fit_max_iter=10,
        primary_model_fit_epsilon=1e-8,
        approximate_Newton=False,
        covariance_update=False,
        # lambda_min=None, lambda_max=None,
        # early_stop=False, n_lambda=100,
        baseline_model=None,
        _estimator_type=None
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
        self.fit_intercept = fit_intercept
        self.n_features_in_: int
        self.n_iter_: int
        self.s_min = s_min
        self.s_max = s_max
        self.A_init = A_init
        self.group = group
        # self.lambda_min = None
        # self.lambda_max = None
        # self.n_lambda = 100
        self.ic_type = ic_type
        self.ic_coef = ic_coef
        self.cv = cv
        self.cv_score = cv_score
        self.screening_size = screening_size
        self.always_select = always_select
        self.primary_model_fit_max_iter = primary_model_fit_max_iter
        self.primary_model_fit_epsilon = primary_model_fit_epsilon
        # self.early_stop = False
        self.approximate_Newton = approximate_Newton
        self.thread = thread
        self.covariance_update = covariance_update
        self.splicing_type = splicing_type
        self.important_search = important_search
        self.baseline_model = baseline_model
        self._estimator_type = _estimator_type
        self.classes_: np.ndarray

    def fit(self,
            X=None,
            y=None,
            is_normal=True,
            sample_weight=None,
            cv_fold_id=None,
            sparse_matrix=False,
            beta_low=None,
            beta_high=None):
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
        sample_weight : array-like, shape (n_samples,), optional
            Individual weights for each sample. Only used for is_weight=True.
            Default=np.ones(n).
        cv_fold_id : array-like, shape (n_samples,), optional, default=None
            An array indicates different folds in CV.
            Samples in the same fold should be given the same number.
        sparse_matrix : bool, optional, default=False
            Set as True to treat X as sparse matrix during fitting.
            It would be automatically set as True when X has the
            sparse matrix type defined in scipy.sparse.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X,
                         y,
                         accept_sparse=True,
                         multi_output=True,
                         #  y_numeric=True,
                         dtype='numeric')

        # Input check & init:
        if isinstance(X, (list, np.ndarray, np.matrix, pd.DataFrame,
                      coo_matrix, csr_matrix)):
            if isinstance(X, (coo_matrix, csr_matrix)):
                sparse_matrix = True

            # Sort for Cox
            if self.model_type == "Cox":
                X = X[y[:, 0].argsort()]
                y = y[y[:, 0].argsort()]
                time = y[:, 0].reshape(-1)
                y = y[:, 1].reshape(-1)

            # Dummy y & classes
            if self.model_type == "Logistic":
                y, self.classes_ = categorical_to_dummy(y.squeeze())
                if self.classes_.size > 2:
                    raise ValueError("Up to 2 classes can be given in y.")
                if self.classes_.size == 1:
                    y = np.zeros(X.shape[0])
                else:
                    y = y[:, 1]
            elif (self.model_type in ("Multinomial", "Ordinal")
                    and (len(y.shape) == 1 or y.shape[1] == 1)):
                y, self.classes_ = categorical_to_dummy(y.squeeze())
                if self.classes_.size == 1:
                    # add a useless label
                    y = np.hstack((np.zeros((X.shape[0], 1)), y))
                    self.classes_ = np.insert(self.classes_, 0, 0)

            # multi_output warning
            if self.model_type in (
                    'Lm', 'Logistic', 'Poisson', 'Gamma'):
                if len(y.shape) > 1:
                    warnings.warn(
                        "A column-vector y was passed "
                        "when a 1d array was expected",
                        DataConversionWarning)
                    y = y.reshape(-1)

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

        # cv
        if (not isinstance(self.cv, int) or self.cv <= 0):
            raise ValueError("cv should be an positive integer.")
        if self.cv > n:
            raise ValueError("cv should be smaller than n.")

        # Ic_type: aic, bic, gic, ebic
        # cv_score: test_loss, roc_auc
        if self.cv == 1:
            if self.ic_type == "loss":
                eval_type_int = 0
            elif self.ic_type == "aic":
                eval_type_int = 1
            elif self.ic_type == "bic":
                eval_type_int = 2
            elif self.ic_type == "gic":
                eval_type_int = 3
            elif self.ic_type == "ebic":
                eval_type_int = 4
            elif self.ic_type == "hic":
                eval_type_int = 5
            else:
                raise ValueError(
                    "ic_type should be \"aic\", \"bic\", \"ebic\","
                    " \"gic\" or \"hic\".")
        else:
            if self.cv_score == "test_loss":
                eval_type_int = 0
            elif self.cv_score == "roc_auc" and self.model_type == "Logistic":
                eval_type_int = 1
            elif (self.cv_score == "roc_auc_ovo" and
                  self.model_type == "Multinomial"):
                eval_type_int = 2
            elif (self.cv_score == "roc_auc_ovr" and
                  self.model_type == "Multinomial"):
                eval_type_int = 3
            else:
                raise ValueError(
                    "cv_score should be \"test_loss\", "
                    "\"roc_auc\"(for logistic), "
                    "\"roc_auc_ovo\"(for multinomial), or "
                    "\"roc_auc_ovr\"(for multinomial).")

        # cv_fold_id
        if cv_fold_id is None:
            cv_fold_id = np.array([], dtype="int32")
        else:
            cv_fold_id = np.array(cv_fold_id, dtype="int32")
            if cv_fold_id.ndim > 1:
                raise ValueError(
                    "cv_fold_id should be an 1D array of integers.")
            if cv_fold_id.size != n:
                raise ValueError(
                    "The length of cv_fold_id should be equal to X.shape[0].")
            if len(set(cv_fold_id)) != self.cv:
                raise ValueError(
                    "The number of different masks should be equal to `cv`.")

        # A_init
        if self.A_init is None:
            A_init_list = np.array([], dtype="int32")
        else:
            A_init_list = np.array(self.A_init, dtype="int32")
            if A_init_list.ndim > 1:
                raise ValueError("The initial active set should be "
                                 "an 1D array of integers.")
            if (A_init_list.min() < 0 or A_init_list.max() >= p):
                raise ValueError("A_init contains out-of-range index.")

        # Group:
        if self.group is None:
            g_index = list(range(p))
        else:
            g = np.array(self.group)
            if g.ndim > 1:
                raise ValueError("group should be an 1D array of integers.")
            if g.size != p:
                raise ValueError(
                    "The length of group should be equal to X.shape[1].")
            group_set = list(set(g))
            g.sort()
            g_index = []
            j = 0
            for i in group_set:
                while g[j] != i:
                    j += 1
                g_index.append(j)

        # sample_weight:
        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            sample_weight = np.array(sample_weight, dtype="float")
            if sample_weight.ndim > 1:
                raise ValueError("sample_weight should be a 1-D array.")
            if sample_weight.size != n:
                raise ValueError(
                    "X.shape[0] should be equal to sample_weight.size")

            useful_index = list()
            for i, w in enumerate(sample_weight):
                if w > 0:
                    useful_index.append(i)
            if len(useful_index) < n:
                X = X[useful_index, :]
                y = y[useful_index, :] if len(y.shape) > 1 else y[useful_index]
                sample_weight = sample_weight[useful_index]
                n = len(useful_index)

        # Path parameters
        if path_type_int == 1:  # seq
            if self.support_size is None:
                if (n == 1 or p == 1):
                    support_sizes = [0, 1]
                else:
                    support_sizes = list(
                        range(0, max(min(
                            p,
                            int(n / (np.log(np.log(n)) * np.log(p)))
                        ), 1)))
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
        if sparse_matrix:
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

        # beta range
        if beta_low is None:
            beta_low = -sys.float_info.max
        if beta_high is None:
            beta_high = sys.float_info.max
        if beta_low > beta_high:
            raise ValueError(
                "Please make sure beta_low <= beta_high.")

        # unused
        n_lambda = 100
        early_stop = False
        self.n_iter_ = self.max_iter

        # wrap with cpp
        # print("wrap enter.")#///
        if n == 1:
            # with only one sample, nothing to be estimated
            result = [np.zeros((p, M)), np.zeros(M), 0, 0, 0]
        else:
            result = pywrap_GLM(
                X, y, sample_weight, n, p, normalize, algorithm_type_int,
                model_type_int,
                self.max_iter, self.exchange_num, path_type_int,
                self.is_warm_start, eval_type_int, self.ic_coef, self.cv,
                g_index,
                support_sizes, alphas, cv_fold_id, new_s_min, new_s_max,
                new_lambda_min, new_lambda_max, n_lambda, self.screening_size,
                always_select_list, self.primary_model_fit_max_iter,
                self.primary_model_fit_epsilon, early_stop,
                self.approximate_Newton, self.thread, self.covariance_update,
                sparse_matrix, self.splicing_type, self.important_search,
                A_init_list, self.fit_intercept, beta_low, beta_high)

        self.coef_ = result[0].squeeze()
        self.intercept_ = result[1].squeeze()
        self.train_loss_ = result[2]
        # self.test_loss_ = result[3]
        # self.ic_ = result[4]
        self.eval_loss_ = result[3] if (self.cv > 1) else result[4]

        if self.model_type == "Cox":
            self.baseline_model.fit(np.dot(X, self.coef_), y, time)
        if self.model_type == "Ordinal" and self.coef_.ndim > 1:
            self.coef_ = self.coef_[:, 0]

        return self
