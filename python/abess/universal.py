from sklearn.base import BaseEstimator
import numpy as np
from .pybind_cabess import pywrap_Universal
from .pybind_cabess import UniversalModel
from .utilities import check_positive_integer, check_non_negative_integer
from jax import jacfwd, jacrev
from jax import grad as jax_grad
import jax.numpy as jnp


class ConvexSparseSolver(BaseEstimator):
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
        default=range(min(n, int(n/(log(log(n))log(p)))))
        An integer vector representing the alternative support sizes.
        Used only when path_type = "seq".
    gs_lower_bound : int, optional, default=0
        The lower bound of golden-section-search for sparsity searching.
        Used only when path_type = "gs".
    gs_higher_bound : int, optional, default=min(n, int(n/(log(log(n))log(p))))
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
    regular_coef : float, optional, default=0.0
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
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        Here are wrong examples: [0,2,1,2](not incremental), [1,2,3,3](not start from 0), [0,2,2,3](there is a gap).
        The variables in the same group must be adjacent, and they will be selected together or not.
        Before use group, it's worth mentioning that the concept "a variable" means "a group of variables" in fact.
        For example, "support_size=[3]" means there will be 3 groups of variables selected rather than 3 variables,
        and "always_include=[0,3]" means the 0-th and 3-th groups must be selected.
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
    coef_ = None
    intercept_ = None
    ic_ = 0
    train_loss_ = 0
    test_loss_ = 0

    def __init__(
        self,
        model_size,
        intercept_size=0,
        sample_size=1,
        max_iter=20,
        max_exchange_num=5,
        splicing_type="halve",
        path_type="seq",
        support_size=None,
        gs_lower_bound=0,
        gs_higher_bound=None,
        ic_type="gic",
        ic_coef=1.0,
        cv=1,
        cv_fold_id=None,
        regular_coef=0.0,
        always_select=None,
        screening_size=-1,
        important_search=128,
        group=None,
        init_active_set=None,
        is_warm_start=True,
        thread=1,
    ):
        self.model = UniversalModel()
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

    def fit(self, data=None):
        r"""
        The fit function is used to transfer
        the information of data and return the fit result.

        Parameters
        ----------
        data : user-defined class
            Any class which is match to model which is also user-defined before fit, denoted as ExternData.
            It cantains all data that model should be known, like samples, responses, weight.
        """
        # data
        if data is None:
            data = self.data

        # model_size
        p = self.model_size
        check_positive_integer(p, "model_size")

        # sample_size
        n = self.sample_size
        check_positive_integer(n, "sample_size")

        # intercept_size
        m = self.intercept_size
        check_non_negative_integer(m, "intercept_size")

        # max_iter
        check_non_negative_integer(self.max_iter, "max_iter")

        # max_exchange_num
        check_positive_integer(self.max_exchange_num, "max_exchange_num")

        # path_type
        if self.path_type == "seq":
            path_type = 1
        elif self.path_type == "gs":
            path_type = 2
        else:
            raise ValueError("path_type should be 'seq' or 'gs'")

        # ic_type
        if self.ic_type == "aic":
            ic_type = 1
        elif self.ic_type == "bic":
            ic_type = 2
        elif self.ic_type == "gic":
            ic_type = 3
        elif self.ic_type == "ebic":
            ic_type = 4
        else:
            raise ValueError('ic_type should be "aic", "bic", "ebic" or "gic"')

        # cv
        check_positive_integer(self.cv, "cv")
        if self.cv > n:
            raise ValueError("cv should not be greater than sample_size")

        # group
        if self.group is None:
            group = np.array(range(p), dtype="int32")
            group_num = p  # len(np.unique(group))
        else:
            group = np.array(self.group)
            if group.ndim > 1:
                raise ValueError("Group should be an 1D array of integers.")
            if group.size != p:
                raise ValueError("The length of group should be equal to model_size.")
            group_num = len(np.unique(group))
            if group[0] != 0:
                raise ValueError("Group should start from 0.")
            if any(group[1:] - group[:-1] < 0):
                raise ValueError("Group should be an incremental integer array.")
            if not group_num == max(group) + 1:
                raise ValueError("There is a gap in group.")
            group = np.array(
                [np.where(group == i)[0][0] for i in range(group_num)], dtype="int32"
            )

        # support_size
        if self.path_type == "gs":
            support_size = np.array([0], dtype="int32")
        else:
            if self.support_size == None:
                if n == 1 or group_num == 1:
                    support_size = np.array([0, 1], dtype="int32")
                else:
                    support_size = np.array(
                        range(
                            max(
                                1,
                                min(
                                    group_num,
                                    int(n / np.log(np.log(n)) / np.log(group_num)),
                                ),
                            )
                        ),
                        dtype="int32",
                    )
            else:
                if isinstance(self.support_size, (int, float)):
                    support_size = np.array([self.support_size], dtype="int32")
                else:
                    support_size = np.array(self.support_size, dtype="int32")
                support_size = np.sort(np.unique(support_size))
                if support_size[0] < 0 or support_size[-1] > group_num:
                    raise ValueError(
                        "All support_size should be between 0 and model_size"
                    )

        # regular_coef
        if self.regular_coef == None:
            regular_coef = np.array([0.0], dtype=float)
        else:
            if isinstance(self.regular_coef, (int, float)):
                regular_coef = np.array([self.regular_coef], dtype=float)
            else:
                regular_coef = np.array(self.regular_coef, dtype=float)
            if any(regular_coef < 0.0):
                raise ValueError("regular_coef should be positive.")

        # gs_bound
        if self.path_type == "seq":
            gs_lower_bound = gs_higher_bound = 0
        else:
            if self.gs_lower_bound is None:
                gs_lower_bound = 0
            else:
                gs_lower_bound = self.gs_lower_bound
            if self.gs_higher_bound is None:
                gs_higher_bound = min(
                    group_num, int(n / (np.log(np.log(n)) * np.log(group_num)))
                )
            else:
                gs_higher_bound = self.gs_higher_bound
            if gs_lower_bound > gs_higher_bound:
                raise ValueError(
                    "gs_higher_bound should be larger than gs_lower_bound."
                )

        # screening_size
        if self.screening_size == -1:
            screening_size = -1
        elif self.screening_size == 0:
            screening_size = min(
                group_num,
                max(
                    max(support_size[-1], gs_higher_bound),
                    int(n / (np.log(np.log(n)) * np.log(group_num))),
                ),
            )
        else:
            screening_size = self.screening_size
            if screening_size > group_num or screening_size < max(
                support_size[-1], gs_higher_bound
            ):
                raise ValueError(
                    "screening_size should be between max(support_size) and model_size."
                )

        # always_select
        if self.always_select is None:
            always_select = np.array([], dtype="int32")
        else:
            always_select = np.sort(np.array(self.always_select, dtype="int32"))
            if len(always_select) > 0 and (
                always_select[0] < 0 or always_select[-1] >= group_num
            ):
                raise ValueError("always_select should be between 0 and model_size.")

        # thread
        check_non_negative_integer(self.thread, "thread")

        # splicing_type
        if self.splicing_type == "halve":
            splicing_type = 0
        elif self.splicing_type == "taper":
            splicing_type = 1
        else:
            raise ValueError('splicing_type should be "halve" or "taper".')

        # important_search
        check_non_negative_integer(self.important_search, "important_search")

        # cv_fold_id
        if self.cv_fold_id is None:
            cv_fold_id = np.array([], dtype="int32")
        else:
            cv_fold_id = np.array(cv_fold_id, dtype="int32")
            if cv_fold_id.ndim > 1:
                raise ValueError("group should be an 1D array of integers.")
            if cv_fold_id.size != n:
                raise ValueError("The length of group should be equal to X.shape[0].")
            if len(set(cv_fold_id)) != self.cv:
                raise ValueError(
                    "The number of different masks should be equal to `cv`."
                )

        # init_active_set
        if self.init_active_set is None:
            init_active_set = np.array([], dtype="int32")
        else:
            init_active_set = np.array(self.init_active_set, dtype="int32")
            if init_active_set.ndim > 1:
                raise ValueError(
                    "The initial active set should be " "an 1D array of integers."
                )
            if init_active_set.min() < 0 or init_active_set.max() >= p:
                raise ValueError("init_active_set contains wrong index.")

        result = pywrap_Universal(
            data,
            self.model,
            p,
            n,
            m,
            self.max_iter,
            self.max_exchange_num,
            path_type,
            self.is_warm_start,
            ic_type,
            self.ic_coef,
            self.cv,
            support_size,
            regular_coef,
            gs_lower_bound,
            gs_higher_bound,
            screening_size,
            group,
            always_select,
            self.thread,
            splicing_type,
            self.important_search,
            cv_fold_id,
            init_active_set,
        )

        self.coef_ = result[0]
        self.intercept_ = result[1].squeeze()
        self.train_loss_ = result[2]
        self.test_loss_ = result[3]
        self.ic_ = result[4]

    def set_model_autodiff(self, loss, gradient, hessian):
        r"""
        Register callback function:

        Parameters
        ----------
        func : function {'para': array-like, 'intercept': array-like, 'data': ExternData, 'return': float}
        """
        self.model.set_loss_of_model(loss)
        self.model.set_gradient_autodiff(gradient)
        self.model.set_hessian_autodiff(hessian)

    def set_model_jax(self, loss):
        r"""
        Register callback function: loss of model.

        Parameters
        ----------
        loss : function {'para': array-like, 'intercept': array-like, 'data': ExternData, 'return': float}
        """
        # the function for differential
        def func_(para_compute, intercept, para, ind, data):
            para_complete = para.at[ind].set(para_compute)
            return loss(para_complete, intercept, data)

        def grad_(para, intercept, data, compute_para_index):
            para_j = jnp.array(para)
            intercept_j = jnp.array(intercept)
            para_compute_j = jnp.array(para[compute_para_index])
            return np.array(
                jnp.append(
                    *jax_grad(func_, (1, 0))(
                        para_compute_j, intercept_j, para_j, compute_para_index, data
                    )
                )
            )

        def hessian_(para, intercept, data, compute_para_index):
            para_j = jnp.array(para)
            intercept_j = jnp.array(intercept)
            para_compute_j = jnp.array(para[compute_para_index])
            return np.array(
                jacfwd(jacrev(func_))(
                    para_compute_j, intercept_j, para_j, compute_para_index, data
                )
            )

        self.model.set_loss_of_model(loss)
        self.model.set_gradient_user_defined(grad_)
        self.model.set_hessian_user_defined(hessian_)

    def set_loss(self, func):
        r"""
        Register callback function: loss of model.

        Parameters
        ----------
        func : function {'para': array-like, 'intercept': array-like, 'data': ExternData, 'return': float}
        """
        self.model.set_loss_of_model(func)

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

    def set_data(self, data):
        r""" """
        self.data = data
