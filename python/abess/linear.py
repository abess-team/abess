from abess.cabess import pywrap_abess
import numpy as np
import math
import types

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
                    print(name)
                    func.__doc__ = parfunc.__doc__ + func.__doc__
    return cls


class bess_base:
    """
    Parameters
    ----------
    max_iter : int, optional
        Max iteration time in PDAS.
        Default: max_iter = 20.
    is_warm_start : bool, optional
        When search the best sparsity,whether use the last parameter as the initial parameter for the next search.
        Default:is_warm_start = False.
    path_type : {"seq", "pgs"}
        The method we use to search the sparsity。
    sequence : array_like, optional
        The  sparsity list for searching. If choose path_type = "seq", we prefer you to give the sequence.If not
        given, we will search all the sparsity([1,2,...,p],p=min(X.shape[0], X.shape[1])).
        Default: sequence = None.
    s_min : int, optional
        The lower bound of golden-section-search for sparsity searching.If not given, we will set s_min = 1.
        Default: s_min = None.
    s_max : int, optional
        The higher bound of golden-section-search for sparsity searching.If not given, we will set s_max = p(p = X.shape[1]).
        Default: s_max = None.
    K_max : int, optional
        The search times of golden-section-search for sparsity searching.If not given, we will set K_max = int(log(p, 2/(math.sqrt(5) - 1))).
        Default: K_max = None.
    epsilon : double, optional
        The stop condition of golden-section-search for sparsity searching.
        Default: epsilon = 0.0001.
    ic_type : {'aic', 'bic', 'gic', 'ebic'}, optional
        The metric when choose the best sparsity.
        Input must be one of the set above. Default: ic_type = 'ebic'.
    is_cv : bool, optional
        Use the Cross-validation method to caculate the loss.
        Default: is_cv = False.
    K : int optional
        The folds number when Use the Cross-validation method to caculate the loss.
        Default: K = 5.

    Atrributes
    ----------
    beta : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the best subset selection problem.


    References
    ----------
    - Wen, C. , Zhang, A. , Quan, S. , & Wang, X. . (2017). [Bess: an r package for best subset selection in linear,
        logistic and coxph models]

    """

    def __init__(self, algorithm_type, model_type, path_type, max_iter=20, exchange_num=5, is_warm_start=True,
                 sequence=None, lambda_sequence=None, s_min=None, s_max=None, K_max=None, epsilon=0.0001, lambda_min=0, lambda_max=0,
                 ic_type="ebic", ic_coef=1.0,
                 is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
                 always_select=[], tau=0.,
                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-8,
                 early_stop=False, approximate_Newton=False):
        self.algorithm_type = algorithm_type
        self.model_type = model_type
        self.path_type = path_type
        self.algorithm_type_int = None
        self.model_type_int = None
        self.path_type_int = None
        self.max_iter = max_iter
        self.exchange_num = exchange_num
        self.is_warm_start = is_warm_start
        self.sequence = sequence
        self.lambda_sequence = lambda_sequence
        self.s_min = s_min
        self.s_max = s_max
        self.K_max = K_max
        self.epsilon = epsilon
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        # to do
        self.n_lambda = 100

        self.ic_type = ic_type
        self.ic_type_int = None
        self.ic_coef = ic_coef
        self.is_cv = is_cv
        self.K = K
        self.path_len = None
        self.p = None
        self.data_type = None
        self.is_screening = is_screening
        self.screening_size = screening_size
        self.powell_path = powell_path
        self.always_select = always_select
        self.tau = tau
        self.primary_model_fit_max_iter = primary_model_fit_max_iter
        self.primary_model_fit_epsilon = primary_model_fit_epsilon
        self.early_stop = early_stop
        self.approximate_Newton = approximate_Newton

        self.beta = None
        self.coef0 = None
        self.train_loss = None
        self.ic = None
        # self.nullloss = None
        # self.ic_sequence = None
        # self.bic_out = None
        # self.gic_out = None
        # self.A_out = None
        # self.l_out = None

        self._arg_check()

    def _arg_check(self):
        """
        Arguments check.

        """
        # print("arg_check")
        if self.algorithm_type == "Pdas":
            self.algorithm_type_int = 1
        elif self.algorithm_type == "GroupPdas":
            self.algorithm_type_int = 2
        elif self.algorithm_type == "L0L2":
            self.algorithm_type_int = 5
        elif self.algorithm_type == "abess":
            self.algorithm_type_int = 6
        else:
            raise ValueError("algorithm_type should not be " +
                             str(self.algorithm_type))

        if self.model_type == "Lm":
            self.model_type_int = 1
        elif self.model_type == "Logistic":
            self.model_type_int = 2
        elif self.model_type == "Poisson":
            self.model_type_int = 3
        elif self.model_type == "Cox":
            self.model_type_int = 4
        else:
            raise ValueError("model_type should not be " +
                             str(self.model_type))

        if self.path_type == "seq":
            # if self.sequence is None:
            #     raise ValueError(
            #         "When you choose path_type = sequence-search, the parameter \'sequence\' should be given.")
            self.path_type_int = 1

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
            self.path_type_int = 2
        else:
            raise ValueError("path_type should be \'seq\' or \'pgs\'")

        if self.ic_type == "aic":
            self.ic_type_int = 1
        elif self.ic_type == "bic":
            self.ic_type_int = 2
        elif self.ic_type == "gic":
            self.ic_type_int = 3
        elif self.ic_type == "ebic":
            self.ic_type_int = 4
        else:
            raise ValueError(
                "ic_type should be \"aic\", \"bic\", \"ebic\" or \"gic\"")

    def fit(self, X, y, is_weight=False, is_normal=True, weight=None, state=None, group=None):
        """
        The fit function is used to transfer the information of data and return the fit result.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y :  array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary. For linear regression problem, y should be a n time 1 numpy array with type \code{double}. For classification problem, \code{y} should be a $n \time 1$ numpy array with values \code{0} or \code{1}. For count data, \code{y} should be a $n \time 1$ numpy array of non-negative integer.
        is_weight : bool 
            whether to weight sample yourself. 
            Default: is$\_$weight = False.
        is_normal : bool, optional
            whether normalize the variables array before fitting the algorithm. 
            Default: is$\_$normal=True.
        weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If set is$\_$weight = True, weight should be given. 
            Default: \code{weight} = \code{numpy.ones(n)}.
        group : int, optional
            The group index for each variable. 
            Default: \code{group} = \code{numpy.ones(p)}.
        """

        self.p = X.shape[1]
        n = X.shape[0]
        p = X.shape[1]

        if self.algorithm_type_int == 2:
            if group is None:
                raise ValueError(
                    "When you choose GroupPdas algorithm, the group information should be given")
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
        else:
            g_index = range(p)

        if self.model_type_int == 4:
            X = X[y[:, 0].argsort()]
            y = y[y[:, 0].argsort()]
            # print(X[104, 252])
            # print(X[:5, :5])
            # print(y)
            y = y[:, 1].reshape(-1)

        if n != y.size:
            raise ValueError("X.shape(0) should be equal to y.size")

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
            state = np.ones(n)

        # path parameter
        if self.path_type_int == 1:
            if self.sequence is None:
                self.sequence = [(i+1)
                                 for i in range(min(p, int(n/np.log(n))))]

            if self.lambda_sequence is None:
                self.lambda_sequence = [0]

            self.s_min = 0
            self.s_max = 0
            self.K_max = 0
            self.lambda_min = 0
            self.lambda_max = 0
            self.path_len = int(len(self.sequence))
        else:
            self.sequence = [1]
            self.lambda_sequence = [0]
            if self.s_min is None:
                self.s_min = 1
            if self.s_max is None:
                self.s_max = p

            if self.K_max is None:
                self.K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))

            if self.lambda_min is None:
                self.lambda_min = 0
            if self.lambda_max is None:
                self.lambda_max = 0

            self.path_len = self.K_max + 2

        if self.is_screening:
            if self.screening_size:
                if(self.screening_size < max(self.sequence)):
                    raise ValueError(
                        "screening size should be more than max(sequence).")
            else:
                self.screening_size = max(p, int(n/np.log(n)))
        else:
            self.screening_size = 1

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
        # print("self.sequence: " + str(self.sequence))
        # print("self.lambda_sequence: " + str(self.lambda_sequence))

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

        # print(X[:10, :10])
        # print(y)

        result = pywrap_abess(X, y, self.data_type, weight,
                              is_normal,
                              self.algorithm_type_int, self.model_type_int, self.max_iter, self.exchange_num,
                              self.path_type_int, self.is_warm_start,
                              self.ic_type_int, self.ic_coef, self.is_cv, self.K,
                              g_index,
                              state,
                              self.sequence,
                              self.lambda_sequence,
                              self.s_min, self.s_max, self.K_max, self.epsilon,
                              self.lambda_min, self.lambda_max, self.n_lambda,
                              self.is_screening, self.screening_size, self.powell_path,
                              self.always_select, self.tau,
                              self.primary_model_fit_max_iter, self.primary_model_fit_epsilon,
                              self.early_stop, self.approximate_Newton,
                              p,
                              1, 1, 1, 1, 1, 1, p
                              )

        # print(2)
        self.beta = result[0]
        self.coef0 = result[1]
        self.train_loss = result[2]
        self.ic = result[3]
        # self.nullloss_out = result[3]
        # self.aic_sequence = result[4]
        # self.bic_sequence = result[5]
        # self.gic_sequence = result[6]
        # self.A_out = result[7]
        # self.l_out = result[8]

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
        if X.shape[1] != self.p:
            raise ValueError("X.shape[1] should be " + str(self.p))

        if self.model_type_int == 1:
            coef0 = np.ones(X.shape[0]) * self.coef0
            return np.dot(X, self.beta) + coef0
        elif self.model_type_int == 2:
            coef0 = np.ones(X.shape[0]) * self.coef0
            xbeta = np.dot(X, self.beta) + coef0

            y = np.zeros(xbeta.size)
            y[xbeta > 0] = 1

            xbeta[xbeta > 25] = 25
            xbeta[xbeta < -25] = -25
            xbeta_exp = np.exp(xbeta)
            pr = xbeta_exp / (xbeta_exp + 1)

            result = dict()
            result["Y"] = y
            result["pr"] = pr
            return result
        elif self.model_type_int == 3:
            coef0 = np.ones(X.shape[0]) * self.coef0
            xbeta_exp = np.exp(np.dot(X, self.beta) + coef0)
            result = dict()
            result["lam"] = xbeta_exp
            return result


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
#     >>> model = PdasLm(path_type="seq", sequence=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasLm(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLm(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     '''

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.):
#         super(PdasLm, self).__init__(
#             algorithm_type="Pdas", model_type="Lm", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = PdasLogistic(path_type="seq", sequence=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasLogistic(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLogistic(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(PdasLogistic, self).__init__(
#             algorithm_type="Pdas", model_type="Logistic", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = PdasPoisson(path_type="seq", sequence=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasPoisson(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasPoisson(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(PdasPoisson, self).__init__(
#             algorithm_type="Pdas", model_type="Poisson", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = PdasCox(path_type="seq", sequence=[5])
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasCox(path_type="seq")
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasCox(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(PdasCox, self).__init__(
#             algorithm_type="Pdas", model_type="Cox", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = PdasLm(path_type="seq", sequence=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasLm(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLm(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Lm, self).__init__(
#             algorithm_type="L0L2", model_type="Lm", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = PdasLm(path_type="seq", sequence=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasLm(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLm(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#         """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Logistic, self).__init__(
#             algorithm_type="L0L2", model_type="Logistic", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = PdasPoisson(path_type="seq", sequence=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasPoisson(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasPoisson(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Poisson, self).__init__(
#             algorithm_type="L0L2", model_type="Poisson", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = PdasCox(path_type="seq", sequence=[5])
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasCox(path_type="seq")
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasCox(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Cox, self).__init__(
#             algorithm_type="L0L2", model_type="Cox", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = GroupPdasLm(path_type="seq", sequence=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = GroupPdasLm(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasLm(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#         """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(GroupPdasLm, self).__init__(
#             algorithm_type="GroupPdas", model_type="Lm", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = GroupPdasLogistic(path_type="seq", sequence=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = GroupPdasLogistic(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasLogistic(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(GroupPdasLogistic, self).__init__(
#             algorithm_type="GroupPdas", model_type="Logistic", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = GroupPdasPoisson(path_type="seq", sequence=[5])
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = GroupPdasPoisson(path_type="seq")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasPoisson(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(GroupPdasPoisson, self).__init__(
#             algorithm_type="GroupPdas", model_type="Poisson", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
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
#     >>> model = GroupPdasCox(path_type="seq", sequence=[5])
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = GroupPdasCox(path_type="seq")
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasCox(path_type="pgs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
#                  K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1
#                  ):
#         super(GroupPdasCox, self).__init__(
#             algorithm_type="GroupPdas", model_type="Cox", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path)
#         self.data_type = 3


@fix_docs
class abessLogistic(bess_base):
    """
    Examples
    --------
    ### Sparsity known
    >>> from bess.linear import *
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
    >>> beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
    >>> xbeta = np.matmul(x, beta)
    >>> p = np.exp(xbeta)/(1+np.exp(xbeta))
    >>> y = np.random.binomial(1, p)
    >>> model = GroupPdasLogistic(path_type="seq", sequence=[5])
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    ### Sparsity unknown
    >>> # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
    >>> model = GroupPdasLogistic(path_type="seq")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)

    >>> # path_type="pgs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = GroupPdasLogistic(path_type="pgs")
    >>> model.fit(X=x, y=y)
    >>> model.predict(x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, sequence=None, lambda_sequence=None, s_min=None, s_max=None,
                 K_max=None, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", ic_coef=1.0, is_cv=False, K=5, is_screening=False, screening_size=None, powell_path=1,
                 always_select=[], tau=0.,
                 primary_model_fit_max_iter=30, primary_model_fit_epsilon=1e-8,
                 early_stop=False, approximate_Newton=False
                 ):
        super(abessLogistic, self).__init__(
            algorithm_type="abess", model_type="Logistic", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, sequence=sequence, lambda_sequence=lambda_sequence, s_min=s_min, s_max=s_max, K_max=K_max,
            epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, ic_coef=ic_coef, is_cv=is_cv, K=K, is_screening=is_screening, screening_size=screening_size, powell_path=powell_path,
            always_select=always_select, tau=tau,
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
            early_stop=early_stop, approximate_Newton=approximate_Newton
        )
        self.data_type = 2
