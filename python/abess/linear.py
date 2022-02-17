import warnings
import numpy as np
from .metrics import concordance_index_censored
from .bess_base import bess_base


def fix_docs(cls):
    # This function is to inherit the docstring from base class
    # and avoid unnecessary duplications on description.
    index = cls.__doc__.find("Examples\n    --------\n")
    if index != -1:
        cls.__doc__ = cls.__doc__[:index] + \
            cls.__bases__[0].__doc__ + cls.__doc__[index:]
    return cls


@ fix_docs
class LogisticRegression(bess_base):
    """
    Adaptive Best-Subset Selection (ABESS) algorithm for logistic regression.

    Parameters
    ----------
    splicing_type: {0, 1}, optional
        The type of splicing in `fit()` (in Algorithm.h).
        "0" for decreasing by half, "1" for decresing by one.
        Default: splicing_type = 0.
    important_search : int, optional
        The size of inactive set during updating active set when splicing.
        It should be a non-positive integer and if important_search=128, it would be set as
        the size of whole inactive set.
        Default: 0.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.linear import LogisticRegression
    >>> from abess.datasets import make_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_glm_data(n = 100, p = 50, k = 10, family = 'binomial')
    >>> model = LogisticRegression(support_size = [10])
    >>> model.fit(data.x, data.y)
    LogisticRegression(always_select=[], support_size=[10])
    >>> model.predict(data.x)[1:10]
    array([0., 0., 1., 1., 0., 1., 0., 1., 0.])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = LogisticRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    LogisticRegression(always_select=[])
    >>> model.predict(data.x)[1:10]
    array([0., 1., 0., 0., 1., 0., 1., 0., 1., 1.])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = LogisticRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    LogisticRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)[1:10]
    array([0., 0., 0., 1., 1., 0., 1., 0., 1., 0.])
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128,
                 ):
        super().__init__(
            algorithm_type="abess", model_type="Logistic", normalize_type=2, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size,
            always_select=always_select,
            primary_model_fit_max_iter=primary_model_fit_max_iter, primary_model_fit_epsilon=primary_model_fit_epsilon,
            approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )

    def predict_proba(self, X):
        """
        The predict_proba function is used to give the probabilities of new data begin assigned to different classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Test data.

        """
        X = self.new_data_check(X)

        intercept_ = np.ones(X.shape[0]) * self.intercept_
        xbeta = X.dot(self.coef_) + intercept_
        return np.exp(xbeta) / (1 + np.exp(xbeta))

    def predict(self, X):
        """
        For Logistic model,
        the predict function returns a \\code{dict} of \\code{pr} and \\code{y}, where \\code{pr} is the probability of response variable is 1 and \\code{y} is predicted to be 1 if \\code{pr} > 0.5 else \\code{y} is 0
        on given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Test data.

        """
        X = self.new_data_check(X)

        intercept_ = np.ones(X.shape[0]) * self.intercept_
        xbeta = X.dot(self.coef_) + intercept_
        y = np.zeros(xbeta.size)
        y[xbeta > 0] = 1
        return y

    def score(self, X, y):
        """
        Give new data, and it returns the entropy function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples, n_features), optional
            Test response (real class).
        """
        X, y = self.new_data_check(X, y)

        intercept_ = np.ones(X.shape[0]) * self.intercept_
        xbeta = X.dot(self.coef_) + intercept_
        xbeta[xbeta > 30] = 30
        xbeta[xbeta < -30] = -30
        pr = np.exp(xbeta) / (1 + np.exp(xbeta))
        return (y * np.log(pr) +
                (np.ones(X.shape[0]) - y) *
                np.log(np.ones(X.shape[0]) - pr)).sum()


@ fix_docs
class LinearRegression(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for linear regression.

    Parameters
    ----------
    splicing_type: {0, 1}, optional
        The type of splicing in `fit()` (in Algorithm.h).
        "0" for decreasing by half, "1" for decresing by one.
        Default: splicing_type = 0.
    important_search : int, optional
        The size of inactive set during updating active set when splicing.
        It should be a non-positive integer and if important_search=128, it would be set as
        the size of whole inactive set.
        Default: 0.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.linear import LinearRegression
    >>> from abess.datasets import make_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_glm_data(n = 100, p = 50, k = 10, family = 'gaussian')
    >>> model = LinearRegression(support_size = [10])
    >>> model.fit(data.x, data.y)
    LinearRegression(always_select=[], support_size=[10])
    >>> model.predict(data.x)[1:4]
    array([   1.42163813,  -43.23929886, -139.79509191,  141.45138403])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = LinearRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    LinearRegression(always_select=[])
    >>> model.predict(data.x)[1:4]
    array([   1.42163813,  -43.23929886, -139.79509191,  141.45138403])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = LinearRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    LinearRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)[1:4]
    array([   1.42163813,  -43.23929886, -139.79509191,  141.45138403])
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 thread=1, covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128,
                 # primary_model_fit_max_iter=10,
                 # primary_model_fit_epsilon=1e-8, approximate_Newton=False
                 ):
        super().__init__(
            algorithm_type="abess", model_type="Lm", normalize_type=1, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size,
            always_select=always_select,
            thread=thread, covariance_update=covariance_update,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )

    def predict(self, X):
        """
        For linear regression problem,
        the predict function returns a numpy array of the prediction of the mean
        on given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Test data.

        """
        X = self.new_data_check(X)

        intercept_ = np.ones(X.shape[0]) * self.intercept_
        return X.dot(self.coef_) + intercept_

    def score(self, X, y):
        """
        Give new data, and it returns the prediction error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples, n_features), optional
            Test response.
        """
        X, y = self.new_data_check(X, y)
        y_pred = self.predict(X)
        return -((y - y_pred) * (y - y_pred)).sum()


@ fix_docs
class CoxPHSurvivalAnalysis(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for COX proportional hazards model.

    Parameters
    ----------
    splicing_type: {0, 1}, optional
        The type of splicing in `fit()` (in Algorithm.h).
        "0" for decreasing by half, "1" for decresing by one.
        Default: splicing_type = 0.
    important_search : int, optional
        The size of inactive set during updating active set when splicing.
        It should be a non-positive integer and if important_search=128, it would be set as
        the size of whole inactive set.
        Default: 0.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.linear import CoxPHSurvivalAnalysis
    >>> from abess.datasets import make_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_glm_data(n = 100, p = 50, k = 10, family = 'cox')
    censoring rate:0.65
    >>> model = CoxPHSurvivalAnalysis(support_size = [10])
    >>> model.fit(data.x, data.y)
    CoxPHSurvivalAnalysis(always_select=[], support_size=[10])
    >>> model.predict(data.x)[1:4]
    array([1.08176927e+00, 6.37029117e-04, 3.64112556e-06, 4.09523406e+05])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = CoxPHSurvivalAnalysis(path_type = "seq")
    >>> model.fit(data.x, data.y)
    CoxPHSurvivalAnalysis(always_select=[], support_size=[10])
    >>> model.predict(data.x)[1:4]
    array([1.08176927e+00, 6.37029117e-04, 3.64112556e-06, 4.09523406e+05])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = CoxPHSurvivalAnalysis(path_type="gs")
    >>> model.fit(data.x, data.y)
    CoxPHSurvivalAnalysis(always_select=[], path_type='gs')
    >>> model.predict(data.x)[1:4]
    array([1.07629689e+00, 6.47263126e-04, 4.30660826e-06, 3.66389638e+05])
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        super().__init__(
            algorithm_type="abess", model_type="Cox", normalize_type=3, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size,
            always_select=always_select,
            primary_model_fit_max_iter=primary_model_fit_max_iter, primary_model_fit_epsilon=primary_model_fit_epsilon,
            approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )

    def predict(self, X):
        """
        For Cox model,
        the predict function returns the time-independent part of hazard function, i.e. :math:`\\exp(X\\beta)`,
        on given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Test data.

        """
        X = self.new_data_check(X)

        return np.exp(X.dot(self.coef_))

    def score(self, X, y):
        """
        Give new data, and it returns C-index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples, n_features), optional
            Test response.
        """
        X, y = self.new_data_check(X, y)
        risk_score = X.dot(self.coef_)
        y = np.array(y)
        result = concordance_index_censored(
            np.array(y[:, 1], np.bool_), y[:, 0], risk_score)
        return result[0]


@ fix_docs
class PoissonRegression(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for Poisson regression.

    Parameters
    ----------
    splicing_type: {0, 1}, optional
        The type of splicing in `fit()` (in Algorithm.h).
        "0" for decreasing by half, "1" for decresing by one.
        Default: splicing_type = 0.
    important_search : int, optional
        The size of inactive set during updating active set when splicing.
        It should be a non-positive integer and if important_search=128, it would be set as
        the size of whole inactive set.
        Default: 0.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.linear import PoissonRegression
    >>> from abess.datasets import make_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_glm_data(n = 100, p = 50, k = 10, family = 'poisson')
    >>> model = PoissonRegression(support_size = [10])
    >>> model.fit(data.x, data.y)
    PoissonRegression(always_select=[], support_size=[10])
    >>> model.predict(data.x)[1:4]
    array([1.06757251e+00, 8.92711312e-01, 5.64414159e-01, 1.35820866e+00])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = PoissonRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    PoissonRegression(always_select=[])
    >>> model.predict(data.x)[1:4]
    array([1.03373139e+00, 4.32229653e-01, 4.48811009e-01, 2.27170366e+00])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = PoissonRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    PoissonRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)[1:4]
    array([1.03373139e+00, 4.32229653e-01, 4.48811009e-01, 2.27170366e+00])
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        super().__init__(
            algorithm_type="abess", model_type="Poisson", normalize_type=2, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size,
            always_select=always_select,
            primary_model_fit_max_iter=primary_model_fit_max_iter, primary_model_fit_epsilon=primary_model_fit_epsilon,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )

    def predict(self, X):
        """
        For Poisson model,
        the predict function returns a numpy array of the prediction of the mean of response,
        on given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Test data.

        """
        X = self.new_data_check(X)

        intercept_ = np.ones(X.shape[0]) * self.intercept_
        xbeta_exp = np.exp(X.dot(self.coef_) + intercept_)
        return xbeta_exp

    def score(self, X, y):
        """
        Give new data, and it returns the prediction error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples, n_features), optional
            Test response.
        """
        X, y = self.new_data_check(X, y)

        intercept_ = np.ones(X.shape[0]) * self.intercept_
        eta = X.dot(self.coef_) + intercept_
        exp_eta = np.exp(eta)
        return (y * eta - exp_eta).sum()


@ fix_docs
class MultiTaskRegression(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for multitasklearning.

    Parameters
    ----------
    splicing_type: {0, 1}, optional
        The type of splicing in `fit()` (in Algorithm.h).
        "0" for decreasing by half, "1" for decresing by one.
        Default: splicing_type = 0.
    important_search : int, optional
        The size of inactive set during updating active set when splicing.
        It should be a non-positive integer and if important_search=128, it would be set as
        the size of whole inactive set.
        Default: 0.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.linear import MultipleLinearRegression
    >>> from abess.datasets import make_multivariate_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_multivariate_glm_data(n = 100, p = 50, k = 10, M = 3, family = 'multigaussian')
    >>> model = MultipleLinearRegression(support_size = [10])
    >>> model.fit(data.x, data.y)
    MultinomialRegression(always_select=[], support_size=[10])
    >>> model.predict(data.x)[1:5, ]
    array([[1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.]])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = MultipleLinearRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    MultinomialRegression(always_select=[])
    >>> model.predict(data.x)[1:5, ]
    array([[1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.]])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = MultipleLinearRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    MultinomialRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)[1:5, ]
    array([[1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.]])
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 thread=1, covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        super().__init__(
            algorithm_type="abess", model_type="Multigaussian", normalize_type=1, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size,
            always_select=always_select,
            thread=thread, covariance_update=covariance_update,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )

    def predict(self, X):
        """
        For Multigaussian model,
        the predict function returns a numpy matrix of the prediction of the mean of responses,
        on given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Test data.

        """
        X = self.new_data_check(X)

        intercept_ = np.repeat(
            self.intercept_[np.newaxis, ...], X.shape[0], axis=0)
        return X.dot(self.coef_) + intercept_

    def score(self, X, y):
        """
        Give new data, and it returns prediction error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples, n_features), optional
            Test response.
        """
        X, y = self.new_data_check(X, y)

        y_pred = self.predict(X)
        return -((y - y_pred) * (y - y_pred)).sum()


@ fix_docs
class MultinomialRegression(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for multiclassification problem.

    Parameters
    ----------
    splicing_type: {0, 1}, optional
        The type of splicing in `fit()` (in Algorithm.h).
        "0" for decreasing by half, "1" for decresing by one.
        Default: splicing_type = 0.
    important_search : int, optional
        The size of inactive set during updating active set when splicing.
        It should be a non-positive integer and if important_search=128, it would be set as
        the size of whole inactive set.
        Default: 0.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.linear import MultinomialRegression
    >>> from abess.datasets import make_multivariate_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_multivariate_glm_data(n = 100, p = 50, k = 10, M = 3, family = 'multinomial')
    >>> model = MultinomialRegression(support_size = [10])
    >>> model.fit(data.x, data.y)
    MultinomialRegression(always_select=[], support_size=[10])
    >>> model.predict(data.x)[1:5, ]
    array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.]])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = MultinomialRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    MultinomialRegression(always_select=[])
    >>> model.predict(data.x)[1:5, ]
    array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.]])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = MultinomialRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    MultinomialRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)[1:5, ]
    array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.]])
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        super().__init__(
            algorithm_type="abess", model_type="Multinomial", normalize_type=2, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size,
            always_select=always_select,
            primary_model_fit_max_iter=primary_model_fit_max_iter, primary_model_fit_epsilon=primary_model_fit_epsilon,
            approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )

    def predict_proba(self, X):
        """
        The predict_proba function is used to give the probabilities of new data begin assigned to different classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Test data.

        """
        X = self.new_data_check(X)

        intercept_ = np.repeat(
            self.intercept_[np.newaxis, ...], X.shape[0], axis=0)
        xbeta = X.dot(self.coef_) + intercept_
        eta = np.exp(xbeta)
        for i in range(X.shape[0]):
            pr = eta[i, :] / np.sum(eta[i, :])
        return pr

    def predict(self, X):
        """
        For Multinomial model,
        the predict function returns return the most possible class the given data may be.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Test data.

        """
        X = self.new_data_check(X)

        intercept_ = np.repeat(
            self.intercept_[np.newaxis, ...], X.shape[0], axis=0)
        xbeta = X.dot(self.coef_) + intercept_
        max_item = np.argmax(xbeta, axis=1)
        y_pred = np.zeros_like(xbeta)
        for i in range(X.shape[0]):
            y_pred[i, max_item[i]] = 1
        return y_pred

    def score(self, X, y):
        """
        Give new data, and it returns the entropy function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples, n_features), optional
            Test response (dummy variables of real class).
        """
        X, y = self.new_data_check(X, y)

        pr = self.predict_proba(X)
        return np.sum(y * np.log(pr))


@ fix_docs
class GammaRegression(bess_base):
    """
    Adaptive Best-Subset Selection(ABESS) algorithm for Gamma regression.

    Parameters
    ----------
    splicing_type: {0, 1}, optional
        The type of splicing in `fit()` (in Algorithm.h).
        "0" for decreasing by half, "1" for decresing by one.
        Default: splicing_type = 0.
    important_search : int, optional
        The size of inactive set during updating active set when splicing.
        It should be a non-positive integer and if important_search=128, it would be set as
        the size of whole inactive set.
        Default: 0.

    Examples
    --------
    >>> ### Sparsity known
    >>>
    >>> from abess.linear import GammaRegression
    >>> from abess.datasets import make_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_glm_data(n = 100, p = 50, k = 10, family = 'gamma')
    >>> model = GammaRegression(support_size = [10])
    >>> model.fit(data.x, data.y)
    GammaRegression(always_select=[], support_size=[10])
    >>> model.predict(data.x)[1:4]
    array([1.34510045e+22, 2.34908508e+30, 1.91570199e+21, 1.29563315e+25])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = GammaRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    GammaRegression(always_select=[])
    >>> model.predict(data.x)[1:4]
    array([7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = GammaRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    GammaRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)[1:4]
    array([7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19])
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq",
                 is_warm_start=True, support_size=None, alpha=None,
                 s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        super().__init__(
            algorithm_type="abess", model_type="Gamma", normalize_type=2,
            path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size,
            alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv,
            screening_size=screening_size,
            always_select=always_select,
            primary_model_fit_max_iter=primary_model_fit_max_iter,
            primary_model_fit_epsilon=primary_model_fit_epsilon,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )

    def predict(self, X):
        """
        For Gamma model,
        the predict function returns a numpy array of the prediction of the mean of response,
        on given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, p_features)
            Test data.

        """
        X = self.new_data_check(X)

        intercept_ = np.ones(X.shape[0]) * self.intercept_
        xbeta_exp = np.exp(X.dot(self.coef_) + intercept_)
        return xbeta_exp

    def score(self, X, y, weights=None):
        """
        Give new data, and it returns the prediction error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples, n_features), optional
            Test response.
        """
        if weights is None:
            X = np.array(X)
            weights = np.ones(X.shape[0])
        X, y, weights = self.new_data_check(X, y, weights)

        def deviance(y, y_pred):
            dev = 2 * (np.log(y_pred / y) + y / y_pred - 1)
            return np.sum(weights * dev)

        y_pred = self.predict(X)
        y_mean = np.average(y, weights=weights)
        dev = deviance(y, y_pred)
        dev_null = deviance(y, y_mean)
        return 1 - dev / dev_null


class abessLogistic(LogisticRegression):
    warning_msg = ("Class ``abessLogistic`` has been renamed to "
                   "``LogisticRegression``. "
                   "The former will be deprecated in version 0.6.0.")
    __doc__ = warning_msg + '\n' + LogisticRegression.__doc__

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq",
                 is_warm_start=True, support_size=None, alpha=None,
                 s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128,
                 ):
        warnings.warn(self.warning_msg, FutureWarning)
        super().__init__(
            path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size,
            alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv,
            screening_size=screening_size,
            always_select=always_select,
            primary_model_fit_max_iter=primary_model_fit_max_iter,
            primary_model_fit_epsilon=primary_model_fit_epsilon,
            approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )


class abessLm(LinearRegression):
    warning_msg = ("Class ``abessLm`` has been renamed to"
                   " ``LinearRegression``. "
                   "The former will be deprecated in version 0.6.0.")
    __doc__ = warning_msg + '\n' + LinearRegression.__doc__

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq",
                 is_warm_start=True, support_size=None, alpha=None,
                 s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 thread=1, covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128,
                 # primary_model_fit_max_iter=10,
                 # primary_model_fit_epsilon=1e-8, approximate_Newton=False
                 ):
        warnings.warn(self.warning_msg, FutureWarning)
        super().__init__(
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


class abessCox(CoxPHSurvivalAnalysis):
    warning_msg = ("Class ``abessCox`` has been renamed to "
                   "``CoxPHSurvivalAnalysis``. "
                   "The former will be deprecated in version 0.6.0.")
    __doc__ = warning_msg + '\n' + CoxPHSurvivalAnalysis.__doc__

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq",
                 is_warm_start=True, support_size=None, alpha=None,
                 s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        warnings.warn(self.warning_msg, FutureWarning)
        super().__init__(
            path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size,
            alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv,
            screening_size=screening_size,
            always_select=always_select,
            primary_model_fit_max_iter=primary_model_fit_max_iter,
            primary_model_fit_epsilon=primary_model_fit_epsilon,
            approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )


class abessPoisson(PoissonRegression):
    warning_msg = ("Class ``abessPoisson`` has been renamed to "
                   "``PoissonRegression``. "
                   "The former will be deprecated in version 0.6.0.")
    __doc__ = warning_msg + '\n' + PoissonRegression.__doc__

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq",
                 is_warm_start=True, support_size=None, alpha=None,
                 s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        warnings.warn(self.warning_msg, FutureWarning)
        super().__init__(
            path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size,
            alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv,
            screening_size=screening_size,
            always_select=always_select,
            primary_model_fit_max_iter=primary_model_fit_max_iter,
            primary_model_fit_epsilon=primary_model_fit_epsilon,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )


class abessMultigaussian(MultiTaskRegression):
    warning_msg = ("Class ``abessMultigaussian`` has been renamed to "
                   "``MultiTaskRegression``. "
                   "The former will be deprecated in version 0.6.0.")
    __doc__ = warning_msg + '\n' + MultiTaskRegression.__doc__

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq",
                 is_warm_start=True, support_size=None, alpha=None,
                 s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 thread=1, covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        warnings.warn(self.warning_msg, FutureWarning)
        super().__init__(
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


class abessMultinomial(MultinomialRegression):
    warning_msg = ("Class ``abessMultinomial`` has been renamed to "
                   "``MultinomialRegression``. "
                   "The former will be deprecated in version 0.6.0.")
    __doc__ = warning_msg + '\n' + MultinomialRegression.__doc__

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq",
                 is_warm_start=True, support_size=None,
                 alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        warnings.warn(self.warning_msg, FutureWarning)
        super().__init__(
            path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size,
            alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv,
            screening_size=screening_size,
            always_select=always_select,
            primary_model_fit_max_iter=primary_model_fit_max_iter,
            primary_model_fit_epsilon=primary_model_fit_epsilon,
            approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )


class abessGamma(GammaRegression):
    warning_msg = ("Class ``abessGamma`` has been renamed to "
                   "``GammaRegression``. "
                   "The former will be deprecated in version 0.6.0.")
    __doc__ = warning_msg + '\n' + GammaRegression.__doc__

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq",
                 is_warm_start=True, support_size=None, alpha=None,
                 s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1,
                 always_select=None,
                 primary_model_fit_max_iter=10,
                 primary_model_fit_epsilon=1e-8,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        warnings.warn(self.warning_msg, FutureWarning)
        super().__init__(
            path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size,
            alpha=alpha, s_min=s_min, s_max=s_max,
            ic_type=ic_type, ic_coef=ic_coef, cv=cv,
            screening_size=screening_size,
            always_select=always_select,
            primary_model_fit_max_iter=primary_model_fit_max_iter,
            primary_model_fit_epsilon=primary_model_fit_epsilon,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )
