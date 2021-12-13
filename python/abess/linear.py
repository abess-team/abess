
from abess.metrics import concordance_index_censored
from .bess_base import bess_base

import numpy as np
import types


def fix_docs(cls):
    # inherit the document from base class
    index = cls.__doc__.find("Examples\n    --------\n")
    if(index != -1):
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


@ fix_docs
class abessLogistic(bess_base):
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
    >>> from abess.linear import abessLogistic
    >>> from abess.datasets import make_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_glm_data(n = 100, p = 50, k = 10, family = 'binomial')
    >>> model = abessLogistic(support_size = [10])
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = abessLogistic(path_type = "seq")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    >>>
    >>> # path_type="gs", 
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = abessLogistic(path_type="gs")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1, 
                 always_select=[], 
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128,
                 ):
        super(abessLogistic, self).__init__(
            algorithm_type="abess", model_type="Logistic", normalize_type=2, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, 
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size, 
            always_select=always_select, 
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
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
        return np.exp(xbeta)/(1 + np.exp(xbeta))

    def predict(self, X):
        """
        For Logistic model, 
        the predict function returns a \code{dict} of \code{pr} and \code{y}, where \code{pr} is the probability of response variable is 1 and \code{y} is predicted to be 1 if \code{pr} > 0.5 else \code{y} is 0
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
        pr = np.exp(xbeta)/(1 + np.exp(xbeta))
        return (y * np.log(pr) + (np.ones(X.shape[0]) - y) * np.log(np.ones(X.shape[0]) - pr)).sum()


@ fix_docs
class abessLm(bess_base):
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
    >>> from abess.linear import abessLm
    >>> from abess.datasets import make_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_glm_data(n = 100, p = 50, k = 10, family = 'gaussian')
    >>> model = abessLm(support_size = [10])
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = abessLm(path_type = "seq")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    >>>
    >>> # path_type="gs", 
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = abessLm(path_type="gs")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1, 
                 always_select=[], 
                 thread=1, covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128,
                 # primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8, approximate_Newton=False
                 ):
        super(abessLm, self).__init__(
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
        return -((y - y_pred)*(y - y_pred)).sum()


@ fix_docs
class abessCox(bess_base):
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
    >>> from abess.linear import abessCox
    >>> from abess.datasets import make_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_glm_data(n = 100, p = 50, k = 10, family = 'cox')
    >>> model = abessCox(support_size = [10])
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = abessCox(path_type = "seq")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    >>>
    >>> # path_type="gs", 
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = abessCox(path_type="gs")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1, 
                 always_select=[], 
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        super(abessCox, self).__init__(
            algorithm_type="abess", model_type="Cox", normalize_type=3, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, 
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size, 
            always_select=always_select, 
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
            approximate_Newton=approximate_Newton,
            thread=thread,
            sparse_matrix=sparse_matrix,
            splicing_type=splicing_type,
            important_search=important_search
        )

    def predict(self, X):
        """
        For Cox model, 
        the predict function returns the time-independent part of hazard function, i.e. :math:`\exp(X\\beta)`, 
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
class abessPoisson(bess_base):
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
    >>> from abess.linear import abessPoisson
    >>> from abess.datasets import make_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_glm_data(n = 100, p = 50, k = 10, family = 'poisson')
    >>> model = abessPoisson(support_size = [10])
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = abessPoisson(path_type = "seq")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    >>>
    >>> # path_type="gs", 
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = abessPoisson(path_type="gs")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1, 
                 always_select=[], 
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        super(abessPoisson, self).__init__(
            algorithm_type="abess", model_type="Poisson", normalize_type=2, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, 
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size, 
            always_select=always_select, 
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
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
class abessMultigaussian(bess_base):
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
    >>> from abess.linear import abessMultigaussian
    >>> from abess.datasets import make_multivariate_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_multivariate_glm_data(n = 100, p = 50, k = 10, M = 3, family = 'multigaussian')
    >>> model = abessMultigaussian(support_size = [10])
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = abessMultigaussian(path_type = "seq")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    >>>
    >>> # path_type="gs", 
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = abessMultigaussian(path_type="gs")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1, 
                 always_select=[], 
                 thread=1, covariance_update=False,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        super(abessMultigaussian, self).__init__(
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
        return -((y - y_pred)*(y - y_pred)).sum()


@ fix_docs
class abessMultinomial(bess_base):
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
    >>> from abess.linear import abessMultinomial
    >>> from abess.datasets import make_multivariate_glm_data
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> data = make_multivariate_glm_data(n = 100, p = 50, k = 10, M = 3, family = 'multinomial')
    >>> model = abessMultinomial(support_size = [10])
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = abessMultinomial(path_type = "seq")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    >>>
    >>> # path_type="gs", 
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = abessMultinomial(path_type="gs")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1, 
                 always_select=[], 
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 approximate_Newton=False,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        super(abessMultinomial, self).__init__(
            algorithm_type="abess", model_type="Multinomial", normalize_type=2, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, 
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size, 
            always_select=always_select, 
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
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
        max_item = np.argmax(xbeta, axis = 1)
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
class abessGamma(bess_base):
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
    >>> from abess.linear import abessGamma
    >>> import numpy as np
    >>> model = abessGamma(support_size = [10])
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = abessGamma(path_type = "seq")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    >>>
    >>> # path_type="gs", 
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = abessGamma(path_type="gs")
    >>> model.fit(data.x, data.y)
    >>> model.predict(data.x)
    """

    def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
                 ic_type="ebic", ic_coef=1.0, cv=1, screening_size=-1, 
                 always_select=[], 
                 primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-8,
                 thread=1,
                 sparse_matrix=False,
                 splicing_type=0,
                 important_search=128
                 ):
        super(abessGamma, self).__init__(
            algorithm_type="abess", model_type="Gamma", normalize_type=2, path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, 
            ic_type=ic_type, ic_coef=ic_coef, cv=cv, screening_size=screening_size, 
            always_select=always_select, 
            primary_model_fit_max_iter=primary_model_fit_max_iter,  primary_model_fit_epsilon=primary_model_fit_epsilon,
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
        if (weights == None):
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

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLm(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     '''

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.):
#         super(PdasLm, self).__init__(
#             algorithm_type="Pdas", model_type="Lm", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.normalize_type = 1


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

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLogistic(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(PdasLogistic, self).__init__(
#             algorithm_type="Pdas", model_type="Logistic", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.normalize_type = 2


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

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasPoisson(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(PdasPoisson, self).__init__(
#             algorithm_type="Pdas", model_type="Poisson", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau
#         )
#         self.normalize_type = 2


# @fix_docs
# class PdasCox(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> data = make_glm_data(100, 200, family="cox", cv=1, rho=0, sigma=1, c=10)
#     >>> model = PdasCox(path_type="seq", support_size=[5])
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasCox(path_type="seq")
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasCox(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(PdasCox, self).__init__(
#             algorithm_type="Pdas", model_type="Cox", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.normalize_type = 3


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

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLm(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Lm, self).__init__(
#             algorithm_type="L0L2", model_type="Lm", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau
#         )
#         self.normalize_type = 1


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

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasLm(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#         """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Logistic, self).__init__(
#             algorithm_type="L0L2", model_type="Logistic", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.normalize_type = 2


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

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasPoisson(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Poisson, self).__init__(
#             algorithm_type="L0L2", model_type="Poisson", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau
#         )
#         self.normalize_type = 2


# @fix_docs
# class L0L2Cox(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> data = make_glm_data(100, 200, family="cox", cv=1, rho=0, sigma=1, c=10)
#     >>> model = PdasCox(path_type="seq", support_size=[5])
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = PdasCox(path_type="seq")
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = PdasCox(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(L0L2Cox, self).__init__(
#             algorithm_type="L0L2", model_type="Cox", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.normalize_type = 3


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

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasLm(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#         """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(GroupPdasLm, self).__init__(
#             algorithm_type="GroupPdas", model_type="Lm", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.normalize_type = 1


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

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasLogistic(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(GroupPdasLogistic, self).__init__(
#             algorithm_type="GroupPdas", model_type="Logistic", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau
#         )
#         self.normalize_type = 2


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

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasPoisson(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1,
#                  always_select=[], tau=0.
#                  ):
#         super(GroupPdasPoisson, self).__init__(
#             algorithm_type="GroupPdas", model_type="Poisson", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path,
#             always_select=always_select, tau=tau)
#         self.normalize_type = 2


# @fix_docs
# class GroupPdasCox(bess_base):
#     """
#     Examples
#     --------
#     ### Sparsity known
#     >>> from bess.linear import *
#     >>> import numpy as np
#     >>> np.random.seed(12345)
#     >>> data = make_glm_data(100, 200, family="cox", cv=1, rho=0, sigma=1, c=10)
#     >>> model = GroupPdasCox(path_type="seq", support_size=[5])
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     ### Sparsity unknown
#     >>> # path_type="seq", Default:support_size=[1,2,...,min(x.shape[0], x.shape[1])]
#     >>> model = GroupPdasCox(path_type="seq")
#     >>> model.fit(data.x, data.y, is_normal=True)
#     >>> model.predict(data.x)

#     >>> # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
#     >>> model = GroupPdasCox(path_type="gs")
#     >>> model.fit(X=x, y=y)
#     >>> model.predict(x)
#     """

#     def __init__(self, max_iter=20, exchange_num=5, path_type="seq", is_warm_start=True, support_size=None, alpha=None, s_min=None, s_max=None,
#                  K_max=1, epsilon=0.0001, lambda_min=None, lambda_max=None, ic_type="ebic", cv=1, screening_size=-1, powell_path=1
#                  ):
#         super(GroupPdasCox, self).__init__(
#             algorithm_type="GroupPdas", model_type="Cox", path_type=path_type, max_iter=max_iter, exchange_num=exchange_num,
#             is_warm_start=is_warm_start, support_size=support_size, alpha=alpha, s_min=s_min, s_max=s_max, K_max=K_max,
#             epsilon=epsilon, lambda_min=lambda_min, lambda_max=lambda_max, ic_type=ic_type, cv=cv, screening_size=screening_size, powell_path=powell_path)
#         self.normalize_type = 3
