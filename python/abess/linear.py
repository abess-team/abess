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
    >>> model.predict(data.x)
    array([0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1.,
       1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0.,
       1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0.,
       0., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
       0., 1., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 0., 1., 1.,
       1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0.])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = LogisticRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    LogisticRegression(always_select=[])
    >>> model.predict(data.x)
    array([0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.,
       1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0.,
       0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0.,
       1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0.,
       0., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1.,
       1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = LogisticRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    LogisticRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)
    array([0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1.,
       1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0.,
       1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0.,
       0., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
       0., 1., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 0., 1., 1.,
       1., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0.])
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
    >>> model.predict(data.x)
    array([   1.42163813,  -43.23929886, -139.79509191,  141.45138403,
        129.6156406 , -533.58644666,  293.93616306,  -46.56937023,
        510.93923901, -187.14612048,  -17.80564527,  -31.59610972,
       -272.47332608,  513.60427185,  -46.811926  , -163.67365947,
        225.881731  ,   -4.38003106,  -42.05847211, -387.58031175,
          1.68296939, -224.54128384,   75.7214898 , -450.45138695,
       -109.10086774,  710.63101439,  -18.08617958, -338.96258389,
         55.11322   , -131.32862924,  169.22857081,  259.68420945,
       -116.48451148, -337.81863738,   22.12358185,   21.01490921,
        -80.16796959, -408.30849929, -115.11938337, -450.20183957,
       -272.03285116,  -78.04106913, -229.14760389,   88.96451949,
        202.03789227,  367.4960523 , -365.26175995,   67.49297407,
       -124.42112229,  484.60139948,  -76.55826781, -553.05715798,
        453.35349285, -488.69395392,    9.42305022,  -10.1588589 ,
        -40.77612885,  123.78210156,  -46.94566911,  229.31774513,
        194.0400254 ,  295.58604997,  317.66574947,  730.03244896,
        242.95121295,  371.11500952, -105.03215459,  -78.94160329,
       -177.9352061 ,  460.05163927,  -17.50182631,  -49.31425363,
        218.91591011,  340.68805324,  -70.37921985,   95.07419884,
        -27.25361885,  113.00807306,  185.02788349,  194.58480369,
        145.05404446, -395.06475378, -527.16966811,  245.23906691,
         -1.20892529,  562.1795358 ,  124.48121227,  -15.36875538,
        196.28045375, -184.79543678, -337.75764605, -677.52259258,
         21.32188449,   -7.79639489,  188.56780716, -207.90034417,
         44.45529748, -561.78857151,  -10.35051552,   -8.11807509])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = LinearRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    LinearRegression(always_select=[])
    >>> model.predict(data.x)
    array([   1.42163813,  -43.23929886, -139.79509191,  141.45138403,
        129.6156406 , -533.58644666,  293.93616306,  -46.56937023,
        510.93923901, -187.14612048,  -17.80564527,  -31.59610972,
       -272.47332608,  513.60427185,  -46.811926  , -163.67365947,
        225.881731  ,   -4.38003106,  -42.05847211, -387.58031175,
          1.68296939, -224.54128384,   75.7214898 , -450.45138695,
       -109.10086774,  710.63101439,  -18.08617958, -338.96258389,
         55.11322   , -131.32862924,  169.22857081,  259.68420945,
       -116.48451148, -337.81863738,   22.12358185,   21.01490921,
        -80.16796959, -408.30849929, -115.11938337, -450.20183957,
       -272.03285116,  -78.04106913, -229.14760389,   88.96451949,
        202.03789227,  367.4960523 , -365.26175995,   67.49297407,
       -124.42112229,  484.60139948,  -76.55826781, -553.05715798,
        453.35349285, -488.69395392,    9.42305022,  -10.1588589 ,
        -40.77612885,  123.78210156,  -46.94566911,  229.31774513,
        194.0400254 ,  295.58604997,  317.66574947,  730.03244896,
        242.95121295,  371.11500952, -105.03215459,  -78.94160329,
       -177.9352061 ,  460.05163927,  -17.50182631,  -49.31425363,
        218.91591011,  340.68805324,  -70.37921985,   95.07419884,
        -27.25361885,  113.00807306,  185.02788349,  194.58480369,
        145.05404446, -395.06475378, -527.16966811,  245.23906691,
         -1.20892529,  562.1795358 ,  124.48121227,  -15.36875538,
        196.28045375, -184.79543678, -337.75764605, -677.52259258,
         21.32188449,   -7.79639489,  188.56780716, -207.90034417,
         44.45529748, -561.78857151,  -10.35051552,   -8.11807509])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = LinearRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    LinearRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)
    array([   1.42163813,  -43.23929886, -139.79509191,  141.45138403,
        129.6156406 , -533.58644666,  293.93616306,  -46.56937023,
        510.93923901, -187.14612048,  -17.80564527,  -31.59610972,
       -272.47332608,  513.60427185,  -46.811926  , -163.67365947,
        225.881731  ,   -4.38003106,  -42.05847211, -387.58031175,
          1.68296939, -224.54128384,   75.7214898 , -450.45138695,
       -109.10086774,  710.63101439,  -18.08617958, -338.96258389,
         55.11322   , -131.32862924,  169.22857081,  259.68420945,
       -116.48451148, -337.81863738,   22.12358185,   21.01490921,
        -80.16796959, -408.30849929, -115.11938337, -450.20183957,
       -272.03285116,  -78.04106913, -229.14760389,   88.96451949,
        202.03789227,  367.4960523 , -365.26175995,   67.49297407,
       -124.42112229,  484.60139948,  -76.55826781, -553.05715798,
        453.35349285, -488.69395392,    9.42305022,  -10.1588589 ,
        -40.77612885,  123.78210156,  -46.94566911,  229.31774513,
        194.0400254 ,  295.58604997,  317.66574947,  730.03244896,
        242.95121295,  371.11500952, -105.03215459,  -78.94160329,
       -177.9352061 ,  460.05163927,  -17.50182631,  -49.31425363,
        218.91591011,  340.68805324,  -70.37921985,   95.07419884,
        -27.25361885,  113.00807306,  185.02788349,  194.58480369,
        145.05404446, -395.06475378, -527.16966811,  245.23906691,
         -1.20892529,  562.1795358 ,  124.48121227,  -15.36875538,
        196.28045375, -184.79543678, -337.75764605, -677.52259258,
         21.32188449,   -7.79639489,  188.56780716, -207.90034417,
         44.45529748, -561.78857151,  -10.35051552,   -8.11807509])
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
    >>> model.predict(data.x)
    array([1.08176927e+00, 6.37029117e-04, 3.64112556e-06, 4.09523406e+05,
       2.71090478e+04, 1.63659726e-17, 1.69558508e+10, 2.59250058e-03,
       4.52145252e+17, 7.14615107e-08, 5.82374001e-03, 3.74249801e-01,
       1.12983775e-08, 3.60562352e+19, 1.08636003e-03, 8.37809345e-07,
       1.43966483e+07, 4.54088621e-01, 5.64110327e-02, 8.23609934e-15,
       7.36918939e-01, 2.38087446e-09, 1.11898995e+02, 8.84803545e-16,
       2.17283174e-05, 7.47446656e+24, 1.84159552e-02, 9.74265912e-12,
       1.71705692e+02, 4.18843521e-06, 3.37568362e+04, 4.02740489e+09,
       2.96641486e-03, 7.55830499e-13, 2.61028112e+01, 1.31456331e+01,
       2.68793461e-03, 1.50575910e-13, 3.57594530e-04, 1.73288436e-14,
       4.59204970e-09, 5.36573918e-03, 2.69183775e-09, 2.58502824e+02,
       1.86705890e+06, 1.38260567e+12, 2.52575258e-12, 2.38097251e+01,
       9.31181662e-04, 1.77087135e+17, 3.97649787e-02, 1.09114549e-20,
       9.48512366e+16, 2.17377932e-19, 9.08183273e+00, 4.56295618e-01,
       1.69478064e-02, 1.02667855e+04, 1.13666164e-03, 1.31836045e+09,
       4.42091092e+07, 1.50074077e+10, 4.67269951e+11, 9.75366320e+25,
       6.97539616e+07, 1.09902665e+12, 5.43637923e-03, 1.23033460e-02,
       2.75655764e-07, 7.63023172e+15, 1.03808041e+01, 1.80535486e-02,
       3.12224049e+07, 1.39274839e+10, 8.33580050e-04, 1.74581194e+03,
       3.59459290e-02, 1.39892413e+05, 7.29131433e+06, 5.58525920e+07,
       1.00413905e+05, 2.87840934e-14, 9.69270679e-20, 1.19771995e+07,
       7.27829134e-01, 3.98550368e+18, 4.54080292e+03, 2.20216519e-01,
       6.15603104e+05, 2.11242310e-06, 8.08901039e-14, 6.13841323e-25,
       4.99696015e+00, 3.97140211e-01, 5.79700304e+06, 2.56921060e-07,
       8.07243685e+00, 2.35128758e-18, 1.13241571e+00, 4.91087312e-02])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = CoxPHSurvivalAnalysis(path_type = "seq")
    >>> model.fit(data.x, data.y)
    CoxPHSurvivalAnalysis(always_select=[], support_size=[10])
    >>> model.predict(data.x)
    array([1.08176927e+00, 6.37029117e-04, 3.64112556e-06, 4.09523406e+05,
       2.71090478e+04, 1.63659726e-17, 1.69558508e+10, 2.59250058e-03,
       4.52145252e+17, 7.14615107e-08, 5.82374001e-03, 3.74249801e-01,
       1.12983775e-08, 3.60562352e+19, 1.08636003e-03, 8.37809345e-07,
       1.43966483e+07, 4.54088621e-01, 5.64110327e-02, 8.23609934e-15,
       7.36918939e-01, 2.38087446e-09, 1.11898995e+02, 8.84803545e-16,
       2.17283174e-05, 7.47446656e+24, 1.84159552e-02, 9.74265912e-12,
       1.71705692e+02, 4.18843521e-06, 3.37568362e+04, 4.02740489e+09,
       2.96641486e-03, 7.55830499e-13, 2.61028112e+01, 1.31456331e+01,
       2.68793461e-03, 1.50575910e-13, 3.57594530e-04, 1.73288436e-14,
       4.59204970e-09, 5.36573918e-03, 2.69183775e-09, 2.58502824e+02,
       1.86705890e+06, 1.38260567e+12, 2.52575258e-12, 2.38097251e+01,
       9.31181662e-04, 1.77087135e+17, 3.97649787e-02, 1.09114549e-20,
       9.48512366e+16, 2.17377932e-19, 9.08183273e+00, 4.56295618e-01,
       1.69478064e-02, 1.02667855e+04, 1.13666164e-03, 1.31836045e+09,
       4.42091092e+07, 1.50074077e+10, 4.67269951e+11, 9.75366320e+25,
       6.97539616e+07, 1.09902665e+12, 5.43637923e-03, 1.23033460e-02,
       2.75655764e-07, 7.63023172e+15, 1.03808041e+01, 1.80535486e-02,
       3.12224049e+07, 1.39274839e+10, 8.33580050e-04, 1.74581194e+03,
       3.59459290e-02, 1.39892413e+05, 7.29131433e+06, 5.58525920e+07,
       1.00413905e+05, 2.87840934e-14, 9.69270679e-20, 1.19771995e+07,
       7.27829134e-01, 3.98550368e+18, 4.54080292e+03, 2.20216519e-01,
       6.15603104e+05, 2.11242310e-06, 8.08901039e-14, 6.13841323e-25,
       4.99696015e+00, 3.97140211e-01, 5.79700304e+06, 2.56921060e-07,
       8.07243685e+00, 2.35128758e-18, 1.13241571e+00, 4.91087312e-02])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = CoxPHSurvivalAnalysis(path_type="gs")
    >>> model.fit(data.x, data.y)
    CoxPHSurvivalAnalysis(always_select=[], path_type='gs')
    >>> model.predict(data.x)
    array([1.07629689e+00, 6.47263126e-04, 4.30660826e-06, 3.66389638e+05,
       2.39269411e+04, 2.46172081e-17, 1.26522793e+10, 2.75706653e-03,
       2.79898097e+17, 8.07647048e-08, 6.64584446e-03, 3.68668433e-01,
       1.33645417e-08, 2.24396923e+19, 1.11346695e-03, 1.08660141e-06,
       1.23553615e+07, 4.66873338e-01, 5.70939765e-02, 1.17110773e-14,
       8.01039282e-01, 2.98695301e-09, 1.12504792e+02, 1.18565799e-15,
       2.65320615e-05, 3.80685000e+24, 2.15146362e-02, 1.18555479e-11,
       1.60098589e+02, 4.65257823e-06, 3.06666016e+04, 3.33704427e+09,
       3.24580009e-03, 1.02393876e-12, 2.32317941e+01, 1.28872243e+01,
       3.03197531e-03, 2.20101072e-13, 3.82339264e-04, 2.16466492e-14,
       5.61899004e-09, 5.67165957e-03, 3.30344297e-09, 2.57291749e+02,
       1.57867826e+06, 1.01380133e+12, 3.45665517e-12, 2.27262251e+01,
       8.77561531e-04, 1.17058233e+17, 3.96296272e-02, 1.71172998e-20,
       6.70779516e+16, 3.37962174e-19, 9.32553609e+00, 4.49270038e-01,
       1.69378489e-02, 8.56447118e+03, 1.32151146e-03, 1.07240238e+09,
       3.50843414e+07, 1.29243065e+10, 3.53339084e+11, 5.08803685e+25,
       5.51765297e+07, 8.44116867e+11, 5.97762597e-03, 1.29236729e-02,
       3.30864742e-07, 4.91729141e+15, 1.00331144e+01, 1.75665690e-02,
       2.51917575e+07, 1.06580316e+10, 9.02799914e-04, 1.71301307e+03,
       3.88670761e-02, 1.24896906e+05, 6.33871443e+06, 4.86596392e+07,
       9.11563213e+04, 3.76352432e-14, 1.41959581e-19, 1.01128655e+07,
       7.08444330e-01, 2.59626290e+18, 4.03792160e+03, 2.24005700e-01,
       5.32729032e+05, 2.34613266e-06, 1.19580956e-13, 9.81842883e-25,
       4.94573346e+00, 4.40823004e-01, 4.86201494e+06, 3.08855760e-07,
       8.08762717e+00, 3.50671471e-18, 1.18728166e+00, 5.16292079e-02])
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
    >>> model.predict(data.x)
    array([1.06757251e+00, 8.92711312e-01, 5.64414159e-01, 1.35820866e+00,
       1.21067754e+00, 2.60579840e-02, 7.21920707e+00, 8.66650213e-01,
       2.04390814e+01, 1.52223750e-01, 5.52244616e-01, 8.44274406e-01,
       4.11383191e-01, 4.82160496e+01, 1.13486310e+00, 9.55649300e-02,
       3.52795251e+00, 9.39826467e-01, 6.95053714e-01, 2.52465619e-02,
       3.85984379e-01, 1.10056651e-01, 1.41947506e+00, 5.23677642e-02,
       2.60386293e-01, 1.40712391e+02, 4.18314916e-01, 2.22684581e-01,
       1.34159973e+00, 2.96683116e-01, 2.51179005e+00, 4.27217158e+00,
       3.51096802e-01, 6.76874457e-02, 1.71801681e+00, 2.36760822e-01,
       7.99067977e-01, 1.70684529e-02, 4.91933693e-01, 1.27756241e-01,
       1.80105426e-01, 5.77643033e-01, 6.74733210e-02, 6.56443669e-01,
       3.36651719e+00, 1.06306769e+01, 4.51628090e-02, 7.00566307e-01,
       8.16692442e-01, 2.68583315e+01, 1.13402304e+00, 1.12770043e-02,
       2.74794207e+01, 3.45698045e-02, 1.38230253e+00, 1.53951180e+00,
       6.48977559e-01, 2.76550904e+00, 2.47569553e-01, 6.26343724e+00,
       6.23555070e+00, 7.08400317e+00, 8.47046077e+00, 1.63395421e+02,
       5.85655133e+00, 8.21005912e+00, 2.06114565e-01, 7.45206199e-01,
       2.18045736e-01, 2.10267891e+01, 2.18878769e+00, 1.09159394e+00,
       2.08254841e+00, 4.69957945e+00, 8.08933356e-01, 8.02319206e-01,
       4.71648368e-01, 3.90947843e+00, 3.91293653e+00, 2.78457540e+00,
       2.86928734e+00, 1.14592547e-01, 4.94166290e-02, 5.69694954e+00,
       9.16453068e-01, 3.05395680e+01, 1.52141897e+00, 3.96718220e-01,
       7.00466655e+00, 3.59243715e-01, 2.47714148e-02, 7.71020419e-03,
       1.41975976e+00, 9.35929658e-01, 1.20318684e+00, 2.22668668e-01,
       5.06622343e-01, 1.55574885e-02, 7.48369574e-01, 7.87738301e-01])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = PoissonRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    PoissonRegression(always_select=[])
    >>> model.predict(data.x)
    array([1.03373139e+00, 4.32229653e-01, 4.48811009e-01, 2.27170366e+00,
       1.86601022e+00, 3.08769054e-02, 6.41303139e+00, 6.22785227e-01,
       2.15778556e+01, 1.96284431e-01, 6.08336296e-01, 9.28374246e-01,
       2.29776339e-01, 4.98795329e+01, 5.66066546e-01, 2.35766232e-01,
       3.65193318e+00, 1.07893449e+00, 7.87826640e-01, 4.13311568e-02,
       7.31133208e-01, 1.37624999e-01, 1.40035404e+00, 5.09464357e-02,
       3.76588688e-01, 1.34338934e+02, 3.56470427e-01, 1.14453114e-01,
       1.19652542e+00, 2.28660049e-01, 1.63920154e+00, 5.15613606e+00,
       5.84468466e-01, 6.70538694e-02, 1.29576702e+00, 7.76138570e-01,
       5.60079414e-01, 4.40447086e-02, 4.52277692e-01, 5.34594254e-02,
       1.76031377e-01, 5.79135268e-01, 1.19120738e-01, 1.31711342e+00,
       2.56642594e+00, 7.86294589e+00, 6.48905173e-02, 1.00960657e+00,
       4.72963666e-01, 3.14884400e+01, 8.57376085e-01, 1.19948342e-02,
       3.04890122e+01, 2.39887107e-02, 1.30382894e+00, 7.22759271e-01,
       5.96088233e-01, 2.14731831e+00, 3.58769952e-01, 6.35076942e+00,
       4.76839752e+00, 6.91141845e+00, 9.20647982e+00, 1.65479030e+02,
       4.60972542e+00, 8.64621022e+00, 4.23817473e-01, 7.38426624e-01,
       2.29332506e-01, 1.98119224e+01, 1.46816542e+00, 5.31082723e-01,
       2.75619973e+00, 7.40106997e+00, 5.49174370e-01, 1.12187065e+00,
       5.49705354e-01, 2.94912450e+00, 3.35133196e+00, 4.51110349e+00,
       2.27075543e+00, 7.92239838e-02, 2.46857554e-02, 3.39002085e+00,
       7.65764093e-01, 3.33618809e+01, 1.63537112e+00, 6.13841273e-01,
       4.69483068e+00, 2.86532536e-01, 4.41088227e-02, 5.13497393e-03,
       1.08329230e+00, 8.34202847e-01, 2.13311076e+00, 2.07639103e-01,
       1.01701396e+00, 2.20754026e-02, 6.93363285e-01, 8.43469475e-01])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = PoissonRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    PoissonRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)
    array([1.03373139e+00, 4.32229653e-01, 4.48811009e-01, 2.27170366e+00,
       1.86601022e+00, 3.08769054e-02, 6.41303139e+00, 6.22785227e-01,
       2.15778556e+01, 1.96284431e-01, 6.08336296e-01, 9.28374246e-01,
       2.29776339e-01, 4.98795329e+01, 5.66066546e-01, 2.35766232e-01,
       3.65193318e+00, 1.07893449e+00, 7.87826640e-01, 4.13311568e-02,
       7.31133208e-01, 1.37624999e-01, 1.40035404e+00, 5.09464357e-02,
       3.76588688e-01, 1.34338934e+02, 3.56470427e-01, 1.14453114e-01,
       1.19652542e+00, 2.28660049e-01, 1.63920154e+00, 5.15613606e+00,
       5.84468466e-01, 6.70538694e-02, 1.29576702e+00, 7.76138570e-01,
       5.60079414e-01, 4.40447086e-02, 4.52277692e-01, 5.34594254e-02,
       1.76031377e-01, 5.79135268e-01, 1.19120738e-01, 1.31711342e+00,
       2.56642594e+00, 7.86294589e+00, 6.48905173e-02, 1.00960657e+00,
       4.72963666e-01, 3.14884400e+01, 8.57376085e-01, 1.19948342e-02,
       3.04890122e+01, 2.39887107e-02, 1.30382894e+00, 7.22759271e-01,
       5.96088233e-01, 2.14731831e+00, 3.58769952e-01, 6.35076942e+00,
       4.76839752e+00, 6.91141845e+00, 9.20647982e+00, 1.65479030e+02,
       4.60972542e+00, 8.64621022e+00, 4.23817473e-01, 7.38426624e-01,
       2.29332506e-01, 1.98119224e+01, 1.46816542e+00, 5.31082723e-01,
       2.75619973e+00, 7.40106997e+00, 5.49174370e-01, 1.12187065e+00,
       5.49705354e-01, 2.94912450e+00, 3.35133196e+00, 4.51110349e+00,
       2.27075543e+00, 7.92239838e-02, 2.46857554e-02, 3.39002085e+00,
       7.65764093e-01, 3.33618809e+01, 1.63537112e+00, 6.13841273e-01,
       4.69483068e+00, 2.86532536e-01, 4.41088227e-02, 5.13497393e-03,
       1.08329230e+00, 8.34202847e-01, 2.13311076e+00, 2.07639103e-01,
       1.01701396e+00, 2.20754026e-02, 6.93363285e-01, 8.43469475e-01])
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
    >>> model.predict(data.x)
    array([[1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = MultipleLinearRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    MultinomialRegression(always_select=[])
    >>> model.predict(data.x)
    array([[1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = MultipleLinearRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    MultinomialRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)
    array([[1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
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
    >>> model.predict(data.x)
    array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = MultinomialRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    MultinomialRegression(always_select=[])
    >>> model.predict(data.x)
    array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = MultinomialRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    MultinomialRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)
    array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
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
    >>> model.predict(data.x)
    array([1.34510045e+22, 2.34908508e+30, 1.91570199e+21, 1.29563315e+25,
       1.08699999e+26, 5.43806194e+12, 4.10451139e+32, 2.30145242e+24,
       2.12461148e+30, 7.78957076e+16, 4.53698616e+22, 5.92053325e+22,
       2.93651132e+10, 7.26552965e+30, 1.00446382e+28, 3.51687438e+15,
       1.11401437e+30, 4.91233127e+21, 3.97013896e+21, 5.75101723e+15,
       1.92514871e+23, 1.70294578e+17, 6.87763311e+26, 4.16607112e+08,
       2.44307315e+23, 5.60562414e+37, 4.31853272e+25, 2.43958940e+15,
       2.60093733e+25, 2.70666135e+24, 6.33978601e+29, 1.66953315e+28,
       1.04539055e+19, 1.05253782e+16, 8.74003777e+21, 8.31448234e+21,
       3.60174883e+25, 5.49111075e+09, 2.71030474e+21, 2.82308414e+10,
       1.23253836e+17, 3.09721743e+21, 1.07274949e+21, 1.92064709e+25,
       7.93874723e+31, 4.90737856e+32, 4.36278242e+14, 1.61239080e+29,
       1.03592107e+15, 1.50250089e+34, 1.76999959e+16, 1.12033788e+12,
       1.22002468e+30, 3.76354787e+15, 3.50118259e+23, 5.28042895e+21,
       9.17295936e+23, 6.88259686e+24, 5.82393657e+25, 5.04183281e+26,
       1.16411269e+27, 5.59147039e+31, 4.08928287e+26, 4.79532141e+39,
       2.34463142e+25, 9.36923884e+34, 5.25932281e+17, 3.03572301e+19,
       3.81736852e+17, 3.82774922e+28, 8.51634859e+20, 1.73618348e+23,
       2.22825880e+25, 2.42085479e+32, 2.38691821e+22, 1.14793887e+32,
       6.05119804e+19, 3.72691367e+24, 1.33723311e+26, 2.84048021e+23,
       4.21249322e+24, 5.40794397e+15, 1.76621204e+05, 4.55979028e+29,
       1.08834809e+21, 1.24034844e+38, 6.55928104e+23, 1.10303854e+23,
       2.47608345e+29, 2.05614511e+17, 1.65803409e+13, 3.17999342e+05,
       1.77063114e+27, 7.62131997e+24, 8.93346538e+24, 4.83211401e+17,
       5.45595802e+26, 1.87265360e+07, 6.11339767e+21, 7.92350465e+24])

    >>> ### Sparsity unknown
    >>>
    >>> # path_type="seq",
    >>> # Default: support_size = list(range(0, max(min(p, int(n / (np.log(np.log(n)) * np.log(p)))), 1))).
    >>> model = GammaRegression(path_type = "seq")
    >>> model.fit(data.x, data.y)
    GammaRegression(always_select=[])
    >>> model.predict(data.x)
    array([7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19])
    >>>
    >>> # path_type="gs",
    >>> # Default: s_min=1, s_max=min(p, int(n / (np.log(np.log(n)) * np.log(p)))), K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
    >>> model = GammaRegression(path_type="gs")
    >>> model.fit(data.x, data.y)
    GammaRegression(always_select=[], path_type='gs')
    >>> model.predict(data.x)
    array([7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19,
       7.03065424e+19, 7.03065424e+19, 7.03065424e+19, 7.03065424e+19])
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
