import numpy as np


def sample(p, k):
    full = np.arange(p)
    select = sorted(np.random.choice(full, k, replace=False))
    return select


def sparse_beta_generator(p, Nonzero, k, M):
    Tbeta = np.zeros([p, M])
    beta_value = beta_generator(k, M)
    Tbeta[Nonzero, :] = beta_value
    return Tbeta


def beta_generator(k, M):
    # # strong_num <- 3
    # # moderate_num <- 7
    # # weak_num <- 5
    # # strong_num <- 10
    # # moderate_num <- 10
    # # weak_num <- 10
    strong_num = int(k * 0.3)
    moderate_num = int(k * 0.4)
    weak_num = k - strong_num - moderate_num
    # signal_num = strong_num + moderate_num + weak_num

    strong_signal = np.random.normal(
        0, 10, strong_num * M).reshape(strong_num, M)
    moderate_signal = np.random.normal(
        0, 5, moderate_num * M).reshape(moderate_num, M)
    weak_signal = np.random.normal(0, 2, weak_num * M).reshape(weak_num, M)
    beta_value = np.concatenate((strong_signal, moderate_signal, weak_signal))

    beta_value = beta_value[sample(k, k), :]

    # beta_value = np.random.normal(size=(k, M))
    return beta_value


class make_glm_data:
    r"""
    Generate a dataset with single response.

    Parameters
    ----------
    n: int
        The number of observations.
    p: int
        The number of predictors of interest.
    k: int
        The number of nonzero coefficients in the
        underlying regression model.
    family: {gaussian, binomial, poisson, gamma, cox}
        The distribution of the simulated response.
        "gaussian" for univariate quantitative response,
        "binomial" for binary classification response,
        "poisson" for counting response,
        "gamma" for positive continuous response,
        "cox" for left-censored response.
    rho: float, optional, default=0
        A parameter used to characterize the pairwise
        correlation in predictors.
    sigma: float, optional, default=1
        The variance of the gaussian noise.
        It would be unused if snr is not None.
    coef_: array_like, optional, default=None
        The coefficient values in the underlying regression model.
    censoring: bool, optional, default=True
        For Cox data, it indicates whether censoring is existed.
    c: int, optional, default=1
        For Cox data and censoring=True, it indicates the maximum
        censoring time.
        So that all observations have chances to be censored at (0, c).
    scal: float, optional, default=10
        The scale of survival time in Cox data.
    snr: float, optional, default=None
        A numerical value controlling the signal-to-noise ratio (SNR)
        in gaussian data.
    class_num: int, optional, default=3
        The number of possible classes in oridinal dataset, i.e.
        :math:`y \in \{0, 1, 2, ..., \text{class_num}-1\}`

    Attributes
    ----------
    x: array-like, shape(n, p)
        Design matrix of predictors.
    y: array-like, shape(n,)
        Response variable.
    coef_: array-like, shape(p,)
        The coefficients used in the underlying regression model.
        It has k nonzero values.

    Notes
    -----
    The output, whose type is named ``data``, contains three elements:
    ``x``, ``y`` and ``coef_``, which correspond the variables, responses
    and coefficients, respectively.

    Each row of ``x`` or ``y`` indicates a sample and is independent to the
    other.

    We denote :math:`x, y, \beta` for one sample in the math formulas below.

    * Linear Regression

        * Usage: ``family='gaussian'[, sigma=...]``
        * Model: :math:`y \sim N(\mu, \sigma^2),\ \mu = x^T\beta`.

            * the coefficient :math:`\beta\sim U[m, 100m]`,
              where :math:`m = 5\sqrt{2\log p/n}`;
            * the variance :math:`\sigma = 1`.

    * Logistic Regression

        * Usage: ``family='binomial'``
        * Model: :math:`y \sim \text{Binom}(\pi),\
          \text{logit}(\pi) = x^T \beta`.

            * the coefficient :math:`\beta\sim U[2m, 10m]`,
              where :math:`m = 5\sqrt{2\log p/n}`.

    * Poisson Regression

        * Usage: ``family='poisson'``
        * Model: :math:`y \sim \text{Poisson}(\lambda),\
          \lambda = \exp(x^T \beta)`.

            * the coefficient :math:`\beta\sim U[2m, 10m]`,
              where :math:`m = 5\sqrt{2\log p/n}`.

    * Gamma Regression

        * Usage: ``family='gamma'``
        * Model: :math:`y \sim \text{Gamma}(k, \theta),\
          k\theta = \exp(x^T \beta + \epsilon), k\sim U[0.1, 100.1]`
          in shape-scale definition.

            * the coefficient :math:`\beta\sim U[m, 100m]`,
              where :math:`m = 5\sqrt{2\log p/n}`.

    * Cox PH Survival Analysis

        * Usage: ``family='cox'[, scal=..., censoring=..., c=...]``
        * Model: :math:`y=\min(t,C)`,
          where :math:`t = \left[-\dfrac{\log U}{\exp(X \beta)}\right]^s,\
          U\sim N(0,1),\ s=\dfrac{1}{\text{scal}}` and
          censoring time :math:`C\sim U(0, c)`.

            * the coefficient :math:`\beta\sim U[2m, 10m]`,
              where :math:`m = 5\sqrt{2\log p/n}`;
            * the scale of survival time :math:`\text{scal} = 10`;
            * censoring is enabled, and max censoring time :math:`c=1`.

    * Ordinal Regression

        * Usage: ``family='ordinal'[, class_num=...]``
        * Model: :math:`y\in \{0, 1, \dots, n_{class}\}`,
          :math:`\mathbb{P}(y\leq i) = \dfrac{1}
          {1+\exp(-x^T\beta - \varepsilon_i)}`,
          where :math:`i\in \{0, 1, \dots, n_{class}\}` and
          :math:`\forall i<j, \varepsilon_i < \varepsilon_j`.

            * the coefficient :math:`\beta\sim U[-M, M]`,
              where :math:`M = 125\sqrt{2\log p/n}`;
            * the intercept: :math:`\forall i,\varepsilon_i\sim U[-M, M]`;
            * the number of classes :math:`n_{class}=3`.

    """

    def __init__(self, n, p, k, family, rho=0, sigma=1, coef_=None,
                 censoring=True, c=1, scal=10, snr=None, class_num=3):
        self.n = n
        self.p = p
        self.k = k
        self.family = family

        zero = np.zeros([n, 1])
        ones = np.ones([n, 1])
        X = np.random.normal(0, 1, n * p).reshape(n, p)
        X = (X - np.matmul(ones, np.array([np.mean(X, axis=0)])))
        normX = np.sqrt(np.matmul(ones.reshape(1, n), X ** 2))
        X = np.sqrt(n) * X / normX

        x = X + rho * \
            (np.hstack((zero, X[:, 0:(p - 2)], zero)) +
             np.hstack((zero, X[:, 2:p], zero)))

        nonzero = sample(p, k)
        Tbeta = np.zeros(p)

        if family == "gaussian":
            m = 5 * np.sqrt(2 * np.log(p) / n)
            M = 100 * m
            if coef_ is None:
                Tbeta[nonzero] = np.random.uniform(m, M, k)
            else:
                Tbeta = coef_

            if snr is None:
                y = np.matmul(x, Tbeta) + sigma * np.random.normal(0, 1, n)
            else:
                y = np.matmul(x, Tbeta)
                power = np.mean(np.square(y))
                npower = power / 10 ** (snr / 10)
                noise = np.random.randn(len(y)) * np.sqrt(npower)
                y += noise

        elif family == "binomial":
            m = 5 * sigma * np.sqrt(2 * np.log(p) / n)
            if coef_ is None:
                Tbeta[nonzero] = np.random.uniform(2 * m, 10 * m, k)
            else:
                Tbeta = coef_

            xbeta = np.matmul(x, Tbeta)
            xbeta[xbeta > 30] = 30
            xbeta[xbeta < -30] = -30

            p = np.exp(xbeta) / (1 + np.exp(xbeta))
            y = np.random.binomial(1, p)

        elif family == "poisson":
            x = x / 16
            m = 5 * sigma * np.sqrt(2 * np.log(p) / n)
            if coef_ is None:
                Tbeta[nonzero] = np.random.uniform(2 * m, 10 * m, k)
                # Tbeta[nonzero] = np.random.normal(0, 4*m, k)
            else:
                Tbeta = coef_

            xbeta = np.matmul(x, Tbeta)
            xbeta[xbeta > 30] = 30
            xbeta[xbeta < -30] = -30

            lam = np.exp(xbeta)
            y = np.random.poisson(lam=lam)

        elif family == "cox":
            m = 5 * sigma * np.sqrt(2 * np.log(p) / n)
            if coef_ is None:
                Tbeta[nonzero] = np.random.uniform(2 * m, 10 * m, k)
            else:
                Tbeta = coef_

            time = np.power(-np.log(np.random.uniform(0, 1, n)) /
                            np.exp(np.matmul(x, Tbeta)), 1 / scal)

            if censoring:
                ctime = c * np.random.uniform(0, 1, n)
                status = (time < ctime) * 1
                censoringrate = 1 - sum(status) / n
                print("censoring rate:" + str(censoringrate))
                for i in range(n):
                    time[i] = min(time[i], ctime[i])
            else:
                status = np.ones(n)
                print("no censoring")

            y = np.hstack((time.reshape((-1, 1)), status.reshape((-1, 1))))

        elif family == "gamma":
            x = x / 16
            m = 5 * np.sqrt(2 * np.log(p) / n)
            if coef_ is None:
                Tbeta[nonzero] = np.random.uniform(m, 100 * m, k)
            else:
                Tbeta = coef_
            # add noise
            eta = x @ Tbeta + np.random.normal(0, sigma, n)
            # set coef_0 as + abs(min(eta)) + 1
            eta = eta + np.abs(np.min(eta)) + 10
            # set the shape para of gamma uniformly in [0.1,100.1]
            shape_para = 100 * np.random.uniform(0, 1, n) + 0.1
            y = np.random.gamma(shape=shape_para, scale=1 /
                                shape_para / eta, size=n)
        elif family == "ordinal":
            M = 125 * np.sqrt(2 * np.log(p) / n)
            if coef_ is None:
                Tbeta[nonzero] = np.random.uniform(-M, M, k)
            else:
                Tbeta = coef_
            intercept = np.sort(np.random.uniform(-M, M, class_num - 1))
            eta = x @ Tbeta[:, np.newaxis] + intercept
            logit = 1 / (1 + np.exp(-eta))
            # prob
            prob = np.zeros((n, class_num))
            prob[:, 0] = logit[:, 0]
            prob[:, 1:class_num - 1] = (logit[:, 1:class_num - 1] -
                                        logit[:, 0:class_num - 2])
            prob[:, class_num - 1] = 1 - logit[:, class_num - 2]
            # y
            y = np.zeros(n)
            for i in range(n):
                y[i] = np.random.choice(np.arange(class_num), 1, p=prob[i, :])
        else:
            raise ValueError(
                "Family should be \'gaussian\', \'binomial\', "
                "\'poisson\', \'gamma\', \'cox\', or \'ordinal\'.")
        self.x = x
        self.y = y
        self.coef_ = Tbeta


class make_multivariate_glm_data:
    r"""
    Generate a dataset with multi-responses.

    Parameters
    ----------
    n: int, optional, default=100
        The number of observations.
    p: int, optional, default=100
        The number of predictors of interest.
    family: {multigaussian, multinomial, poisson}, optional
        default="multigaussian".
        The distribution of the simulated multi-response.
        "multigaussian" for multivariate quantitative responses,
        "multinomial" for multiple classification responses,
        "poisson" for counting responses.
    k: int, optional, default=10
        The number of nonzero coefficients in the underlying regression model.
    M: int, optional, default=1
        The number of responses.
    rho: float, optional, default=0.5
        A parameter used to characterize the pairwise correlation
        in predictors.
    coef_: array_like, optional, default=None
        The coefficient values in the underlying regression model.
    sparse_ratio: float, optional, default=None
        The sparse ratio of predictor matrix (x).

    Attributes
    ----------
    x: array-like, shape(n, p)
        Design matrix of predictors.
    y: array-like, shape(n, M)
        Response variable.
    coef_: array-like, shape(p, M)
        The coefficients used in the underlying regression model.
        It is rowwise sparse, with k nonzero rows.

    Notes
    -----

    The output, whose type is named ``data``, contains three elements:
    ``x``, ``y`` and ``coef_``, which correspond the variables, responses
    and coefficients, respectively.

    Note that the ``y`` and ``coef_`` here are both matrix:

    1. each row of ``x`` and ``y`` indicates a sample;
    2. each column of ``coef_`` corresponds to the effect on one response.
       It is rowwise sparsity. Under this setting, a "useful" variable is
       relevant to all responses.

    We :math:`x, y, \beta` for one sample in the math formulas below.

    * Multitask Regression

        * Usage: ``family='multigaussian'``
        * Model: :math:`y \sim MVN(\mu, \Sigma),\ \mu^T=x^T \beta`.

            * the variance :math:`\Sigma = \text{diag}(1, 1, \cdots, 1)`;
            * the coefficient :math:`\beta` contains 30% "strong" values, 40%
              "moderate" values and the rest are "weak". They come from
              :math:`N(0, 10)`, :math:`N(0, 5)` and :math:`N(0, 2)`,
              respectively.

    * Multinomial Regression

        * Usage: ``family='multinomial'``
        * Model: :math:`y` is a "0-1" array with only one "1". Its index is
          chosed under probabilities :math:`\pi = \exp(x^T \beta)`.

            * the coefficient :math:`\beta` contains 30% "strong" values, 40%
              "moderate" values and the rest are "weak". They come from
              :math:`N(0, 10)`, :math:`N(0, 5)` and :math:`N(0, 2)`,
              respectively.

    """

    def __init__(self,
                 n=100, p=100, k=10, family="multigaussian", rho=0.5,
                 coef_=None, M=1, sparse_ratio=None):
        Sigma = np.ones(p * p).reshape(p, p) * rho
        ones = np.ones([n, 1])
        for i in range(p):
            Sigma[i, i] = 1
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean=mean, cov=Sigma, size=(n,))

        # Sigma[Sigma < 1e-10] = 0
        X = (X - np.matmul(ones, np.array([np.mean(X, axis=0)])))
        normX = np.sqrt(np.matmul(ones.reshape(1, n), X ** 2) / (n - 1))

        X = X / normX

        if sparse_ratio is not None:
            sparse_size = int((1 - sparse_ratio) * n * p)
            position = sample(n * p, sparse_size)
            print(position)
            for i in range(sparse_size):
                X[int(position[i] / p), position[i] % p] = 0

        Nonzero = sample(p, k)
        # Nonzero = np.array([0, 1, 2])
        # Nonzero[:k] = 1
        if coef_ is None:
            Tbeta = sparse_beta_generator(p, Nonzero, k, M)
        else:
            Tbeta = coef_

        if family in ("multigaussian", "gaussian"):
            eta = np.matmul(X, Tbeta)
            y = eta + np.random.normal(0, 1, n * M).reshape(n, M)

        elif family in ("multinomial", "binomial"):
            for i in range(M):
                Tbeta[:, i] = Tbeta[:, i] - Tbeta[:, M - 1]
            eta = np.exp(np.matmul(X, Tbeta))
            # y2 = np.zeros(n)
            y = np.zeros([n, M])
            index = np.linspace(0, M - 1, M)
            for i in range(n):
                p = eta[i, :] / np.sum(eta[i, :])
                j = np.random.choice(index, size=1, replace=True, p=p)
                # print(j)
                y[i, int(j[0])] = 1
                # y2[i] = j

        elif family == "poisson":
            X = X / 16
            eta = np.matmul(X, Tbeta)
            eta[eta > 30] = 30
            eta[eta < -30] = -30
            lam = np.exp(eta)
            y = np.random.poisson(lam=lam)

        else:
            raise ValueError(
                "Family should be \'gaussian\', \'multigaussian\', "
                "or \'multinomial\'.")

        self.x = X
        self.y = y
        self.coef_ = Tbeta
