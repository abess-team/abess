import numpy as np
import random
from abess.cabess import ising_sample_by_conf_wrap, ising_gibbs_wrap
from collections import namedtuple

class data:
    def __init__(self, x, y, coef_):
        self.x = x
        self.y = y
        self.coef_ = coef_


def sample(p, k):
    full = np.arange(p)
    select = np.random.choice(full, k, replace=False)
    select.sort()
    return select


def make_glm_data(n, p, k, family, rho=0, sigma=1, coef_=None, censoring=True, c=1, scal=10, snr=None):
    """
    Generate a dataset with single response.

    Parameters
    ----------
    n: int
        The number of observations.
    p: int
        The number of predictors of interest.
    k: int
        The number of nonzero coefficients in the underlying regression model. 
    family: {gaussian, binomial, poisson, cox}
        The distribution of the simulated response. 
        "gaussian" for univariate quantitative response, 
        "binomial" for binary classification response, 
        "poisson" for counting response, 
        "cox" for left-censored response.
    rho: float, optional
        A parameter used to characterize the pairwise correlation in predictors. 
        Default: rho = 0.
    sigma: float, optional
        The variance of the gaussian noise. 
        It would be unused if `snr` is not None.
        Default: sigma = 1.
    coef\_: array_like, optional
        The coefficient values in the underlying regression model. 
        Default: coef\_ = None.
    censoring: bool, optional
        For Cox data, it indicates whether censoring is existed.
        Default: censoring = True
    c: int, optional
        For Cox data and `censoring=True`, it indicates the maximum censoring time.
        So that all observations have chances to be censored at (0, c).
        Default: c = 1.
    scal: float, optional
        The scale of survival time in Cox data.
        Default: scal = 10.
    snr: float, optional
        A numerical value controlling the signal-to-noise ratio (SNR) in gaussian data.
        Default: snr = None.

    Returns
    -------
    x: array_like, shape(n, p)
        Design matrix of predictors.
    y: array_like, shape(n,)
        Response variable.
    coef\_: array_like, shape(p,)
        The coefficients used in the underlying regression model.
    """
    zero = np.zeros([n, 1])
    ones = np.ones([n, 1])
    X = np.random.normal(0, 1, n*p).reshape(n, p)
    X = (X - np.matmul(ones, np.array([np.mean(X, axis=0)])))
    normX = np.sqrt(np.matmul(ones.reshape(1, n), X ** 2))
    X = np.sqrt(n) * X / normX

    x = X + rho * \
        (np.hstack((zero, X[:, 0:(p-2)], zero)) +
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
            npower = power / 10 ** (snr/10)
            noise = np.random.randn(len(y)) * np.sqrt(npower)
            y += noise

        return data(x, y, Tbeta)

    elif family == "binomial":
        m = 5 * sigma * np.sqrt(2 * np.log(p) / n)
        if coef_ is None:
            Tbeta[nonzero] = np.random.uniform(2*m, 10*m, k)
        else:
            Tbeta = coef_

        xbeta = np.matmul(x, Tbeta)
        xbeta[xbeta > 30] = 30
        xbeta[xbeta < -30] = -30

        p = np.exp(xbeta)/(1+np.exp(xbeta))
        y = np.random.binomial(1, p)
        return data(x, y, Tbeta)

    elif family == "poisson":
        x = x / 16
        m = 5 * sigma * np.sqrt(2 * np.log(p) / n)
        if coef_ is None:
            Tbeta[nonzero] = np.random.uniform(2*m, 10*m, k)
            # Tbeta[nonzero] = np.random.normal(0, 4*m, k)
        else:
            Tbeta = coef_

        xbeta = np.matmul(x, Tbeta)
        xbeta[xbeta > 30] = 30
        xbeta[xbeta < -30] = -30

        lam = np.exp(xbeta)
        y = np.random.poisson(lam=lam)
        return data(x, y, Tbeta)

    elif family == "cox":
        m = 5 * sigma * np.sqrt(2 * np.log(p) / n)
        if coef_ is None:
            Tbeta[nonzero] = np.random.uniform(2*m, 10*m, k)
        else:
            Tbeta = coef_

        time = np.power(-np.log(np.random.uniform(0, 1, n)) /
                        np.exp(np.matmul(x, Tbeta)), 1/scal)

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

        return data(x, y, Tbeta)
    else:
        raise ValueError(
            "Family should be \'gaussian\', \'binomial\', \'possion\', or \'cox\'")


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


def make_multivariate_glm_data(n=100, p=100, k=10, family="gaussian", SNR=1, rho=0.5, coef_=None, M=1, sparse_ratio=None):
    """
    Generate a dataset with multi-responses.

    Parameters
    ----------
    n: int, optional
        The number of observations.
        Default: n = 100.
    p: int, optional
        The number of predictors of interest.
        Default: p = 100.
    family: {gaussian, binomial, poisson}, optional
        The distribution of the simulated multi-response. 
        "gaussian" for univariate quantitative response, 
        "binomial" for binary classification response, 
        "poisson" for counting response.
        Default: family = "gaussian".
    k: int, optional
        The number of nonzero coefficients in the underlying regression model. 
        Default: k = 10.
    M: int, optional
        The number of multi-responses. 
        Default: M = 1.
    rho: float, optional
        A parameter used to characterize the pairwise correlation in predictors. 
        Default: rho = 0.5.
    coef\_: array_like, optional
        The coefficient values in the underlying regression model. 
        Default: coef\_ = None.
    sparse_ratio: float, optional
        The sparse ratio of predictor matrix (x).
        Default: sparse_ratio = None.

    Returns
    -------
    x: array_like, shape(n, p)
        Design matrix of predictors.
    y: array_like, shape(n, M)
        Response variable.
    coef\_: array_like, shape(p, M)
        The coefficients used in the underlying regression model.
    """
    Sigma = np.ones(p*p).reshape(p, p) * rho
    ones = np.ones([n, 1])
    for i in range(p):
        Sigma[i, i] = 1
    mean = np.zeros(p)
    X = np.random.multivariate_normal(mean=mean, cov=Sigma, size=(n,))

    # Sigma[Sigma < 1e-10] = 0
    X = (X - np.matmul(ones, np.array([np.mean(X, axis=0)])))
    normX = np.sqrt(np.matmul(ones.reshape(1, n), X ** 2) / (n-1))

    X = X / normX

    if sparse_ratio != None:
        sparse_size = int((1 - sparse_ratio) * n * p)
        position = sample(n*p, sparse_size)
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

    if family == "multigaussian" or family == "gaussian":
        eta = np.matmul(X, Tbeta)
        y = eta + np.random.normal(0, 1, n*M).reshape(n, M)
        return data(X, y, Tbeta)
    elif family == "multinomial" or family == "binomial":
        for i in range(M):
            Tbeta[:, i] = Tbeta[:, i] - Tbeta[:, M-1]
        eta = np.exp(np.matmul(X, Tbeta))
        # y2 = np.zeros(n)
        y = np.zeros([n, M])
        index = np.linspace(0, M-1, M)
        for i in range(n):
            p = eta[i, :] / np.sum(eta[i, :])
            j = np.random.choice(index, size=1, replace=True, p=p)
            # print(j)
            y[i, int(j[0])] = 1
            # y2[i] = j
        return data(X, y, Tbeta)
    elif family == "poisson":
        eta = np.matmul(X, Tbeta)
        eta[eta > 30] = 30
        eta[eta < -30] = -30
        lam = np.exp(eta)
        y = np.random.poisson(lam=lam)
        return data(X, y, Tbeta)
    else:
        raise ValueError(
            "Family should be \'gaussian\', \'multigaussian\', or \'multinomial\'")


def make_ising_data(n = 1000, p = 10, theta = None, seed = None, type = 1, alpha = 0.4, beta = None, degree = 3):
    """
    Generate dataset for ising model.

    Parameters
    ----------
    n: int, optional
        Sample size.
        Default: n = 1000
    p: int, optional
        The number of nodes in graph. It would be ignored if theta is given.
        Default: p = 10
    theta: array_like, optional
        The theta for generation. If not given, it would be generated by other parameters. 
        Default: theta = None
    seed: int, optional
        Random seed.
        Default: seed = None
    type: int, optional
        The shape of graph. (Not support yet.)
        `1` for linear; `2` for ...; `3` for 4NN; `4` for 8NN; `5` for star; 
        `6``7``8``9``10``11``12`
        Default: type = 1
    alpha: double, optional
        Generation parameter. (Not support yet.)
        Default: alpha = 0.4
    beta: double, optional
        Generation parameter. (Not support yet.)
        Default: beta = None
    degree: int, optional
        Generation parameter. (Not support yet.)
        Default: degree = 3

    Returns
    -------
    theta\_: array_like, shape(p, p)
        The theta used for generation.
    train: array_like, shape(_, p)
        Sample matrix for training.
    valid: array_like, shape(_, p)
        Sample matrix for validation.
    train_freq: array_like, shape(train.shape[0],)
        The frequence for each train samples.
    train_freq: array_like, shape(test.shape[0],)
        The frequence for each valid samples.

    """
    # seed
    if seed is not None:
        np.random.seed(seed)
        using_seed = True
    else:
        seed = 0
        using_seed = False

    # gene theta 
    if theta is not None:
        # check ?
        theta = np.array(theta, dtype = 'float')
        p = theta.shape[1]
    else:
        # todo
        theta = np.zeros((p, p))
        raise ValueError("Sorry! Generation for theta have not finished yet. It should be given at present.")
        
    # gene data
    # if (p <= 20):
    #     seed = random.sample(range(1, 31415926), 2)
    #     m = np.power(2, p)
    #     result = ising_sample_by_conf_wrap(
    #         n, theta, seed[0], seed[1], 
    #         int(m*(p + 1)), int(m*(p + 1))
    #     )
    #     train = result[0].reshape(m, p + 1)
    #     valid = result[1].reshape(m, p + 1)
    #     train = train[train[:, 0] != 0, :]
    #     valid = valid[valid[:, 0] != 0, :]
    #     train_freq = train[:, 0].reshape(-1)
    #     valid_freq = valid[:, 0].reshape(-1)
    #     train = train[:, 1:]
    #     valid = valid[:, 1:]
    # else:
    # Gibbs
    burn = int(1e5)
    skip = 50
    value = np.array([-1, 1], dtype = 'float')
    result = ising_gibbs_wrap(
        theta, n, burn, skip, value, using_seed, seed,
        int(2*n*p)
    )
    dt = result.reshape(2*n, p)
    train = dt[0:2*n:2, :]
    valid = dt[1:2*n:2, :]
    train_freq = np.ones(n)
    valid_freq = np.ones(n)

    # print('train\n', train)
    # print('train_freq\n', train_freq)
    # print('valid\n', valid)
    # print('valid_freq\n', valid_freq)

    Ising_Data = namedtuple('Ising_Data', 'train train_freq valid valid_freq theta_')
    output = Ising_Data(train, train_freq, valid, valid_freq, theta)
    return output