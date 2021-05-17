import numpy as np


class data:
    def __init__(self, x, y, beta):
        self.x = x
        self.y = y
        self.beta = beta


def sample(p, k):
    full = np.arange(p)
    select = np.random.choice(full, k, replace=False)
    select.sort()
    return select


def gen_data(n, p, family, k, rho=0, sigma=1, beta=None, censoring=True, c=1, scal=10):
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
        if beta is None:
            Tbeta[nonzero] = np.random.uniform(m, M, k)
        else:
            Tbeta = beta

        y = np.matmul(x, Tbeta) + sigma * np.random.normal(0, 1, n)
        return data(x, y, Tbeta)

    elif family == "binomial":
        m = 5 * sigma * np.sqrt(2 * np.log(p) / n)
        if beta is None:
            Tbeta[nonzero] = np.random.uniform(2*m, 10*m, k)
        else:
            Tbeta = beta

        xbeta = np.matmul(x, Tbeta)
        xbeta[xbeta > 30] = 30
        xbeta[xbeta < -30] = -30

        p = np.exp(xbeta)/(1+np.exp(xbeta))
        y = np.random.binomial(1, p)
        return data(x, y, Tbeta)

    elif family == "poisson":
        x = x / 16
        m = 5 * sigma * np.sqrt(2 * np.log(p) / n)
        if beta is None:
            Tbeta[nonzero] = np.random.uniform(2*m, 10*m, k)
            # Tbeta[nonzero] = np.random.normal(0, 4*m, k)
        else:
            Tbeta = beta

        xbeta = np.matmul(x, Tbeta)
        xbeta[xbeta > 30] = 30
        xbeta[xbeta < -30] = -30

        lam = np.exp(xbeta)
        y = np.random.poisson(lam=lam)
        return data(x, y, Tbeta)

    elif family == "cox":
        m = 5 * sigma * np.sqrt(2 * np.log(p) / n)
        if beta is None:
            Tbeta[nonzero] = np.random.uniform(2*m, 10*m, k)
        else:
            Tbeta = beta

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

    # print(beta_value)
    for i in range(M):
        beta_value[:, i] = beta_value[:, i] - beta_value[:, M-1]

    beta_value = beta_value[sample(k, k), :]

    # beta_value = np.random.normal(size=(k, M))
    return beta_value


def gen_data_splicing(family="gaussian", n=100, p=100, k=10, SNR=1, rho=0.5, beta=None, M=1, sparse_ratio=None):
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
    if beta is None:
        Tbeta = sparse_beta_generator(p, Nonzero, k, M)
    else:
        Tbeta = beta

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
    else:
        raise ValueError(
            "Family should be \'gaussian\', \'multigaussian\', or \'multinomial\'")
