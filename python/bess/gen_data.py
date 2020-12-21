import numpy as np


class data:
    def __init__(self, x, y, beta):
        self.x = x
        self.y = y
        self.beta = beta


def sample(p, k):
    full = np.arange(p)
    select = np.zeros(k, int)
    for i in range(k):
        z = np.random.choice(full, 1)
        index = np.where(full == z)
        select[i] = z
        full = np.delete(full, index)
    return select


def gen_data(n, p, family, k, rho=0, sigma=1, beta=None, censoring =True, c=1, scal=10):
    zero = np.zeros([n, 1])
    ones = np.ones([n, 1])
    X = np.random.normal(0, 1, n*p).reshape(n, p)
    X = (X - np.matmul(ones, np.array([np.mean(X, axis=0)])))
    normX = np.sqrt(np.matmul(ones.reshape(1, n), X ** 2))
    X = np.sqrt(n) * X / normX

    x = X + rho * (np.hstack((zero, X[:, 0:(p-2)], zero)) + np.hstack((zero, X[:, 2:p], zero)))

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

        time = np.power(-np.log(np.random.uniform(0, 1, n)) / np.exp(np.matmul(x, Tbeta)), 1/scal)

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
        raise ValueError("Family should be \'gaussian\', \'binomial\', \'possion\', or \'cox\'")
