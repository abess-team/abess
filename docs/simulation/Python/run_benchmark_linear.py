import sys
import warnings
from time import time
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LassoCV
from sklearn.linear_model import OrthogonalMatchingPursuitCV
# from spams import fistaFlat
# from sklearn.model_selection import GridSearchCV
# from glmnet import ElasticNet
# import statsmodels.api as sm
# from l0bnb import fit_path
from abess.linear import LinearRegression
from abess.datasets import make_glm_data

warnings.filterwarnings("ignore", category=FutureWarning)


def metrics(coef, pred, test):
    pred_err = np.linalg.norm((pred - test.y))
    coef_err = np.linalg.norm(coef - test.coef_)

    p = abs(coef) > 1e-5
    r = abs(test.coef_) > 1e-5
    tpr = sum(r & p) / sum(r)
    fpr = sum(~r & p) / sum(~r)
    mcc = matthews_corrcoef(r, p)

    return np.array([pred_err, coef_err, tpr, fpr, mcc])


n = 500
p = 8000
M = 20
rho = float(sys.argv[1])
model_name = "Lm"
method = [
    "lasso",
    "omp",
    # "statsmodels",
    # "glmnet",
    # "l0bnb",
    # "spams",
    "abess",
]
res_output = True
data_output = True

# pred_err, coef_err, tpr, fpr, mcc, time
met = np.zeros((len(method), M, 6))
res = np.zeros((len(method), 12))

print('===== Testing ' + model_name + " - " + str(rho) + ' =====')
for m in range(M):
    ind = -1
    if m % 10 == 0:
        print(" --> iter: " + str(m))

    # data gene
    np.random.seed(m)
    train = make_glm_data(n=n, p=p, k=10, family="gaussian", rho=rho)
    np.random.seed(m + M)
    test = make_glm_data(
        n=n,
        p=p,
        k=10,
        family="gaussian",
        rho=rho,
        coef_=train.coef_)

    # lasso
    if "lasso" in method:
        ind += 1

        t_start = time()
        model = LassoCV(cv=5, n_jobs=5)
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind, m, 0:5] = metrics(fit.coef_, fit.predict(test.x), test)
        met[ind, m, 5] = t_end - t_start
        # print("     --> SKL time: " + str(t_end - t_start))

    # omp
    if "omp" in method:
        ind += 1
        t_start = time()

        model = OrthogonalMatchingPursuitCV(cv=5, n_jobs=5, max_iter=100)
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind, m, 0:5] = metrics(fit.coef_, fit.predict(test.x), test)
        met[ind, m, 5] = t_end - t_start
        # print("     --> OMP time: " + str(t_end - t_start))

    # statsmodels
    if "statsmodels" in method:
        ind += 1

        t_start = time()
        model = sm.OLS(train.y, train.x)
        fit = model.fit_regularized(alpha=1, L1_wt=1)
        t_end = time()

        met[ind, m, 0:5] = metrics(fit.params, fit.predict(test.x), test)
        met[ind, m, 5] = t_end - t_start
        # print("     --> STATS time: " + str(t_end - t_start))

    # glmnet
    if "glmnet" in method:
        ind += 1

        t_start = time()
        model = ElasticNet(n_jobs=8)
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind, m, 0:5] = metrics(fit.coef_, fit.predict(test.x), test)
        met[ind, m, 5] = t_end - t_start
        # print("     --> GLMNET time: " + str(t_end - t_start))

    # l0bnb
    if "l0bnb" in method:
        ind += 1

        t_start = time()
        fit = fit_path(train.x, train.y, max_nonzeros=99)
        t_end = time()

        pred = np.dot(test.x, fit[4]['B']) + fit[4]['B0']
        met[ind, m, 0:5] = metrics(fit[4]['B'], pred, test)
        met[ind, m, 5] = t_end - t_start
        # print("     --> L0BNB time: " + str(t_end - t_start))

    # spams
    if "spams" in method:
        ind += 1

        W0 = np.zeros((p + 1, 1))
        X0 = np.asfortranarray(np.hstack(((train.x), np.ones((n, 1)))))
        Y0 = np.asfortranarray(train.y.reshape(len(train.y), 1))
        t_start = time()
        fit = fistaFlat(
            Y0,
            X0,
            W0,
            regul='l0',
            loss='square',
            lambda1=1000,
            intercept=True)
        t_end = time()

        fit = fit.reshape(-1)
        pred = np.dot(test.x, fit[0: p]) + fit[p]
        met[ind, m, 0:5] = metrics(fit[0:p], pred, test)
        met[ind, m, 5] = t_end - t_start
        # print("     --> SPAMS time: " + str(t_end - t_start))

    # abess
    if "abess" in method:
        ind += 1

        t_start = time()
        model = LinearRegression(cv=5, support_size=range(100), thread=5)
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind, m, 0:5] = metrics(fit.coef_, fit.predict(test.x), test)
        met[ind, m, 5] = t_end - t_start
        # print("     --> ABESS time: " + str(t_end - t_start))


for ind in range(0, len(method)):
    m = met[ind].mean(axis=0)
    se = met[ind].std(axis=0) / np.sqrt(M - 1)
    res[ind] = np.hstack((m, se))

print("===== Results " + model_name + " - " + str(rho) + " =====")
print("Method: \n", method)
print("Metrics: \n", res[:, 0:6])
print("Err: \n", res[:, 6:12])

if res_output:
    np.save(model_name + str(rho) + "_res.npy", res)
    print("Result saved.")

if data_output:
    np.save(model_name + str(rho) + "_data.npy", met)
    print("Data saved.")
