import sys
from time import time
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
# from glmnet import LogitNet
# import statsmodels.api as sm
from abess.linear import abessLogistic
from abess.datasets import make_glm_data


def metrics(coef, pred, test):
    auc = roc_auc_score(test.y, pred)
    coef_err = np.linalg.norm(coef - test.coef_)

    p = abs(coef) > 1e-5
    r = abs(test.coef_) > 1e-5
    tpr = sum(r & p) / sum(r)
    fpr = sum(~r & p) / sum(~r)
    mcc = matthews_corrcoef(r, p)

    return np.array([auc, coef_err, tpr, fpr, mcc])


n = 500
p = 8000
M = 20
rho = float(sys.argv[1])
model_name = "Logistic"
method = [
    "lasso",
    # "statsmodels",
    # "glmnet",
    "abess",
]
res_output = True
data_output = True

# auc, coef_err, tpr, fpr, mcc, time
met = np.zeros((len(method), M, 6))
res = np.zeros((len(method), 12))

print('===== Testing ' + model_name + " - " + str(rho) + ' =====')
for m in range(M):
    ind = -1
    if m % 10 == 0:
        print(" --> iter: " + str(m))

    # data gene
    np.random.seed(m)
    train = make_glm_data(n=n, p=p, k=10, family="binomial", rho=rho)
    np.random.seed(m + M)
    test = make_glm_data(
        n=n,
        p=p,
        k=10,
        family="binomial",
        rho=rho,
        coef_=train.coef_)

    # lasso
    if "lasso" in method:
        ind += 1

        t_start = time()
        model = LogisticRegressionCV(
            penalty="l1", solver="liblinear", cv=5, n_jobs=5)
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind, m, 0:5] = metrics(
            fit.coef_.flatten(), fit.predict(test.x), test)
        met[ind, m, 5] = t_end - t_start
        # print("     --> SKL time: " + str(t_end - t_start))

    # statsmodels
    if "statsmodels" in method:
        ind += 1

        t_start = time()
        model = sm.Logit(train.y, train.x)
        fit = model.fit_regularized(alpha=1, L1_wt=1)
        t_end = time()

        met[ind, m, 0:5] = metrics(fit.params, fit.predict(test.x), test)
        met[ind, m, 5] = t_end - t_start
        # print("     --> STATS time: " + str(t_end - t_start))

    # glmnet
    if "glmnet" in method:
        ind += 1

        t_start = time()
        model = LogitNet(n_jobs=8)
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind, m, 0:5] = metrics(
            fit.coef_.flatten(), fit.predict(test.x), test)
        met[ind, m, 5] = t_end - t_start
        # print("     --> GLMNET time: " + str(t_end - t_start))

    # abess
    if "abess" in method:
        ind += 1

        t_start = time()
        # model = abessLogistic(is_cv = True, path_type = "pgs", s_min = 0, s_max = 99, thread = 0)
        model = abessLogistic(cv=5, support_size=range(100), thread=5,
                              approximate_Newton=True, primary_model_fit_epsilon=1e-6)
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
