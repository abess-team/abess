import sys
import numpy as np
from time import time
from abess.linear import abessLm
from abess.datasets import make_glm_data
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LassoCV

def metrics(coef, pred, test):
    pred_err = np.linalg.norm(pred - test.y)

    p = coef != 0
    r = test.coef_ != 0
    tpr = sum(r & p) / sum(r)
    fpr = sum(~r & p) / sum(~r)
    mcc = matthews_corrcoef(r, p)

    return np.array([pred_err, tpr, fpr, mcc])

n = 500
p = 1000
M = 100
rho = float(sys.argv[1])
model_name = "Lm"
method = [
    "abess", 
    "sklearn", 
]
file_output = True

# pred_err, tpr, fpr, mcc, time
met = np.zeros((len(method), M, 5))
res = np.zeros((len(method), 10))

print('===== Testing '+ model_name + " - " + str(rho) + ' =====')
for m in range(M):
    ind = -1
    if (m % 10 == 0):
        print(" --> iter: " + str(m))

    ## data gene
    np.random.seed(m)
    train = make_glm_data(n = n, p = p, k = 10, family = "gaussian", rho = rho)
    np.random.seed(m + M)
    test = make_glm_data(n = n, p = p, k = 10, family = "gaussian", rho = rho, coef_ = train.coef_)

    ## abess
    if "abess" in method:
        ind += 1

        t_start = time()
        model = abessLm(is_cv = True, path_type = "pgs", s_min = 0, s_max = 99, thread = 0)
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind, m, 0:4] = metrics(fit.coef_, fit.predict(test.x), test)
        met[ind, m, 4] = t_end - t_start
        # print("     --> ABESS time: " + str(t_end - t_start))

    ## sklearn
    if "sklearn" in method:
        ind += 1

        t_start = time()
        model = LassoCV(n_jobs = -1)
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind, m, 0:4] = metrics(fit.coef_, fit.predict(test.x), test)
        met[ind, m, 4] = t_end - t_start
        # print("     --> SKL time: " + str(t_end - t_start))
    
for ind in range(0, len(method)):
    m = met[ind].mean(axis = 0)
    se = met[ind].std(axis = 0) / np.sqrt(M - 1)
    res[ind] = np.hstack((m, se))

print("===== Results " + model_name + " - " + str(rho) + " =====")
print("Method: \n", method)
print("Metrics: \n", res[:, 0:5])
print("Err", res[:, 5:10])

if (file_output):
    np.savetxt(model_name + str(rho) + "_res.csv", res, delimiter=",", encoding = "utf-8", fmt = "%.8f")
    print("Saved in file.")
