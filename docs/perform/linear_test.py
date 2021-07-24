import numpy as np
from time import time
from abess.linear import abessLm
from abess.gen_data import gen_data
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LassoCV
from glmnet import ElasticNet

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
rho = 0.1 
model_name = "Lm"
method = [
    "abess", 
    "sklearn", 
    "glmnet", 
]
file_output = True

# time
ti = np.zeros(len(method))
# pred_err, tpr, fpr, mcc
met = np.zeros((len(method), 4))


for m in range(M):
    ind = -1
    if (m % 10 == 0):
        print(" --> iter: " + str(m))

    ## data gene
    np.random.seed(m)
    train = gen_data(n = n, p = p, k = 10, family = "gaussian", rho = rho)
    np.random.seed(m + 100)
    test = gen_data(n = n, p = p, k = 10, family = "gaussian", rho = rho, coef_ = train.coef_)

    ## abess
    if "abess" in method:
        ind += 1

        t_start = time()
        model = abessLm(is_cv = True, path_type = "pgs", s_min = 0, s_max = 99)
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind] += metrics(fit.coef_, fit.predict(test.x), test)
        ti[ind] += t_end - t_start
        # print("     --> ABESS time: " + str(t_end - t_start))

    ## sklearn
    if "sklearn" in method:
        ind += 1

        t_start = time()
        model = LassoCV()
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind] += metrics(fit.coef_, fit.predict(test.x), test)
        ti[ind] += t_end - t_start
        # print("     --> SKL time: " + str(t_end - t_start))
    
    ## glmnet
    if "glmnet" in method:
        ind += 1

        t_start = time()
        model = ElasticNet()
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind] += metrics(fit.coef_, fit.predict(test.x), test)
        ti[ind] += t_end - t_start
        # print("     --> GLMNET time: " + str(t_end - t_start))


met /= M
ti /= M

print("===== " + model_name + " - " + str(rho) + " =====")
print("Method: \n", method)
print("Metrics: \n", met)
print("Time: \n", ti)

if (file_output):
    temp = ti.reshape(len(method), 1)
    temp = np.hstack((met, temp)).T
    np.savetxt(model_name + "_res.csv", temp, delimiter=",", encoding = "utf-8", fmt = "%.8f")
    print("Saved in file.")
