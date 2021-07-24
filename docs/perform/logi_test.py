import numpy as np
from time import time
from abess.linear import abessLogistic
from abess.gen_data import gen_data
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from glmnet import LogitNet

def metrics(coef, pred, test):
    auc = roc_auc_score(test.y, pred)

    p = coef != 0
    r = test.coef_ != 0
    tpr = sum(r & p) / sum(r)
    fpr = sum(~r & p) / sum(~r)
    mcc = matthews_corrcoef(r, p)

    return np.array([auc, tpr, fpr, mcc])

n = 500
p = 1000
M = 100
rho = 0.1 
model_name = "Logistic"
method = [
    "abess",
    "sklearn",
    "glmnet"
]
file_output = True

# time
ti = np.zeros(len(method))
# auc, tpr, fpr, mcc
met = np.zeros((len(method), 4))


for m in range(M):
    ind = -1
    if (m % 10 == 0):
        print(" --> iter: " + str(m))

    ## data gene
    np.random.seed(m)
    train = gen_data(n = n, p = p, k = 10, family = "binomial", rho = rho)
    np.random.seed(m + 100)
    test = gen_data(n = n, p = p, k = 10, family = "binomial", rho = rho, coef_ = train.coef_)

    ## abess
    if "abess" in method:
        ind += 1

        t_start = time()
        model = abessLogistic(is_cv = True, path_type = "pgs", s_min = 0, s_max = 99)
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind] += metrics(fit.coef_, fit.predict(test.x), test)
        ti[ind] += t_end - t_start
        # print("     --> ABESS time: " + str(t_end - t_start))
    
    if "sklearn" in method:
        ind += 1

        t_start = time()
        model = LogisticRegressionCV(penalty = "l1", solver = "liblinear")
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind] += metrics(fit.coef_.flatten(), fit.predict(test.x), test)
        ti[ind] += t_end - t_start
        # print("     --> SKL time: " + str(t_end - t_start))
    
    ## glmnet
    if "glmnet" in method:
        ind += 1

        t_start = time()
        model = LogitNet()
        fit = model.fit(train.x, train.y)
        t_end = time()

        met[ind] += metrics(fit.coef_.flatten(), fit.predict(test.x), test)
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
