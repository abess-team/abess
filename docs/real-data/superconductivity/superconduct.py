# %%
import numpy as np
from time import time
from abess.linear import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, OrthogonalMatchingPursuitCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from celer import LassoCV as celerLassoCV
from sklearn.model_selection import train_test_split
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%
## preprocess superconduct data: create a high-dimensional data
data = pd.read_csv("train.csv", header=0)
data = data.loc[data["number_of_elements"] == 3, :]
data = data.drop(columns=['number_of_elements'])
y = data.loc[:, 'critical_temp']
X = data.drop(columns=['critical_temp'])

# %%
feature = PolynomialFeatures(include_bias=False, degree=3, interaction_only=True)
X = feature.fit_transform(X)
y = np.reshape(y, -1)
print("sample size: {}, dimension: {}".format(X.shape[0], X.shape[1]))

# %%
def metrics(coef, pred, real):
    auc = mean_squared_error(real, pred)
    nnz = len(np.nonzero(coef)[0])
    return np.array([auc, nnz])

M = 20
model_name = "Linear"
method = [
    "lasso",
    "celer",
    # "omp",  # uncomment this line because of memory leak
    "abess",
]
res_output = True
data_output = False
verbose = True

# MSE, NNZ, time
met = np.zeros((len(method), M, 3))
res = np.zeros((len(method), 6))

# Test
print('===== Testing '+ model_name + ' =====')
for m in range(M):
    ind = -1
    print(" --> Replication: " + str(m+1))

    trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.1, random_state=m)

    if "lasso" in method:
        ind += 1

        # transform via StandardScaler and increase tol because of ConvergenceWarning
        scaler = StandardScaler()
        scaler.fit(trainx)
        lasso_trainx = scaler.transform(trainx)
        lasso_testx = scaler.transform(testx)
        t_start = time()
        model = LassoCV(cv=5, n_jobs=5, random_state=0, tol=2e-2)
        fit = model.fit(lasso_trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(fit.coef_, fit.predict(lasso_testx), testy)
        met[ind, m, 2] = t_end - t_start
        print("     --> SKL time: " + str(t_end - t_start))
        print("     --> SKL err : " + str(met[ind, m, 0]))
        print("     --> SKL NNZ : " + str(met[ind, m, 1]))
    
    if "celer" in method:
        ind += 1

        t_start = time()
        model = celerLassoCV(cv=5, n_jobs=5)
        fit = model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(fit.coef_, fit.predict(testx), testy)
        met[ind, m, 2] = t_end - t_start
        print("     --> CELER time: " + str(t_end - t_start))
        print("     --> CELER err : " + str(met[ind, m, 0]))
        print("     --> CELER NNZ : " + str(met[ind, m, 1]))
    
    ## omp
    if "omp" in method:
        ind += 1

        t_start = time()
        model = OrthogonalMatchingPursuitCV(cv=5, n_jobs=5, max_iter=100)
        fit = model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(fit.coef_, fit.predict(testx), testy)
        met[ind, m, 2] = t_end - t_start
        print("     --> OMP time: " + str(t_end - t_start))
        print("     --> OMP err : " + str(met[ind, m, 0]))
        print("     --> OMP NNZ : " + str(met[ind, m, 1]))

    ## abess
    if "abess" in method:
        ind += 1
        max_supp = np.min([100, trainx.shape[1]])

        t_start = time()
        model = LinearRegression(cv=5, support_size=range(max_supp), thread=5)
        model.fit(trainx, trainy)
        t_end = time()

        met[ind, m, 0:2] = metrics(model.coef_, model.predict(testx), testy)
        met[ind, m, 2] = t_end - t_start
        print("     --> ABESS time: " + str(t_end - t_start))
        print("     --> ABESS err : " + str(met[ind, m, 0]))
        print("     --> ABESS NNZ : " + str(met[ind, m, 1]))

for ind in range(0, len(method)):
    m = met[ind].mean(axis = 0)
    se = met[ind].std(axis = 0) / np.sqrt(M - 1)
    res[ind] = np.hstack((m, se))
    res = np.around(res, decimals=2)

print("===== Results " + model_name + " =====")
print("Method: \n", method)
print("Metrics (MSE, NNZ, Runtime): \n", res[:, 0:3])

if (res_output):
    np.save(model_name + "_res.npy", res)
    print("Result saved.")
