import pandas as pd
from bess.linear import *
import numpy as np
from time import time

import os 

# x = pd.read_csv("cox_x.csv", index_col=0)
# y = pd.read_csv("cox_x.csv.csv", index_col=0)

x = pd.read_csv("cox_x.csv")
y = pd.read_csv("cox_y.csv")

print(x.shape)
x = x.values
# y = y.values.reshape(1, -1)[0]
y = y.values

# x_sort = x[y[:,0].argsort()]
# y_sort = y[y[:,0].argsort()]
# # print(X[104, 252])
# # print(X[:5, :5])
# print(y)

# time = y[:, 0]
# time_set = list(set(time))
# print(len(time))
# print(len(time_set))
# y_sort = y_sort[:, 1].reshape(-1)

# df_x_sort = pd.DataFrame(x_sort)
# df_y_sort = pd.DataFrame(y_sort)

# df_x_sort.to_csv("x_sort.csv", index=0)
# df_y_sort.to_csv("y_sort.csv", index=0)

# print(x)
# print(y)

# # print(x.head())
# # print(y.head())

# # x = x.values
# # # y = y.values.reshape(1, -1)[0]
# # y = y.values
# print(y)

for i in range(1):
    print(i)
    # model = PdasLm(path_type="seq", sequence=[100], ic_type="ebic")
    # model = PdasLogistic(path_type="seq", sequence=range(1,100), ic_type="ebic")
    # model = PdasCox(path_type="seq", sequence=[100], ic_type="ebic")
    # model = PdasPoisson(path_type="seq", sequence=range(1,100), ic_type="ebic")

    # lambda_sequence = np.exp(np.linspace(np.log(100), np.log(0.01), 10))
    # group = np.array(range(0, x.shape[1]))
    # # print(lambda_sequence)
    # model = GroupPdasLogistic(path_type="seq", sequence=range(1, 30), lambda_sequence=lambda_sequence, ic_type="gic", is_screening=True, screening_size=100, K_max=10, epsilon=10, powell_path=2, s_min= 1, s_max= 100, lambda_min=0.01, lambda_max=100, is_cv=True, K=5)
    # # model = L0L2Lm(path_type="pgs", s_min=1, s_max=30, lambda_min=0.01, lambda_max=100, is_cv=True, is_screening=True, screening_size=30, powell_path=2)
    # start = time()
    # #print(np.exp(np.linspace(np.log(10), np.log(0.1), 10)))
    # #print(range(1,11))
    # #model = L0L2Cox(path_type="seq", sequence=range(1,20),lambda_sequence=np.exp(np.linspace(np.log(10), np.log(0.1), 10)), is_cv=True,K=5)
    # model.fit(x,y,group=group)
    # print(np.nonzero(model.beta))
    # stop = time()
    # print("new time: " + str(stop-start))
    # print(model.beta[np.nonzero(model.beta)])

    start = time()
    group = np.array(range(0, x.shape[1]))
    start = time()
    model = PdasCox(path_type="seq", sequence=[x.shape[1]], ic_type="aic")
    model.fit(x,y)
    stop = time()
    print("new time: " + str(stop-start))
    print(np.nonzero(model.beta))
    print(model.beta[np.nonzero(model.beta)])

    # model = PdasCox(path_type="pgs", s_min=1, s_max=8, ic_type="aic")
    # model.fit(x,y)
    # print(np.nonzero(model.beta))
    # print(model.beta[np.nonzero(model.beta)])


    # lambda_sequence = np.exp(np.linspace(np.log(100), np.log(0.01), 50))
    # # group = np.array(range(0, x.shape[1]))
    # # print(lambda_sequence)
    # model = PdasLogistic(path_type="pgs", sequence=range(1, 30), lambda_sequence=lambda_sequence, ic_type="gic", is_screening=True, screening_size=100, K_max=10, epsilon=10, powell_path=2, s_min= 1, s_max= 100, lambda_min=0.01, lambda_max=100, is_cv=True, K=5)
    # # model = L0L2Lm(path_type="pgs", s_min=1, s_max=30, lambda_min=0.01, lambda_max=100, is_cv=True, is_screening=True, screening_size=30, powell_path=2)
    # start = time()
    # #print(np.exp(np.linspace(np.log(10), np.log(0.1), 10)))
    # #print(range(1,11))
    # #model = L0L2Cox(path_type="seq", sequence=range(1,20),lambda_sequence=np.exp(np.linspace(np.log(10), np.log(0.1), 10)), is_cv=True,K=5)
    # model.fit(x,y)
    # print(np.nonzero(model.beta))
    # stop = time()
    # print("old time: " + str(stop-start))
    # print(model.beta[np.nonzero(model.beta)])

    # group = np.array(range(0, x.shape[1]))

    # # print(x.shape)
    # group = np.array(range(0, x.shape[1]))
    # # model = L0L2Cox(path_type="pgs", s_min=1, s_max=30, lambda_min=0.01, lambda_max=100, is_cv=True, ic_type="gic", is_screening=True, screening_size=30, powell_path=2, K_max=10, epsilon=10)
    # # lambda_sequence = [0.1, 0.2, 0.3, 0.4, 0.5]
    # model = GroupPdasLm(path_type="seq", sequence=range(1, 30), lambda_sequence=lambda_sequence, ic_type="gic", is_screening=True, screening_size=30, K_max=10, epsilon=10, powell_path=1, s_min= 1, s_max= 30, lambda_min=0.01, lambda_max=100, is_cv=True, K=5)
    # start = time()
    # model.fit(x,y,group=group)
    # stop = time()
    # print("new time: " + str(stop-start))
    # print(model.beta[np.nonzero(model.beta)])
    # print(np.nonzero(model.beta))



