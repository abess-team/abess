import numpy as np
from bess.linear import PdasLm, PdasLogistic, PdasPoisson, PdasCox, L0L2Lm, L0L2Logistic, L0L2Poisson, L0L2Cox, GroupPdasLm, GroupPdasLogistic,GroupPdasPoisson, GroupPdasCox

from bess.gen_data import gen_data
import pandas as pd
import os
from time import time

n=100
p=10
k=2
family = "poisson"
rho=0.5
sigma=1

def mse(x,Tbeta,beta):
    xTbeta = np.matmul(x, Tbeta)
    xbeta = np.matmul(x, beta) 
    return np.sum((xTbeta - xbeta)**2) / np.sum(xTbeta**2)

# np.random.seed(123)
data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
test_data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma, beta=data.beta)

# np.random.seed(12345)
mse_list = []
for i in range(1):
    print(i)
    # data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
    # print(np.nonzero(data.beta))
    # print(data.beta[np.nonzero(data.beta)])

    # print(data.x[:5,:])

    model = PdasPoisson(path_type="seq", sequence=range(1, p+1), ic_type='ebic', is_screening=False, screening_size=30, K_max=10, epsilon=10, powell_path=2, s_min= 1, s_max= p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5)
    # model = L0L2Cox(path_type="seq", sequence=range(1,p),lambda_sequence=np.exp(np.linspace(np.log(0.1), np.log(10), 10)),ic_type="ebic")
    start = time()
    model.fit(data.x, data.y)
    stop = time()
    print("seq: " + str(stop-start))
    print(np.nonzero(model.beta))
    print(np.nonzero(data.beta))
    print(model.beta[np.nonzero(model.beta)])
    print(model.coef0)

    
    model = PdasPoisson(path_type="pgs", sequence=range(1, p+1), ic_type='ebic', is_screening=False, screening_size=30, K_max=10, epsilon=10, powell_path=2, s_min= 1, s_max= p, lambda_min=0.01, lambda_max=100, is_cv=False, K=5)
    # model = L0L2Cox(path_type="seq", sequence=range(1,p),lambda_sequence=np.exp(np.linspace(np.log(0.1), np.log(10), 10)),ic_type="ebic")
    start = time()
    model.fit(data.x, data.y)
    stop = time()
    print("pgs: " + str(stop-start))
    print(np.nonzero(model.beta))
    print(np.nonzero(data.beta))
    print(data.beta[np.nonzero(data.beta)])
    print(model.beta[np.nonzero(model.beta)])
    print(model.coef0)

    # model = GroupPdasLm(path_type="seq", sequence=range(1, 30), ic_type='ebic', is_screening=False, screening_size=30, K_max=10, epsilon=10, powell_path=2, s_min= 1, s_max= 30, lambda_min=0.01, lambda_max=100, is_cv=False, K=3)
    # # model = L0L2Cox(path_type="seq", sequence=range(1,p),lambda_sequence=np.exp(np.linspace(np.log(0.1), np.log(10), 10)),ic_type="ebic")
    # start = time()
    # model.fit(data.x, data.y)
    # stop = time()
    # print("seq: " + str(stop-start))
    # print(np.nonzero(model.beta))
    # print(np.nonzero(data.beta))
    # # print(model.beta[np.nonzero(model.beta)])
    # # print(model.coef0)


    # model = L0L2Lm(path_type="seq", sequence=range(1, min(p, int(n/np.log(n)))), lambda_sequence=[0], ic_type='ebic')
    # start = time()
    # model.fit(data.x, data.y)
    # stop = time()
    # print("L0L2")
    # print("time: " + str(stop-start))
    # # print(np.nonzero(data.beta))
    # # print(np.nonzero(model.beta))
    # # print(data.beta[np.nonzero(data.beta)])
    # print(model.beta[np.nonzero(model.beta)])

    # model = PdasLogistic(path_type="seq", sequence=range(1, min(p, int(n/np.log(n)))), ic_type='ebic')           
#    time_begin = time()
#    model_pdas.fit(data.x, data.y)


    # lambda_sequence = np.exp(np.linspace(np.log(50), np.log(0.01), 50))
    # # model = PdasLogistic(path_type="pgs", s_min=1, s_max=int(min(p, int(n/np.log(n)))), is_cv=True)
    # model = GroupPdasLm(path_type="seq", sequence=range(1, 30), ic_type='ebic', is_screening=False, screening_size=30, K_max=10, epsilon=10, powell_path=2, s_min= 1, s_max= 30, lambda_min=0.01, lambda_max=100, is_cv=True, K=3)
    # start = time()
    # model.fit(data.x, data.y)
    # stop = time()
    # print("Pdas")
    # print("time: " + str(stop-start))
    # print(np.nonzero(data.beta))
    # print(np.nonzero(model.beta))
    # print(data.beta[np.nonzero(data.beta)])
    # print(model.beta[np.nonzero(model.beta)])
    # print(np.sum((model.beta - data.beta)**2))
    # print("-" * 20)
    # mse_list.append(mse(test_data.x,data.beta,model.beta))


    # model = GroupPdasLm(path_type="seq", sequence=range(1, 30), ic_type='ebic', is_screening=False, screening_size=30, K_max=10, epsilon=10, powell_path=2, s_min= 1, s_max= 30, lambda_min=0.01, lambda_max=100, is_cv=True, K=5)
    # start = time()
    # group = np.linspace(1,p,p)
    # model.fit(data.x, data.y, group=group)
    # stop = time()
    # print("GroupPdas")
    # print("time: " + str(stop-start))
    # # print(np.nonzero(data.beta))
    # # print(np.nonzero(model.beta))
    # print(data.beta[np.nonzero(data.beta)])
    # print(model.beta[np.nonzero(model.beta)])
    # # print(np.sum((model.beta - data.beta)**2))
    # # print("-" * 20)
    # # modelbeta2 = model.beta


    # print(np.nonzero(modelbeta1)[0] == np.nonzero(modelbeta2)[0])


# for i in range(50):
#     data = gen_data(500, 300, family="binomial", k=10, rho=0.5)
#     # print(np.nonzero(data.beta))
#     # print(data.beta[np.nonzero(data.beta)])
#     lambdamax = 11.12745
#     lambdamin = 0.01*lambdamax
#     model = L0L2Logistic(path_type="seq", sequence=range(1,20), lambda_sequence=np.linspace(lambdamin, lambdamax), ic_type="gic")
#     model.fit(data.x, data.y)
#     print(1)
#     model = L0L2Logistic(path_type="pgs", s_min=1, s_max=20, lambda_min=lambdamin, lambda_max=lambdamax, ic_type="gic")
#     start = time()
#     model.fit(data.x, data.y)
#     # stop = time()
#     # print("seq: " + str(stop-start))
#     print(np.nonzero(data.beta))
#     print(np.nonzero(model.beta))
#     # print(data.beta[np.nonzero(data.beta)])
#     # print(model.beta[np.nonzero(model.beta)])


#     model = L0L2Lm(path_type="pgs", s_min=1, s_max=20, lambda_min=0.001*26.82 / 100, lambda_max=26.82 / 100, ic_type="gic")
#     start = time()
#     model.fit(data.x, data.y)
#     stop = time()
#     print("powell: " + str(stop-start))

#     print(np.nonzero(model.beta))
#     print(model.beta[np.nonzero(model.beta)])

#     print("-" * 20)





# # ##### 推送测试 #####
# import numpy as np
# np.random.seed(12345)
# x = np.random.normal(0, 1, 200 * 150).reshape((200, 150))
# beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
# noise = np.random.normal(0, 1, 200)
# y = np.matmul(x, beta) + noise
# from bess.linear import PdasLm
# model = PdasLm(path_type="seq", sequence = [5])
# model.fit(X=x, y=y)
# print(np.nonzero(model.beta))
# print(model.beta[:5])

# for i in range(100):
#     import numpy as np
#     # np.random.seed(12345)
#     x = np.random.normal(0, 1, 200 * 150).reshape((200, 150))
#     beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
#     xbeta = np.matmul(x, beta)
#     p = np.exp(xbeta)/(1+np.exp(xbeta))
#     y = np.random.binomial(1, p)
#     from bess.linear import PdasLogistic
#     model = PdasLogistic(path_type="seq", sequence=[5])
#     model.fit(X=x, y=y)
#     print(np.nonzero(model.beta))
#     print(model.beta[:5])



# np.random.seed(11122)
# result_list = []
# ic_list = []
# for i in range(30):
#     x = np.random.normal(0, 1, 300 * 1000).reshape((300, 1000))
#     beta = np.hstack((np.array([1, 1, -1, -1, -1, 1, 1, -1, -1, -1]), np.zeros(990)))
#     lam = np.exp(np.matmul(x, beta))
#     y = np.random.poisson(lam=lam)
#     model = PdasPoisson(path_type="seq", sequence=[10])
#     model.fit(x, y)
#     result_list.append(np.sum(np.array(np.nonzero(model.beta)) < 10))
#     ic_list.append(model.ic)


    # x = np.random.normal(0, 1, 300 * 1000).reshape((300, 1000))
    # beta = np.hstack((np.array([1, 1, -1, -1, -1, 1, 1, -1, -1, -1]), np.zeros(990)))
    # noise = np.random.normal(0, 1, 300)
    # y = np.matmul(x, beta) + noise
    # model = PdasLm(path_type="seq", sequence=[10])
    # model.fit(X=x, y=y)
    # result_list.append(np.sum(np.array(np.nonzero(model.beta)) < 10))
    # ic_list.append(model.ic)

    # x = np.random.normal(0, 1, 100 * 1000).reshape((100, 1000))
    # beta = np.hstack((np.array([1, 1, -1, -1, -1, 1, 1, -1, -1, -1]), np.zeros(990)))
    # xbeta = np.matmul(x, beta)
    # p = np.exp(xbeta)/(1+np.exp(xbeta))
    # y = np.random.binomial(1, p)
    # # print(x[:3, :6])
    # # print(beta[:10])
    # # print(y[:3])
    # model = PdasLogistic(path_type="seq", sequence=[10])
    # model.fit(X=x, y=y)
    # result_list.append(np.sum(np.array(np.nonzero(model.beta)) < 10))
    # ic_list.append(model.ic)

# print(np.mean(result_list) / 10)
# print(np.mean(ic_list))


# np.random.seed(12345)
# for i in range(1000):
#     x = np.random.normal(0, 1, 200 * 600).reshape((200, 600))
#     beta = np.hstack((np.ones(5), -1 * np.ones(5),  np.zeros(400)))
#     y = np.matmul(x, beta)
#     model = PdasLm(path_type="seq", sequence=[5])
#     model.fit(X=x, y=y)


# x = np.random.normal(0, 1, 100 * 1000).reshape((100, 1000))
# beta = np.hstack((np.array([1, 1, -1, -1, -1, 1, 1, -1, -1, -1]), np.zeros(990)))
# noise = np.random.normal(0, 1, 100)
# y = np.matmul(x, beta)
# model = PdasLm(path_type="seq", sequence=range(1, 100))
# model.fit(X=x, y=y)
# for i in np.array(np.nonzero(model.beta)):
#     print("beta: " + str(model.beta[i]))
#     print("特征选择 ： " + str(np.array(np.nonzero(model.beta))))
# print("aic: "  + str(model.ic))
# print("train_loss: " + str(model.train_loss))


# np.random.seed(12345)
# x = np.random.normal(0, 1, 100 * 1000).reshape((100, 1000))
# beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(995)))
# xbeta = np.matmul(x, beta)
# p = np.exp(xbeta)/(1+np.exp(xbeta))
# y = np.random.binomial(1, p)
# model = PdasLogistic(path_type="seq", sequence=range(1, 100))
# model.fit(X=x, y=y)
# for i in np.array(np.nonzero(model.beta)):
#     print(model.beta[i])
#     print(np.array(np.nonzero(model.beta)))


### Sparsity unknown

# for i in range(100):
#     x = np.random.normal(0, 1, 100 * 5).reshape((100, 5))
#     # beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(195)))
#     beta = np.array([100, 100, -100, -100, -100])
#     xbeta = np.matmul(x, beta)
#     p = np.exp(xbeta)/(1+np.exp(xbeta))
#     y = np.random.binomial(1, p)
#     model = PdasLogistic(path_type="seq", sequence=[5], ic_type="gic", is_warm_start=True)
#     model.fit(X=x, y=y, is_normal=True)
#
#     xbeta = np.matmul(x, model.beta)
#     xbeta[xbeta>30] = 30
#     xbeta[xbeta < -30] = -30
#     p = np.exp(xbeta) / (1 + np.exp(xbeta))
#     print(np.sum(y * np.log(p) + (np.ones(100) - y) * np.log(np.ones(100) - p)))
#     # print(p[:5])
#     if abs(np.sum(y * np.log(p) + (np.ones(100) - y) * np.log(np.ones(100) - p))) < 10:
#         print("find")
#         print(np.sum(y * np.log(p) + (np.ones(100) - y) * np.log(np.ones(100) - p)))

        # for j in np.array(np.nonzero(model.beta)):
        #     print(model.beta[i])
#


# np.random.seed(12345)
# x = np.random.normal(0, 1, 100 * 200).reshape((100, 200))
# beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(195)))
# y = np.matmul(x, beta)
# model = PdasLm(path_type="seq", sequence=range(1, 10), ic_type="gic")
# model.fit(X=x, y=y)
# print(np.array(np.nonzero(model.beta)))
# # print(1)
# print(model.coef0)
# print(model.train_loss)
# print(model.ic)
# print(model.beta[:10])
# self.nullloss
# print(model.A)
# print(model.l)
# print(model.fact_path_size)
# print(model.beta_matrix)
# print(model.coef0_sequence)
# print(model.train_loss_sequence)
# print(model.ic_sequence)
# print(model.l_sequence)
# print(model.A_matrix)
# for i in np.array(np.nonzero(model.beta)):
#     print(model.beta[i])

# data = pd.read_csv("G:\\bess-0.0.8\\test\\data.csv")
# print(data)
# x = data.iloc[:,:5].values
# y = data.iloc[:,5].values
# model = PdasLogistic(path_type="seq", sequence=range(1,6), ic_type="gic", is_warm_start=True)
# model.fit(X=x, y=y, is_normal=True)
# for i in np.array(np.nonzero(model.beta)):
#     print(model.beta[i])
#
# xbeta = np.matmul(x, model.beta) + np.ones(100) * model.coef0
# xbeta[xbeta>30] = 30
# xbeta[xbeta<-30] = -30
# p = np.exp(xbeta)/(1+np.exp(xbeta))
# # print(p[:5])
# print(np.sum(y*np.log(p) + (np.ones(100) - y)*np.log(np.ones(100) - p)))

# print(os.getcwd())




# data = pd.read_csv("data.csv")
# print(data)
# x = data.iloc[:100, :200].values
# y = data.iloc[:100, 200].values
# beta = data.iloc[:, 201].values
# model = PdasLm(path_type="seq", sequence=range(1, 10), ic_type="gic", is_warm_start=True)
# model.fit(x, y, is_normal=True)
# print(np.mean(x, axis=0))
# print(np.std(x, axis=0))
# print(beta[np.nonzero(beta)])

# np.random.seed(12345)
# data = gen_data(1000, 20, family="gaussian", k=4, rho=0.5, sigma=1)
# model = L0L2Lm(path_type="seq", sequence=range(1,20), lambdaic_type="gic")
# model.fit(data.x, data.y)
# print(np.nonzero(data.beta))
# print(np.nonzero(model.beta))
# print(data.beta[np.nonzero(data.beta)])
# print(model.beta[np.nonzero(model.beta)])
# print(model.train_loss)

# model = PdasLogistic(path_type="seq", sequence=range(1,50), ic_type="ebic", is_warm_start=True)
# model.fit(data.x, data.y)
# print(np.nonzero(data.beta))
# print(np.nonzero(model.beta))
# print(data.beta[np.nonzero(data.beta)])
# print(model.beta[np.nonzero(model.beta)])
# print(model.train_loss)


# np.random.seed(12345)
# data = gen_data(10000, 20000, family="gaussian", k=20, rho=0, sigma=1)
# model = PdasLm(path_type="gs", s_min=1, s_max=50, ic_type="ebic")
# model.fit(data.x, data.y)
# print(np.nonzero(data.beta))
# print(np.nonzero(model.beta))
# print(data.beta[np.nonzero(data.beta)])
# print(model.beta[np.nonzero(model.beta)])


# np.random.seed(12345)
# data = gen_data(10000, 20000, family="binomial", k=20, rho=0, sigma=1)
# model = PdasLogistic(path_type="gs", s_min=1, s_max=50, ic_type="ebic", is_warm_start=True)
# model.fit(data.x, data.y)
# print(np.nonzero(data.beta))
# print(np.nonzero(model.beta))
# print(data.beta[np.nonzero(data.beta)])
# print(model.beta[np.nonzero(model.beta)])


# np.random.seed(123)
# data = gen_data(100, 1000, family="poisson", k=5, rho=0, sigma=1)
# model = PdasPoisson(path_type="seq", sequence=range(1,11))
# model.fit(data.x, data.y)
# print(np.nonzero(data.beta))
# print(np.nonzero(model.beta))
# print(data.beta[np.nonzero(data.beta)])
# print(model.beta[np.nonzero(model.beta)])

# np.random.seed(12345)
# data = gen_data(100, 200, family="cox", k=5, rho=0, sigma=1, c=10)
# print(data.y)
# print(data.x.shape)


# data = pd.read_csv("cox_data.csv")
# print(data.head())
# x = data.iloc[:100, :200].values
# y = data.iloc[:100, 200:202].values
# beta = data.iloc[:,-1].values
#
# model = PdasCox(path_type="seq", sequence=range(1,11))
# model.fit(x, y, is_normal=True)
# print(np.nonzero(beta))
# print(np.nonzero(model.beta))
# print(beta[np.nonzero(beta)])
# print(model.beta[np.nonzero(model.beta)])

# import time
# data = gen_data(200, 100, family="gaussian", k=5, rho=0, sigma=1, c=10)
# print(data.y)
# print(data.x.shape)
#
# model = PdasLm(path_type="seq", sequence=range(1,10), is_cv=True)
# start = time.time()
# model.fit(data.x, data.y)
# print(time.time() - start)
#
# print(np.nonzero(data.beta))
# print(np.nonzero(model.beta))
# print(data.beta[np.nonzero(data.beta)])
# print(model.beta[np.nonzero(model.beta)])


# np.random.seed(123)
# data = gen_data(100, 200, family="cox", k=5, rho=0, sigma=1)
#
# model = PdasCox(path_type="seq", sequence=[5])
# model.fit(data.x, data.y, is_normal=True)
# print(np.nonzero(data.beta))
# print(np.nonzero(model.beta))
# print(data.beta[np.nonzero(data.beta)])
# print(model.beta[np.nonzero(model.beta)])


# ### PdasPoisson sample
# from bess.linear import *
# import numpy as np
# np.random.seed(12345)
# x = np.random.normal(0, 1, 100 * 1000).reshape((100, 1000))
# # print(x[:3, :5])
# beta = np.hstack((np.array([1, 1, 1, -1, -1]), np.zeros(995)))
# lam = np.exp(np.matmul(x, beta))
# # print(lam)
# y = np.random.poisson(lam=lam)
# # print(y)
#
# ### Sparsity known
# model = PdasPoisson(path_type="seq", sequence=range(1,11), ic_type='ebic')
# model.fit(X=x, y=y)
# # print(np.nonzero(data.beta))
# print(np.nonzero(model.beta))
# # print(data.beta[np.nonzero(data.beta)])
# print(model.beta[np.nonzero(model.beta)])
# print(model.coef0)

# from bess.linear import *
# import numpy as np
# np.random.seed(123)
# x = np.random.normal(0, 1, 100 * 1000).reshape((100, 1000))
# # print(x[:3, :5])
# beta = np.hstack((np.array([1, 1, 1, -1, -1]), np.zeros(995)))
#
# time = np.power(-np.log(np.random.uniform(0, 1, 100)) / np.exp(np.matmul(x, beta)), 1/10)
# ctime = 10 * np.random.uniform(0, 1, 100)
# status = (time < ctime) * 1
# for i in range(100):
#     time[i] = min(time[i], ctime[i])
# y = np.hstack((time.reshape((-1, 1)), status.reshape((-1, 1))))
#
# model = PdasCox(path_type="seq", sequence=range(1,11))
# model.fit(x, y)
# print(np.nonzero(beta))
# print(np.nonzero(model.beta))
# print(beta[np.nonzero(beta)])
# print(model.beta[np.nonzero(model.beta)])


# np.random.seed(123)
# x = np.random.normal(0, 1, 100 * 1000).reshape((100, 1000))
# beta = np.hstack((np.array([1, 1, 1, -1, -1]), np.zeros(995)))
# time = np.power(-np.log(np.random.uniform(0, 1, 100)) / np.exp(np.matmul(x, beta)), 1/10)
# ctime = 10 * np.random.uniform(0, 1, 100)
# status = (time < ctime) * 1
# for i in range(100):
#     time[i] = min(time[i], ctime[i])
# y = np.hstack((time.reshape((-1, 1)), status.reshape((-1, 1))))



# ### real data
# data = pd.read_csv("C:\\Users\\jtwok\\Documents\\WeChat Files\\wxid_70qqws672a5322\\FileStorage\\File\\2020-06\\Bike-Sharing-Dataset\\day.csv")
# print(data)
# x = data.iloc[:, 2:15].values
# y = data.iloc[:, 15].values
#
# print(x)
# print(y)
#
# model = PdasPoisson(path_type="seq", sequence=range(1,10))
# model.fit(x, y, is_normal=True)
# print(np.nonzero(model.beta))
# print(model.beta[np.nonzero(model.beta)])









###### 说明文档测试 ######
#
# ### PdasLm sample
# from bess.linear import *
# import numpy as np
#
# np.random.seed(12345)   # fix seed to get the same result
# x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
# beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
# noise = np.random.normal(0, 1, 100)
# y = np.matmul(x, beta) + noise
#
# ### Sparsity known
# model = PdasLm(path_type="seq", sequence=[5])
# model.fit(X=x, y=y)
# model.predict(x)
#
# ### Sparsity unknown
# # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
# model = PdasLm(path_type="seq", sequence=range(1,10))
# model.fit(X=x, y=y)
# model.predict(x)
#
# # # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
# # model = PdasLm(path_type="gs")
# # model.fit(X=x, y=y)
# # model.predict(x)
#
#
# ### PdasLogistic sample
# from bess.linear import *
# import numpy as np
# np.random.seed(12345)
# x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
# beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
# xbeta = np.matmul(x, beta)
# p = np.exp(xbeta)/(1+np.exp(xbeta))
# y = np.random.binomial(1, p)
#
# ### Sparsity known
# model = PdasLogistic(path_type="seq", sequence=[5])
# model.fit(X=x, y=y)
# model.predict(x)
#
# ### Sparsity unknown
# # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
# model = PdasLogistic(path_type="seq", sequence=range(1,10))
# model.fit(X=x, y=y)
# model.predict(x)
#
# # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
# model = PdasLogistic(path_type="gs")
# model.fit(X=x, y=y)
# model.predict(x)
#
#
# ### PdasPoisson sample
# from bess.linear import *
# import numpy as np
# np.random.seed(12345)
# x = np.random.normal(0, 1, 100 * 150).reshape((100, 150))
# beta = np.hstack((np.array([1, 1, -1, -1, -1]), np.zeros(145)))
# lam = np.exp(np.matmul(x, beta))
# y = np.random.poisson(lam=lam)
#
# ### Sparsity known
# model = PdasPoisson(path_type="seq", sequence=[5])
# model.fit(X=x, y=y)
# model.predict(x)
#
# ### Sparsity unknown
# # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
# model = PdasPoisson(path_type="seq", sequence=range(1,10))
# model.fit(X=x, y=y)
# model.predict(x)
#
# # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
# model = PdasPoisson(path_type="gs")
# model.fit(X=x, y=y)
# model.predict(x)
#
#
# ### PdasCox sample
# from bess.linear import *
# from bess.gen_data import gen_data
# import numpy as np
# np.random.seed(12345)
# data = gen_data(100, 200, family="cox", k=5, rho=0, sigma=1, c=10)
#
# ### Sparsity known
# model = PdasCox(path_type="seq", sequence=[5])
# model.fit(data.x, data.y, is_normal=True)
# model.predict(data.x)
#
# ### Sparsity unknown
# # path_type="seq", Default:sequence=[1,2,...,min(x.shape[0], x.shape[1])]
# model = PdasCox(path_type="seq", sequence=range(1,10))
# model.fit(data.x, data.y)
# model.predict(data.x)
#
# # path_type="gs", Default:s_min=1, s_max=X.shape[1], K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
# model = PdasCox(path_type="gs")
# model.fit(data.x, data.y)
# model.predict(data.x)



