import numpy as np
from bess.linear import PdasLm, PdasLogistic, PdasPoisson, PdasCox, GroupPdasCox
from bess.gen_data import gen_data
import pandas as pd
from time import time
import logging
import scipy
import multiprocessing
#from glmnet import ElasticNet, LogitNet
# import pycasso
# from sklearn.model_selection import KFold
# import glmnet_python
# from glmnet import glmnet; from glmnetPlot import glmnetPlot
# from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
# from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
# from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
import warnings
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

# def CV(data, kfold, lamb):
#     kf = KFold(n_splits=kfold)
#     kf.get_n_splits(data.x) # returns the number of splitting iterations in the cross-validator
#     pre_loss = []
#     for train_index, test_index in kf.split(data.x):
#         # print("TRAIN:", train_index, "TEST:", test_index)
#         x_train, x_test = data.x[train_index], data.x[test_index]
#         y_train, _ = data.y[train_index], data.y[test_index]
#         fit = glmnet(x=x_train.copy(), y=y_train.copy(), family="cox", lambdau=np.array([lamb]), nlambda=1)
#         model_beta = fit["beta"].reshape(-1)      
#         pre_loss.append(mse(x_test, data.beta, model_beta))
#     return np.mean(pre_loss)

def mse(x,Tbeta,beta):
    xTbeta = np.matmul(x, Tbeta)
    xbeta = np.matmul(x, beta) 
    return np.sum((xTbeta - xbeta)**2) / np.sum(xTbeta**2)

def beta2result(testx, Tbeta, beta):
    coef_mse = np.mean(np.square(Tbeta[np.nonzero(Tbeta)] - beta[np.nonzero(Tbeta)]))
    fact_var_list = np.array(np.nonzero(Tbeta))[0]
    pre_var_list = np.array(np.nonzero(beta))[0]
    TP = 0
    for v in pre_var_list:
        if v in fact_var_list:
            TP += 1
    FP = len(pre_var_list) - TP
    FN = len(fact_var_list) - TP
    TN = p - len(pre_var_list) - (len(fact_var_list) - TP)
    mcc = (TP * TN - FP * FN) / np.sqrt(
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            
    pre_loss = mse(testx,Tbeta,beta)
    return [TP, FP, mcc, pre_loss]

def sim(n,p,k,j,family,sigma,rho):
    np.random.seed(j)
    logging.info("n = " + str(n) + "p = " + str(p) + "k = " + str(k) + "j = " + str(j))
    # print("n = " + str(n) + "p = " + str(p) + "k = " + str(k) + " j= " + str(j))
    data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma, c=50, scal=10)
    test_data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma, beta=data.beta, c=50, scal=10)
    
    model_pdas = PdasCox(path_type="seq", sequence=range(1, min(p, int(144))), ic_type="ebic")         
    time_begin = time()
    model_pdas.fit(data.x, data.y)
    t_pdas = time() - time_begin
    pdas_result = beta2result(test_data.x, data.beta, model_pdas.beta)
    pdas_result.append(t_pdas)
            
    model_gpdas = PdasCox(path_type="pgs", s_min=1, s_max=int(min(p, int(144))), ic_type="ebic")
    time_begin = time()
    model_gpdas.fit(data.x, data.y)
    t_gpdas = time() - time_begin
    gpdas_result = beta2result(test_data.x, data.beta, model_gpdas.beta)
    gpdas_result.append(t_gpdas)
    
    #########################################################################    
    # logging.info("lasso")
    # time_begin = time()
    # lamb_list = np.exp(np.linspace(np.log(0.001), np.log(100), 100))
    # cv_result_list = []
    # for lamb in lamb_list:
    #     cv_result_list.append(CV(data, 5, lamb))
    # lamb_best = lamb_list[cv_result_list.index(min(cv_result_list))]
    # fit = glmnet(x=data.x, y=data.y, family=family, lambdau= np.array([lamb_best]), nlambda=1)
    # model_beta = fit["beta"].reshape(-1)
    # t_lasso = time() - time_begin
    # lasso_result = beta2result(test_data.x, data.beta, model_beta)
    # lasso_result.append(t_lasso)

    # logging.info("lasso")
    # time_begin = time()
    # lamb_list = np.exp(np.linspace(np.log(0.001), np.log(10), 100))
    # cv = KFold(n_splits=5, shuffle=True, random_state=0)
    # y_surv = []
    # for yi in  data.y:
    #     y_surv.append((bool(yi[1]), yi[0]))
    # y_surv = np.array(y_surv, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    # cv = KFold(n_splits=5, shuffle=True, random_state=0)
    # gcv = GridSearchCV(
    #     make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=1.0, max_iter=10000)),
    #     param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in lamb_list]},
    #     cv=cv,
    #     error_score=0.5,
    #     n_jobs=5).fit(data.x, y_surv)
    # model_beta = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"].coef_.reshape(-1)
    # t_lasso = time() - time_begin
    # lasso_result = beta2result(test_data.x, data.beta, model_beta)
    # lasso_result.append(t_lasso)

    # return  np.array([pdas_result, gpdas_result, lasso_result]).reshape(-1)
    return np.array([pdas_result, gpdas_result]).reshape(-1)
    # return np.array([lasso_result]).reshape(-1)

if __name__ == "__main__":
    parameter_len = 3
    family = "cox"
    sigma = 1
    rho = 0.5
    sim_time = 10

    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='simulate_' + family + '_' + str(parameter_len) + '_'+str(sim_time) + '_pdas.log',
                filemode='w')

    file_name= 'simulate_' + family + '_' + str(parameter_len) + '_'+ str(sim_time) + '_pdas.csv'
    with open(file_name,'a') as f:
        f.write(",TP_pdas_list,FP_pdas_list,mcc_pdas_list,pre_loss_pdas_list,t_pdas_list,TP_gpdas_list,FP_gpdas_list,mcc_gpdas_list,pre_loss_gpdas_list,t_gpdas_list,TP_lasso_list,FP_lasso_list,mcc_lasso_list,pre_loss_lasso_list,t_lasso_list")
        f.write("\n")
    for i in range(parameter_len):
        n = 1000
        p = 10 ** (i + 2)
        k = 20
        sim_result = []
        
        # pool = multiprocessing.Pool(processes = 4)

        for j in range(sim_time):
            sim_result.append(sim(n=n, p=p, k=k, j=j, family=family, sigma=sigma, rho=rho))
            # sim_result.append(pool.apply_async(sim, (n,p,k,j,family,sigma,rho)))

        sim_result_array = np.array(sim_result)       
        with open(file_name,'a') as f:        
            f.write("n=" + str(n) + "p=" + str(p) + "k=" + str(k))
            for col in range(sim_result_array.shape[1]):
                f.write(",")
                f.write(str(np.mean(sim_result_array[:, col])))
            f.write("\n")

