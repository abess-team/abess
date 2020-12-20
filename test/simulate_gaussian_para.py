import numpy as np
from bess.linear import PdasLm, PdasLogistic, PdasPoisson ,PdasCox
from bess.gen_data import gen_data
import pandas as pd
from time import time
import logging
import scipy
import multiprocessing

import glmnet_python
from glmnet import glmnet; from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict

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
    logging.info("n = " + str(n) + "p = " + str(p) + "k = " + str(k) + "j = " + str(j))
    # print("n = " + str(n) + "p = " + str(p) + "k = " + str(k) + " j= " + str(j))
    np.random.seed(j)
    data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma)
    test_data = gen_data(n, p, family=family, k=k, rho=rho, sigma=sigma, beta=data.beta)
    
    # # model_pdas = PdasLm(path_type="seq", sequence=range(1, min(p, int(n/np.log(n)))), ic_type="ebic")
    # model_pdas = PdasLm(path_type="seq", sequence=range(1, min(p, int(n/np.log(n)))), is_cv=True, K=5)           
    # time_begin = time()
    # model_pdas.fit(data.x, data.y)
    # t_pdas = time() - time_begin
    # pdas_result = beta2result(test_data.x, data.beta, model_pdas.beta)
    # pdas_result.append(t_pdas)
            
    # # model_gpdas = PdasLm(path_type="pgs", s_min=1, s_max=int(min(p, int(n/np.log(n)))), ic_type="ebic")
    # model_gpdas = PdasLm(path_type="pgs", s_min=1, s_max=int(min(p, int(n/np.log(n)))), is_cv=True, K=5)
    # time_begin = time()
    # model_gpdas.fit(data.x, data.y)
    # t_gpdas = time() - time_begin
    # gpdas_result = beta2result(test_data.x, data.beta, model_gpdas.beta)
    # gpdas_result.append(t_gpdas)
    
    ##########################################################################    
    logging.info("lasso")
    time_begin = time()
    model_lasso = cvglmnet(x=data.x.copy(), y=np.array(data.y.copy(), dtype=np.float), family=family, lambda_min=np.array([0.01]), nlambda=int(n/np.log(n)), nfolds=5)       
    t_lasso = time() - time_begin
    model_lasso_beta = cvglmnetCoef(model_lasso, s='lambda_min')[1:].reshape(-1)
    # print(model_glmnet_beta)
    lasso_result = beta2result(test_data.x, data.beta, model_lasso_beta)
    lasso_result.append(t_lasso)

    # return  np.array([pdas_result, gpdas_result, lasso_result]).reshape(-1)
    # return  np.array([pdas_result, gpdas_result]).reshape(-1)
    return np.array([lasso_result]).reshape(-1)


if __name__ == "__main__":
    parameter_len = 10
    family = "gaussian"
    sigma = 9
    rho = 0.5
    sim_time = 100

    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='simulate_' + family + '_' + str(parameter_len) + '_'+str(sim_time) + '.log',
                filemode='w')

    file_name= 'simulate_' + family + '_' + str(parameter_len) + '_'+str(sim_time) + '.csv'
    with open(file_name,'a') as f:
        f.write(",TP_pdas_list,FP_pdas_list,mcc_pdas_list,pre_loss_pdas_list,t_pdas_list,TP_gpdas_list,FP_gpdas_list,mcc_gpdas_list,pre_loss_gpdas_list,t_gpdas_list,TP_lasso_list,FP_lasso_list,mcc_lasso_list,pre_loss_lasso_list,t_lasso_list")
        f.write("\n")
    for i in range(parameter_len):
        n = 1000
        p = (i + 1) * 200
        k = 20   
        sim_result = []

        # pool = multiprocessing.Pool(processes = 4)
        
        for j in range(sim_time):
            # sim_result.append(pool.apply_async(sim, (n,p,k,j,family,sigma,rho,)))
            sim_result.append(sim(n=n, p=p, k=k, j=j, family=family, sigma=sigma, rho=rho))

        sim_result_array = np.array(sim_result)       
        with open(file_name,'a') as f:        
            f.write("n=" + str(n) + "p=" + str(p) + "k=" + str(k))
            for col in range(sim_result_array.shape[1]):
                f.write(",")
                f.write(str(np.mean(sim_result_array[:, col])))
            f.write("\n")

