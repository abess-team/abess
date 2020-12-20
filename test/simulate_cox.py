import numpy as np
from bess.linear import PdasLm, PdasLogistic, PdasPoisson ,PdasCox
from bess.gen_data import gen_data
import pandas as pd
from time import time
#from glmnet import ElasticNet, LogitNet
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
import logging


def mse(x,Tbeta,beta):
    xTbeta = np.matmul(x, Tbeta)
    xbeta = np.matmul(x, beta)  
    return np.sum((xTbeta - xbeta)**2) / np.sum(xTbeta**2)

def acc(y, y_):
    return np.sum(y==y_) / len(y)

def CV(model, data, kfold):
    kf = KFold(n_splits=kfold)
    kf.get_n_splits(data.x) # returns the number of splitting iterations in the cross-validator
    pre_loss = []
    for train_index, test_index in kf.split(data.x):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = data.x[train_index], data.x[test_index]
        y_train, _ = data.y[train_index], data.y[test_index]

        survival = pd.DataFrame()
        for i in range(x_train.shape[1]):
            survival["Var" + str(i)] = x_train[:,i]
        
        survival["T"] = y_train[:,0]
        survival["E"] = y_train[:,1]

        model.fit(survival, 'T', event_col='E')
        
        pre_loss.append(mse(x_test, data.beta, model.params_))
    return np.mean(pre_loss)

family = "cox"
parameter_len = 10
sim_time = 100

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='simulate_' + family + '_' + str(parameter_len) + '_'+str(sim_time) + '_1.log',
                filemode='w')

t_pdas_list = np.zeros(parameter_len)
TP_pdas_list = np.zeros(parameter_len)
TN_pdas_list = np.zeros(parameter_len)
FP_pdas_list = np.zeros(parameter_len)
FN_pdas_list = np.zeros(parameter_len)
ic_pdas_list = np.zeros(parameter_len)
loss_pdas_list = np.zeros(parameter_len)
mcc_pdas_list = np.zeros(parameter_len)
coef_mse_pdas_list = np.zeros(parameter_len)
pre_loss_pdas_list = np.zeros(parameter_len)

t_std_pdas_list = np.zeros(parameter_len)
TP_std_pdas_list = np.zeros(parameter_len)
TN_std_pdas_list = np.zeros(parameter_len)
FP_std_pdas_list = np.zeros(parameter_len)
FN_std_pdas_list = np.zeros(parameter_len)
ic_std_pdas_list = np.zeros(parameter_len)
loss_std_pdas_list = np.zeros(parameter_len)
mcc_std_pdas_list = np.zeros(parameter_len)
coef_mse_std_pdas_list = np.zeros(parameter_len)
pre_loss_std_pdas_list = np.zeros(parameter_len)


t_gpdas_list = np.zeros(parameter_len)
TP_gpdas_list = np.zeros(parameter_len)
TN_gpdas_list = np.zeros(parameter_len)
FP_gpdas_list = np.zeros(parameter_len)
FN_gpdas_list = np.zeros(parameter_len)
ic_gpdas_list = np.zeros(parameter_len)
loss_gpdas_list = np.zeros(parameter_len)
mcc_gpdas_list = np.zeros(parameter_len)
coef_mse_gpdas_list = np.zeros(parameter_len)
pre_loss_gpdas_list = np.zeros(parameter_len)

t_std_gpdas_list = np.zeros(parameter_len)
TP_std_gpdas_list = np.zeros(parameter_len)
TN_std_gpdas_list = np.zeros(parameter_len)
FP_std_gpdas_list = np.zeros(parameter_len)
FN_std_gpdas_list = np.zeros(parameter_len)
ic_std_gpdas_list = np.zeros(parameter_len)
loss_std_gpdas_list = np.zeros(parameter_len)
mcc_std_gpdas_list = np.zeros(parameter_len)
coef_mse_std_gpdas_list = np.zeros(parameter_len)
pre_loss_std_gpdas_list = np.zeros(parameter_len)


t_glmnet_list = np.zeros(parameter_len)
TP_glmnet_list = np.zeros(parameter_len)
TN_glmnet_list = np.zeros(parameter_len)
FP_glmnet_list = np.zeros(parameter_len)
FN_glmnet_list = np.zeros(parameter_len)
ic_glmnet_list = np.zeros(parameter_len)
loss_glmnet_list = np.zeros(parameter_len)
mcc_glmnet_list = np.zeros(parameter_len)
coef_mse_glmnet_list = np.zeros(parameter_len)
pre_loss_glmnet_list = np.zeros(parameter_len)

t_std_glmnet_list = np.zeros(parameter_len)
TP_std_glmnet_list = np.zeros(parameter_len)
TN_std_glmnet_list = np.zeros(parameter_len)
FP_std_glmnet_list = np.zeros(parameter_len)
FN_std_glmnet_list = np.zeros(parameter_len)
ic_std_glmnet_list = np.zeros(parameter_len)
loss_std_glmnet_list = np.zeros(parameter_len)
mcc_std_glmnet_list = np.zeros(parameter_len)
coef_mse_std_glmnet_list = np.zeros(parameter_len)
pre_loss_std_glmnet_list = np.zeros(parameter_len)


#np.random.seed(12345)
for i in range(parameter_len):
    n = 1000
    # p = 20 + i * 10 
    # k = 4

    p = (i +1) * 200
    k = 20

    sim_time = 10
    
    t_pdas = np.zeros(sim_time)
    TP_pdas = np.zeros(sim_time)
    TN_pdas = np.zeros(sim_time)
    FP_pdas = np.zeros(sim_time)
    FN_pdas = np.zeros(sim_time)
    ic_pdas = np.zeros(sim_time)
    loss_pdas = np.zeros(sim_time)
    mcc_pdas = np.zeros(sim_time)
    coef_mse_pdas = np.zeros(sim_time)
    pre_loss_pdas = np.zeros(sim_time)
     
    t_gpdas = np.zeros(sim_time)
    TP_gpdas = np.zeros(sim_time)
    TN_gpdas = np.zeros(sim_time)
    FP_gpdas = np.zeros(sim_time)
    FN_gpdas = np.zeros(sim_time)
    ic_gpdas = np.zeros(sim_time)
    loss_gpdas = np.zeros(sim_time)
    mcc_gpdas = np.zeros(sim_time)
    coef_mse_gpdas = np.zeros(sim_time)
    pre_loss_gpdas = np.zeros(sim_time)
    
    t_glmnet = np.zeros(sim_time)
    TP_glmnet = np.zeros(sim_time)
    TN_glmnet = np.zeros(sim_time)
    FP_glmnet = np.zeros(sim_time)
    FN_glmnet = np.zeros(sim_time)
    ic_glmnet = np.zeros(sim_time)
    loss_glmnet = np.zeros(sim_time)
    mcc_glmnet = np.zeros(sim_time)
    coef_mse_glmnet = np.zeros(sim_time)
    pre_loss_glmnet = np.zeros(sim_time)
    
    for j in range(sim_time):
        logging.info("n = " + str(n) + "p = " + str(p) + "k = " + str(k) + "j = " + str(j))
        # print("n = " + str(n) + "p = " + str(p) + "k = " + str(k) + "j= " + str(j))
        data = gen_data(n, p, family=family, k=k, rho=0.50, sigma = 1)
        test_data = gen_data(n, p, family=family, k=k, rho=0.50, sigma = 1, beta=data.beta)
        
        model_pdas = PdasCox(path_type="seq", sequence=range(1, min(int(n/np.log(n)), p)), ic_type="ebic")         
        time_begin = time()
        model_pdas.fit(data.x, data.y)
        t_pdas[j] = time() - time_begin
        coef_mse_pdas[j] = np.mean(np.square(data.beta[np.nonzero(data.beta)] - model_pdas.beta[np.nonzero(data.beta)]))
        fact_var_pdas_list = np.array(np.nonzero(data.beta))[0]
        pre_var_pdas_list = np.array(np.nonzero(model_pdas.beta))[0]
        for v in pre_var_pdas_list:
             if v in fact_var_pdas_list:
                 TP_pdas[j] += 1
        FP_pdas[j] = len(pre_var_pdas_list) - TP_pdas[j]
        FN_pdas[j] = len(fact_var_pdas_list) - TP_pdas[j]
        TN_pdas[j] = p - len(pre_var_pdas_list) - (len(fact_var_pdas_list) - TP_pdas[j])
        ic_pdas[j] = model_pdas.ic
        loss_pdas[j] = model_pdas.train_loss
        mcc_pdas[j] = (TP_pdas[j] * TN_pdas[j] - FP_pdas[j] * FN_pdas[j]) / np.sqrt(
             (TP_pdas[j] + FP_pdas[j]) * (TP_pdas[j] + FN_pdas[j]) * (TN_pdas[j] + FP_pdas[j]) * (TN_pdas[j] + FN_pdas[j]))
        pre_loss_pdas[j] = mse(test_data.x,data.beta,model_pdas.beta)
        
        
        model_gpdas = PdasCox(path_type="pgs", s_min=1, s_max=int(min(int(n/np.log(n)), p)), ic_type="ebic")
        time_begin = time()
        model_gpdas.fit(data.x, data.y)
        t_gpdas[j] = time() - time_begin
        coef_mse_gpdas[j] = np.mean(np.square(data.beta[np.nonzero(data.beta)] - model_gpdas.beta[np.nonzero(data.beta)]))
        fact_var_gpdas_list = np.array(np.nonzero(data.beta))[0]
        pre_var_gpdas_list = np.array(np.nonzero(model_gpdas.beta))[0]
        for v in pre_var_gpdas_list:
            if v in fact_var_gpdas_list:
                TP_gpdas[j] += 1
        FP_gpdas[j] = len(pre_var_gpdas_list) - TP_gpdas[j]
        FN_gpdas[j] = len(fact_var_gpdas_list) - TP_gpdas[j]
        TN_gpdas[j] = p - len(pre_var_gpdas_list) - (len(fact_var_gpdas_list) - TP_gpdas[j])
        ic_gpdas[j] = model_gpdas.ic
        loss_gpdas[j] = model_gpdas.train_loss
        mcc_gpdas[j] = (TP_gpdas[j] * TN_gpdas[j] - FP_gpdas[j] * FN_gpdas[j]) / np.sqrt(
            (TP_gpdas[j] + FP_gpdas[j]) * (TP_gpdas[j] + FN_gpdas[j]) * (TN_gpdas[j] + FP_gpdas[j]) * (TN_gpdas[j] + FN_gpdas[j]))
        pre_loss_gpdas[j] = mse(test_data.x,data.beta,model_gpdas.beta)
        
    #     time_begin = time()
    #     lamb_list = np.exp(np.linspace(np.log(0.001), np.log(100), 100))
    #     cv_result_list = []
    #     for lamb in lamb_list:
    #         cph = CoxPHFitter(penalizer=lamb, l1_ratio=1.0)
    #         cv_result_list.append(CV(cph, data, 5))
    #     lamb_best = lamb_list[cv_result_list.index(min(cv_result_list))]
    #     cph = CoxPHFitter(penalizer=lamb_best, l1_ratio=1.0)
    #     survival = pd.DataFrame()
    #     for col in range(data.x.shape[1]):
    #         survival["Var" + str(col)] = data.x[:,col]    
    #     survival["T"] = data.y[:,0]
    #     survival["E"] = data.y[:,1]
    #     cph.fit(survival, 'T', event_col='E')
    #     t_glmnet[j] = time() - time_begin

    #     model_glmnet_beta = cph.params_.values
    #     coef_mse_glmnet[j] = np.mean(np.square(data.beta[np.nonzero(data.beta)] - model_glmnet_beta[np.nonzero(data.beta)]))
    #     fact_var_glmnet_list = np.array(np.nonzero(data.beta))[0]
    #     pre_var_glmnet_list = np.array(np.nonzero(model_glmnet_beta))[0]
    #     for v in pre_var_glmnet_list:
    #         if v in fact_var_glmnet_list:
    #             TP_glmnet[j] += 1
    #     FP_glmnet[j] = len(pre_var_glmnet_list) - TP_glmnet[j]
    #     FN_glmnet[j] = len(fact_var_glmnet_list) - TP_glmnet[j]
    #     TN_glmnet[j] = p - len(pre_var_glmnet_list) - (len(fact_var_glmnet_list) - TP_glmnet[j])
    # #        ic_glmnet[j] = model_glmnet.ic
    # #        loss_glmnet[j] = model_glmnet.train_loss
    #     mcc_glmnet[j] = (TP_glmnet[j] * TN_glmnet[j] - FP_glmnet[j] * FN_glmnet[j]) / np.sqrt(
    #         (TP_glmnet[j] + FP_glmnet[j]) * (TP_glmnet[j] + FN_glmnet[j]) * (TN_glmnet[j] + FP_glmnet[j]) * (TN_glmnet[j] + FN_glmnet[j]))
    #     pre_loss_glmnet[j] = mse(test_data.x,data.beta,model_glmnet_beta)

    t_pdas_list[i] = np.mean(t_pdas)
    TP_pdas_list[i] = np.mean(TP_pdas)
    TN_pdas_list[i] = np.mean(TN_pdas)
    FP_pdas_list[i] = np.mean(FP_pdas)
    FN_pdas_list[i] = np.mean(FN_pdas)
    ic_pdas_list[i] = np.mean(ic_pdas)
    loss_pdas_list[i] = np.mean(loss_pdas)
    mcc_pdas_list[i] = np.mean(mcc_pdas)
    coef_mse_pdas_list[i] = np.mean(coef_mse_pdas)
    pre_loss_pdas_list[i] = np.mean(pre_loss_pdas)

    t_std_pdas_list[i] = np.std(t_pdas)
    TP_std_pdas_list[i] = np.std(TP_pdas)
    TN_std_pdas_list[i] = np.std(TN_pdas) 
    FP_std_pdas_list[i] = np.std(FP_pdas)
    FN_std_pdas_list[i] = np.std(FN_pdas)
    ic_std_pdas_list[i] = np.std(ic_pdas)
    loss_std_pdas_list[i] = np.std(loss_pdas)
    mcc_std_pdas_list[i] = np.std(mcc_pdas)
    coef_mse_std_pdas_list[i] = np.std(coef_mse_pdas)
    pre_loss_std_pdas_list[i] = np.std(pre_loss_pdas)
    
    
    t_gpdas_list[i] = np.mean(t_gpdas)
    TP_gpdas_list[i] = np.mean(TP_gpdas)
    TN_gpdas_list[i] = np.mean(TN_gpdas)
    FP_gpdas_list[i] = np.mean(FP_gpdas)
    FN_gpdas_list[i] = np.mean(FN_gpdas)
    ic_gpdas_list[i] = np.mean(ic_gpdas)
    loss_gpdas_list[i] = np.mean(loss_gpdas)
    mcc_gpdas_list[i] = np.mean(mcc_gpdas)
    coef_mse_gpdas_list[i] = np.mean(coef_mse_gpdas)
    pre_loss_gpdas_list[i] = np.mean(pre_loss_gpdas)

    t_std_gpdas_list[i] = np.std(t_gpdas)
    TP_std_gpdas_list[i] = np.std(TP_gpdas)
    TN_std_gpdas_list[i] = np.std(TN_gpdas)
    FP_std_gpdas_list[i] = np.std(FP_gpdas)
    FN_std_gpdas_list[i] = np.std(FN_gpdas)
    ic_std_gpdas_list[i] = np.std(ic_gpdas)
    loss_std_gpdas_list[i] = np.std(loss_gpdas)
    mcc_std_gpdas_list[i] = np.std(mcc_gpdas)
    coef_mse_std_gpdas_list[i] = np.std(coef_mse_gpdas)
    pre_loss_std_gpdas_list[i] = np.std(pre_loss_gpdas)
  
    
    t_glmnet_list[i] = np.mean(t_glmnet)
    TP_glmnet_list[i] = np.mean(TP_glmnet)
    TN_glmnet_list[i] = np.mean(TN_glmnet)
    FP_glmnet_list[i] = np.mean(FP_glmnet)
    FN_glmnet_list[i] = np.mean(FN_glmnet)
    ic_glmnet_list[i] = np.mean(ic_glmnet)
    loss_glmnet_list[i] = np.mean(loss_glmnet)
    mcc_glmnet_list[i] = np.mean(mcc_glmnet)
    coef_mse_glmnet_list[i] = np.mean(coef_mse_glmnet)
    pre_loss_glmnet_list[i] = np.mean(pre_loss_glmnet)

    t_std_glmnet_list[i] = np.std(t_glmnet)
    TP_std_glmnet_list[i] = np.std(TP_glmnet)
    TN_std_glmnet_list[i] = np.std(TN_glmnet)
    FP_std_glmnet_list[i] = np.std(FP_glmnet)
    FN_std_glmnet_list[i] = np.std(FN_glmnet)
    ic_std_glmnet_list[i] = np.std(ic_glmnet)
    loss_std_glmnet_list[i] = np.std(loss_glmnet)
    mcc_std_glmnet_list[i] = np.std(mcc_glmnet)
    coef_mse_std_glmnet_list[i] = np.std(coef_mse_glmnet)
    pre_loss_pdas_list[i] = np.std(pre_loss_pdas)
    pre_loss_std_glmnet_list[i] = np.std(pre_loss_glmnet)


result = pd.DataFrame()
result['t_pdas_list'] = t_pdas_list
result['TP_pdas_list'] = TP_pdas_list
result['FP_pdas_list'] = FP_pdas_list
result['mcc_pdas_list'] = mcc_pdas_list
result['pre_loss_pdas_list'] = pre_loss_pdas_list

result['t_gpdas_list'] = t_gpdas_list
result['TP_gpdas_list'] = TP_gpdas_list
result['FP_gpdas_list'] = FP_gpdas_list
result['mcc_gpdas_list'] = mcc_gpdas_list
result['pre_loss_gpdas_list'] = pre_loss_gpdas_list

result['t_glmnet_list'] = t_glmnet_list
result['TP_glmnet_list'] = TP_glmnet_list
result['FP_glmnet_list'] = FP_glmnet_list
result['mcc_glmnet_list'] = mcc_glmnet_list
result['pre_loss_glmnet_list'] = pre_loss_glmnet_list

result.to_csv('simulate_' + family + '_' + str(parameter_len) + '_'+str(sim_time) + '_1.csv')