# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:08:54 2020

@author: public
"""

from lifelines.datasets import load_regression_dataset
regression_dataset = load_regression_dataset()
from bess.linear import PdasCox
from bess.gen_data import gen_data
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
import numpy  as np


def AUC(x, Tbeta, beta):
    pass


def CV(model, data, kfold):
    kf = KFold(n_splits=kfold)
    kf.get_n_splits(data.x) # returns the number of splitting iterations in the cross-validator
    pre_loss = []
    for train_index, test_index in kf.split(data.x):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = data.x[train_index], data.x[test_index]
        y_train, _ = data.y[train_index], data.y[test_index]
        model.fit(x_train, y_train)
        pre_loss.append(AUC(x_test, data.Tbeta, model.beta))
    return np.mean(pre_loss)
    

if __name__ == "__main__":
    n=1000
    p=100
    k=20
    data = gen_data(n, p, family="cox", k=k, rho=0.50)
    
    survival = pd.DataFrame()
    for i in range(data.x.shape[1]):
        survival["Var" + str(i)] = data.x[:,i]
        
    survival["T"] = data.y[:,0]
    survival["E"] = data.y[:,1]


    for lamb in np.linspace(0, 1, 100):
        cph = CoxPHFitter(penalizer=lamb, l1_ratio=1.0)

    cph.fit(survival, 'T', event_col='E')
    cph.print_summary()
    print(cph.params_.values)
    print(cph.params_.values[np.nonzero(data.beta)])
    print(np.nonzero(cph.params_.values))
    # cph.plot()
    
    