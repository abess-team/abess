# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 20:58:19 2020

@author: public
"""

from sklearn.model_selection import KFold # import KFold
import numpy as np

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array
y = np.array([1, 2, 3, 4]) # Create another array
kf = KFold(n_splits=2) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
