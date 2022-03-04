"""
=================
Linear Regression
=================

In this tutorial, we are going to demonstrate how to use the ``abess`` package to carry out best subset selection
in linear regression with both simulated data and real data.
"""

###############################################################################
#
# Our package ``abess`` implements a polynomial algorithm in the following best-subset selection problem:
#
# .. math::
#     \min_{\beta\in \mathbb{R}^p} \frac{1}{2n} ||y-X\beta||^2_2,\quad \text{s.t.}\ ||\beta||_0\leq s,
#
#
# where :math:`\| \cdot \|_2` is the :math:`\ell_2` norm, :math:`\|\beta\|_0=\sum_{i=1}^pI( \beta_i\neq 0)`
# is the :math:`\ell_0` norm of :math:`\beta`, and the sparsity level :math:`s`
# is usually an unknown non-negative integer.
# Next, we present an example to show how to use the ``abess`` package to solve a simple problem.
#
# Simulated Data Example
# ^^^^^^^^^^^^^^^^^^^^^^
# Fixed Support Size Best Subset Selection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We generate a design matrix :math:`X` containing 300 observations and each observation has 1000 predictors.
# The response variable :math:`y` is linearly related to the first, second, and fifth predictors in :math:`X`:
#
# .. math::
#     y = 3X_1 + 1.5X_2 + 2X_5 + \epsilon,
#
# where :math:`\epsilon` is a standard normal random variable.


import matplotlib.pyplot as plt
import os
import pandas as pd
from abess import LinearRegression
import numpy as np
from abess.datasets import make_glm_data
np.random.seed(0)

n = 300
p = 1000
k = 3
real_coef = np.zeros(p)
real_coef[[0, 1, 4]] = 3, 1.5, 2
data1 = make_glm_data(n=n, p=p, k=k, family="gaussian", coef_=real_coef)

print(data1.x.shape)
print(data1.y.shape)
# %%
# Use ``LinearRegression`` to fit the data, with a fixed support size:

model = LinearRegression(support_size=3)
model.fit(data1.x, data1.y)

# %%
# After fitting, the predicted coefficients are stored in ``model.coef_``:

print("shape:", model.coef_.shape)
ind = np.nonzero(model.coef_)
print("predicted non-zero: ", ind)
print("predicted coef: ", model.coef_[ind])

# %%
# From the result, we know that ``abess`` found which 3 predictors are useful among all 1000 variables.
# Besides, the predicted coefficients of them are quite close to the real ones.
#
# Adaptive Best Subset Selection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# However, we may not know the true sparsity level in real world data,
# and thus we need to determine the most proper one from a large range.
# Suppose that we believe the real sparsity level is between 0 and 30 (so
# that ``range(0, 31)``):


model = LinearRegression(support_size=range(31))
model.fit(data1.x, data1.y)

ind = np.nonzero(model.coef_)
print("predicted non-zero: ", ind)
print("predicted coef: ", model.coef_[ind])

# %%
# The program can adaptively choose the sparsity level that best fits the
# data. It is not surprising that it chooses 3 variables, the same as the
# last section.

###############################################################################
# Real data example
# ^^^^^^^^^^^^^^^^^
#
# Hitters Dataset
# ~~~~~~~~~~~~~~~
# Now we focus on real data on the `Hitters` dataset: https://www.kaggle.com/floser/hitters.
# We hope to use several predictors related to the performance of
# the baseball athletes last year to predict their salary.
#
# First, let's have a look at this dataset. There are 19 variables except
# `Salary` and 322 observations.


data2 = pd.read_csv(os.path.join(os.getcwd(), 'Hitters.csv'))
print(data2.shape)
print(data2.head(5))

# %%
# Since the dataset contains some missing values, we simply drop those rows with missing values.
# Then we have 263 observations remain:


data2 = data2.dropna()
print(data2.shape)

# %%
# What is more, before fitting, we need to transfer the character
# variables to dummy variables:


data2 = pd.get_dummies(data2)
data2 = data2.drop(['League_A', 'Division_E', 'NewLeague_A'], axis=1)
print(data2.shape)
print(data2.head(5))

###############################################################################
# Model Fitting
# ~~~~~~~~~~~~~
# As what we do in simulated data, an adaptive best subset can be formed
# easily:

x = np.array(data2.drop('Salary', axis=1))
y = np.array(data2['Salary'])

model = LinearRegression(support_size=range(20))
model.fit(x, y)

# %%
# The result can be shown as follows:


ind = np.nonzero(model.coef_)
print("non-zero:\n", data2.columns[ind])
print("coef:\n", model.coef_)

# %%
# Automatically, variables `Hits`, `CRBI`, `PutOuts`, `League\_N` are
# chosen in the model (the chosen sparsity level is 4).

###############################################################################
# More on the results
# ~~~~~~~~~~~~~~~~~~~
# We can also plot the path of abess process:


coef = np.zeros((20, 19))
ic = np.zeros(20)
for s in range(20):
    model = LinearRegression(support_size=s)
    model.fit(x, y)
    coef[s, :] = model.coef_
    ic[s] = model.ic_

for i in range(19):
    plt.plot(coef[:, i], label=i)

plt.xlabel('support_size')
plt.ylabel('coefficients')
# plt.legend() # too long to plot
plt.show()

# %%
# Besides, we can also generate a graph about the tuning parameter.
# Remember that we used the default EBIC to tune the support size.

plt.plot(ic, 'o-')
plt.xlabel('support_size')
plt.ylabel('EBIC')
plt.show()

# %%
# In EBIC criterion, a subset with the support size 4 has the lowest value,
# so the process adaptively chooses 4 variables.
# Note that under other information criteria, the result may be different.

###############################################################################
# R tutorial
# ^^^^^^^^^^
# For R tutorial, please view
# https://abess-team.github.io/abess/articles/v01-abess-guide.html.
