"""
================================
Multi-Response Linear Regression
================================
"""
###############################################################################
# Introduction: model setting
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Multi-response linear regression (a.k.a., multi-task learning)
# aims at predicting multiple responses at the same time,
# and thus, it is a natural extension for classical linear regression where the response is univariate.
# Multi-response linear regression (MRLR) is very helpful for the analysis of
# correlated response such as chemical measurements for soil samples and
# microRNAs associated with Glioblastoma multiforme cancer.
# Suppose :math:`y` is an :math:`m`-dimensional response variable,
# :math:`x` is :math:`p`-dimensional predictors,
# :math:`B \in R^{m \times p}` is the coefficient matrix,
# the MMLR model for the multivariate response is given by
#
# .. math::
#   y = B x + \epsilon,
#
# where :math:`\epsilon` is an :math:`m`-dimensional random noise variable with zero mean.
#
# Due to the Occam`s razor principle or the high-dimensionality of predictors,
# it is meaningful to use a small amount of predictors to conduct multi-task learning.
# For example, understanding the relationship between gene expression and symptoms of a disease
# has significant importance in identifying potential markers. Many diseases usually
# involve multiple manifestations and those manifestations are usually related.
# In some cases, it makes sense to predict those manifestations using a small but the same set of predictors.
# The best subset selection problem under the MMLR model is formulated as
#
# .. math::
#   \frac{1}{2n} \| Y - XB \|_{2}^2, \text{ subject to: } \| B \|_{0, 2} \leq s,
#
# where, :math:`Y \in R^{n \times m}` and :math:`X \in R^{n \times p}` record
# :math:`n` observations` response and predictors, respectively.
# Here :math:`\| B \|_{0, 2} = \sum_{i = 1}^{p} I(B_{i\cdot} = {\bf 0})`,
# where :math:`B_{i\cdot}` is the :math:`i`-th row of coefficient matrix :math:`B` and
# :math:`{\bf 0} \in R^{m}` is an all-zero vector.
#
# Simulated Data Example
# ~~~~~~~~~~~~~~~~~~~~~~
# We use an artificial dataset to demonstrate how to solve best subset selection problem for MMLR with ``abess`` package.
# The ``make_multivariate_glm_data()`` function provides a simple way to generate suitable dataset for this task.
# The synthetic data have 100 observations with 3-dimensional responses and 20-dimensional predictors.
# Note that there are three predictors having an impact on the responses.


import matplotlib.pyplot as plt
from abess import MultiTaskRegression
from abess.datasets import make_multivariate_glm_data
import numpy as np
np.random.seed(0)

n = 100
p = 20
M = 3
k = 3

data = make_multivariate_glm_data(n=n, p=p, M=M, k=k, family='multigaussian')
print(data.y[0:5, ])

print(data.coef_)
print("non-zero: ", set(np.nonzero(data.coef_)[0]))

###############################################################################
# Model Fitting
# """""""""""""
# To carry out sparse mutli-task learning, we can call the
# ``MultiTaskRegression`` like:


model = MultiTaskRegression()
model.fit(data.x, data.y)

# %%
# After fitting, ``model.coef_`` contains the predicted coefficients:


print(model.coef_)
print("non-zero: ", set(np.nonzero(model.coef_)[0]))

# %%
# The outputs show that the support set is correctly identifying and the parameter estimation approaches to the truth.
#
# More on the results
# """""""""""""""""""
# Since there are three responses, we have three solution paths, which correspond to three responses, respectively.
# To plot the figure, we can fix the ``support_size`` at different levels:


coef = np.zeros((3, 21, 20))
for s in range(21):
    model = MultiTaskRegression(support_size=s)
    model.fit(data.x, data.y)

    for y in range(3):
        coef[y, s, :] = model.coef_[:, y]


for i in range(20):
    plt.plot(coef[0, :, i])
plt.xlabel('support_size')
plt.ylabel('value')
plt.title('the 1st response\\`s coefficients')
plt.show()


for i in range(20):
    plt.plot(coef[1, :, i])
plt.xlabel('support_size')
plt.ylabel('value')
plt.title('the 2nd response\\`s coefficients')
plt.show()


for i in range(20):
    plt.plot(coef[2, :, i])
plt.xlabel('support_size')
plt.ylabel('value')
plt.title('the 3rd response\\`s coefficients')
plt.show()

###############################################################################
# The ``abess`` R package also supports MRLR.
# For R tutorial, please view https://abess-team.github.io/abess/articles/v06-MultiTaskLearning.html.
#
