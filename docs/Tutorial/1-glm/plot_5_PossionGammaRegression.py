"""
===============================================
Positive Responses: Poisson & Gamma Regressions
===============================================
"""
###############################################################################
# Poisson Regression
# ^^^^^^^^^^^^^^^^^^
#
# Poisson Regression involves regression models in which the response variable is in the form of counts.
# For example, the count of number of car accidents or number of customers in line at a reception desk.
# The response variables is assumed to follow a Poisson distribution.
#
# The general mathematical equation for Poisson regression is
#
# .. math::
#   \log(E(y)) = \beta_0 + \beta_1 X_1+\beta_2 X_2+\dots+\beta_p X_p.
#
#
# Simulated Data Example
# ~~~~~~~~~~~~~~~~~~~~~~
#
# We generate some artificial data using this logic.
# Consider a dataset containing ``n=100`` observations with ``p=6`` variables.
# The ``make_glm_data()`` function allows uss to generate simulated data.
# By specifying ``k = 3``, we set only 3 of the 6 variables to have effect on the expectation of the response.
#

import numpy as np
from abess.datasets import make_glm_data
np.random.seed(0)

n = 100
p = 6
k = 3
data = make_glm_data(n=n, p=p, k=k, family="poisson")
print("the first 5 x observation:\n", data.x[0:5, ])
print("the first 5 y observation:\n", data.y[0:5])


# %%
# Notice that, the response have non-negative integer value.
# 
# The effective predictors and real coefficients are:
# 

print("non-zero:\n", np.nonzero(data.coef_))
print("real coef:\n", data.coef_)

###############################################################################
# Model Fitting
# ~~~~~~~~~~~~~
# The ``PoissonRegression()`` function in the ``abess`` package allows you to perform
# best subset selection in a highly efficient way.
# We can call the function using formula like:

from abess.linear import PoissonRegression

model = PoissonRegression(support_size=range(7))
model.fit(data.x, data.y)

# %%
# where ``support_size`` contains the level of sparsity we consider,
# and the program can adaptively choose the "best" one.
# The result of coefficients can be viewed through ``model.coef_``:

print(model.coef_)

# %%
# So that the first, third and last variables are thought to be useful in the model (the chosen sparsity is 3),
# which is the same as "real" variables. What's more, the predicted coefficients are also close to the real ones.
#
# More on the Results
# ~~~~~~~~~~~~~~~~~~~
# Actually, we can also plot the path of coefficients in ``abess`` process.
# This can be computed by fixing the ``support_size`` as one number from 0
# to 6 each time:

import matplotlib.pyplot as plt

coef = np.zeros((7, 6))
ic = np.zeros(7)
for s in range(7):
    model = PoissonRegression(support_size=s)
    model.fit(data.x, data.y)
    coef[s, :] = model.coef_
    ic[s] = model.eval_loss_

for i in range(6):
    plt.plot(coef[:, i], label=i)

plt.xlabel('support_size')
plt.ylabel('coefficients')
plt.title('ABESS path for Poisson regression')
plt.legend()
plt.show()

# %%
# And the evolution of information criterion (by default, we use EBIC):

plt.plot(ic, 'o-')
plt.xlabel('support_size')
plt.ylabel('EBIC')
plt.title('Model selection via EBIC')
plt.show()

# %%
# The lowest point is shown on ``support_size=3`` and that's why the program chooses 3 variables as output.
#
# Gamma Regression
# ^^^^^^^^^^^^^^^^
# Gamma regression can be used when you have positive continuous response variables such as payments for insurance claims,
# or the lifetime of a redundant system.
# It is well known that the density of Gamma distribution can be represented as a function of
# a mean parameter (:math:`\mu`) and a shape parameter (:math:`\alpha`), respectively,
#
# .. math::
#   f(y \mid \mu, \alpha)=\frac{1}{y \Gamma(\alpha)}\left(\frac{\alpha y}{\mu}\right)^{\alpha} e^{-\alpha y / \mu} {I}_{(0, \infty)}(y),
#
# where :math:`I(\cdot)` denotes the indicator function. In the Gamma regression model,
# response variables are assumed to follow Gamma distributions. Specifically,
#
# .. math::
#   y_i \sim Gamma(\mu_i, \alpha),
#
#
# where :math:`-1/\mu_i = x_i^T\beta`.
#
# Compared with Poisson regression, this time we consider the response variables as (continuous) levels of satisfaction.
#
# Simulated Data Example
# ~~~~~~~~~~~~~~~~~~~~~~
# Firstly, we also generate data from ``make_glm_data()``, but ``family =
# "gamma"`` is given this time:

np.random.seed(1)

n = 100
p = 6
k = 3
data = make_glm_data(n=n, p=p, k=k, family="gamma")
print("the first 5 x:\n", data.x[0:5, ])
print("the first 5 y:\n", data.y[0:5])

# %%
# Notice that the response `y` is positive **continuous** value, which is different from the Poisson regression for integer value. 
# 
# The effective predictors and their effects are presented below:

nnz_ind = np.nonzero(data.coef_)
print("non-zero:\n", nnz_ind)
print("non-zero coef:\n", data.coef_[nnz_ind])

###############################################################################
# Model Fitting
# ~~~~~~~~~~~~~
# We apply the above procedure for gamma regression simply by using ``GammaRegression()`` in ``abess.linear``.
# It has similar member functions for fitting.
# We use five fold cross validation (CV) for selecting the model size:

from abess.linear import GammaRegression

model = GammaRegression(support_size=range(7), cv=5) 
model.fit(data.x, data.y)

# %%
# The fitted coefficients:

print(model.coef_)


###############################################################################
# More on the Results
# ~~~~~~~~~~~~~~~~~~~
# We can also plot the path of coefficients in abess process.


coef = np.zeros((7, 6))
loss = np.zeros(7)
for s in range(7):
    model = GammaRegression(support_size=s)
    model.fit(data.x, data.y)
    coef[s, :] = model.coef_
    loss[s] = model.eval_loss_

for i in range(6):
    plt.plot(coef[:, i], label=i)

plt.xlabel('support_size')
plt.ylabel('coefficients')
plt.title('ABESS path for gamma regression')
plt.legend()
plt.show()

###############################################################################
# The ``abess`` R package also supports Poisson regression and Gamma regression.
# For R tutorial, please view
# https://abess-team.github.io/abess/articles/v04-PoissonGammaReg.html.
