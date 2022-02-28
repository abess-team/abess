"""
==============================================
Classification: Logistic Regression and Beyond
==============================================
We would like to use an example to show how the best subset selection for logistic regression works in our program.
"""

###############################################################################
# Real Data Example
# ^^^^^^^^^^^^^^^^^
# Titanic Dataset
# """""""""""""""
# Consider the Titanic dataset obtained from the Kaggle competition: https://www.kaggle.com/c/titanic/data.
# The dataset consists of data of 889 passengers, and the goal of the competition is to
# predict the survival status (yes/no) based on features including the
# class of service, the sex, the age, etc.


from abess.linear import MultinomialRegression
from abess.datasets import make_multivariate_glm_data
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from abess.linear import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

dt = pd.read_csv("train.csv")
print(dt.head(5))

# %%
# We only focus on some numerical or categorical variables:
#
# - predictor variables: `Pclass`,  `Sex`,  `Age`,  `SibSp`,  `Parch`,  `Fare`,  `Embarked`;
# - response variable is `Survived`.


dt = dt.iloc[:, [1, 2, 4, 5, 6, 7, 9, 11]]  # variables interested
dt['Pclass'] = dt['Pclass'].astype(str)
print(dt.head(5))

# %%
# However, some rows contain missing values (NaN) and we need to drop them.

dt = dt.dropna()
print('sample size: ', dt.shape)

# %%
# Then use dummy variables to replace classification variables:


dt1 = pd.get_dummies(dt)
print(dt1.head(5))

# %%
# Now we split `dt1` into training set and testing set:


X = np.array(dt1.drop('Survived', axis=1))
Y = np.array(dt1.Survived)

train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.33, random_state=0)
print('train size: ', train_x.shape[0])
print('test size:', test_x.shape[0])

# %%
# Here ``train_x`` contains:
#
# - V0: dummy variable, 1st ticket class (1-yes, 0-no)
# - V1: dummy variable, 2nd ticket class (1-yes, 0-no)
# - V2: dummy variable, sex (1-male, 0-female)
# - V3: Age
# - V4: # of siblings / spouses aboard the Titanic
# - V5: # of parents / children aboard the Titanic
# - V6: Passenger fare
# - V7: dummy variable, Cherbourg for embarkation (1-yes, 0-no)
# - V8: dummy variable, Queenstown for embarkation (1-yes, 0-no)
#
# And ``train_y`` indicates whether the passenger survived (1-yes, 0-no).

print('train_x:\n', train_x[0:5, :])
print('train_y:\n', train_y[0:5])

###############################################################################
# Model Fitting
# """""""""""""
# The ``LogisticRegression()`` function in the ``abess.linear`` allows us to perform
# best subset selection in a highly efficient way.
# For example, in the Titanic sample, if you want to look for a best subset with no more than 5 variables
# on the logistic model, you can call:


s = 5   # max target sparsity
model = LogisticRegression(support_size=range(0, s + 1))
model.fit(train_x, train_y)

# %%
# Now the ``model.coef_`` contains the coefficients of logistic model with no more than 5 variables.
# That is, those variables with a coefficient 0 is unused in the model:

print(model.coef_)

# %%
# By default, the ``LogisticRegression`` function set ``support_size = range(0, min(p,n/log(n)p)``
# and the best support size is determined by the Extended Bayesian Information Criteria (EBIC).
# You can change the tuning criterion by specifying the argument ``ic_type``.
# The available tuning criteria now are ``"gic"``, ``"aic"``, ``"bic"``, ``"ebic"``.
#
# For a quicker solution, you can change the tuning strategy to a golden section path which
# tries to find the elbow point of the tuning criterion over the hyperparameter space.
# Here we give an example.


model_gs = LogisticRegression(path_type="gs", s_min=0, s_max=s)
model_gs.fit(train_x, train_y)
print(model_gs.coef_)

# %%
# where ``s_min`` and ``s_max`` bound the support size and this model gives the same answer as before.
#
# More on the Results
# """""""""""""""""""
# After fitting with ``model.fit()``, we can further do more exploring work to interpret it.
# As we show above, ``model.coef_`` contains the sparse coefficients of variables and
# those non-zero values indicate "important" variables chosen in the model.


print('Intercept: ', model.intercept_)
print('coefficients: \n', model.coef_)
print('Used variables\' index:', np.nonzero(model.coef_ != 0)[0])

# %%
# The training loss and the score under information criterion:


print('Training Loss: ', model.train_loss_)
print('IC: ', model.ic_)

# %%
# Prediction is allowed for the estimated model. Just call
# ``model.predict()`` function like:

fitted_y = model.predict(test_x)
print(fitted_y)

# %%
# Besides, we can also call for the survival probability of each observation by ``model.predict_proba()``.
# Actually, those people with a probability greater than 0.5 are
# classified as "1" (survived).


fitted_p = model.predict_proba(test_x)
print(fitted_p)

# %%
# We can also generate an ROC curve and calculate tha AUC value. On this dataset,
# the AUC is 0.817, which is quite close to 1.


fpr, tpr, _ = roc_curve(test_y, fitted_p)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()

print('AUC: ', auc(fpr, tpr))

###############################################################################
# Extension: Multi-class Classification
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Multinomial logistic regression
# """""""""""""""""""""""""""""""
# When the number of classes is more than 2, we call it multi-class classification task.
# Logistic regression can be extended to model several classes of events such as determining
# whether an image contains a cat, dog, lion, etc. Each object being detected in the image
# would be assigned a probability between 0 and 1, with a sum of one. The extended model is multinomial logistic regression.
#
# To arrive at the multinomial logistic model, one can imagine, for :math:`K` possible classes,
# running :math:`K−1` independent logistic regression models, in which one class is chosen as
# a "pivot" and then the other :math:`K−1` classes are separately regressed against the pivot outcome.
# This would proceed as follows, if class :math:`K` (the last outcome) is chosen as the pivot:
#
# .. math::
#     \ln (\mathbb{P}(y=1)/\mathbb{P}(y=K)) = x^T\beta^{(1)},\\
#     \dots\ \dots\\
#     \ln (\mathbb{P}(y=K-1)/\mathbb{P}(y=K)) = x^T\beta^{(K-1)}.
#
#
# Then, the probability to choose the :math:`j`-th class can be easily derived to be:
#
# .. math::
#     \mathbb{P}(y=j) = \frac{\exp(x^T\beta^{(j)})}{1+\sum_{k=1}^{K-1} \exp(x^T\beta^{(k)})},
#
#
# and subsequently, we would that the object belongs to the :math:`j^*`-th class if
# the :math:`j^*=\arg\max_j \mathbb{P}(y=j)`. Notice that, for :math:`K` possible classes case,
# there are :math:`p\times(K−1)` unknown parameters: :math:`\beta^{(1)},\dots,\beta^{(K−1)}` to be estimated.
# Because the number of parameters increases as :math:`K`, it is even more urgent to constrain the model complexity.
# And the best subset selection for multinomial logistic regression aims to maximize the log-likelihood function and control the model complexity by restricting :math:`B=(\beta^{(1)},\dots,\beta^{(K−1)})` with :math:`||B||_{0,2}\leq s` where :math:`||B||_{0,2}=\sum_{i=1}^p I(B_{i\cdot}=0)`, :math:`B_{i\cdot}` is the :math:`i`-th row of coefficient matrix :math:`B` and :math:`0\in R^{K-1}` is an all zero vector. In other words, each row of :math:`B` would be either all zero or all non-zero.
#
# Simulated Data Example
# ~~~~~~~~~~~~~~~~~~~~~~
# We shall conduct Multinomial logistic regression on an artificial dataset for demonstration.
# The ``make_multivariate_glm_data()`` provides a simple way to generate suitable dataset for this task.
#
# The assumption behind this model is that the response vector follows a multinomial distribution.
# The artificial dataset contains 100 observations and 20 predictors but only five predictors
# have influence on the three possible classes.


n = 100  # sample size
p = 20  # all predictors
k = 5   # real predictors
M = 3   # number of classes

np.random.seed(0)
dt = make_multivariate_glm_data(n=n, p=p, k=k, family="multinomial", M=M)
print(dt.coef_)
print('real variables\' index:\n', set(np.nonzero(dt.coef_)[0]))

# %%
# To carry out best subset selection for multinomial logistic regression,
# we can call the ``MultinomialRegression()``. Here is an example.


s = 5
model = MultinomialRegression(support_size=range(0, s + 1))
model.fit(dt.x, dt.y)

# %%
# Its use is quite similar to ``LogisticRegression``. We can get the
# coefficients to recognize "in-model" variables.


print('intercept:\n', model.intercept_)
print('coefficients:\n', model.coef_)

# %%
# So those variables used in model can be recognized and we can find that
# they are the same as the data's "real" coefficients we generate.


print('used variables\' index:\n', set(np.nonzero(model.coef_)[0]))

###############################################################################
# The ``abess`` R package also supports classification tasks.
# For R tutorial, please view
# https://abess-team.github.io/abess/articles/v03-classification.html.
