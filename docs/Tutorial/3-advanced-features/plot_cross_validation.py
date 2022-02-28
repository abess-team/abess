"""
Cross-Validation Division
=========================
"""

########################################################
# User-specified cross validation division
# ----------------------------------------
# Sometimes, especially when running a test, we would like to fix the train and valid data used in cross validation,
# instead of choosing them randomly.
# One simple method is to fix a random seed, such as ``numpy.random.seed()``.
# But in some cases, we would also like to specify which samples would be in the same "fold", which has great flexibility.
#
# In our program, an additional argument ``cv_fold_id`` is for this user-specified cross validation division.
# An integer ``numpy`` array with the same size of input samples can be given,
# and those with same integer would be assigned to the same "fold" in
# K-fold CV.

import numpy as np
from abess.datasets import make_glm_data
from abess.linear import LinearRegression
n = 100
p = 1000
k = 3
np.random.seed(2)

data = make_glm_data(n=n, p=p, k=k, family='gaussian')

# cv_fold_id has a size of `n`
# cv_fold_id has `cv` different integers
cv_fold_id = [1 for i in range(30)] + \
    [2 for i in range(30)] + [3 for i in range(40)]

model = LinearRegression(support_size=range(0, 5), cv=3)
model.fit(data.x, data.y, cv_fold_id=cv_fold_id)
print('fitted coefficients\' indexes:', np.nonzero(model.coef_)[0])

# %%
# The ``abess`` R package also supports user-defined cross-validation division.
# For R tutorial, please view https://abess-team.github.io/abess/articles/v07-advancedFeatures.html.
#
