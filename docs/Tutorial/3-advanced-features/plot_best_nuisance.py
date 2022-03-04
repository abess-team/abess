"""
Nuisance Regression
===================
"""

# %%
# Introduction
# ------------
# Nuisance regression refers to best subset selection with some prior information that some variables are required to stay in the active set.
# For example, if we are interested in a certain gene and want to find out what other genes are associated with the response when this particular gene shows effect.
#
# Using: nuisance regression
# --------------------------
# In the ``LinearRegression()`` (or other methods), the argument ``always_select`` is designed to realize this goal.
# User can pass a vector containing the indexes of the target variables to
# ``always_select``. Here is an example.

import numpy as np
from abess.datasets import make_glm_data
from abess.linear import LinearRegression

np.random.seed(0)

# gene data
n = 100
p = 20
k = 5
data = make_glm_data(n=n, p=p, k=k, family='gaussian')
print('real coefficients:\n', data.coef_, '\n')
print('real coefficients\' indexes:\n', np.nonzero(data.coef_)[0])

model = LinearRegression(support_size=range(0, 6))
model.fit(data.x, data.y)
print('fitted coefficients:\n', model.coef_, '\n')
print('fitted coefficients\' indexes:\n', np.nonzero(model.coef_)[0])

# %%
# The coefficients are located in \[2, 5, 10, 11, 18\].
# But if we suppose that the 7th and 8th variables are worthy to be
# included in the model, we can call:

model = LinearRegression(support_size=range(0, 6), always_select=[7, 8])
model.fit(data.x, data.y)
print('fitted coefficients:\n', model.coef_, '\n')
print('fitted coefficients\' indexes:\n', np.nonzero(model.coef_)[0])

# %%
# Now the variables we chosen are always in the model.
#
# The ``abess`` R package also supports nuisance regression.
# For R tutorial, please view
# https://abess-team.github.io/abess/articles/v07-advancedFeatures.html.

# sphinx_gallery_thumbnail_path = '_static/nuisance_cover.png'