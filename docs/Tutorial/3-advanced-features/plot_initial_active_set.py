"""
Initial Active Set
==================
"""

########################################################
# User-specified initial active set
# ---------------------------------
# We believe that it worth allowing given an initial active set so that the splicing process starts from this set for each sparsity.
# It might come from prior analysis, whose result is not quite precise but better than random selection,
# so the algorithm can run more efficiently. Or you just want to give different initial sets to test the stability of the algorithm.
#
# Note that this is **NOT** equivalent to ``always_select``, since they can be exchanged to inactive set when splicing.
#
# To specify initial active set, an additive argument ``A_init`` should be
# given in ``fit()``.

import numpy as np
from abess.datasets import make_glm_data
from abess.linear import LinearRegression
n = 100
p = 10
k = 3
np.random.seed(2)

data = make_glm_data(n=n, p=p, k=k, family='gaussian')

model = LinearRegression(support_size=range(0, 5))
model.fit(data.x, data.y, A_init=[0, 1, 2])

# %%
# Some strategies for initial active set are:
#
# - If ``sparsity = len(A_init)``, the splicing process would start from ``A_init``.
# - If ``sparsity > len(A_init)``, the initial set includes ``A_init`` and other variables with larger forward sacrifices chooses.
# - If ``sparsity < len(A_init)``, the initial set includes part of ``A_init``.
# - If both ``A_init`` and ``always_select`` are given, ``always_select`` first.
# - For warm-start, ``A_init`` will only affect splicing under the first sparsity in ``support_size``.
# - For CV, ``A_init`` will affect each fold but not the re-fitting on full data.
#
# The ``abess`` R package also supports user-defined initial active set.
# For R tutorial, please view
# https://abess-team.github.io/abess/articles/v07-advancedFeatures.html.
