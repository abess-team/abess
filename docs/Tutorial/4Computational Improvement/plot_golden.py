"""
Golden-section searching
=========================
"""
#%%
# Here we generate a simple example and draw the path of scores of information criterion. Typically, the curve should be a strictly unimodal function achieving minimum at the true subset size.

import numpy as np
import matplotlib.pyplot as plt
from abess.datasets import make_glm_data

np.random.seed(0)
data = make_glm_data(n = 100, p = 20, k = 5, family = 'gaussian')

ic = np.zeros(21)
for sz in range(21):
    model = LinearRegression(support_size = [sz], ic_type = 'ebic')
    model.fit(data.x, data.y)
    ic[sz] = model.ic_

print("lowest point: ", np.argmin(ic))
plt.plot(ic, 'o-')
plt.xlabel('support_size')
plt.ylabel('EBIC')
plt.show()

#%%
# Here the generated data contains 100 observations with 20 predictors, while 5 of them are useful (should be non-zero). The default information criterion is EBIC. From the figure, we can find that "support_size = 5" is the lowest point.
# 
# Compared with searching the optimal support size one by one from a candidate set with :math:`O(s_{max})` complexity, **golden-section** reduce the time complexity to :math:`O(ln(s_{max}))`, giving a significant computational improvement.
# 
# In `abess` package, this can be easily formed like:



model = LinearRegression(path_type = 'gs', s_min = 0, s_max = 20)
model.fit(data.x, data.y)
print("real coef:\n", np.nonzero(data.coef_)[0])
print("predicted coef:\n", np.nonzero(model.coef_)[0])

#%%
# where `path_type = gs` means golden-section and `s_min`, `s_max` indicates the left and right bound of range of the support size. Note that in golden-section searching, we should not give `support_size`, which is only useful for sequential strategy.
# 
# The output of golden-section strategy suggests the optimal model size is accurately detected. Compare to the sequential searching, the golden section reduce the runtime because it skip some support sizes which are likely to be a non-optimal one:



from time import time

t1 = time()
model = LinearRegression(support_size = range(21))
model.fit(data.x, data.y)
print("sequential time: ", time() - t1)

t2 = time()
model = LinearRegression(path_type = 'gs', s_min = 0, s_max = 20)
model.fit(data.x, data.y)
print("golden-section time: ", time() - t2)

#%%
# The golden-section runs much faster than sequential method, espectially when the range of support size is large.

###############################################################################
# R tutorial
# -------------
#
# For R tutorial, please view [https://abess-team.github.io/abess/articles/v09-fasterSetting.html](https://abess-team.github.io/abess/articles/v09-fasterSetting.html).
