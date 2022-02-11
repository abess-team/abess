"""
p>>N (Important Search)
-------------------
Suppose that there are only a few variables are important (i.e. too many noise variables), it may be a vise choice to focus on some important variables in splicing process. This can save a lot of time, especially under a large :math:`p`.

In abess package, an argument called `important_search` is used for it, which means the size of inactive set for each splicing process. By default, this argument is set as 0, and the total inactive variables would be contained in the inactive set. But if an positive integer is given, the splicing process would focus on active set and the most important `important_search` inactive variables.
 
However, after convergence on this subset, we check if the chosen variables are still the most important ones by recomputing on the full set with the new active set. If not, we update the subset and splicing again. On our testing, it would not iterate many time to reach a stable subset. After that, the active set on the stable subset would be treated as that on the full set.

Here we take `LogisticRegression` for an example. 
"""


from abess.linear import LogisticRegression
from abess.datasets import make_glm_data
from time import time
import numpy as np

data = make_glm_data(n = 500, p = 10000, k = 10, family = "binomial")

t1 = time()
model = LogisticRegression()
model.fit(data.x, data.y)
t2 = time()

print("non_zero :\n", np.nonzero(model.coef_)[0])
print("time : ", t2 - t1)

#%%
# However, if we only focus on 500 important inactive variables when searching:


t1 = time()
model2 = LogisticRegression(important_search = 500)
model2.fit(data.x, data.y)
t2 = time()

print("non_zero :\n", np.nonzero(model2.coef_)[0])
print("time : ", t2 - t1)

#%%
# It takes much less time to reach the same result. We recommend use this method for large :math:`p` situation, but in small one, it may not be faster than the primary fitting.
# 
# Here we compare the AUC and runtime for `LogisticRegression` under different `important_search` and the test code can be found [here](https://github.com/abess-team/abess/blob/master/docs/simulation/Python/impsearch.py).
# 
# ![](../fig/impsearch.png)
# 
# At a low level of `important_search`, however, the performance (AUC) has been very good. In this situation, a lower `important_search` can save lots of time and space.

###############################################################################
# R tutorial
# -------------
#
# For R tutorial, please view [https://abess-team.github.io/abess/articles/v09-fasterSetting.html](https://abess-team.github.io/abess/articles/v09-fasterSetting.html).
