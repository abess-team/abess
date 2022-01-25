"""
Computational Improvement
============================
 The generic splicing technique certifiably guarantees the best subset can be selected in a polynomial time. In practice, the computational efficiency can be improved to handle large scale datasets. The tips for computational improvement include:
 
 - exploit sparse strucute of input matrix;
 - use golden-section to search best support size;
 - focus on important variables when splicing;
 - early-stop scheme;
 - sure independence screening;
 - warm-start initialization;
 - parallel computing when performing cross validation;
 - covariance update for `LinearRegression` or `MultiTaskRegression`;
 - approximate Newton iteration for `LogisticRegression`, `PoissonRegression`, `CoxRegression`.
 
 This vignette illustrate the first two tips. For the other tips, they have been efficiently implemented and set as the default in abess package.
"""


###############################################################################
#  Sparse matrix
# ------------------------
#
# We sometimes meet with problems where the :math:`N×p` input matrix :math:`X` is extremely sparse, i.e., many entries in :math:`X:math:` have zero values. A notable example comes from document classification: aiming to assign classes to a document, making it easier to manage for publishers and news sites. The input variables for characterizing documents are generated from a so called "bag-of-words" model. In this model, each variable is scored for the presence of each of the words in the entire dictionary under consideration. Since most words are absent, the input variables for each document is mostly zero, and so the entire matrix is mostly zero. 
# 
# For example, we create a sparse matrix like:

from scipy.sparse import coo_matrix
import numpy as np

row  = np.array([0, 1, 2, 3, 4, 4,  5, 6, 7, 7, 8, 9])
col  = np.array([0, 3, 1, 2, 4, 3, 5, 2, 3, 1, 5, 2])
data = np.array([4, 5, 7, 9, 1, 23, 4, 5, 6, 8, 77, 100])
x = coo_matrix((data, (row, col)))

print(x.toarray())

##%
# The sparse matrix can be directly used in `abess` pacakages. We just need to set argument `sparse_matrix = T`. Note that if the input matrix is not sparse matrix, the program would automatically transfer it into the sparse one, so this argument can also make some improvement.


from abess import LinearRegression

coef = np.array([1, 1, 1, 0, 0, 0])
y = x.dot(coef)
model = LinearRegression(sparse_matrix = True)
model.fit(x, y)

print("real coef: \n", coef)
print("pred coef: \n", model.coef_)

##%
# We compare the runtime when the input matrix is dense matrix:


from time import time

t = time()
model = LinearRegression()
model.fit(x.toarray(), y)
print("dense matrix:  ", time() - t)

t = time()
model = LinearRegression(sparse_matrix = True)
model.fit(x, y)
print("sparse matrix:  ", time() - t)

##%
# From the comparison, we see that the time required by sparse matrix is smaller, and this sould be more visible when the sparse imput matrix is large. Hence, we suggest to assign a sparse matrix to `abess` when the input matrix have a lot of zero entries.

###############################################################################
# Golden-section searching
# -----------------------------
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
# Important Search
# -------------------
# Suppose that there are only a few variables are important (i.e. too many noise variables), it may be a vise choice to focus on some important variables in splicing process. This can save a lot of time, especially under a large :math:`p`.
# 
# In abess package, an argument called `important_search` is used for it, which means the size of inactive set for each splicing process. By default, this argument is set as 0, and the total inactive variables would be contained in the inactive set. But if an positive integer is given, the splicing process would focus on active set and the most important `important_search` inactive variables.
# 
# However, after convergence on this subset, we check if the chosen variables are still the most important ones by recomputing on the full set with the new active set. If not, we update the subset and splicing again. On our testing, it would not iterate many time to reach a stable subset. After that, the active set on the stable subset would be treated as that on the full set.
# 
# Here we take `LogisticRegression` for an example. 



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
# ![](./fig/impsearch.png)
# 
# At a low level of `important_search`, however, the performance (AUC) has been very good. In this situation, a lower `important_search` can save lots of time and space.

###############################################################################
# R tutorial
# -------------
#
# For R tutorial, please view [https://abess-team.github.io/abess/articles/v09-fasterSetting.html](https://abess-team.github.io/abess/articles/v09-fasterSetting.html).
