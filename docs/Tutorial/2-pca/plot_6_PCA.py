"""
Principal Component Analysis
============================
This notebook introduces what is adaptive best subset selection principal component analysis (SparsePCA) and uses a real data example to show how to use it.
"""

###############################################################################
# PCA
# ---
# Principal component analysis (PCA) is an important method in the field of data science,
# which can reduce the dimension of data and simplify our model. It actually solves an optimization problem like:
#
# .. math::
#     \max_{v} v^{\top}\Sigma v,\qquad s.t.\quad v^Tv=1.
#
#
# where :math:`\Sigma = X^TX / (n-1)` and :math:`X` is the **centered** sample matrix.
# We also denote that :math:`X` is a :math:`n\times p` matrix, where each row is an observation and each column is a variable.
#
# Then, before further analysis, we can project :math:`X` to :math:`v` (thus dimension reduction),
# without losing too much information.
#
# However, consider that:
#
# - The PC is a linear combination of all primary variables (:math:`Xv`),
#   but sometimes we may tend to use less variables for clearer interpretation
#   (and less computational complexity);
#
# - It has been proved that if :math:`p/n` does not converge to :math:`0`,
#   the classical PCA is not consistent, but this would happen in some high-dimensional data analysis.
#
# For example, in gene analysis, the dataset may contain plenty of genes (variables)
# and we would like to find a subset of them, which can explain most information.
# Compared with using all genes, this small subset may perform better on interpretation,
# without losing much information. Then we can focus on these variables in further analysis.
#
# When we are trapped by these problems, a classical PCA may not be a best choice, since it uses all variables.
# One of the alternatives is `SparsePCA`, which is able to seek for principal component with a sparsity limitation:
#
# .. math::
#     \max_{v} v^{\top}\Sigma v,\qquad s.t.\quad v^Tv=1,\ ||v||_0\leq s.
#
#
# where :math:`s` is a non-negative integer, which indicates how many primary variables are used in principal component.
# With `SparsePCA`, we can search for the best subset of variables to form principal component and
# it retains consistency even under :math:`p>>n`.
# And we make two remarks:
#
# - Clearly, if :math:`s` is equal or larger than the number of primary variables,
#   this sparsity limitation is actually useless, so the problem is equivalent to a classical PCA.
#
# - With less variables, the PC must have lower explained variance.
#   However, this decrease is slight if we choose a good :math:`s` and at this price,
#   we can interpret the PC much better. It is worthy.
#
# In the next section, we will show how to perform `SparsePCA`.
#
# Real Data Example (Communities and Crime Dataset)
# -------------------------------------------------
#
# Here we will use real data analysis to show how to perform `SparsePCA`. The data we use is from
# `UCI: Communities and Crime Data Set <https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime>`__
# and we pick up its 99 predictive variables as our samples.
#
# Firstly, we read the data and pick up those variables we are interested in.


import matplotlib.pyplot as plt
import numpy as np
from abess.decomposition import SparsePCA

X = np.genfromtxt('communities.data', delimiter=',')
X = X[:, 5:127]                         # numeric predictiors
X = X[:, ~np.isnan(X).any(axis=0)]    # drop variables with nan

n, p = X.shape
print(n)
print(p)

###############################################################################
# Model fitting
# ^^^^^^^^^^^^^
# To build an SparsePCA model, we need to give the target sparisty to its
# `support_size` argument. Our program supports adaptively finding a best sparisty in a given range.
#
# Fixed sparsity
# """"""""""""""
# If we only focus on one fixed sparsity, we can simply give a single
# integer to fit on this situation. And then the fitted sparse principal
# component is stored in `SparsePCA.coef_`:


model = SparsePCA(support_size=20)

# %%
# Give either :math:`X` or :math:`\Sigma` to `model.fit()`, the fitting
# process will start. The argument `is_normal = False` here means that the
# program will not normalize :math:`X`. Note that if both :math:`X` and
# :math:`Sigma` are given, the program prefers to use :math:`X`.


model.fit(X=X, is_normal=False)
# model.fit(Sigma = np.cov(X.T))

# %%
# After fitting, `model.coef_` returns the sparse principal component and
# its non-zero positions correspond to variables used.


temp = np.nonzero(model.coef_)[0]
print('sparsity: ', temp.size)
print('non-zero position: \n', temp)
print(model.coef_.T)

###############################################################################
# Adaptive sparsity
# """""""""""""""""
# What's more, **abess** also support a range of sparsity and adaptively choose the best-explain one.
# However, usually a higher sparsity level would lead to better explaination.
#
# Now, we need to build an :math:`s_{max} \times 1` binomial matrix, where
# :math:`s_{max}` indicates the max target sparsity and each row indicates
# one sparsity level (i.e. start from :math:`1`, till :math:`s_{max}`).
# For each position with :math:`1`, **abess** would try to fit the model
# under that sparsity and finally give the best one.


# fit sparsity from 1 to 20
support_size = np.ones((20, 1))
# build model
model = SparsePCA(support_size=support_size)
model.fit(X, is_normal=False)
# results
temp = np.nonzero(model.coef_)[0]
print('chosen sparsity: ', temp.size)
print('non-zero position: \n', temp)
print(model.coef_.T)

# %%
# Because of warm-start, the results here may not be the same as fixed sparsity.
#
# Then, the explained variance can be computed by:


Xc = X - X.mean(axis=0)
Xv = Xc @ model.coef_
explained = Xv.T @ Xv                   # explained variance (information)
total = sum(np.diag(Xc.T @ Xc))         # total variance (information)
print('explained ratio: ', explained / total)

###############################################################################
# More on the results
# ^^^^^^^^^^^^^^^^^^^
# We can give different target sparsity (change `s_begin` and `s_end`) to get different sparse loadings.
# Interestingly, we can seek for a smaller sparsity which can explain most of the variance.
#
# In this example, if we try sparsities from :math:`0` to :math:`p`, and
# calculate the ratio of explained variance:


num = 30
i = 0
sparsity = np.linspace(1, p - 1, num, dtype='int')
explain = np.zeros(num)
Xc = X - X.mean(axis=0)
for s in sparsity:
    model = SparsePCA(
        support_size=np.ones((s, 1)),
        exchange_num=int(s),
        max_iter=50
    )
    model.fit(X, is_normal=False)
    Xv = Xc @ model.coef_
    explain[i] = Xv.T @ Xv
    i += 1

print('80%+ : ', sparsity[explain > 0.8 * explain[num - 1]])
print('90%+ : ', sparsity[explain > 0.9 * explain[num - 1]])

# %%
# If we denote the explained ratio from all 99 variables as 100%, the
# curve indicates that at least 31 variables can reach 80% (blue dashed
# line) and 41 variables can reach 90% (red dashed line).


plt.plot(sparsity, explain)
plt.xlabel('Sparsity')
plt.ylabel('Explained variance')

ind = np.where(explain > 0.8 * explain[num - 1])[0][0]
plt.plot([0, sparsity[ind]], [explain[ind], explain[ind]], 'b--')
plt.plot([sparsity[ind], sparsity[ind]], [0, explain[ind]], 'b--')
plt.text(sparsity[ind], 0, str(sparsity[ind]))
plt.text(0, explain[ind], '80%')

ind = np.where(explain > 0.9 * explain[num - 1])[0][0]
plt.plot([0, sparsity[ind]], [explain[ind], explain[ind]], 'r--')
plt.plot([sparsity[ind], sparsity[ind]], [0, explain[ind]], 'r--')
plt.text(sparsity[ind], 0, str(sparsity[ind]))
plt.text(0, explain[ind], '90%')

plt.plot([0, p], [explain[num - 1], explain[num - 1]],
         color='gray', linestyle='--')
plt.text(0, explain[num - 1], '100%')

plt.show()

# %%
# This result shows that using less than half of all 99 variables can be
# close to perfect. For example, if we choose sparsity 31, the used
# variables are:


model = SparsePCA(support_size=31)
model.fit(X, is_normal=False)
temp = np.nonzero(model.coef_)[0]
print('non-zero position: \n', temp)

###############################################################################
# Extension: Group PCA
# --------------------
# Group PCA
# ^^^^^^^^^
# Furthermore, in some situations, some variables may need to be considered together,
# that is, they should be "used" or "unused" for PC at the same time, which we call "group information".
# The optimization problem becomes:
#
# .. math::
#     \max_{v} v^{\top}\Sigma v,\qquad s.t.\quad v^Tv=1,\ \sum_{g=1}^G I(||v_g||\neq 0)\leq s.
#
#
# where we suppose there are :math:`G` groups, and the :math:`g`-th one correspond to :math:`v_g`,
# :math:`v = [v_1^{\top},v_2^{\top},\cdots,v_G^{\top}]^{\top}`. Then we are interested to find :math:`s` (or less) important groups.
#
# Group problem is extraordinarily important in real data analysis. Still take gene analysis as an example,
# several sites would be related to one character, and it is meaningless to consider each of them alone.
#
# `SparsePCA` can also deal with group information. Here we make sure that variables in the same group address close to each other
# (if not, the data should be sorted first).
#
# Simulated Data Example
# ^^^^^^^^^^^^^^^^^^^^^^
# Suppose that the data above have group information like:
#
# - Group 0: {the 1st, 2nd, ..., 6th variable};
#
# - Group 1: {the 7th, 8th, ..., 12th variable};
#
# - ...
#
# - Group 15: {the 91st, 92nd, ..., 96th variable};
#
# - Group 16: {the 97th, 98th, 99th variables}.
#
# Denote different groups as different numbers:


g_info = np.arange(17)
g_info = g_info.repeat(6)
g_info = g_info[0:99]

print(g_info)

# %%
# And fit a group sparse PCA model with additional argument `group=g_info`:


model = SparsePCA(support_size=np.ones((6, 1)))
model.fit(X, group=g_info, is_normal=False)

# %%
# The result comes to:


print(model.coef_.T)

temp = np.nonzero(model.coef_)[0]
temp = np.unique(g_info[temp])

print('non-zero group: \n', temp)
print('chosen sparsity: ', temp.size)

# %%
# Hence we can focus on variables in Group 0, 8, 9, 10, 11, 15.
#
#
# Extension: Multiple principal components
# ----------------------------------------
#
# Multiple principal components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In some cases, we may seek for more than one principal components under sparsity.
# Actually, we can iteratively solve the largest principal component and then mapping
# the covariance matrix to its orthogonal space:
#
# .. math::
#     \Sigma' = (1-vv^{\top})\Sigma(1-vv^{\top})
#
#
# where :math:`\Sigma` is the currect covariance matrix and :math:`v` is its (sparse) principal component.
# We map it into :math:`\Sigma'`, which indicates the orthogonal space of :math:`v`, and then solve the sparse principal component again.
#
# By this iteration process, we can acquire multiple principal components and they are sorted from the largest to the smallest.
# In our program, there is an additional argument `number`, which indicates the number of principal components we need, defaulted by 1.
# Now the `support_size` is shaped in :math:`s_{max}\times \text{number}`
# and each column indicates one principal component.


model = SparsePCA(support_size=np.ones((31, 3)))
model.fit(X, is_normal=False, number=3)
model.coef_.shape

# %%
# Here, each column of the `model.coef_` is a sparse PC (from the largest
# to the smallest), for example the second one is that:


model.coef_[:, 1]

# %%
# If we want to compute the explained variance of them, it is also quite easy:


Xv = Xc.dot(model.coef_)
explained = np.sum(np.diag(Xv.T.dot(Xv)))
print('explained ratio: ', explained / total)

###############################################################################
# R tutorial
# ----------
# For R tutorial, please view
# https://abess-team.github.io/abess/articles/v08-sPCA.html.
