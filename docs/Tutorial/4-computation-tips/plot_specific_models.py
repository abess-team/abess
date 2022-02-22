"""
Specific Models
==============
"""

##########################################
# Introduction
# ^^^^^^^^^^^^^^^^
# From the algorithm preseneted in “ABESS algorithm: details”, 
# one of the bottleneck in algorithm is the computation of forward and backward sacrifices, 
# which requires conducting iterative algorithms or frequently visiting :math:`p` variables. 
# To improve computational efficiency, 
# we designed specialize strategies for computing forward and backward sacrifices for different models.
# The specialize strategies is roughly divide into two classes: (i) covariance update for (multivariate) linear model;
# (ii) quasi Newton iteration for non-linear model (e.g., logistic regression).
# We going to specify the two strategies as follows.
# 
# Covariance update
# ^^^^^^^^^^^^^^^^
# Under linear model, the core bottleneck is computing backward sacrifices, i.e., 
# 
#
# Quasi Newton iteration 
# ^^^^^^^^^^^^^^^^
# In the fourth step in Algorithm 2, we need to solve a convex optimization problem:
# 
# .. math::
#     \tilde{\beta} = \arg\min_{\text{supp}(\beta) = \tilde{\mathcal{A}} }  l_n(\beta ).
# 
# 
# But generally, it has no closed-form solution, and has to be solved via iterative algorithm.
# A natural method for solving this problem is Netwon method, i.e.,
# conduct the update:
# 
# .. math::
#     \beta_{\tilde{\mathcal{A}} }^{m+1} \leftarrow \boldsymbol  \beta_{\tilde{\mathcal{A}} }^m - \Big( \left.\frac{\partial^2 l_n( \boldsymbol  \beta )}{ (\partial \boldsymbol  \beta_{\tilde{\mathcal{A}}}  )^2 }\right|_{\boldsymbol  \beta = \boldsymbol  \beta^m} \Big)^{-1} \Big( \left.\frac{\partial  l_n( \boldsymbol  \beta )}{  \partial \boldsymbol  \beta_{\tilde{\mathcal{A}}}    }\right|_{\boldsymbol  \beta = \boldsymbol  \beta^m}  \Big),
# 
# 
# until :math:`\| \beta_{\tilde{\mathcal{A}} }^{m+1} - \beta_{\tilde{\mathcal{A}} }^{m}\|_2 \leq \epsilon` or :math:`m \geq k`,
# where :math:`\epsilon, k` are two user-specific parameters. 
# Generally, setting :math:`\epsilon = 10^{-6}` and :math:`k = 80` achieves desirable estimation.
# Generally, the inverse of second derivative is computationally intensive, and thus,
# we approximate it with its diagonalized version. Then, the update formulate changes to:
# 
# .. math::
#     \beta_{\tilde{\mathcal{A}} }^{m+1} \leftarrow \boldsymbol  \beta_{\tilde{\mathcal{A}} }^m -  \rho D \Big( \left.\frac{\partial  l_n( \boldsymbol  \beta )}{  \partial \boldsymbol  \beta_{\tilde{\mathcal{A}}}    }\right|_{\boldsymbol  \beta = \boldsymbol  \beta^m}  \Big),
# 
# 
# where :math:`D = \textup{diag}( (\left.\frac{\partial^2 l_n( \boldsymbol  \beta )}{ (\partial \boldsymbol  \beta_{\tilde{\mathcal{A}_{1}}}  )^2 }\right|_{\boldsymbol  \beta = \boldsymbol  \beta^m} )^{-1}, \ldots, (\left.\frac{\partial^2 l_n( \boldsymbol  \beta )}{ (\partial \boldsymbol  \beta_{\tilde{\mathcal{A}}_{|A|}}  )^2 }\right|_{\boldsymbol  \beta = \boldsymbol  \beta^m} )^{-1})`
# and :math:`\rho`` is step size.
# Although using the approximation may increase the iteration time,
# it avoids a large computational complexity when computing the matrix inversion.
# Furthermore, we use a heuristic strategy to reduce the iteration time.
# Observing that not every new support after exchanging the elements in active set and inactive set
# may not reduce the loss function,
# we can early stop the newton iteration on these support.
# Specifically, support :math:`l_1 = L({\beta}^{m}), l_2 = L({\beta}^{m+1})`,
# if :math:`l_1 - (k - m - 1) \times (l_2 - l_1)) > L - \tau`,
# then we can expect the new support cannot lead to a better loss after :math:`k` iteration,
# and hence, it is no need to conduct the remaining :math:`k - m - 1` times Newton update.
# This heuristic strategy is motivated by the convergence rate of Netwon method is linear at least.
# |image0|
# 
# 
# 
# The ``abess`` R package also supports covariance update and quasi Newton iteration. 
# For R tutorial, please view https://abess-team.github.io/abess/articles/v09-fasterSetting.html
# 
# .. |image0| image:: ../../Tutorial/figure/convergence_rates.png