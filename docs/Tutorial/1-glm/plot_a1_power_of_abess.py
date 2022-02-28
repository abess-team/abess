"""
==========================
Power of **abess** Library
==========================

"""

######################################
# Introduction
# ^^^^^^^^^^^^
# In this part, we are going to explore the power of the ``abess`` package
# using simulated data. We compare the abess package with popular Python
# packages:
# `scikit-learn <https://scikit-learn.org/stable/supervised_learning.html#supervised-learning>`__
# for linear and logistic regressions in the following section. Actually,
# we also compare with
# `python-glmnet <https://github.com/civisanalytics/python-glmnet>`__,
# `statsmodels <https://github.com/statsmodels/statsmodels>`__ and
# `L0bnb <https://github.com/alisaab/l0bnb>`__, but the python-glmnet
# presents a poor prediction error, the statsmodels runs slow and the
# L0bnb cannot adaptively choose sparsity level. So their results are not
# showed here.

######################################
# Simulation
# ^^^^^^^^^^
# Setting
# ~~~~~~~
# Both packages are compared in three aspects including the prediction
# performance, the variable selection performance, and the computation
# efficiency.
#
# -  The prediction performance of the linear model is measured by
#    :math:`||y−\hat{y}||_2` on a test set and for logistic regression
#    this is measured by the area under the ROC Curve (AUC).
# -  For the variable selection performance, we compute the coefficient
#    error :math:`||\beta - \hat{\beta}||_2`, true positive rate (TPR,
#    which is the proportion of varibales in the active set that are
#    correctly identified) and the false positive rate (FPR, which is the
#    proportion of the varibales in the inactive set that are falsely
#    identified as a signal).
# -  Timings of the CPU execution are recorded in seconds and all the
#    performances are averaged over 20 replications on a sequence of 100
#    regularization parameters.
#
# The simulated data are made by ``abess.datasets.make_glm_data()``. The
# number of predictors is :math:`p=8000` and the size of data is
# :math:`n=500`. The true coefficient contains :math:`k=10` nonzero
# entries uniformly distributed in :math:`[b,B]`. For linear (gaussian)
# data, we set :math:`b = 5\sqrt{2\ln p / n}` and :math:`B = 100b`. For
# logistic (binomial) data, we set :math:`b = 10\sqrt{2\ln p / n}` and
# :math:`B = 5b`. In each regression, we test for both low
# (:math:`\rho=0.1`) and high correlation (:math:`\rho=0.7`) scenarios.
# What’s more, a random noise generated from a standard Gaussian
# distribution is added to the linear predictor :math:`x′β` for linear
# regression.
#
# All experiments are evaluated on a Ubuntu platform with Intel(R)
# Core(TM) i9-9940X CPU @ 3.30GHz and 48 RAM.
#
# .. code:: bash
#
#    $ python abess/docs/simulation/Python/perform.py

##############################################################
# Results
# ~~~~~~~
#
# For linear regression, we compare three methods in the two packages:
# Lasso, OMP and abess. For logistic regression, we compare two
# methods: lasso and abess.
# The results are presented in the following pictures. The first column is
# the result of linear regression and the second one is of logistic
# regression.
#
# -  Firstly, among all of the methods implemented in different packages,
#    the estimator obtained by the abess package shows both the best
#    prediction performance and the best coefficient error.
#
# -  Secondly, the estimator obtained by the abess package can reasonably
#    control FPR in a low level while the TPR stays at 1. (Since all
#    methods’ TPR are 1, the figure is not plotted.)
#
# -  Furthermore, our abess package is highly efficient compared with
#    other packages, especially in the linear regression.
#
# |image0|
# |image1|
#
# For ``abess`` R library's performance, please view
# https://abess-team.github.io/abess/articles/v11-power-of-abess.html.
#
# .. |image0| image:: ../../Tutorial/figure/perform.png
# .. |image1| image:: ../../Tutorial/figure/timings.png

# sphinx_gallery_thumbnail_path = '_static/timings.png'
