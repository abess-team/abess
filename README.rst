Welcome to abess's documentation!
==========================================================================

.. raw:: html

   <!-- badges: start -->

|Codacy| |Travis build status| |codecov| |docs| |cran| |pypi| |pyversions| |License|

**abess** (Adaptive BEst Subset Selection) aims to find a small subset of predictors such
that the resulting linear model is expected to have the most desirable
prediction accuracy. This project implements a polynomial algorithm proposed to solve these problems. It supports:

-  linear regression
-  classification (binary or multi-class)
-  counting-response modeling
-  censored-response modeling
-  multi-response modeling (multi-tasks learning)
-  group best subset selection
-  nuisance penalized regression
-  sure independence screening

Installation
============

To install abess, please view chapter: `Installation`_.

.. _Installation: https://abess.readthedocs.io/en/latest/Installation.html


Performance
===========

To show the computational efficiency of abess, 
we compare abess R package with popular R libraries: glmnet, ncvreg, picasso for linear, logistic and poisson regressions; 
Timings of the CPU execution are recorded in seconds and averaged over 100 replications on a sequence
of 100 regularization parameters.

The designed matrix is formed by i.i.d sample generated from a multivariate normal distribution with mean 0 and covariance matrix :math:`\Sigma = (\sigma_{ij})`. We consider two settings—low correlation and high correlation. 
For the low correlation scenario, we set :math:`\sigma_{ij} = 0.1^{|i-j|}` and for the high correlation :math:`\sigma_{ij} = 0.7`. The number of predictors is 1000. 
The true coefficient :math:`\beta^*` is a vector with 10 nonzero entries uniformly distributed in :math:`[b,B]`. We set :math:`b=5\sqrt{2\log(p)/n}`, :math:`B = 100b` for linear regression :math:`b = 10\sqrt{2\log(p)/n}`, :math:`B = 5*b` for 
logistic regression and :math:`b = -10 \sqrt{2  \log(p) / n}`, :math:`B=10 \sqrt{2 \log(p) / n}` for poisson regression. A random noise generated from a standard Gaussian distribution is added to the linear predictor :math:`x^\prime\beta` for linear regression. 
The size of training data is 500.

All experiments are evaluated on an Intel(R) Xeon(R) 
CPU E5-2620 v4 @ 2.10GHz and under R version 3.6.1. 
for 100 replicas.

As a package solving best subset selection, abess reaches a high efficient performance especially in linear regression where it gives the fastest solution.

.. image:: ./Tutorial/fig/readmeTiming.png


Reference
=========

| Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang. A polynomial algorithm for best-subset selection problem. Proceedings of the National Academy of Sciences, 117(52):33117-33123, 2020.

| Fan, J. and Lv, J. (2008), Sure independence screening for ultrahigh dimensional feature space. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 70: 849-911. https://doi.org/10.1111/j.1467-9868.2008.00674.x

| Qiang Sun & Heping Zhang (2020) Targeted Inference Involving High-Dimensional Data Using Nuisance Penalized Regression, Journal of the American Statistical Association, DOI: 10.1080/01621459.2020.1737079

| Zhang, Y., Zhu, J., Zhu, J. and Wang, X., 2021. Certifiably Polynomial Algorithm for Best Group Subset Selection. arXiv preprint arXiv:2104.12576.

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/3f6e60a3a3e44699a033159633981b76 
   :target: https://www.codacy.com/gh/abess-team/abess/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=abess-team/abess&amp;utm_campaign=Badge_Grade
.. |Travis build status| image:: https://travis-ci.com/abess-team/abess.svg?branch=master
   :target: https://travis-ci.com/abess-team/abess
.. |codecov| image:: https://codecov.io/gh/abess-team/abess/branch/master/graph/badge.svg?token=LK56LHXV00
   :target: https://codecov.io/gh/abess-team/abess
.. |docs| image:: https://readthedocs.org/projects/abess/badge/?version=latest
   :target: https://abess.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |cran| image:: https://img.shields.io/cran/v/abess?logo=R
   :target: https://cran.r-project.org/package=abess
.. |pypi| image:: https://badge.fury.io/py/abess.svg
   :target: https://badge.fury.io/py/abess
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/abess
.. |License| image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg 
   :target: http://www.gnu.org/licenses/gpl-3.0

    
