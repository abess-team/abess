abess: An R & Python package for Best-Subset Selection in Polynomial Time
==========================================================================

.. raw:: html

   <!-- badges: start -->

|Travis build status| |codecov| |docs|

Best-subset selection aims to find a small subset of predictors such
that the resulting linear model is expected to have the most desirable
prediction accuracy. This project implements a polynomial algorithm
proposed by Zhu et al (2020) to solve the problem. It supports:

-  various model:
-  linear regression
-  classification (binary or multi-class)
-  counting-response modeling
-  censored-response modeling
-  multi-response modeling (multi-tasks learning)
-  sure independence screening
-  nuisance penalized regression

Installation
============

You can install the newest version of abess from
`pypi <https://pypi.org>`__ with:

.. code-block:: console

    $pip install abess

Reference
=========

| A polynomial algorithm for best-subset selection problem. Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang. Proceedings of the National Academy of Sciences Dec 2020, 117 (52) 33117-33123; DOI:10.1073/pnas.2014241117

| Fan, J. and Lv, J. (2008), Sure independence screening for ultrahigh dimensional feature space. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 70: 849-911. https://doi.org/10.1111/j.1467-9868.2008.00674.x

| Qiang Sun & Heping Zhang (2020) Targeted Inference Involving High-Dimensional Data Using Nuisance Penalized Regression, Journal of the American Statistical Association, DOI: 10.1080/01621459.2020.1737079

.. |Travis build status| image:: https://travis-ci.com/abess-team/abess.svg?branch=master
   :target: https://travis-ci.com/abess-team/abess
.. |codecov| image:: https://codecov.io/gh/abess-team/abess/branch/master/graph/badge.svg?token=LK56LHXV00
   :target: https://codecov.io/gh/abess-team/abess
.. |docs| image:: https://readthedocs.org/projects/abess-test/badge/?version=master
   :target: https://abess-test.readthedocs.io/en/master/?badge=master
   :alt: Documentation Status

